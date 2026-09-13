from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import time

from mmseg.apis import init_model as init_seg_model
from mmseg.apis import inference_model as inference_seg_model


# =========================
# Model configuration
# =========================
SEG_CONFIG_FILE = None
SEG_CHECKPOINT_FILE = None
DEVICE = 'cuda:0'


# =========================
# Camera configuration
# =========================
CAMERA_ID = 0
CAMERA_W = 1920
CAMERA_H = 1080

# Full-screen display size
SHOW_W = 1920
SHOW_H = 1080

# Crop region (used for inference only)
Y1, Y2 = 36, 1046
X1, X2 = 700, 1860

# Run inference every N frames
INFER_INTERVAL = 4

# FPS update interval
FPS_UPDATE_INTERVAL = 60

# =========================
# Segmentation stabilization parameters
# =========================
THRESHOLD = 0.5
TEMPORAL_ALPHA = 0.16
MIN_AREA = 80
KEEP_ONLY_LARGEST = True

# =========================
# Curve stabilization parameters
# =========================
CURVE_DEGREE = 4
CURVE_SAMPLES = 200
CURVE_BIN_SIZE = 4
CURVE_COEFF_ALPHA = 0.80
LINE_THICKNESS = 5
LINE_COLOR = (0, 255, 255)  # BGR


def keep_largest_component(mask):
    """
    mask: uint8, 0/1
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + np.argmax(areas)

    out = np.zeros_like(mask, dtype=np.uint8)
    out[labels == largest_idx] = 1
    return out


def filter_small_components(mask, min_area=80):
    """
    mask: uint8, 0/1
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask, dtype=np.uint8)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 1

    return out


def get_foreground_probability(result):
    """
    Extract the foreground probability map from an MMSeg result
    Defaults:
      - For two or more channels, class 1 is foreground
      - For one channel, apply sigmoid
    Return float32 [H, W] values in the range 0 to 1
    """
    score = result.seg_logits.data  # [C, H, W]

    if score.ndim != 3:
        raise ValueError(f"Unexpected seg_logits.data dimensions: {score.shape}")

    if score.shape[0] >= 2:
        prob = torch.softmax(score, dim=0)
        fg_prob = prob[1]
    else:
        fg_prob = torch.sigmoid(score[0])

    return fg_prob.detach().cpu().numpy().astype(np.float32)


def build_stable_binary_mask(prob_map):
    """
    prob_map: float32 [H, W], 0~1
    Return a stabilized binary mask as uint8 values 0 or 1
    """
    # 1) Spatial smoothing
    prob_smooth = cv2.GaussianBlur(prob_map, (5, 5), 0)

    # 2) Thresholding
    mask = (prob_smooth > THRESHOLD).astype(np.uint8)

    # 3) Morphological denoising
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # 4) Connected components
    if KEEP_ONLY_LARGEST:
        mask = keep_largest_component(mask)
    else:
        mask = filter_small_components(mask, min_area=MIN_AREA)

    return mask


def fit_single_arc_from_mask(mask, prev_curve_state=None):
    """
    Fit a single arc from a binary mask.
    Return:
      curve_pts: ndarray, shape [N,1,2], int32
      curve_state: dict or None
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 20:
        return None, prev_curve_state

    # Determine the dominant direction:
    # Fit x = f(y) when the y span is larger
    # Otherwise fit y = f(x)
    range_y = ys.max() - ys.min()
    range_x = xs.max() - xs.min()
    fit_x_as_func_of_y = (range_y >= range_x)

    if fit_x_as_func_of_y:
        indep = ys.astype(np.float32)
        dep = xs.astype(np.float32)
        mode = 'x=f(y)'
    else:
        indep = xs.astype(np.float32)
        dep = ys.astype(np.float32)
        mode = 'y=f(x)'

    # Aggregate bins to reduce local spikes
    vmin = int(indep.min())
    vmax = int(indep.max())

    indep_samples = []
    dep_samples = []

    for start in range(vmin, vmax + 1, CURVE_BIN_SIZE):
        end = start + CURVE_BIN_SIZE
        sel = (indep >= start) & (indep < end)
        if np.count_nonzero(sel) < 3:
            continue

        indep_bin = indep[sel]
        dep_bin = dep[sel]

        indep_samples.append(float(np.mean(indep_bin)))
        dep_samples.append(float(np.median(dep_bin)))

    indep_samples = np.asarray(indep_samples, dtype=np.float32)
    dep_samples = np.asarray(dep_samples, dtype=np.float32)

    if len(indep_samples) < CURVE_DEGREE + 2:
        return None, prev_curve_state

    # Polynomial fitting
    coeff = np.polyfit(indep_samples, dep_samples, deg=CURVE_DEGREE).astype(np.float32)

    # Temporally smooth curve parameters
    if prev_curve_state is not None and prev_curve_state['mode'] == mode:
        prev_coeff = prev_curve_state['coeff']
        prev_vmin = prev_curve_state['vmin']
        prev_vmax = prev_curve_state['vmax']

        coeff = CURVE_COEFF_ALPHA * prev_coeff + (1.0 - CURVE_COEFF_ALPHA) * coeff
        vmin = int(CURVE_COEFF_ALPHA * prev_vmin + (1.0 - CURVE_COEFF_ALPHA) * vmin)
        vmax = int(CURVE_COEFF_ALPHA * prev_vmax + (1.0 - CURVE_COEFF_ALPHA) * vmax)

    # Sample curve points
    indep_draw = np.linspace(vmin, vmax, CURVE_SAMPLES, dtype=np.float32)
    dep_draw = np.polyval(coeff, indep_draw)

    if mode == 'x=f(y)':
        y_draw = indep_draw
        x_draw = dep_draw
    else:
        x_draw = indep_draw
        y_draw = dep_draw

    # Clip to image bounds
    h, w = mask.shape[:2]
    x_draw = np.clip(x_draw, 0, w - 1)
    y_draw = np.clip(y_draw, 0, h - 1)

    curve_pts = np.stack([x_draw, y_draw], axis=1).astype(np.int32).reshape(-1, 1, 2)

    curve_state = {
        'mode': mode,
        'coeff': coeff,
        'vmin': vmin,
        'vmax': vmax,
    }

    return curve_pts, curve_state


def offset_curve_points(curve_pts, offset_x, offset_y):
    """
    Translate ROI points back to full-image coordinates
    """
    if curve_pts is None:
        return None

    pts = curve_pts.reshape(-1, 2).astype(np.int32).copy()
    pts[:, 0] += offset_x
    pts[:, 1] += offset_y
    return pts.reshape(-1, 1, 2)


def transform_points_to_display(curve_pts, scale, offset_x, offset_y):
    """
    Map full-image points to display-canvas coordinates
    """
    if curve_pts is None:
        return None

    pts = curve_pts.reshape(-1, 2).astype(np.float32).copy()
    pts[:, 0] = pts[:, 0] * scale + offset_x
    pts[:, 1] = pts[:, 1] * scale + offset_y
    return pts.astype(np.int32).reshape(-1, 1, 2)


def main():
    parser = ArgumentParser(description='Run Mask2Former inference on a camera.')
    parser.add_argument('config', nargs='?', default=SEG_CONFIG_FILE,
                        help='Path to the model config file')
    parser.add_argument('checkpoint', nargs='?', default=SEG_CHECKPOINT_FILE,
                        help='Path to the model checkpoint')
    parser.add_argument('--device', default=DEVICE,
                        help='Inference device, for example cuda:0 or cpu')
    parser.add_argument('--camera-id', type=int, default=CAMERA_ID,
                        help='OpenCV camera ID')
    args = parser.parse_args()
    if not args.config or not args.checkpoint:
        parser.error('config and checkpoint are required')

    # 1. Initialize the segmentation model
    seg_model = init_seg_model(args.config, args.checkpoint, device=args.device)

    # 2. Open the external camera
    cap = cv2.VideoCapture(args.camera_id)

    if not cap.isOpened():
        print(f"Unable to open camera: {args.camera_id}")
        return

    # Request a camera resolution of 1920x1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)

    # Optionally request MJPG when supported
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # Read the actual resolution
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Actual camera resolution: {actual_w} x {actual_h}")

    # 3. Create a full-screen window
    window_name = "Segmentation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    frame_idx = 0

    # Caches
    cached_prob_map = None
    cached_curve_pts = None          # Curve in ROI coordinates
    cached_curve_state = None

    # FPS statistics
    display_fps = 0.0
    fps_timer_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read a camera frame")
            break

        frame_idx += 1

        # Prevent out-of-bounds cropping when the frame is too small
        if frame.shape[0] < Y2 or frame.shape[1] < X2:
            print(f"Frame is too small; actual frame size: {frame.shape[1]}x{frame.shape[0]}")
            break

        # Keep the full frame for display
        full_frame = frame.copy()

        # Crop only the ROI for inference
        roi = full_frame[Y1:Y2, X1:X2]

        # Infer every INFER_INTERVAL frames and reuse the previous result otherwise
        need_infer = ((frame_idx - 1) % INFER_INTERVAL == 0) or (cached_curve_pts is None)

        if need_infer:
            # Run segmentation on the ROI
            with torch.inference_mode():
                result = inference_seg_model(seg_model, roi)

            # Foreground probability for the current frame
            fg_prob = get_foreground_probability(result)

            # Temporal smoothing of the probability map
            if cached_prob_map is None:
                smooth_prob = fg_prob
            else:
                smooth_prob = TEMPORAL_ALPHA * cached_prob_map + (1.0 - TEMPORAL_ALPHA) * fg_prob

            cached_prob_map = smooth_prob

            # Stabilize the binary mask
            mask = build_stable_binary_mask(smooth_prob)

            # Fit one arc from the mask
            curve_pts, curve_state = fit_single_arc_from_mask(mask, cached_curve_state)

            # Update caches only after a successful fit
            if curve_pts is not None:
                cached_curve_pts = curve_pts
                cached_curve_state = curve_state

        # =========================
        # Display
        # Display the full frame and draw the ROI result at its original position
        # =========================
        display_frame = full_frame.copy()
        src_h, src_w = display_frame.shape[:2]

        # Keep displaying the cached curve when available
        full_curve_pts = None
        if cached_curve_pts is not None:
            full_curve_pts = offset_curve_points(cached_curve_pts, X1, Y1)

        # Scale the full frame to the display canvas
        scale = min(SHOW_W / float(src_w), SHOW_H / float(src_h))
        new_w = int(src_w * scale)
        new_h = int(src_h * scale)

        resized_frame = cv2.resize(display_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Center the frame on a black canvas
        canvas = np.zeros((SHOW_H, SHOW_W, 3), dtype=np.uint8)
        pad_x = (SHOW_W - new_w) // 2
        pad_y = (SHOW_H - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_frame

        # Map and draw the curve on the display canvas
        if full_curve_pts is not None:
            draw_pts = transform_points_to_display(full_curve_pts, scale, pad_x, pad_y)
            cv2.polylines(
                canvas,
                [draw_pts],
                isClosed=False,
                color=LINE_COLOR,
                thickness=LINE_THICKNESS,
                lineType=cv2.LINE_AA
            )

        # Update FPS
        if frame_idx % FPS_UPDATE_INTERVAL == 0:
            now = time.time()
            elapsed = now - fps_timer_start
            display_fps = FPS_UPDATE_INTERVAL / max(elapsed, 1e-6)
            fps_timer_start = now

        cv2.putText(
            canvas,
            f"FPS: {display_fps:.2f}",
            (200, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            3
        )

        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
