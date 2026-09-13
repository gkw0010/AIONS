"""Run Mask2Former inference on every image in a directory."""

from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from mmengine.model import revert_sync_batchnorm
from mmseg.apis import inference_model, init_model
from skimage.morphology import dilation, disk, skeletonize


IMAGE_EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('config', help='Path to the model config file')
    parser.add_argument('checkpoint', help='Path to the model checkpoint')
    parser.add_argument('input_dir', help='Directory containing input images')
    parser.add_argument('output_dir', help='Directory for rendered images')
    parser.add_argument('--device', default='cuda:0',
                        help='Inference device, for example cuda:0 or cpu')
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f'Input directory does not exist: {input_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)

    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    structuring_element = disk(5)
    image_paths = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f'Failed to read image: {image_path}')
            continue

        result = inference_model(model, str(image_path))
        prediction = result.pred_sem_seg.data.cpu().numpy()[0]
        skeleton = skeletonize(prediction > 0)
        mask = dilation(skeleton, structuring_element).astype(np.uint8)

        overlay = np.zeros_like(image)
        overlay[mask > 0] = (0, 255, 255)
        rendered = cv2.addWeighted(image, 1.0, overlay, 1.0, 0)
        output_path = output_dir / image_path.name
        if not cv2.imwrite(str(output_path), rendered):
            print(f'Failed to write image: {output_path}')
        else:
            print(f'Saved: {output_path}')


if __name__ == '__main__':
    main()
