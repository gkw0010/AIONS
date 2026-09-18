# AINOS

## Installation

Create the dedicated Conda environment named `AINOS`:

```bash
conda create -n AINOS python=3.9 -y
conda activate AINOS

# Install the CUDA 11.8 PyTorch build (use the matching command for your GPU).
pip install torch==2.2.1 torchvision==0.17.1 \
  --index-url https://download.pytorch.org/whl/cu118

pip install -U openmim
mim install "mmcv==2.1.0" "mmengine==0.10.7" "mmdet==3.3.0"

cd AIONS
pip install -e .
```

The versions above are the versions used to verify image inference. For a
CPU-only machine, install a CPU-compatible PyTorch build instead and use
`--device cpu` when running the demos.

If one of the OpenMMLab dependencies is missing, install it with OpenMIM:

```bash
pip install -U openmim
mim install "mmcv>=2.0.0rc4,<2.2.0" "mmengine>=0.5.0,<1.0.0" "mmdet>=3.0.0,<4.0.0"
```

## Image inference

Put the configuration and checkpoint anywhere on the machine, then run the
batch demo. Each supported image in the input directory is saved to the
output directory with a yellow skeleton overlay.

```bash
python image_inference.py \
  /path/to/model.py \
  /path/to/checkpoint.pth \
  /path/to/images \
  /path/to/results \
  --device cuda:0
```

## Video inference

`video_demo.py` runs the existing real-time camera pipeline: ROI inference,
temporal mask stabilization, arc fitting, and full-screen display.

```bash
python video_demo.py \
  /path/to/model.py \
  /path/to/checkpoint.pth \
  --device cuda:0 \
  --camera-id 0
```

Press `q` or `Esc` to stop the camera demo. The camera crop and display
parameters are defined near the top of `video_demo.py`.
