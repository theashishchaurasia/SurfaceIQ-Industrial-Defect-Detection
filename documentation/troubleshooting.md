# Troubleshooting

- No images: verify dataset.root points to the image-containing folder.
- Unknown labels: check folder names or Pascal VOC XML object names.
- Torch mismatch: reinstall a compatible torch/torchvision pair in the venv.
- Missing checkpoint: train before inference.
- YOLO failure: verify Ultralytics and normalized YOLO labels/YAML.
- No U-Net pairs: add real same-stem pixel-level PNG masks.
- Slow CPU: reduce image size, batch size and epochs; optionally use Colab GPU.
