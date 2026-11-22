# Industrial Surface Defect Inspection Platform

Python-only, CPU-friendly B.Tech Deep Learning PBL for the six NEU-DET steel defect classes.

**Honest status:** source code and tests are packaged; no training was performed because your dataset is on your PC and is excluded. No performance metrics are fabricated. YOLO training needs valid bounding boxes; U-Net training needs pixel-level masks. The Python-only choice does not satisfy any C/C++/Java-specific PBL rubric requirement; discuss this deviation with your professor.

## Quick start (PowerShell)
```powershell
cd .\Industrial-Defect-Detection
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
# Edit config.yaml: dataset.root = absolute path to your NEU-DET folder
python -m src.cli inspect-data
python -m src.cli train --model cnn --epochs 10
python -m src.cli train --model resnet18 --epochs 8
python -m src.cli predict --image "C:/path/to/image.jpg" --model cnn
streamlit run app/streamlit_app.py
```
Or run `run_app.bat` after installing dependencies.

## Dataset setup
Edit `config.yaml` → `dataset.root` to point to your existing `datasets/raw/NEU-DET` folder. Recursive discovery supports JPG/JPEG/PNG/BMP, recognized class folder names, and Pascal VOC XML object labels. Run `inspect-data`; review `results/dataset_summary.json`, `dataset_manifest.csv`, and `class_distribution.png`. Unknown labels are not silently guessed. The original dataset is never modified.

Classes: crazing, inclusion, patches, pitted surface, rolled-in scale, scratches.

## Included status
- Dataset discovery, validation, hashes/duplicates, class distribution: implemented; run on local data.
- Custom CNN and ResNet18 classifier, training, checkpoints, prediction: implemented; requires local training.
- Grad-CAM and Streamlit upload/report workflow: implemented; requires trained checkpoint.
- YOLO VOC annotation audit and Ultralytics training/inference wrapper: implemented; valid boxes and YOLO-format dataset YAML required.
- U-Net and paired-mask dataset: implemented; training pending pixel masks.
- Fundamentals, optimizer comparisons, tensor/SVD, learning paradigms, VAE and activation demos: implemented.
- Transformer report encoder: optional, requires `transformers` and pretrained model download.
- GAN training: deliberately deferred rather than adding an unvalidated/fake feature.

NEU-DET boxes are not segmentation masks. Provide one pixel-level mask per image, same stem as the image, PNG format, background=0 and defect pixels nonzero. Configure `dataset.segmentation_masks`.

## Compatibility
Python 3.11 or 3.12 is recommended for predictable Windows wheels. Python 3.13 package support varies. Use a clean project venv; do not rely on an existing environment with mismatched Torch/torchvision. CPU is supported. For GPU experiments, see `documentation/installation.md`.

## CLI
`python -m src.cli inspect-data`
`python -m src.cli train --model cnn --epochs 10`
`python -m src.cli train --model resnet18 --epochs 8`
`python -m src.cli predict --image sample.jpg --model cnn`
`python -m src.cli fundamentals`

No weights or dataset are bundled. Use results from a held-out test split for academic claims. Softmax confidence is not proof of correctness; retain human review.
