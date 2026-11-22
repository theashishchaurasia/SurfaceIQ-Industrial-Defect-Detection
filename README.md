# **Industrial Surface Defect Inspection Platform**

I am developing a **Python-only, CPU-friendly Deep Learning project** for my B.Tech PBL, focused on identifying six types of steel surface defects using the NEU-DET dataset.

**Project status:** I have packaged the source code and tests. I have not performed model training yet because my dataset is stored locally on my PC and is not included in the package. I have not fabricated any performance metrics. YOLO training requires valid bounding-box annotations, while U-Net training requires pixel-level segmentation masks.

I have chosen Python exclusively for implementation. However, I understand that this does not fulfil the C/C++/Java-specific requirements mentioned in the official PBL rubric, so I will discuss this deviation with my professor.

## **Quick start (PowerShell)**

I will follow these steps to set up and run the project:

```
cd .\Industrial-Defect-Detection
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
# Edit config.yaml: dataset.root = absolute path to my NEU-DET folder
python -m src.cli inspect-data
python -m src.cli train --model cnn --epochs 10
python -m src.cli train --model resnet18 --epochs 8
python -m src.cli predict --image "C:/path/to/image.jpg" --model cnn
streamlit run app/streamlit_app.py
```

Alternatively, I can run `run_app.bat` after installing the dependencies.

## **Dataset setup**

I will configure `config.yaml` and set `dataset.root` to the absolute path of my existing `datasets/raw/NEU-DET` folder.

The project supports recursive discovery of JPG, JPEG, PNG and BMP images, recognized defect-class folder names, and Pascal VOC XML object annotations.

After running `inspect-data`, I will review:

- `results/dataset_summary.json`
- `dataset_manifest.csv`
- `class_distribution.png`

The project also checks file hashes and duplicate images. Unknown labels are not silently guessed, and my original dataset remains unchanged.

The six defect classes I am working with are:

1. Crazing
2. Inclusion
3. Patches
4. Pitted Surface
5. Rolled-in Scale
6. Scratches

## **Implemented project components**

- **Dataset processing:** I have implemented dataset discovery, validation, duplicate checking, hashing and class-distribution analysis. I need to run these on my local dataset.
- **Defect classification:** I have implemented a custom CNN and ResNet18 classifier, along with training, checkpoint saving and prediction functionality.
- **Explainable AI:** I have implemented Grad-CAM to visualize the image regions influencing a model's prediction.
- **Web dashboard:** I have implemented a Streamlit interface for image uploads and prediction reports. It requires a trained checkpoint.
- **Defect detection:** I have implemented a YOLO wrapper and Pascal VOC annotation audit. Training requires valid bounding boxes and a YOLO-format dataset YAML file.
- **Defect segmentation:** I have implemented a U-Net model and paired-mask dataset pipeline. Training is pending suitable pixel-level masks.
- **Deep Learning fundamentals:** I have included tensor operations, SVD, optimizer comparisons, learning paradigms, VAE and activation demonstrations.
- **Transformer-based report encoder:** I have included an optional implementation that requires the `transformers` package and a pretrained model download.
- **GAN:** I have deliberately deferred GAN training rather than claiming an unvalidated feature as complete.

## **Segmentation requirements**

I understand that NEU-DET bounding boxes are not the same as segmentation masks.

For U-Net training, I need one pixel-level mask for each image, with the same filename stem and PNG format. Background pixels must be labelled `0`, while defect pixels must have nonzero values. I will configure the mask directory using `dataset.segmentation_masks`.

## **Compatibility and hardware**

I recommend Python 3.11 or 3.12 for more predictable Windows package compatibility. Python 3.13 support varies across packages.

I will use a clean project virtual environment rather than relying on an existing environment with potentially mismatched PyTorch and torchvision versions.

My project supports CPU execution. For heavier GPU experiments, I can refer to `documentation/installation.md` and use a suitable GPU environment.

## **Available CLI commands**

I can use the following commands to operate the project:

```
python -m src.cli inspect-data
python -m src.cli train --model cnn --epochs 10
python -m src.cli train --model resnet18 --epochs 8
python -m src.cli predict --image sample.jpg --model cnn
python -m src.cli fundamentals
```

No trained weights or dataset are bundled with the project. I will use results from a held-out test split when reporting academic performance. I also understand that softmax confidence does not guarantee a correct prediction, so human review remains important.
