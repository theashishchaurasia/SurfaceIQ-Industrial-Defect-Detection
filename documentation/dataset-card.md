
# SurfaceIQ – Dataset Card

## 1. Dataset Overview

**Dataset:** NEU Surface Defect Database (NEU-DET)  
**Domain:** Industrial steel surface inspection  
**Task:** Defect classification and object detection  
**Number of classes:** 6  
**Image type:** Steel surface images  
**Annotation format:** Pascal VOC XML bounding boxes

The dataset is used for academic experimentation and evaluation of deep learning models for steel surface defect inspection.

The dataset was obtained and stored locally. It is not redistributed with this project.

## 2. Defect Categories

| Class | Description |
|---|---|
| Crazing | Fine network-like surface cracks |
| Inclusion | Foreign material or inclusions |
| Patches | Irregular patch-like surface defects |
| Pitted Surface | Small pits or depressions |
| Rolled-in Scale | Scale embedded during rolling |
| Scratches | Linear surface marks |

## 3. Dataset Organisation

The classification pipeline discovers images and labels using folder names or Pascal VOC XML annotations.

The classification dataset contains approximately 1,800 labelled images.

The YOLO detection preparation produced:

| Partition | Images | Annotated objects |
|---|---:|---:|
| Training | 1,440 | 3,335 |
| Validation | 360 | 854 |
| Total | 1,800 | 4,189 |

The detection annotations include multiple defects in some images.

## 4. Classification Data Split

The classification pipeline uses stratified splitting with random seed 42.

| Partition | Fraction | Images |
|---|---:|---:|
| Training | 70% | 1,260 |
| Validation | 15% | 270 |
| Test | 15% | 270 |
| Total | 100% | 1,800 |

The classifier metrics in the model card are calculated on the 270-image validation partition.

The test partition has not been used to claim performance results.

## 5. Preprocessing

### Classification

1. Load image.
2. Convert to RGB.
3. Resize to 200 × 200 pixels.
4. Convert to PyTorch tensor.
5. Normalise each channel using mean 0.5 and standard deviation 0.5.

### Detection

1. Convert Pascal VOC XML annotations into YOLO bounding-box format.
2. Prepare training and validation image directories.
3. Resize model inputs to 320 × 320 during YOLO evaluation and inference.

### VAE Demonstration

Images are converted to grayscale and resized to 28 × 28 pixels.

## 6. Annotation Format

The classification pipeline uses defect labels from folder names or XML object annotations.

The detection pipeline uses Pascal VOC bounding boxes.

Bounding boxes describe rectangular defect locations. They do not provide pixel-level segmentation masks.

Consequently, this dataset cannot directly support supervised semantic segmentation without additional mask annotations.

## 7. Quality Checks

The project includes dataset inspection utilities for:

- Corrupt image detection.
- Unknown class labels.
- Duplicate image hashes.
- Class distribution.
- XML annotation validation.
- Bounding-box coordinate validation.

## 8. Known Risks and Limitations

- Dataset size is limited compared with real industrial production datasets.
- Class imbalance may affect model performance.
- Image acquisition conditions may be correlated with defect categories.
- Lighting, camera and material changes may cause domain shift.
- Near-duplicate images and acquisition correlations require further auditing.
- No independent external industrial dataset has been used.
- The dataset does not provide pixel-level segmentation masks.

## 9. Ethical and Usage Considerations

The dataset is intended for academic and research experimentation.

Users should verify the dataset's original licence and redistribution permissions before sharing images or annotations.

Model predictions must not be treated as final industrial quality-control decisions without human review and independent validation.

## 10. Intended Applications

- Academic deep learning projects.
- Industrial surface defect classification research.
- Object detection experiments.
- Explainable AI demonstrations.
- Comparison of CNN, ResNet and YOLO architectures.
