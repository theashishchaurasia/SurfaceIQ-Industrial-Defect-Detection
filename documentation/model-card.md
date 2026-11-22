
# SurfaceIQ – Model Card

## 1. Project Overview

**Project:** SurfaceIQ – Industrial Surface Defect Inspection Platform  
**Domain:** Deep Learning / Computer Vision  
**Framework:** Python, PyTorch, Torchvision, Ultralytics YOLO, Streamlit  
**Dataset:** NEU-DET  
**Purpose:** Educational research and steel surface defect inspection demonstrations.

The platform combines image classification, object detection, explainable AI, neural-network fundamentals and experimental generative architectures.

It is not intended for autonomous safety-critical industrial decisions.

## 2. Dataset and Classes

The project uses six NEU-DET defect categories:

1. Crazing
2. Inclusion
3. Patches
4. Pitted Surface
5. Rolled-in Scale
6. Scratches

The classification pipeline uses a 70% training, 15% validation and 15% test split, with stratification and random seed 42.

The classifier evaluation reported here uses the 270-image validation partition.

The YOLO pipeline uses its separately prepared detection dataset, with 1,440 training images and 360 validation images.

## 3. Classification Models

### Custom CNN

- Architecture: Three convolutional feature-extraction blocks.
- Batch normalization and ReLU activations.
- Adaptive average pooling.
- Fully connected six-class output layer.
- Input size: 200 × 200 RGB.

**Validation results:**

| Metric | Result |
|---|---:|
| Accuracy | 84.44% |
| Macro F1 | 84.44% |
| Weighted F1 | 84.44% |
| Validation images | 270 |

### ResNet18

- Architecture: Torchvision ResNet18.
- Final fully connected layer modified for six classes.
- Input size: 200 × 200 RGB.
- Task-specific training performed from randomly initialized weights.

**Validation results:**

| Metric | Result |
|---|---:|
| Accuracy | 92.59% |
| Macro F1 | 92.57% |
| Weighted F1 | 92.57% |
| Validation images | 270 |

ResNet18 achieved 8.15 percentage points higher validation accuracy than the custom CNN.

These are validation results, not independent test-set results.

## 4. Object Detection Model

### YOLOv8n

- Architecture: YOLOv8 nano.
- Task: Multi-object steel defect detection.
- Input size: 320 × 320.
- Dataset: Prepared NEU-DET bounding-box annotations.
- Training: 30 epochs on Google Colab T4 GPU.

**Validation results:**

| Metric | Result |
|---|---:|
| Precision | 67.17% |
| Recall | 62.67% |
| mAP@50 | 71.17% |
| mAP@50–95 | 39.40% |

### Local CPU Benchmark

| Measurement | Result |
|---|---:|
| Hardware | Ryzen 7 7730U CPU |
| Images benchmarked | 30 |
| Average latency | 44.35 ms |
| Average speed | 22.55 FPS |
| Input size | 320 × 320 |

FPS is a local single-image inference benchmark, including preprocessing and prediction overhead. It is not a production throughput measurement.

## 5. Explainability

Grad-CAM is used to generate class activation visualisations for classifier predictions.

These heatmaps provide an indication of image regions influencing a prediction. They do not guarantee causal explanations or model correctness.

## 6. Advanced Architecture Demonstrations

### Transformer Report Encoder

- Custom PyTorch Transformer encoder.
- Token embeddings and positional embeddings.
- Multi-head self-attention with four heads.
- 64-dimensional representation.
- Attention visualisation generated.

Status: Untrained educational architecture demonstration. It is not a trained language model or production report generator.

### Variational Autoencoder (VAE)

- Model: TinyVAE.
- Input: 28 × 28 grayscale images.
- Latent dimension: 8.
- Training: 5 epochs.
- Training subset: 600 images.
- Validation subset: 180 images.

Outputs include image reconstructions, latent-space visualisation and validation loss.

Status: Trained educational reconstruction experiment, not a full-resolution industrial defect generator.

## 7. Mathematical Foundations

The project includes educational implementations and demonstrations of:

- Tensor operations and Singular Value Decomposition.
- McCulloch-Pitts and sigmoid neurons.
- Multilayer perceptrons and backpropagation.
- Batch Gradient Descent and Stochastic Gradient Descent.
- Momentum and Nesterov optimisation.
- Hebbian, Oja, competitive and error-correction learning.
- Simplified Boltzmann learning.

## 8. Evaluation and Testing

- Classification accuracy, precision, recall and F1.
- Confusion matrices and per-class reports.
- YOLO precision, recall and mAP.
- Local CPU inference latency and FPS.
- Automated project tests: 5 passed.

No independent test-set performance is claimed in this model card.

## 9. Limitations

- Dataset size and category coverage are limited.
- Lighting, camera and production-environment shifts may reduce performance.
- Segmentation is not implemented because pixel-level masks are unavailable in the current dataset.
- False-reject rate has not been established because a suitable normal-image evaluation set and acceptance policy are unavailable.
- Transformer demonstration is untrained.
- VAE is a small grayscale educational experiment.
- No production deployment or safety certification has been performed.

## 10. Intended Use

SurfaceIQ is intended for:

- Academic demonstrations.
- Deep Learning experimentation.
- Steel surface defect classification and detection research.
- Model comparison and explainability demonstrations.

Predictions should be reviewed by a human operator before any real-world decision.

