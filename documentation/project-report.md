# SurfaceIQ: Industrial Surface Defect Inspection Platform

## 1. Abstract

SurfaceIQ is a Python-based Deep Learning platform for detecting and analysing defects in industrial steel surfaces. It uses the NEU-DET dataset and combines image classification, object detection, explainable AI and experimental neural-network architectures.

The platform implements custom CNN and ResNet18 classifiers, a trained YOLOv8n detector, Grad-CAM visualisations and a Streamlit dashboard. Additional demonstrations cover neural-network fundamentals, optimisation algorithms, Transformer self-attention and Variational Autoencoders.

The project aims to demonstrate how different Deep Learning concepts can be integrated into an industrial visual inspection application.

## 2. Problem Statement

Manual inspection of industrial metal surfaces can be time-consuming and dependent on human judgement. Surface defects such as cracks, inclusions, patches, pits, rolled-in scale and scratches may affect product quality.

The objective of SurfaceIQ is to develop an educational prototype that can classify and localise visible surface defects and provide visual explanations of model predictions.

## 3. Project Objectives

- Develop an image-classification pipeline for six steel defect categories.
- Compare a custom CNN with ResNet18.
- Implement YOLO-based defect detection.
- Integrate Grad-CAM for model explainability.
- Demonstrate neural-network fundamentals and optimisation techniques.
- Demonstrate Transformer and VAE architectures.
- Develop a Streamlit interface for inspection and result visualisation.
- Evaluate models using appropriate classification and detection metrics.

## 4. Dataset Description

The project uses the NEU-DET steel surface defect dataset.

The six categories are:

1. Crazing
2. Inclusion
3. Patches
4. Pitted Surface
5. Rolled-in Scale
6. Scratches

The dataset contains approximately 1,800 labelled images.

For classification, images are converted to RGB, resized to 200 × 200 pixels, converted into tensors and normalised using a mean and standard deviation of 0.5.

The classification split uses stratified sampling with random seed 42:

| Partition | Images |
|---|---:|
| Training | 1,260 |
| Validation | 270 |
| Test | 270 |
| Total | 1,800 |

The YOLO detection dataset contains 1,440 training images and 360 validation images, with 4,189 annotated defect objects in total.

Pascal VOC XML bounding-box annotations are used for object detection. Pixel-level segmentation masks are not available in the current dataset.

## 5. System Architecture

SurfaceIQ consists of the following components:

- **Data processing:** Image discovery, label validation, preprocessing and dataset preparation.
- **Classification:** Custom CNN and ResNet18.
- **Detection:** YOLOv8n with bounding-box predictions.
- **Explainability:** Grad-CAM activation visualisations.
- **Mathematical sandbox:** Tensor operations, neural networks, backpropagation and optimisation demonstrations.
- **Advanced architectures:** Transformer encoder and VAE experiments.
- **User interface:** Streamlit dashboard.
- **Reporting:** Classification metrics, detection results and CSV inspection reports.

## 6. Implementation Methodology

### 6.1 Classification

Two six-class image classifiers were implemented:

- Custom CNN with convolutional blocks, batch normalisation, ReLU activations and adaptive average pooling.
- ResNet18 with a modified fully connected output layer.

Both models were evaluated on the same 270-image validation partition.

### 6.2 Object Detection

YOLOv8n was trained using the prepared NEU-DET bounding-box dataset.

The detector was trained for 30 epochs using a Google Colab T4 GPU, with an input image size of 320 pixels.

### 6.3 Explainable AI

Grad-CAM was integrated to highlight image regions contributing to classification predictions.

### 6.4 Mathematical Foundations

Four educational notebooks demonstrate:

- Tensor operations, broadcasting, SVD and gradient checking.
- McCulloch-Pitts neurons, sigmoid neurons, MLPs and backpropagation.
- Batch GD, SGD, Momentum, Nesterov optimisation and regularisation.
- Hebbian learning, Oja's rule, competitive learning, error-correction learning and simplified Boltzmann learning.

### 6.5 Transformer Demonstration

A custom PyTorch Transformer encoder demonstrates token embeddings, positional embeddings, four-head self-attention and feed-forward processing.

It produces a 64-dimensional report representation and attention visualisation.

This is an untrained architectural demonstration, not a trained defect-reporting language model.

### 6.6 Variational Autoencoder

A small fully connected VAE was trained on 28 × 28 grayscale steel defect images.

Configuration:

- Latent dimension: 8.
- Training subset: 600 images.
- Validation subset: 180 images.
- Training duration: 5 epochs.
- Optimiser: Adam.

The experiment produces reconstructed images and a two-dimensional PCA visualisation of latent representations.

## 7. Experimental Results

### 7.1 Classification Results

| Metric | Custom CNN | ResNet18 |
|---|---:|---:|
| Validation accuracy | 84.44% | 92.59% |
| Macro F1-score | 84.44% | 92.57% |
| Weighted F1-score | 84.44% | 92.57% |
| Validation images | 270 | 270 |

ResNet18 achieved an 8.15 percentage-point improvement in validation accuracy over the custom CNN.

### 7.2 YOLOv8n Detection Results

| Metric | Result |
|---|---:|
| Precision | 67.17% |
| Recall | 62.67% |
| mAP@50 | 71.17% |
| mAP@50–95 | 39.40% |

### 7.3 YOLO CPU Benchmark

The detector was benchmarked locally using 30 images after a warm-up period.

| Measurement | Result |
|---|---:|
| Hardware | AMD Ryzen 7 7730U CPU |
| Average latency | 44.35 ms |
| Average speed | 22.55 FPS |
| Input size | 320 × 320 |

These figures represent local single-image inference performance, including preprocessing and prediction overhead.

### 7.4 VAE Training

Training loss per image decreased over five epochs:

| Epoch | Loss |
|---|---:|
| 1 | 545.7939 |
| 2 | 539.6878 |
| 3 | 534.5006 |
| 4 | 529.6889 |
| 5 | 524.9749 |

This indicates decreasing training loss during the demonstration. It does not independently establish reconstruction quality or generative performance.

### 7.5 Software Testing

Automated tests were executed using pytest.

**Result: 5 tests passed in 11.27 seconds.**

## 8. Dashboard and User Interface

The Streamlit application provides:

- Overview.
- Dataset Explorer.
- Classification.
- Detection.
- Segmentation status.
- Explainable AI.
- Model Performance.
- Inspection Reports.

Users can upload images, view model predictions, inspect detection results and export inspection reports.

Pixel-level segmentation remains pending because the current dataset does not contain segmentation masks.

## 9. Limitations

- The dataset is relatively small and may not represent real industrial production conditions.
- Lighting, camera and material variations may affect performance.
- Reported classification results are validation results, not independent test-set results.
- Detection performance depends on the quality and coverage of bounding-box annotations.
- Segmentation is not implemented due to the absence of pixel-level masks.
- False-reject rate has not been established because a suitable normal-image evaluation set and acceptance policy are unavailable.
- The Transformer is an untrained architectural demonstration.
- The VAE is a small grayscale reconstruction experiment, not a production-grade defect generator.
- The system has not undergone independent industrial validation or safety certification.

## 10. Academic Deviation

The original project specification mentions C, C++ or Java as implementation languages. The delivered implementation is Python-based.

This is a deviation from the stated language requirement and requires instructor approval where that requirement is mandatory.

The project demonstrates the requested Deep Learning concepts through Python implementations and educational experiments, but this should not be represented as full compliance with a mandatory C/C++/Java requirement.

## 11. Future Scope

- Implement supervised segmentation using a dataset with pixel-level masks.
- Develop and train a defect-report Transformer on a suitable labelled text dataset.
- Explore conditional GAN-based defect synthesis.
- Expand evaluation using independent industrial datasets.
- Perform controlled ablation experiments.
- Improve deployment performance and model calibration.
- Integrate human review and production monitoring.

## 12. Conclusion

SurfaceIQ demonstrates an integrated approach to industrial steel surface defect inspection using Deep Learning.

The project combines classification, object detection, explainability, mathematical neural-network demonstrations, Transformer architecture exploration and VAE-based image reconstruction.

ResNet18 achieved 92.59% validation accuracy, while YOLOv8n achieved 71.17% mAP@50 and a measured local CPU inference speed of 22.55 FPS.

The completed prototype provides a practical academic demonstration of how Deep Learning techniques can be applied to industrial surface inspection, while documenting the limitations and areas requiring further development.