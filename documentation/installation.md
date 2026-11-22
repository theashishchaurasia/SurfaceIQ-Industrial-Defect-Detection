
# SurfaceIQ – Installation and Setup Guide

## 1. System Requirements

Recommended environment:

| Component | Requirement |
|---|---|
| Operating System | Windows 10/11 |
| Python | 3.11 or 3.12 recommended |
| RAM | 8 GB minimum, 16 GB recommended |
| Storage | 5 GB or more free space |
| Processor | Modern multi-core CPU |
| GPU | Optional |
| Framework | PyTorch |

A CUDA-compatible NVIDIA GPU is optional. CPU inference is supported.

## 2. Project Setup

Open PowerShell in the project root directory.

Create a virtual environment:

```powershell
py -3.11 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## 3. PyTorch Configuration

Install a compatible PyTorch and Torchvision combination for your operating system.

For CPU-only systems, use the official PyTorch installation selector:

https://pytorch.org/get-started/locally/

Verify installation:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

A CPU-only system should report `False` for CUDA availability.

## 4. Dataset Configuration

Place the NEU-DET dataset in an accessible local directory.

Edit `config.yaml` and update:

```yaml
dataset:
  root: "C:/path/to/NEU-DET"
```

Use forward slashes in Windows paths.

Ensure that the dataset contains the required images, labels and annotations.

## 5. Model Checkpoints

Place trained checkpoints in the `models/` directory:

- `cnn_best.pt`
- `resnet18_best.pt`
- `yolo_best.pt`

The VAE demonstration checkpoint is generated separately by the VAE training script.

## 6. Launch the Application

From the project root, activate the virtual environment and execute:

```powershell
python -m streamlit run app/streamlit_app.py
```

Streamlit will provide a local browser address for the dashboard.

## 7. Run Automated Tests

Execute:

```powershell
python -m pytest -q
```

The tests verify selected mathematical fundamentals and evaluation utilities.

## 8. Run Evaluation Scripts

Classification evaluation:

```powershell
python -m src.evaluation.evaluate_classifiers
```

YOLO CPU benchmark:

```powershell
python -m src.evaluation.benchmark_yolo
```

Transformer demonstration:

```powershell
python -m src.advanced.transformer_demo
```

VAE training and demonstration:

```powershell
python -m src.generative.vae_demo
```

Generated reports, metrics and visualisations are stored in `results/`.

## 9. Optional Google Colab GPU Training

For additional model training:

1. Upload the source code and dataset to Google Drive.
2. Open Google Colab.
3. Select a GPU runtime.
4. Install compatible project dependencies.
5. Mount Google Drive.
6. Configure the dataset path.
7. Run the appropriate training module.
8. Download the resulting model checkpoint.

The local application can use downloaded model weights without requiring a CUDA-enabled laptop.

## 10. Troubleshooting

- If package installation fails, verify Python and PyTorch compatibility.
- If model loading fails, check checkpoint paths and architecture compatibility.
- If images cannot be found, verify `dataset.root`.
- If PowerShell blocks virtual environment activation, use the process-scoped execution policy command.
- If the application cannot find YOLO, verify that Ultralytics is installed.
