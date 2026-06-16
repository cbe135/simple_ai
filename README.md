# D3/D4 Hands-on: CNN Classification

Multi-task CNN classification pipeline supporting both D3 (Liver CT) and D4 (Chest X-ray Hackathon) using MONAI + PyTorch + timm.

**2026 Winter — Last modified: 2025/12/17**

## Overview

This repository contains hands-on tutorials for medical image classification using CNNs:

| Task | Data | Format | Description |
|---|---|---|---|
| **D3 Liver CT** | NIfTI (.nii.gz) | CT images + masks | Classify liver tumors |
| **D4 Hackathon** | JPG images | Chest X-rays | Classify 5 diseases (Atelectasis, Cardiomegaly, Colon_Polyp, Diabetic_Retinopathy, Melanoma) |

Both tasks share the same training pipeline but differ in data loading, preprocessing, and augmentation.

## Quick Start

### Prerequisites

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### One-liner (D3 Liver CT)

```bash
git clone https://github.com/cbe135/d3-hands-on-liver-classification.git && cd d3-hands-on-liver-classification && uv sync && uv run python src/main.py
```

### One-liner (D4 Hackathon)

```bash
git clone https://github.com/cbe135/d3-hands-on-liver-classification.git && cd d3-hands-on-liver-classification && uv sync && uv run python src/main.py --task d4_hackathon --group-num 8
```

### Step-by-step

```bash
# 1. Clone the repo
git clone https://github.com/cbe135/d3-hands-on-liver-classification.git
cd d3-hands-on-liver-classification

# 2. Install dependencies
uv sync

# 3a. Run D3 (Liver CT)
uv run python src/main.py

# 3b. Run D4 (Chest X-ray, group 8)
uv run python src/main.py --task d4_hackathon --group-num 8
```

Or with pip:

```bash
pip install -r requirements.txt
python src/main.py --task d4_hackathon --group-num 8
```

## CLI Arguments

| Argument | Choices | Default | Description |
|---|---|---|---|
| `--task` | `d3_liver_ct`, `d4_hackathon` | `d3_liver_ct` | Task type |
| `--group-num` | 1–15 | 1 | Group number for D4 (determines disease class) |
| `--config` | file path | None | Override with custom config YAML |

### D4 Disease Mapping

| Group | Disease |
|---|---|
| 1, 6, 11 | Atelectasis |
| 2, 7, 12 | Cardiomegaly |
| 3, 8, 13 | Colon_Polyp |
| 4, 9, 14 | Diabetic_Retinopathy |
| 5, 10, 15 | Melanoma |

## Project Structure

```
├── README.md
├── pyproject.toml                    # uv project config
├── requirements.txt                  # pip dependencies
├── config.yaml                       # D3 default hyperparameters
├── config_d4_hackathon.yaml          # D4 preset
├── environment.md                    # Environment setup guide
├── 01-introduction.md                # Background & learning objectives
├── 02-setup-and-dependencies.md      # Install & imports
├── 03-configuration.md               # Hyperparameters & config
├── 04-data-preparation.md            # Download, unzip, inspect data
├── 05-preprocessing.md               # Resize, window, normalize, mask
├── 06-augmentation.md                # Data augmentation
├── 07-training.md                    # Model creation & training loop
├── 08-evaluation.md                  # ROC, AUC, confusion matrix, Grad-CAM
├── src/
│   ├── __init__.py
│   ├── env_setup.py                  # Environment detection & data setup
│   ├── config.py                     # Config load/save
│   ├── data.py                       # Data loading & splitting
│   ├── transforms.py                 # Task-aware transform pipelines
│   ├── model.py                      # Model & optimizer creation
│   ├── train.py                      # Training loop
│   ├── evaluate.py                   # Inference, ROC, Grad-CAM
│   ├── utils.py                      # Plotting helpers
│   └── main.py                       # CLI entry point with --task support
└── D3_Hands_on_2026_v2.ipynb         # Launcher notebook (task-selectable)
```

## Task Differences

| Aspect | D3 (Liver CT) | D4 (Chest X-ray) |
|---|---|---|
| Data format | ZIP → NIfTI | tar.gz → JPG |
| Loader | `LoadImaged(keys=['image','mask'])` | `LoadImaged(keys=['image'], reader='pilreader')` |
| Has masks | Yes | No |
| Preprocessing | Resize + CT windowing + MaskIntensity + RepeatChanneld | Resize + RepeatChanneld |
| Augmentation | RandAffined (prob=0) | RandAffined (prob=0.5) + GaussianNoise |
| Data source | Google Drive ZIP (3 links) | gdown tar.gz by group number |

## Supported Environments

| Environment | Status | Notes |
|---|---|---|
| Google Colab | Supported | D3: Drive mount; D4: auto-download via gdown |
| Kaggle | Supported | Upload data as Kaggle Dataset |
| Local | Supported | Auto-downloads data via gdown |

See [environment.md](environment.md) for detailed setup instructions.

## Adding a New Task

To add a new dataset/task:

1. **Add a config file** `config_<task_name>.yaml` with your hyperparameters
2. **Add `task` key** to the config: `task: <task_name>`
3. **Update `src/transforms.py`**: Add a branch in `get_loaders()`, `get_preprocess()`, and `get_augmentation()` for your task
4. **Update `src/env_setup.py`**: Add a `_setup_data_<task>()` function
5. **Update `src/main.py`**: Add defaults in `get_default_args()`
6. **Run**: `python src/main.py --task <task_name>`

The modular `src/` design makes this straightforward — each module checks `args["task"]` to dispatch to the correct logic.

## Configuration

All parameters are defined in `config.yaml` or in the `args` dictionary. Key parameters:

| Parameter | D3 Default | D4 Default | Description |
|---|---|---|---|
| `task` | `d3_liver_ct` | `d4_hackathon` | Task type |
| `seed` | 888 | 42 | Random seed |
| `spatial_size` | [250, 250] | [224, 224] | Image resize dimensions |
| `a_min/a_max` | -125/200 | -125/200 | CT window range (HU) |
| `num_epoch` | 3 | 10 | Training epochs |
| `batch_size` | 128 | 16 | Batch size |
| `lr` | 0.001 | 0.001 | Learning rate |
| `timm_model` | resnet18 | resnet18 | Model architecture |
| `threshold` | 0.5 | 0.5 | Classification threshold |

## Datasets

- **D3**: [Medical Segmentation Decathlon](http://medicaldecathlon.com/) — Liver CT data via Google Drive
- **D4**: Chest X-ray data (Atelectasis, Cardiomegaly, etc.) via gdown

## License

This project is for educational purposes.
