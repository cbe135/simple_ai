# D3/D4 Hands-on: CNN Classification

Multi-task CNN classification pipeline using MONAI + PyTorch + timm. **All behavior is data-driven** — no hardcoded task names. The pipeline reads configuration from YAML files and derives transform pipelines from the data itself.

**2026 Winter — Last modified: 2025/12/17**

## How It Works

Every dataset ships with:

```
data_dir/
├── data_list.yaml        # image/mask/label entries
├── dataset_info.yaml     # modality: CT | X-ray | MRI | ...
├── images/               # image files
└── masks/                # (optional) mask files
```

The pipeline automatically derives:
- **Reader** — from file extensions (`.nii.gz` → NIfTI, `.jpg`/`.png` → PIL)
- **Has masks** — from whether `data_list.yaml` entries contain a `"mask"` key
- **Preprocessing** — from modality in `dataset_info.yaml` (CT → windowing + mask + resize; X-ray → resize only)

No `if task == ...` anywhere in code.

## Quick Start

### Prerequisites

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### D3 Liver CT (default)

```bash
git clone https://github.com/cbe135/d3-hands-on-liver-classification.git
cd d3-hands-on-liver-classification
uv sync
uv run python src/main.py
```

### D4 Chest X-ray Hackathon

```bash
git clone https://github.com/cbe135/d3-hands-on-liver-classification.git
cd d3-hands-on-liver-classification
uv sync
uv run python src/main.py --config config_d4_hackathon.yaml
```

### One-liner

```bash
git clone https://github.com/cbe135/d3-hands-on-liver-classification.git && cd d3-hands-on-liver-classification && uv sync && uv run python src/main.py --config config_d4_hackathon.yaml
```

## CLI

```
python src/main.py [--config CONFIG_FILE]
```

| Argument | Description |
|---|---|
| `--config` | Path to config YAML (default: `config.yaml`) |

## Project Structure

```
├── README.md
├── pyproject.toml                    # uv project config
├── requirements.txt                  # pip dependencies
├── config.yaml                       # D3 Liver CT defaults
├── config_d4_hackathon.yaml          # D4 Chest X-ray preset
├── dataset_info_d3_liver_ct.yaml     # Example: modality: CT
├── dataset_info_d4_hackathon.yaml    # Example: modality: X-ray
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
│   ├── env_setup.py                  # Environment detection + data download
│   ├── config.py                     # Config load/save
│   ├── data.py                       # Data loading & splitting
│   ├── transforms.py                 # Data-driven transform pipelines
│   ├── model.py                      # Model & optimizer creation
│   ├── train.py                      # Training loop
│   ├── evaluate.py                   # Inference, ROC, Grad-CAM
│   ├── utils.py                      # Plotting helpers
│   └── main.py                       # CLI entry point
└── D3_Hands_on_2026_v2.ipynb         # Launcher notebook
```

## Data Setup

Each dataset needs:

### 1. `data_list.yaml`

```yaml
data:
  - image: "path/to/image1.nii.gz"
    mask: "path/to/mask1.nii.gz"    # omit if no masks
    label: 0
  - image: "path/to/image2.nii.gz"
    mask: "path/to/mask2.nii.gz"
    label: 1
```

### 2. `dataset_info.yaml`

```yaml
modality: CT    # CT | X-ray | MRI | ...
```

The modality determines preprocessing:

| Modality | Preprocessing |
|---|---|
| `CT` | Resize + CT windowing (`a_min`/`a_max`) + MaskIntensity + RepeatChannel |
| `X-ray` / `MRI` / other | Resize + RepeatChannel |

### 3. Config YAML

Specify data source for auto-download:

```yaml
environ:
  data_name: liver_data
  data_source:
    file_ids: ["1LNkFfchl4YwKzLJ5SVDovhyvmw6vUUMf"]
    archive_format: zip    # zip or tar.gz
```

Or place your data manually in the working directory.

## Adding a New Dataset

1. Prepare `data_list.yaml` + `dataset_info.yaml` + image files
2. Create a config YAML with your data source and hyperparameters
3. Run: `python src/main.py --config your_config.yaml`

No code changes needed.

## Supported Environments

| Environment | Status | Notes |
|---|---|---|
| Google Colab | Supported | Auto-downloads via gdown; GPU recommended |
| Kaggle | Supported | Upload data as Kaggle Dataset |
| Local | Supported | Auto-downloads via gdown or place data manually |

## Configuration

All parameters are in the config YAML. Key parameters:

| Parameter | D3 Default | D4 Default | Description |
|---|---|---|---|
| `data_name` | `liver_data` | `Atelectasis` | Data directory name |
| `data_source.file_ids` | 3 IDs | 1 ID | Google Drive file IDs |
| `data_source.archive_format` | `zip` | `tar.gz` | Archive type |
| `seed` | 888 | 42 | Random seed |
| `spatial_size` | [250, 250] | [224, 224] | Image resize dimensions |
| `a_min/a_max` | -125/200 | -125/200 | CT window range (HU) |
| `num_epoch` | 3 | 10 | Training epochs |
| `batch_size` | 128 | 16 | Batch size |
| `lr` | 0.001 | 0.001 | Learning rate |
| `timm_model` | resnet18 | resnet18 | Model architecture |

## Dependencies

| Package | Purpose |
|---|---|
| MONAI | Medical imaging preprocessing & data loading |
| PyTorch | Deep learning framework |
| timm | Pretrained models (ResNet-18) |
| scikit-learn | Evaluation metrics |
| matplotlib | Visualization |
| gdown | Google Drive data download |

## License

This project is for educational purposes.
