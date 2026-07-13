# CNN Classification Pipeline

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

### Setup (separate steps)

```bash
git clone https://github.com/cbe135/simple_ai.git
cd simple_ai
uv sync
```

### Prepare data (run once)

The pipeline does **not** download data. The directory you pass to
`--data-dir` must already contain:

- `data_list.yaml` (or `data_list.json`) — the image/mask/label entries
- `dataset_info.yaml` — the modality (`CT` | `X-ray` | `MRI` | ...)
- `images/` (and optional `masks/`)

If you need to fetch the example dataset from Google Drive, use the
standalone preparation script (idempotent — safe to re-run):

```bash
python src/prepare_data.py \
    --data-dir /content/liver_data \
    --file-ids 1LNkF... 1vki3... 1ueP6... \
    --archive-format zip
```

Or place your data manually so that `--data-dir` points at the folder
holding `data_list.yaml` / `dataset_info.yaml`.

### Run

```bash
python src/main.py --config config.yaml --data-dir /content/liver_data
```

Or with uv:

```bash
uv run python src/main.py --config config.yaml --data-dir /content/liver_data
```

### One-liner (run only)

```bash
python src/main.py --config config.yaml --data-dir /content/liver_data
```

## CLI

```
python src/main.py --config CONFIG_FILE --data-dir DATA_DIR [--output-dir OUTPUT_DIR]
```

| Argument | Description |
|---|---|
| `--config` | Path to config YAML (default: `config.yaml`) |
| `--data-dir` | **Required.** Directory containing `data_list.yaml` (or `data_list.json`) and `dataset_info.yaml`, e.g. `/content/liver_data`. |
| `--output-dir` | Parent directory for run outputs. A timestamped subdirectory (`YYYYMMDD_HHMMSS`) is created here holding the weights, loss curve, config, and ROC PNGs. Defaults to the current working directory. |

### Examples

```bash
# Local run, data lives in ./my_data and outputs go to ./results
python src/main.py --config config.yaml --data-dir ./my_data --output-dir ./results

# Colab: data already in /content/liver_data, outputs in /content
python src/main.py --config config.yaml --data-dir /content/liver_data
```

### Run output

Each run creates a timestamped directory (e.g. `./20260712_211300/`) under `--output-dir` containing:

| File | Description |
|---|---|
| `best_weights.pth` | Best model weights (lowest validation loss) |
| `loss_curve.png` | Training / validation loss curve |
| `roc_train.png` / `roc_validation.png` / `roc_test.png` | ROC curves for each split |
| `config.yaml` | The resolved config used for this run |

## Project Structure

```
├── README.md
├── pyproject.toml                    # uv project config (single source of dependencies)
├── config.yaml                       # Default config (example: image classification)
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
│   ├── env_setup.py                  # Environment detection + data dir discovery
│   ├── prepare_data.py               # Standalone: download + extract data (run once)
│   ├── config.py                     # Config load/save
│   ├── data.py                       # Data loading & splitting
│   ├── transforms.py                 # Data-driven preset transforms + config extras
│   ├── model.py                      # Model & optimizer creation
│   ├── train.py                      # Training loop
│   ├── evaluate.py                   # Inference, ROC, Grad-CAM
│   ├── utils.py                      # Plotting helpers
│   └── main.py                       # CLI entry point
└── example.ipynb                     # Launcher notebook
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
  data_name: dataset
  data_source:
    file_ids: ["1LNkFfchl4YwKzLJ5SVDovhyvmw6vUUMf"]
    archive_format: zip    # zip or tar.gz
```

Or place your data manually in the working directory.

## Adding a New Dataset

No code changes needed. Follow these steps:

### Step 1: Prepare your data directory

```
your_data/
├── data_list.yaml        # required
├── dataset_info.yaml     # required
├── images/               # required
│   ├── img_001.nii.gz    # (or .jpg, .png, .dcm, ...)
│   ├── img_002.nii.gz
│   └── ...
└── masks/                # optional (omit if no segmentation masks)
    ├── mask_001.nii.gz
    └── ...
```

### Step 2: Create `data_list.yaml`

List every sample with its image path, optional mask, and label.

**With masks (e.g. CT segmentation):**

```yaml
data:
  - image: "path/to/img_001.nii.gz"
    mask: "path/to/mask_001.nii.gz"
    label: 0
  - image: "path/to/img_002.nii.gz"
    mask: "path/to/mask_002.nii.gz"
    label: 1
```

**Without masks (e.g. X-ray classification):**

```yaml
data:
  - image: "path/to/img_001.jpg"
    label: 0
  - image: "path/to/img_002.jpg"
    label: 1
```

Paths can be absolute or relative to the data directory.

### Step 3: Create `dataset_info.yaml`

One line — the modality of your data:

```yaml
modality: CT
```

Supported modalities and their automatic preprocessing:

| Modality | Preprocessing applied |
|---|---|
| `CT` | Resize → CT windowing (`a_min`/`a_max`) → MaskIntensity (if masks exist) → RepeatChannel |
| `X-ray` | Resize → RepeatChannel |
| `MRI` | Resize → RepeatChannel |

Other values (e.g. `Ultrasound`, `Pathology`) also work — they get the non-CT default pipeline (resize + repeat).

### Step 4: Create a config YAML

Copy an existing config and customize. Example `config_my_data.yaml`:

```yaml
environ:
  data_name: your_data          # must match your data directory name
  seed: 42
  data_source:
    file_ids:
      - "your_google_drive_file_id_here"
    archive_format: zip          # zip or tar.gz

data:
  spatial_size: [224, 224]       # resize target (match your model's expected input)
  repeats: 3                     # number of channels to repeat to
  a_min: -125                    # CT window min (only used if modality: CT)
  a_max: 200                     # CT window max (only used if modality: CT)
  affine_prob: 0.5               # probability of random affine augmentation
  flip_prob: 0.5                 # probability of random flip
  cache_rate: 1                  # dataset caching rate
  train_percentage: 0.7
  val_percentage: 0.15
  test_percentage: 0.15

training:
  num_epoch: 10
  batch_size: 16
  lr: 0.001
  timm_model: resnet18           # any timm model (e.g. efficientnet_b0)
  num_classes: 1                 # binary classification

threshold: 0.5
```

If you don't have Google Drive file IDs, place the data manually in your working directory under `your_data/`.

### Step 5: Run

```bash
python src/main.py --config config_my_data.yaml
```

Or in the notebook:

```python
args = load_config("config_my_data.yaml")
```

### Step 6: Verify

The pipeline will print what it derived:

```
INFO - Data: your_data
INFO - Modality: CT
INFO - Has masks: True
INFO - Reader: monai default
INFO - Number of images: 500
INFO - Number of masks: 500
INFO - 350 training, 75 validation, 75 testing
```

### Quick reference: what gets auto-derived

| Property | Source | Example |
|---|---|---|
| **Reader** | File extension of first image in `data_list.yaml` | `.nii.gz` → MONAI NIfTI; `.jpg` → PIL |
| **Has masks** | Whether `data_list.yaml` entries have a `"mask"` key | `"mask" in data_dicts[0]` |
| **Preprocessing** | `dataset_info.yaml` → `modality` field | `CT` → windowing + mask; `X-ray` → resize |
| **Data source** | Config YAML → `data_source.file_ids` + `archive_format` | Google Drive download via gdown |

## Supported Environments

| Environment | Status | Notes |
|---|---|---|
| Google Colab | Supported | Auto-downloads via gdown; GPU recommended |
| Kaggle | Supported | Upload data as Kaggle Dataset |
| Local | Supported | Auto-downloads via gdown or place data manually |

## Configuration

All parameters are in the config YAML. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `data_name` | `dataset` | Data directory name |
| `data_source.file_ids` | 3 IDs | Google Drive file IDs (example) |
| `data_source.archive_format` | `zip` | Archive type |
| `seed` | 888 | Random seed |
| `spatial_size` | [250, 250] | Image resize dimensions |
| `a_min/a_max` | -125/200 | CT window range (HU) |
| `num_epoch` | 3 | Training epochs |
| `batch_size` | 128 | Batch size |
| `lr` | 0.001 | Learning rate |
| `timm_model` | resnet18 | Model architecture |

### Transforms (extras)

The pipeline builds a **preset** transform set automatically from your data (`dataset_info.yaml` modality, whether masks exist, and file extensions). You can append your own MONAI transforms via the `transforms` block in the config, written in [MONAI bundle](https://docs.monai.io/en/stable/mb/config_syntax.html) format:

```yaml
transforms:
  loaders_extra: []        # after LoadImaged / EnsureTyped
  preprocess_extra: []     # after Resize / window / MaskIntensity / RepeatChannel
  augmentation_extra: []   # after RandAffine / RandFlip / GaussianNoise (train only)
```

Example — add Gaussian smoothing to preprocessing (references `@data.*` from the config):

```yaml
transforms:
  preprocess_extra:
    - _target_: monai.transforms.RandGaussianSmoothd
      keys: ["image"]
      sigma_x: [0.5, 1.0]
```

Use the fully-qualified `_target_` (e.g. `monai.transforms.RandFlipd`). Leave the lists empty to use only the presets.

## Autonomous Research (`autoresearch`)

`simple_ai_autoresearch_train` lets an LLM agent improve training by editing
**only** `config.yaml`. Each run:

1. The LLM proposes a new `config.yaml` (given the current one + history).
2. The pipeline trains for a short budget (`uv run python src/main.py ...`).
3. The final `Validation loss:` line is parsed — **lower is better**.
4. If it improves on the best known loss, the config is committed; otherwise it
   is discarded (`git checkout config.yaml`). Every run is logged to
   `experiments.tsv`. The LLM call and training are fully sequential.

Two backends are supported:

- **Local Ollama (default, Colab T4)** — this is the default. Run the one-time
  setup first to install Ollama and verify your GPU/driver:
  `simple_ai_autoresearch_setup`. The training CLI starts `ollama serve` if
  needed, pulls the model, and shuts it down on completion/interrupt.
- **OpenRouter (free tier)** — opt in with `--remote`; needs
  `OPENROUTER_API_KEY` (env or `.env`). Pass `--model <id>`.

### Prerequisites & caveats

- Install `uv` and the project (`uv pip install -e .`). Training runs as
  `uv run python src/main.py ...`.
- OpenRouter free-tier models are rate-limited (~20 req/min, ~200 req/day).
  Both limits apply to **all** `:free` models on the same key, so keep `--runs`
  modest and avoid concurrent jobs.
- **Colab T4 + Ollama GPU requirement:** Ollama needs a CUDA ≥ 12 runtime. In
  Colab, choose *Runtime ▸ Change runtime type ▸ T4 GPU* and a **CUDA 12.x**
  image. With an older CUDA driver, Ollama silently falls back to **CPU-only**
  (the CLI warns when it detects this) and training is very slow — in that case
  use OpenRouter instead. (TPU v5e runtimes are **not** supported by Ollama.)
- Use `--unload-between-runs` on T4 to free VRAM between the LLM call and
  training.
- Keep secrets only in `simple_ai/.env` (already gitignored). Never commit API
  keys.

### Setup (run once, Colab T4)

Installs Ollama if missing, starts it, verifies your GPU/driver match, and
pre-pulls the default model (`qwen2.5-coder:7b`) so the long optimization loop
starts clean:

```bash
simple_ai_autoresearch_setup            # install + verify GPU + pull model
simple_ai_autoresearch_setup --no-pull  # skip the ~5GB model download
```

If setup reports CPU-only Ollama, switch to a CUDA 12.x T4 runtime (see
caveats below) or use OpenRouter with `--remote`.

### Background serving (run server in its own cell)

`ollama serve` is long-running, so it must not block the notebook cell. Use the
dedicated serve command, which starts the server **detached** and returns
immediately; the training command then reuses the already-running server:

```python
# Cell 1 — returns at once; Ollama keeps serving in the background
!simple_ai_autoresearch_serve

# Cell 2 — training reuses the running server (no second spawn)
!simple_ai_autoresearch_train --data-dir /content/dataset --runs 12
```

`--models-dir PATH` selects where Ollama stores models (e.g. a Google Drive
mount) instead of the default `~/.ollama/models`. The server only starts the
API — the model itself is pulled and loaded on demand by the training command.

#### Warm start and exposing the server to another Colab instance

- **`--warm-start`** also loads the model into GPU memory (the cell blocks until
  ready) and sets `OLLAMA_KEEP_ALIVE=-1` so it stays resident — useful when
  another instance will drive this server.
- **`--expose {proxy,localtunnel}`** (default: **`localtunnel`**) prints an
  external URL so a *different* Colab instance (or any machine) can reach this
  Ollama server:

  ```python
  # This instance — serve, warm the model, and expose it (localtunnel by default)
  !simple_ai_autoresearch_serve --warm-start --expose
  # or explicitly: !simple_ai_autoresearch_serve --warm-start --expose proxy
  ```

  - `proxy` (Method 1): uses Colab's native `google.colab.kernel.proxyPort`. The
    URL is only reachable by **your Google account**. If the command can't access
    `google.colab` (e.g. run via `!`), it prints the exact Python-cell snippet to
    generate the URL instead.
  - `localtunnel` (Method 2): spins up a **public** `https://….loca.lt` URL via
    `npx localtunnel` (installs `nodejs`/`npm` if missing). Anyone with the URL
    can reach the server.

  The printed URL (append `/v1` for the OpenAI-compatible endpoint) is what the
  other instance references. Bind address stays `127.0.0.1:11434`; both methods
  forward to it locally.

  The **other** instance consumes it by passing that URL to the training command
  (skips starting its own server):

  ```bash
  # Instance B — drive instance A's exposed Ollama from instance B's training
  simple_ai_autoresearch_train --local --ollama-base-url https://abc123.loca.lt/v1 \
      --data-dir /content/dataset --runs 12
  ```

### Examples

```bash
# Local Ollama is the default (download data, then optimize)
simple_ai_autoresearch_setup
simple_ai_autoresearch_train --gdown-id <gdrive-id> --runs 12

# OpenRouter free tier (opt in with --remote)
export OPENROUTER_API_KEY=sk-or-...
simple_ai_autoresearch_train --remote --data-dir /content/dataset \
    --model meta-llama/llama-3.1-8b-instruct:free --runs 12
```

Options: `--data-dir`, `--config` (default `config.yaml`),
`--experiments` (default `experiments.tsv`), `--model`, `--base-url`,
`--ollama-base-url` (remote Ollama URL with --local, e.g. `https://x.loca.lt/v1`),
`--local` (default), `--remote` (use OpenRouter instead),
`--unload-between-runs`, `--timeout` (default 700s),
`--runs` (default 10), `--gdown-id`, `--data-name`.

## Dependencies

| Package | Purpose |
|---|---|
| MONAI | Medical imaging preprocessing & data loading |
| PyTorch | Deep learning framework |
| timm | Pretrained models (ResNet-18) |
| scikit-learn | Evaluation metrics |
| matplotlib | Visualization |
| gdown | Google Drive data download |
| openai | OpenAI-compatible LLM client (autoresearch) |
| python-dotenv | Load `.env` for API keys (autoresearch) |

## License

This project is for educational purposes.
