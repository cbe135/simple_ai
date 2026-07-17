# VLM Fine-tune & Inference Track (`vlm/`)

Medical Vision-Language Model fine-tuning and inference for **classification
tasks**, framed as **VQA-style text generation** (the model writes the class as
text). One task runs at a time. The approach mirrors the notebooks in
[`sayedmohamedscu/Vision-language-models-VLM`](https://github.com/sayedmohamedscu/Vision-language-models-VLM)
(MedGemma / PaliGemma / Florence-2 QLoRA fine-tuning), but is built
data-driven like the rest of this repo: it reuses the `data_list.yaml`
convention from `src/`.

**Last modified: 2026-07-16**

## Tasks supported

| Task | Modality | Recommended VLM (fine-tune, VQA) | Specialist alternative |
|---|---|---|---|
| Atelectasis | Chest X-ray | `Qwen/Qwen2.5-VL-7B-Instruct` | `MusinguziDenis/PaliGemma-CXR` (CXR-native) |
| Cardiomegaly | Chest X-ray | `Qwen/Qwen2.5-VL-7B-Instruct` | `PaliGemma-CXR` |
| Colon_Polyps | Endoscopy crops | `Qwen/Qwen2.5-VL-7B-Instruct` (Kvasir/PolypGen) | CNNs still beat VLMs here; `EndoViT` encoder |
| Melanoma | Skin lesion | `Qwen/Qwen2.5-VL-7B-Instruct` (ISIC) | `BiomedCLIP`, `MM-Skin/SkinVL` |
| Diabetic Retinopathy | Retinal fundus | `Qwen/Qwen2.5-VL-7B-Instruct` (APTOS/IDRID) | `sStonemason/RET-CLIP` (SOTA retinal, CLIP-style) |

**Backbone default:** `Qwen/Qwen2.5-VL-7B-Instruct` for all five — a strong,
**non-gated** VLM (Apache-2.0) that runs with no Hugging Face login. For the
best medical accuracy you can swap to `google/medgemma-4b-it` (gated — see
"Gated models (MedGemma)" below): its SigLIP encoder was pre-trained on chest
X-ray, dermatology, ophthalmology, and histopathology, so it works **zero-shot**
across these modalities (e.g. MedGemma 4B zero-shot: dermatology 71.8%,
diabetic retinopathy 64.9%, histopathology 69.8%, cardiomegaly AUC 0.904).
Fine-tuning is recommended for strong per-task accuracy. Swap the backbone by
editing `model.model_id` in a config.

## Install

```bash
uv sync                          # installs everything, no --extra needed
```

## Global CLI flags (all `simple_ai_vlm_*` commands)

Every VLM command accepts two global flags:

- `--help` — print the full usage and parameter list, then exit.
- `--default` — print the **default value of every CLI parameter** and exit.
  These are the command's own argument defaults (e.g. `--split test`,
  `--backend hf`); they do **not** include the values from your config YAML.

```bash
simple_ai_vlm_train --help
simple_ai_vlm_infer --default
simple_ai_vlm_save  --default
```

## Cache the base model (so you don't re-download it every run)

`simple_ai_vlm_save` downloads the Hugging Face base once and copies it to a
persistent directory (Google Drive on Colab by default, else `./vlm_models`).
Later `train`/`infer` runs load from that cache with `--base-dir` (or the
`SIMPLE_AI_VLM_BASE_DIR` env var), offline.

```bash
# On Colab (after mounting Drive in a Python cell):
simple_ai_vlm_save                          # caches Qwen/Qwen2.5-VL-7B-Instruct to Drive
simple_ai_vlm_save --ollama                 # also caches the Ollama base (qwen2.5vl:7b)

# Local:
simple_ai_vlm_save --models-dir /path/to/vlm_models
export SIMPLE_AI_VLM_BASE_DIR=/path/to/vlm_models
```

**Parameters** (`simple_ai_vlm_save --help` for the live list):

| Argument | Default | Description |
|---|---|---|
| `--model-id` | `Qwen/Qwen2.5-VL-7B-Instruct` | HF base model id to cache. |
| `--models-dir` | `None` | Cache directory (Drive on Colab, else `./vlm_models`). |
| `--ollama-model` | `None` | Also cache this Ollama base (e.g. `qwen2.5vl:7b`). |
| `--ollama` | `False` | Shorthand to also cache `qwen2.5vl:7b` via Ollama. |
| `--default` | `False` | Print CLI parameter defaults and exit. |
| `--help` | — | Print usage and exit. |

## Gated models (MedGemma)

The default model (`Qwen/Qwen2.5-VL-7B-Instruct`) is **not** gated, so no login
is required. If you switch `model.model_id` to `google/medgemma-4b-it` for
higher medical accuracy, it is **gated** on Hugging Face — you must grant access
once, or `simple_ai_vlm_save` / `simple_ai_vlm_infer` will fail with
`GATED MODEL ACCESS DENIED ... (HTTP 401 Unauthorized)` and print these exact
steps:

1. Visit https://huggingface.co/google/medgemma-4b-it and click
   **"Agree and access repository"** to accept its license (needs a free HF
   account).
2. Authenticate in the environment where you run the command (one of):
   - `huggingface-cli login` (then paste your token), or
   - `export HF_TOKEN=hf_xxx` (token from https://huggingface.co/settings/tokens).
   The token's account must be the one that accepted the license.
3. Re-run `simple_ai_vlm_save` (to cache) and then `simple_ai_vlm_infer`.

## Fine-tune (one task)

```bash
simple_ai_vlm_train \
    --config vlm/configs/melanoma.yaml \
    --data-dir /path/to/isic_data \
    --base-dir /path/to/vlm_models
```

**Parameters** (`simple_ai_vlm_train --help` for the live list):

| Argument | Default | Description |
|---|---|---|
| `--config` | *(required)* | Path to a vlm config YAML (e.g. `vlm/configs/melanoma.yaml`). |
| `--data-dir` | *(required)* | Directory containing `data_list.yaml` + `images/`. |
| `--base-dir` | `None` | Cached base-model dir (from `simple_ai_vlm_save`). Falls back to the Hub. |
| `--device` | `None` | `cuda` \| `mps` \| `cpu` (default: auto-detect). |
| `--quantize` | `None` | `4bit` \| `none` — overrides the config. |
| `--output-dir` | `None` | Overrides `output_dir` from the config. |
| `--default` | `False` | Print CLI parameter defaults and exit. |
| `--help` | — | Print usage and exit. |

- `quantize: 4bit` (QLoRA) is used on **CUDA**; on Apple Silicon / CPU it
  auto-falls back to **pure LoRA** (`quantize: none`) since bitsandbytes 4-bit
  is CUDA-only. Override with `--quantize none`.
- Output: `outputs/<task>/<timestamp>/adapter/` (the fine-tuned LoRA weights)
  + `vlm_config.yaml`.

## Inference + metrics

```bash
# Fine-tuned (Hugging Face backend, loads base + adapter):
simple_ai_vlm_infer \
    --config vlm/configs/melanoma.yaml \
    --data-dir /path/to/isic_data \
    --adapter outputs/melanoma/<timestamp>/adapter \
    --split test

# Zero-shot base via Ollama (no adapter; needs `ollama serve` running):
simple_ai_vlm_infer \
    --config vlm/configs/melanoma.yaml \
    --data-dir /path/to/isic_data \
    --backend ollama
```

**Parameters** (`simple_ai_vlm_infer --help` for the live list):

| Argument | Default | Description |
|---|---|---|
| `--config` | *(required)* | Path to a vlm config YAML. |
| `--data-dir` | *(required)* | Directory containing `data_list.yaml` + `images/`. |
| `--base-dir` | `None` | Cached base-model dir (from `simple_ai_vlm_save`). |
| `--device` | `None` | `cuda` \| `mps` \| `cpu` (default: auto-detect). |
| `--adapter` | `None` | Path to a saved LoRA adapter (hf backend). |
| `--split` | `test` | `train` \| `val` \| `test`. |
| `--backend` | `hf` | `hf` \| `ollama`. |
| `--quantize` | `None` | `4bit` \| `none` — overrides the config. |
| `--default` | `False` | Print CLI parameter defaults and exit. |
| `--help` | — | Print usage and exit. |

Artifacts written to `<cwd>/vlm/`: `predictions.csv`, `confusion_matrix.png`,
`metrics.txt`, and (binary tasks) `roc.png` with AUC from yes/no token logits.

## Data format (reuses `src/` convention)

```
your_data/
├── data_list.yaml        # required: a `data` list of {image, label}
└── images/               # or wherever the image paths point
```

```yaml
data:
  - image: "images/case_001.jpg"
    label: 1
  - image: "images/case_002.jpg"
    label: 0
```

The `label` integer maps to the answer text via `label_map` in the config
(e.g. `0: "no"`, `1: "yes"`). Masks are ignored by the VLM path.

## Config reference

| Key | Meaning |
|---|---|
| `model.model_id` | Hugging Face base model id |
| `model.ollama_model` | Ollama base tag (ollama backend only) |
| `model.quantize` | `4bit` (QLoRA, CUDA) or `none` (pure LoRA) |
| `data.image_size` | Images are resized to this (896 works for MedGemma; Qwen accepts variable sizes) |
| `prompt` | VQA template; `{modality}`/`{condition}` substituted |
| `modality` / `condition` | substituted into `prompt` |
| `label_map` | int label → answer string the VLM generates/parses |
| `training.*` | epochs, batch size, lr, LoRA `r`/`alpha`/`dropout` |

## Notes

- **License:** The default model `Qwen/Qwen2.5-VL-7B-Instruct` is Apache-2.0.
  If you instead use `google/medgemma-4b-it`, it is released under the Health
  AI Developer Foundations terms of use — research / non-clinical. Not a
  medical device.
- **GGUF/Ollama fine-tune serving:** fine-tuned LoRA adapters are served via
  Hugging Face directly. Exporting a *fine-tuned* model to Ollama GGUF (merge +
  `llama.cpp` convert) is intentionally out of scope; the Ollama backend is for
  zero-shot base inference.
- **Autoresearch:** the existing LLM-driven config optimizer (`src/autoresearch.py`)
  can later be pointed at these `vlm/configs/*.yaml` to tune hyperparameters.

## Acknowledgements

- Approach based on
  [`sayedmohamedscu/Vision-language-models-VLM`](https://github.com/sayedmohamedscu/Vision-language-models-VLM)
  (MedGemma / PaliGemma / Florence-2 fine-tuning notebooks).
- Base model: Google `medgemma-4b-it` (Health AI Developer Foundations).
