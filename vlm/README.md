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
| Atelectasis | Chest X-ray | `google/medgemma-4b-it` | `MusinguziDenis/PaliGemma-CXR` (CXR-native) |
| Cardiomegaly | Chest X-ray | `google/medgemma-4b-it` | `PaliGemma-CXR` |
| Colon_Polyps | Endoscopy crops | `google/medgemma-4b-it` (Kvasir/PolypGen) | CNNs still beat VLMs here; `EndoViT` encoder |
| Melanoma | Skin lesion | `google/medgemma-4b-it` (ISIC) | `BiomedCLIP`, `MM-Skin/SkinVL` |
| Diabetic Retinopathy | Retinal fundus | `google/medgemma-4b-it` (APTOS/IDRID) | `sStonemason/RET-CLIP` (SOTA retinal, CLIP-style) |

**Backbone default:** `google/medgemma-4b-it` for all five. The SigLIP
encoder was pre-trained on chest X-ray, dermatology, ophthalmology, and
histopathology, so it works **zero-shot** across all five modalities (e.g.
MedGemma 4B zero-shot: dermatology 71.8%, diabetic retinopathy 64.9%,
histopathology 69.8%, cardiomegaly AUC 0.904). Fine-tuning is recommended for
strong per-task accuracy. Swap the backbone by editing `model.model_id` in a
config.

## Install

```bash
uv sync                          # installs everything, no --extra needed
```

## Cache the base model (so you don't re-download it every run)

`simple_ai_vlm_save` downloads the Hugging Face base once and copies it to a
persistent directory (Google Drive on Colab by default, else `./vlm_models`).
Later `train`/`infer` runs load from that cache with `--base-dir` (or the
`SIMPLE_AI_VLM_BASE_DIR` env var), offline.

```bash
# On Colab (after mounting Drive in a Python cell):
simple_ai_vlm_save                          # caches google/medgemma-4b-it to Drive
simple_ai_vlm_save --ollama                 # also caches the Ollama base (medgemma:4b)

# Local:
simple_ai_vlm_save --models-dir /path/to/vlm_models
export SIMPLE_AI_VLM_BASE_DIR=/path/to/vlm_models
```

## Fine-tune (one task)

```bash
simple_ai_vlm_train \
    --config vlm/configs/melanoma.yaml \
    --data-dir /path/to/isic_data \
    --base-dir /path/to/vlm_models
```

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
| `data.image_size` | Images are resized to this (MedGemma expects 896) |
| `prompt` | VQA template; `{modality}`/`{condition}` substituted |
| `modality` / `condition` | substituted into `prompt` |
| `label_map` | int label → answer string the VLM generates/parses |
| `training.*` | epochs, batch size, lr, LoRA `r`/`alpha`/`dropout` |

## Notes

- **License:** MedGemma is released under the Health AI Developer Foundations
  terms of use — research / non-clinical. Not a medical device.
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
