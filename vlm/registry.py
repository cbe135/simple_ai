"""Per-task recommendations and default specialisms for the vlm track.

This is documentation/reference data, not control flow. The actual run config
lives in ``vlm/configs/<task>.yaml``. Listed here so the README and CLI can
surface the recommended model and specialist alternatives without hardcoding
them elsewhere.
"""

from __future__ import annotations

# Default backbone across all 5 tasks (one code path, swappable via config).
# Non-gated, works with no Hugging Face login.
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_OLLAMA_MODEL = "qwen2.5vl:7b"


def gated_access_help(model_id: str) -> str:
    """Exact steps to get through a Hugging Face gated-repo 401."""
    return (
        f"\nGATED MODEL ACCESS DENIED: {model_id} (HTTP 401 Unauthorized)\n"
        "This model is gated on Hugging Face. To use it you must:\n"
        "  1. Visit https://huggingface.co/" + model_id + " and click "
        '"Agree and access repository" to accept its license\n'
        "     (requires a free Hugging Face account).\n"
        "  2. Authenticate in THIS environment (one of):\n"
        "       huggingface-cli login        # then paste your token when prompted\n"
        "       export HF_TOKEN=hf_xxx       # token from "
        "https://huggingface.co/settings/tokens\n"
        "     The token's account must be the one that accepted the license.\n"
        "  3. Re-run this command.\n"
        "The default model (Qwen/Qwen2.5-VL-7B-Instruct) is NOT gated and needs no "
        "login.\n"
        "To switch, set model.model_id in your config (or pass --model-id to "
        "simple_ai_vlm_save).\n"
    )


def raise_if_gated(model_id: str, exc: Exception):
    """Re-raise as a clear SystemExit if ``exc`` is a gated/401 access error."""
    s = str(exc).lower()
    if any(k in s for k in ("gated", "401", "unauthorized", "restricted")):
        raise SystemExit(gated_access_help(model_id))
    raise

# task -> (modality, condition, label_map, prompt, specialist note)
TASKS = {
    "atelectasis": {
        "modality": "chest x-ray",
        "condition": "atelectasis",
        "label_map": {0: "no", 1: "yes"},
        "prompt": "Does this {modality} show {condition}? Answer yes or no.",
        "specialist": "PaliGemma-CXR (MusinguziDenis/PaliGemma-CXR) for a CXR-native multitask model.",
    },
    "cardiomegaly": {
        "modality": "chest x-ray",
        "condition": "cardiomegaly",
        "label_map": {0: "no", 1: "yes"},
        "prompt": "Does this {modality} show {condition}? Answer yes or no.",
        "specialist": "PaliGemma-CXR (MusinguziDenis/PaliGemma-CXR).",
    },
    "colon_polyps": {
        "modality": "endoscopic",
        "condition": "polyp",
        "label_map": {0: "no", 1: "yes"},
        "prompt": "Does this {modality} image contain a {condition}? Answer yes or no.",
        "specialist": "CNNs (ResNet50) still beat VLMs on polyp classification; MedGemma is the open VLM route. EndoViT as an encoder alt.",
    },
    "melanoma": {
        "modality": "skin lesion",
        "condition": "melanoma",
        "label_map": {0: "no", 1: "yes"},
        "prompt": "Is this {modality} malignant melanoma? Answer yes or no.",
        "specialist": "BiomedCLIP or MM-Skin/SkinVL (if weights released) as dermatology specialists.",
    },
    "diabetic_retinopathy": {
        "modality": "retinal fundus",
        "condition": "diabetic retinopathy",
        "label_map": {
            0: "grade 0", 1: "grade 1", 2: "grade 2", 3: "grade 3", 4: "grade 4",
        },
        "prompt": "What is the {condition} grade (0-4) of this {modality} image? Respond with only the grade.",
        "specialist": "RET-CLIP (sStonemason/RET-CLIP) is SOTA for retinal but is CLIP-style (not generative); use for a non-VLM baseline.",
    },
}


def known_task(name: str) -> bool:
    return name in TASKS
