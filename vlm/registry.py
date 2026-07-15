"""Per-task recommendations and default specialisms for the vlm track.

This is documentation/reference data, not control flow. The actual run config
lives in ``vlm/configs/<task>.yaml``. Listed here so the README and CLI can
surface the recommended model and specialist alternatives without hardcoding
them elsewhere.
"""

from __future__ import annotations

# Default backbone across all 5 tasks (one code path, swappable via config).
DEFAULT_MODEL_ID = "google/medgemma-4b-it"
DEFAULT_OLLAMA_MODEL = "medgemma:4b"

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
