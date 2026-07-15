"""vlm: medical Vision-Language Model fine-tune + inference track.

A data-driven counterpart to the referenced
`sayedmohamedscu/Vision-language-models-VLM` notebooks, built on the same
MedGemma/PaliGemma QLoRA recipe but reusing this repo's data_list.yaml
convention. Supports fine-tune (`simple_ai_vlm_train`), inference
(`simple_ai_vlm_infer`), and base-model caching (`simple_ai_vlm_save`).
"""

from .config import load_config, save_config, DEFAULT_CONFIG  # noqa: F401
from .registry import TASKS, DEFAULT_MODEL_ID, DEFAULT_OLLAMA_MODEL  # noqa: F401
