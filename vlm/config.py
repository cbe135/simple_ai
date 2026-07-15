"""Configuration loading/saving for the vlm fine-tune + inference track.

Mirrors the data-driven style of ``src/config.py``: a YAML file is the single
source of truth for one task run, merged over defaults. No task names are
hardcoded in code.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULT_CONFIG = {
    "task": "melanoma",
    "model": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",  # non-gated default; swap to google/medgemma-4b-it (gated) for medical accuracy
        "ollama_model": "qwen2.5vl:7b",         # used only by the Ollama inference backend
        "quantize": "4bit",                     # "4bit" (QLoRA, CUDA only) | "none" (pure LoRA)
        "attn_implementation": "eager",
        "trust_remote_code": False,
    },
    "data": {
        "train_percentage": 0.8,
        "val_percentage": 0.1,
        "test_percentage": 0.1,
        "image_size": 896,       # resized before encoding (896 works for MedGemma; Qwen accepts variable sizes)
        "num_workers": 4,
        "seed": 888,
        "stratify": True,
    },
    # VQA prompt. {modality} and {condition} are substituted from the keys below.
    "prompt": "Does this {modality} image show {condition}? Answer yes or no.",
    "modality": "skin",
    "condition": "melanoma",
    # int label -> answer string produced/parsed by the VLM
    "label_map": {0: "no", 1: "yes"},
    "training": {
        "num_epoch": 3,
        "batch_size": 4,
        "lr": 1e-4,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "warmup_ratio": 0.03,
        "gradient_accumulation_steps": 1,
        "max_length": 512,
        "weight_decay": 0.0,
    },
    "output_dir": "outputs",
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(config_path=None, overrides=None):
    """Load YAML config merged over DEFAULT_CONFIG; apply overrides dict."""
    args = copy.deepcopy(DEFAULT_CONFIG)
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as fp:
            file_config = yaml.safe_load(fp) or {}
        args = _deep_merge(args, file_config)
    if overrides:
        args = _deep_merge(args, overrides)
    # Normalize label_map keys to int (YAML may load them as strings).
    if "label_map" in args and args["label_map"]:
        args["label_map"] = {int(k): str(v) for k, v in args["label_map"].items()}
    return args


def save_config(config: dict, path) -> None:
    with open(path, "w") as fp:
        yaml.safe_dump(config, fp, sort_keys=False)
