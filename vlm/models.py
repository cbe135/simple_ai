"""Model/processor loading for the vlm track.

Two quantization paths:
  * ``quantize: 4bit`` on CUDA -> QLoRA (4-bit base via bitsandbytes + LoRA).
  * ``quantize: none`` (or any non-CUDA device) -> pure LoRA in full precision.

``simple_ai_vlm_save`` caches the Hugging Face base weights (optionally on
Google Drive) so training/inference can load them offline instead of pulling
from the Hub every run. ``resolve_base_dir`` redirects the model id to that
cache when present.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def auto_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_base_dir(model_id: str, base_dir: str | None) -> str:
    """If ``base_dir`` (or the default models dir) holds a cached copy of
    ``model_id``, use it; else fall back to the Hub id."""
    candidates = [base_dir] if base_dir else []
    try:
        from .save import default_models_dir
        candidates.append(str(default_models_dir()))
    except Exception:
        pass
    for d in candidates:
        if not d:
            continue
        cached = Path(d) / model_id.replace("/", "--")
        if cached.exists() and any(cached.iterdir()):
            logger.info("Loading base model from cache: %s", cached)
            return str(cached)
    return model_id


def _load_base_model(model_id: str, cfg: dict, device: str, quantize: str):
    """Load the (un-wrapped) base model + processor from ``model_id``."""
    import torch
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
    )

    trust = bool(cfg["model"].get("trust_remote_code", False))
    # Force sdpa: the "eager" backend builds a 5D causal mask that breaks
    # Qwen2.5-VL (RuntimeError in sdpa_mask.expand). sdpa avoids that path.
    attn = "sdpa"
    from .registry import raise_if_gated

    logger.info("Loading base model %s on %s (quantize=%s)…", model_id, device, quantize)

    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust)
    except Exception as exc:
        raise_if_gated(model_id, exc)

    use_4bit = quantize == "4bit" and device == "cuda"
    if use_4bit:
        logger.info("Loading base model in 4-bit (QLoRA)…")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                quantization_config=bnb,
                device_map="auto",
                attn_implementation=attn,
                trust_remote_code=trust,
            )
        except Exception as exc:
            raise_if_gated(model_id, exc)
    else:
        if quantize == "4bit":
            logger.warning(
                "quantize=4bit requested but device=%s is not CUDA; "
                "bitsandbytes 4-bit is CUDA-only, falling back to full-precision "
                "LoRA (quantize=none).",
                device,
            )
        if device == "cpu":
            logger.warning(
                "Running on CPU: a 7B model loads in fp32 (~28GB) and is very slow / "
                "may run out of memory. Enable a GPU runtime for 4-bit QLoRA."
            )
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=dtype,
                attn_implementation=attn,
                trust_remote_code=trust,
            )
        except Exception as exc:
            raise_if_gated(model_id, exc)
            model.to(device)
    if use_4bit:
        _assert_4bit_active(model)
    logger.info("Base model loaded on %s.", device)
    return model, processor


def _assert_4bit_active(model):
    """Fail loudly if a 4-bit QLoRA request did not actually quantize.

    A silent fallback to bf16 would make a 7B model load in ~14GB and OOM
    on a T4 later with a confusing traceback. Catch it up front instead.
    """
    loaded_4bit = getattr(model, "is_loaded_in_4bit", False)
    if not loaded_4bit:
        try:
            from bitsandbytes.nn import Linear4bit
        except Exception:
            Linear4bit = None
        if Linear4bit is not None:
            loaded_4bit = any(
                isinstance(m, Linear4bit) for m in model.modules()
            )
    if not loaded_4bit:
        raise RuntimeError(
            "Requested 4-bit QLoRA but the model is NOT loaded in 4-bit "
            "(bitsandbytes likely failed or was skipped). A 7B model in full "
            "precision will OOM on a T4. Fix bitsandbytes on the GPU runtime, "
            "use a larger GPU (L4 24GB / A100), or switch to a smaller model."
        )


def _lora_targets(model) -> list[str]:
    """Pick LoRA target modules that actually exist in the model."""
    candidates = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "linear_proj", "up_proj", "down_proj", "gate_proj",
    ]
    present = set()
    for name, _ in model.named_modules():
        for c in candidates:
            if name.endswith(c):
                present.add(c)
    return list(present) or ["q_proj", "k_proj", "v_proj", "o_proj"]


def apply_lora(model, cfg: dict):
    from peft import LoraConfig, get_peft_model

    t = cfg["training"]
    targets = _lora_targets(model)
    logger.info("LoRA target modules: %s", targets)
    lora = LoraConfig(
        r=int(t.get("lora_r", 8)),
        lora_alpha=int(t.get("lora_alpha", 16)),
        lora_dropout=float(t.get("lora_dropout", 0.05)),
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora)


def load_base(cfg: dict, base_dir: str | None = None, device: str | None = None,
              quantize: str | None = None):
    """Load base model + processor (no LoRA). ``quantize`` overrides config."""
    device = device or auto_device()
    quantize = (quantize or cfg["model"].get("quantize", "4bit")).lower()
    model_id = resolve_base_dir(cfg["model"]["model_id"], base_dir)
    model, processor = _load_base_model(model_id, cfg, device, quantize)
    return model, processor, device


def load_for_training(cfg: dict, base_dir: str | None = None, device: str | None = None,
                      quantize: str | None = None):
    model, processor, device = load_base(cfg, base_dir, device, quantize)
    quantize_eff = (quantize or cfg["model"].get("quantize", "4bit")).lower()
    if quantize_eff == "4bit":
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    model = apply_lora(model, cfg)
    model.print_trainable_parameters() if hasattr(model, "print_trainable_parameters") else None
    return model, processor, device


def load_for_inference(cfg: dict, adapter_path: str | None = None,
                       base_dir: str | None = None, device: str | None = None,
                       quantize: str | None = None):
    """Load base (+ LoRA adapter if ``adapter_path`` given) for inference."""
    model, processor, device = load_base(cfg, base_dir, device, quantize)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        logger.info("Loaded LoRA adapter from %s", adapter_path)
    return model, processor, device
