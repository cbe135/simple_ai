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
import time
from pathlib import Path

import requests
try:
    from huggingface_hub import HfHubHTTPError
except ImportError:
    try:
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError:
        from huggingface_hub.utils import HfHubHTTPError

logger = logging.getLogger(__name__)


def _is_transient_hub_error(exc) -> bool:
    """True for retryable HF/network errors (504/502/503/500/429, timeouts, conn resets)."""
    if isinstance(exc, HfHubHTTPError):
        resp = getattr(exc, "response", None)
        return resp is not None and resp.status_code in (429, 500, 502, 503, 504)
    if isinstance(exc, (requests.exceptions.ConnectionError,
                        requests.exceptions.Timeout,
                        requests.exceptions.ChunkedEncodingError)):
        return True
    if isinstance(exc, requests.exceptions.HTTPError):
        resp = getattr(exc, "response", None)
        return resp is not None and resp.status_code in (429, 500, 502, 503, 504)
    return False


def _load_with_retry(loader, max_retries=5, base_wait=5, **kwargs):
    """Call ``loader(**kwargs)``, retrying on transient HF/network errors with exp backoff."""
    last = None
    for attempt in range(max_retries):
        try:
            return loader(**kwargs)
        except Exception as exc:
            if _is_transient_hub_error(exc) and attempt < max_retries - 1:
                wait = base_wait * (2 ** attempt)
                logger.warning(
                    "Transient HF/network error (%s); retrying in %ss (attempt %d/%d)...",
                    exc, wait, attempt + 1, max_retries,
                )
                time.sleep(wait)
                last = exc
                continue
            raise
    raise last


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

    # If resolve_base_dir redirected to a locally cached copy, load fully offline
    # (no HEAD request to the Hub, so a Hub outage 504 can never break a cached run).
    local_only = model_id != cfg["model"]["model_id"]

    try:
        processor = _load_with_retry(
            AutoProcessor.from_pretrained,
            pretrained_model_name_or_path=model_id,
            trust_remote_code=trust,
            local_files_only=local_only,
        )
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
            model = _load_with_retry(
                AutoModelForImageTextToText.from_pretrained,
                pretrained_model_name_or_path=model_id,
                quantization_config=bnb,
                device_map="auto",
                attn_implementation=attn,
                trust_remote_code=trust,
                local_files_only=local_only,
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
            model = _load_with_retry(
                AutoModelForImageTextToText.from_pretrained,
                pretrained_model_name_or_path=model_id,
                torch_dtype=dtype,
                attn_implementation=attn,
                trust_remote_code=trust,
                local_files_only=local_only,
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


def _downcast_lm_head(model):
    """Replace the fp32 CastToFloat lm_head with a plain bf16 Linear.

    Saves ~1.1 GiB on a 7B VLM. Safe because the head is never updated during
    QLoRA (only the LoRA-adapter proj layers train); we just need logits.
    """
    import torch

    try:
        from peft.utils import CastToFloat
    except Exception:
        try:
            from peft.utils.other import CastToFloat
        except Exception:
            CastToFloat = None

    lm = model.get_output_embeddings()
    if CastToFloat is None or not isinstance(lm, CastToFloat):
        return
    orig = lm.float16_linear
    device = orig.weight.device
    new_lm = torch.nn.Linear(orig.in_features, orig.out_features, bias=orig.bias is not None)
    with torch.no_grad():
        new_lm.weight.copy_(orig.weight.to(torch.bfloat16))
        if orig.bias is not None:
            new_lm.bias.copy_(orig.bias.to(torch.bfloat16))
    new_lm = new_lm.to(torch.bfloat16).to(device)
    model.set_output_embeddings(new_lm)
    logger.info("lm_head recast to bf16 (~1.1 GiB saved).")


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
        # prepare_model_for_kbit_training wraps the lm_head in fp32 (CastToFloat),
        # i.e. ~2.18 GiB for a 7B VLM. We don't train the head (LoRA targets the
        # attention/MLP proj layers), so keep it in bf16 to reclaim ~1.1 GiB — the
        # difference between OOM and fitting on a 14 GiB T4.
        _downcast_lm_head(model)
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
