"""Inference + evaluation for the vlm track.

Two backends:
  * ``hf``   (default) -- loads base (+ LoRA adapter) via transformers and generates.
  * ``ollama`` -- sends image + prompt to a running Ollama server (zero-shot base only).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from PIL import Image

from .data import build_samples, load_data_list, split_indices
from .evaluate import evaluate
from .models import load_for_inference
from .prompts import build_messages, parse_prediction

logger = logging.getLogger(__name__)

SPLIT_TO_IDX = {"train": 0, "val": 1, "test": 2}


def _ollama_available():
    try:
        import ollama  # noqa: F401
        return True
    except Exception:
        return False


def infer_hf(cfg, data_dir, split, adapter, base_dir, device, quantize, run_dir):
    import torch

    data_dir = Path(data_dir)
    entries = load_data_list(data_dir)
    labels = [int(e["label"]) for e in entries]
    idx = split_indices(cfg, len(entries), labels, cfg["data"]["seed"])[SPLIT_TO_IDX[split]]
    samples = build_samples(cfg, entries, idx, data_dir)
    label_map = {int(k): str(v) for k, v in cfg["label_map"].items()}
    image_size = int(cfg["data"]["image_size"])

    model, processor, device = load_for_inference(
        cfg, adapter_path=adapter, base_dir=base_dir, device=device, quantize=quantize
    )
    model.eval()
    tokenizer = processor.tokenizer

    logger.info("Running hf inference over %d %s samples on %s…", len(samples), split, device)

    yes_id = no_id = None
    is_binary = len(label_map) == 2
    if is_binary:
        for i, a in label_map.items():
            if a.lower() == "yes":
                yes_id = tokenizer.convert_tokens_to_ids("yes")
            if a.lower() == "no":
                no_id = tokenizer.convert_tokens_to_ids("no")

    rows, scores = [], []
    from tqdm import tqdm

    for s in tqdm(samples, desc="infer"):
        image = Image.open(s["image"]).convert("RGB").resize((image_size, image_size))
        messages = build_messages(s["prompt"])
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, images=[image],
        ).to(device)

        with torch.inference_mode():
            if is_binary and yes_id is not None and no_id is not None:
                out = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                                     output_scores=True, return_dict_in_generate=True)
                sequences = out.sequences
                scores = out.scores
            else:
                out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
                sequences = out
                scores = None

        gen = sequences[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen, skip_special_tokens=True).strip()
        pred = parse_prediction(text, label_map)

        score = None
        if scores is not None:
            try:
                logits = scores[0][0]
                psy = float(torch.softmax(logits.float(), dim=-1)[yes_id])
                pno = float(torch.softmax(logits.float(), dim=-1)[no_id])
                score = psy / (psy + pno + 1e-9)
            except Exception:
                score = None
        rows.append((s["filename"], s["label"], text, pred))
        scores.append(score)

    return evaluate(rows, label_map, run_dir, scores if is_binary else None, backend="hf")


def infer_ollama(cfg, data_dir, split, ollama_model, run_dir):
    if not _ollama_available():
        raise SystemExit("`ollama` python package not installed (pip install ollama).")
    import ollama

    data_dir = Path(data_dir)
    entries = load_data_list(data_dir)
    labels = [int(e["label"]) for e in entries]
    idx = split_indices(cfg, len(entries), labels, cfg["data"]["seed"])[SPLIT_TO_IDX[split]]
    samples = build_samples(cfg, entries, idx, data_dir)
    label_map = {int(k): str(v) for k, v in cfg["label_map"].items()}
    image_size = int(cfg["data"]["image_size"])

    import base64
    import io

    rows = []
    for s in samples:
        image = Image.open(s["image"]).convert("RGB").resize((image_size, image_size))
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        resp = ollama.chat(
            model=ollama_model,
            messages=[{"role": "user", "content": s["prompt"], "images": [b64]}],
        )
        text = resp["message"]["content"].strip()
        pred = parse_prediction(text, label_map)
        rows.append((s["filename"], s["label"], text, pred))

    return evaluate(rows, label_map, run_dir, None, backend="ollama")


def infer_pipeline(cfg, data_dir, split="test", adapter=None, base_dir=None,
                   device=None, quantize=None, backend="hf", run_dir=None):
    run_dir = Path(run_dir) if run_dir else Path.cwd()
    (run_dir / "vlm").mkdir(parents=True, exist_ok=True)
    run_dir = run_dir / "vlm"

    if backend == "ollama":
        return infer_ollama(cfg, data_dir, split, cfg["model"].get("ollama_model", "qwen2.5vl:7b"), run_dir)

    adapter = Path(adapter) if adapter else None
    if adapter and not adapter.exists():
        raise SystemExit(f"Adapter path not found: {adapter}")
    return infer_hf(cfg, data_dir, split, adapter, base_dir, device, quantize, run_dir)
