"""Data handling for the vlm track.

Reuses the same ``data_list.yaml`` convention as ``src/``: a ``data`` list of
per-sample dicts with ``image`` (path) and ``label`` (int). Masks, if present,
are ignored by the VLM path. Everything else (prompt, label_map, splits) is
driven by the vlm config YAML.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

from PIL import Image

from .prompts import build_messages, format_prompt

logger = logging.getLogger(__name__)


def load_data_list(data_dir) -> list[dict]:
    """Return the ``data`` list from data_list.{yaml,yml,json} in ``data_dir``."""
    data_dir = Path(data_dir)
    for name in ("data_list.yaml", "data_list.yml", "data_list.json"):
        p = data_dir / name
        if p.exists():
            with open(p, "r") as fp:
                raw = yaml_safe_load(fp, p.suffix)
            entries = (raw or {}).get("data") or (raw or {}).get("entries") or []
            if not entries:
                raise SystemExit(f"{p} has no 'data' list.")
            for e in entries:
                e.setdefault("filename", e.get("image"))
                lab = e.get("label")
                if isinstance(lab, list):
                    if not lab:
                        raise SystemExit("Each data entry 'label' must not be an empty list.")
                    lab = lab[0]
                if lab is None:
                    raise SystemExit("Each data entry must contain a 'label'.")
                e["label"] = int(lab)
            logger.info("Loaded %d entries from %s", len(entries), p)
            return entries
    raise SystemExit(f"No data_list.yaml/json found in {data_dir}")


def yaml_safe_load(fp, suffix: str):
    if suffix == ".json":
        return json.load(fp)
    import yaml

    return yaml.safe_load(fp)


def split_indices(cfg: dict, n: int, labels: list[int], seed: int):
    """Stratified train/val/test split by percentage. Returns (train, val, test) index lists."""
    d = cfg["data"]
    train_p, val_p, test_p = (
        float(d["train_percentage"]),
        float(d["val_percentage"]),
        float(d["test_percentage"]),
    )
    rng = random.Random(seed)

    if d.get("stratify", True) and len(set(labels)) > 1:
        by_label: dict[int, list[int]] = {}
        for i, lab in enumerate(labels):
            by_label.setdefault(lab, []).append(i)
        train_idx, val_idx, test_idx = [], [], []
        for lab, idxs in by_label.items():
            rng.shuffle(idxs)
            k = len(idxs)
            nt = int(round(train_p * k))
            nv = int(round(val_p * k))
            train_idx += idxs[:nt]
            val_idx += idxs[nt:nt + nv]
            test_idx += idxs[nt + nv:]
        rng.shuffle(train_idx)
    else:
        idxs = list(range(n))
        rng.shuffle(idxs)
        nt = int(round(train_p * n))
        nv = int(round(val_p * n))
        train_idx, val_idx, test_idx = idxs[:nt], idxs[nt:nt + nv], idxs[nt + nv:]

    logger.info(
        "Split -> train: %d, val: %d, test: %d", len(train_idx), len(val_idx), len(test_idx)
    )
    return train_idx, val_idx, test_idx


def build_samples(cfg: dict, entries: list[dict], indices: list[int], data_dir) -> list[dict]:
    """Build per-sample dicts: resolved image path, label, prompt text, target text."""
    ptext = format_prompt(cfg["prompt"], cfg["modality"], cfg["condition"])
    label_map = {int(k): str(v) for k, v in cfg["label_map"].items()}
    data_dir = Path(data_dir)
    out = []
    for i in indices:
        e = entries[i]
        img = str(e["image"])
        if not os.path.isabs(img):
            img = str(data_dir / img)
        lab = int(e["label"])
        out.append({
            "image": img,
            "label": lab,
            "filename": e.get("filename", img),
            "prompt": ptext,
            "target": label_map.get(lab, str(lab)),
        })
    return out


class VLMDataset:
    """Loads images and builds Gemma3 chat messages on the fly.

    A plain map-style dataset (no torch import needed at definition time).
    """

    def __init__(self, samples: list[dict], image_size: int):
        self.samples = samples
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: str):
        return Image.open(path).convert("RGB").resize((self.image_size, self.image_size))

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = self._load_image(s["image"])
        user = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": s["prompt"]}]}]
        full = user + [{"role": "assistant", "content": s["target"]}]
        return {"image": image, "user_messages": user, "messages": full,
                "filename": s["filename"], "label": s["label"]}


class VLMCollator:
    """Tokenize VLM examples and mask the prompt so loss is on the answer only.

    Requires ``transformers`` >= 4.45 (``apply_chat_template`` with ``images``)
    and ``torch`` (both imported lazily at call time).
    """

    def __init__(self, processor, max_length: int):
        import torch

        self.processor = processor
        self.max_length = max_length
        self._torch = torch
        tok = processor.tokenizer
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def _tokenize(self, messages, image, add_generation_prompt: bool) -> dict:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        out = self.processor(text=text, images=[image], return_tensors="pt")
        ids = out["input_ids"][0]
        am = out.get("attention_mask")
        if am is None:
            am = self._torch.ones_like(ids)
        return {
            "input_ids": ids.tolist(),
            "attention_mask": am.tolist(),
            "pixel_values": out.get("pixel_values"),
            "image_grid_thw": out.get("image_grid_thw"),
        }

    def _pad(self, seqs, value):
        torch = self._torch
        maxlen = min(self.max_length, max(len(s) for s in seqs))
        out = []
        for s in seqs:
            pad = maxlen - len(s)
            out.append(s + [value] * pad)
        return torch.tensor(out)

    def __call__(self, batch):
        torch = self._torch
        input_ids, labels, attn, pixel_values, grids = [], [], [], [], []
        for ex in batch:
            full = self._tokenize(ex["messages"], ex["image"], add_generation_prompt=False)
            prompt_len = len(self._tokenize(ex["user_messages"], ex["image"], add_generation_prompt=True)["input_ids"])

            ids = full["input_ids"]
            lbl = list(ids)
            lbl[:prompt_len] = [-100] * prompt_len

            input_ids.append(ids)
            labels.append(lbl)
            attn.append(full.get("attention_mask", [1] * len(ids)))
            pv = full.get("pixel_values")
            pixel_values.append(pv if pv is not None else torch.zeros(1))
            g = full.get("image_grid_thw")
            grids.append(g if g is not None else torch.zeros(1, 3, dtype=torch.long))

        return {
            "input_ids": self._pad(input_ids, self.pad_id),
            "attention_mask": self._pad(attn, 0),
            "labels": self._pad(labels, -100),
            "pixel_values": torch.cat(pixel_values, dim=0) if pixel_values[0].dim() > 0 else None,
            "image_grid_thw": torch.cat(grids, dim=0) if grids[0].dim() > 0 else None,
        }
