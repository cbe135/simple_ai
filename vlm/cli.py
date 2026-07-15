"""CLI entry points for the vlm track:

  simple_ai_vlm_train   -- fine-tune (QLoRA / pure LoRA) on one task
  simple_ai_vlm_infer   -- run inference + metrics (hf or ollama backend)
  simple_ai_vlm_save    -- cache the HF base model (see vlm/vlm/save.py)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger("vlm")

# Make the repo root importable (so `vlm` and `src` resolve regardless of cwd).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _setup_logging():
    load_dotenv()  # pick up .env (e.g. HF_TOKEN) for huggingface_hub
    level = getattr(logging, os.environ.get("SIMPLE_AI_LOG_LEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _base_dir_arg(args) -> str | None:
    return args.base_dir or os.environ.get("SIMPLE_AI_VLM_BASE_DIR")


def _common(p: argparse.ArgumentParser):
    p.add_argument("--config", required=True, help="Path to a vlm config YAML (e.g. vlm/configs/melanoma.yaml).")
    p.add_argument("--data-dir", required=True, help="Directory containing data_list.yaml + images/.")
    p.add_argument("--base-dir", default=None, help="Cached base-model dir (from simple_ai_vlm_save).")
    p.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])


def train_cmd():
    _setup_logging()
    p = argparse.ArgumentParser(description="Fine-tune a medical VLM (one task).")
    _common(p)
    p.add_argument("--quantize", default=None, choices=["4bit", "none"], help="Override config quantize.")
    p.add_argument("--output-dir", default=None, help="Override config output_dir.")
    args = p.parse_args()

    from .config import load_config
    from .train import train_pipeline

    cfg = load_config(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    run_dir, adapter = train_pipeline(
        cfg, args.data_dir, base_dir=_base_dir_arg(args), device=args.device, quantize=args.quantize
    )
    logger.info("Done. Adapter: %s", adapter)


def infer_cmd():
    _setup_logging()
    p = argparse.ArgumentParser(description="Run VLM inference + metrics (one task).")
    _common(p)
    p.add_argument("--adapter", default=None, help="Path to a saved LoRA adapter (hf backend).")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--backend", default="hf", choices=["hf", "ollama"])
    p.add_argument("--quantize", default=None, choices=["4bit", "none"])
    args = p.parse_args()

    from .config import load_config
    from .infer import infer_pipeline

    cfg = load_config(args.config)
    infer_pipeline(
        cfg, args.data_dir, split=args.split, adapter=args.adapter,
        base_dir=_base_dir_arg(args), device=args.device, quantize=args.quantize,
        backend=args.backend,
    )


def save_cmd():
    _setup_logging()
    # `save` has its own argument set (see vlm/vlm/save.py).
    from .save import save_cmd as _save

    _save()


if __name__ == "__main__":
    # Allow `python -m vlm.cli train|infer|save ...`
    _setup_logging()
    which = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("train", "infer", "save") else "train"
    sys.argv = [sys.argv[0]] + sys.argv[2:] if which != "train" else sys.argv
    if which == "train":
        train_cmd()
    elif which == "infer":
        infer_cmd()
    else:
        save_cmd()
