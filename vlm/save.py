"""simple_ai_vlm_save -- cache the Hugging Face base model so users don't
re-download it every run.

Default backend ``hf``: download ``model_id`` once (via huggingface_hub) into a
persistent directory -- by default Google Drive on Colab
(``/content/drive/MyDrive/vlm_models``), else a local ``./vlm_models``. Future
``simple_ai_vlm_train`` / ``simple_ai_vlm_infer`` runs then pass ``--base-dir``
(or set ``SIMPLE_AI_VLM_BASE_DIR``) to load from that cache, offline.

Optional ``--ollama`` additionally pulls the Ollama base model and copies the
Ollama store to Drive so zero-shot Ollama inference also survives restarts.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def default_models_dir() -> Path:
    """Drive on Colab, else a local ``./vlm_models``."""
    if os.path.exists("/content/drive/MyDrive"):
        return Path("/content/drive/MyDrive/vlm_models")
    return Path.cwd() / "vlm_models"


def save_hf_base(model_id: str, models_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    dest = models_dir / model_id.replace("/", "--")
    if dest.exists() and any(dest.iterdir()):
        logger.info("HF base already cached at %s -- skipping download.", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading HF base %s -> %s (this may take a while)...", model_id, dest)
    from .registry import raise_if_gated

    try:
        snapshot_download(repo_id=model_id, local_dir=str(dest))
    except Exception as exc:
        raise_if_gated(model_id, exc)
    return dest


def save_ollama_base(ollama_model: str, models_dir: Path) -> None:
    """Pull the Ollama base model and copy its store into ``models_dir`` (Drive)."""
    if shutil.which("ollama") is None:
        logger.warning("`ollama` not found; skipping Ollama base cache.")
        return
    logger.info("Pulling Ollama base %s ...", ollama_model)
    subprocess.run(["ollama", "pull", ollama_model], check=False)

    # Reuse the autoresearch Ollama store logic if available.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src.autoresearch_paths import apply_models_dir
        store = apply_models_dir({"models_dir": str(models_dir / "ollama_models")})
    except Exception:
        store = os.environ.get("OLLAMA_MODELS", str(Path.home() / ".ollama" / "models"))

    src = Path(store)
    if src.exists():
        dst = models_dir / "ollama_models"
        dst.mkdir(parents=True, exist_ok=True)
        logger.info("Copying Ollama store %s -> %s", src, dst)
        shutil.copytree(src, dst, dirs_exist_ok=True)


def save_model(model_id: str, ollama_model: str | None, models_dir: Path) -> None:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    dest = save_hf_base(model_id, models_dir)

    if ollama_model:
        save_ollama_base(ollama_model, models_dir)

    logger.info("Base model cached under: %s", models_dir)
    logger.info("Reuse it with:  export SIMPLE_AI_VLM_BASE_DIR=%s", models_dir)
    logger.info("   or pass:  --base-dir %s", models_dir)


def save_cmd():
    import argparse

    from src.cli_help import add_default_flag, parse_with_default

    p = argparse.ArgumentParser(description="Cache the VLM base model (HF, optionally Ollama).")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct", help="HF base model id.")
    p.add_argument("--models-dir", default=None,
                   help="Directory to cache the base model in (default: Drive on Colab, else ./vlm_models).")
    p.add_argument("--ollama-model", default=None, help="Also cache this Ollama base (e.g. qwen2.5vl:7b).")
    p.add_argument("--ollama", action="store_true", help="Shorthand to also cache `qwen2.5vl:7b` via Ollama.")
    add_default_flag(p)
    args = parse_with_default(p)

    models_dir = Path(args.models_dir) if args.models_dir else default_models_dir()
    ollama_model = args.ollama_model or ("qwen2.5vl:7b" if (args.ollama or args.ollama_model) else None)
    save_model(args.model_id, ollama_model, models_dir)
