"""Console-script entry point: ``simple_ai_autoresearch_save``.

Persist the Ollama models store to Google Drive so future Colab sessions reuse
the weights instead of re-downloading them. Behavior:

  - The default model is ``qwen2.5-coder:7b``; pass ``--model`` one or more times
    (or space-separated) to save other models instead, e.g.
    ``--model qwen2.5-coder:7b llama3.2``.
  - If a local Ollama store already exists (~/.ollama/models or $OLLAMA_MODELS),
    the requested model(s) are pulled into it (skipping any already present) and
    the whole store is copied to the Drive folder (default
    ``/content/drive/MyDrive/ollama_models``).
  - If there is no local store, ``OLLAMA_MODELS`` is just pointed at the Drive
    folder so a later ``ollama pull`` lands on Drive directly.

The ``--models-dir`` source argument intentionally does NOT auto-default to the
Drive path (unlike setup/serve/train) so the *source* stays the local store.

Examples
--------
    simple_ai_autoresearch_save                                  # save default model to Drive
    simple_ai_autoresearch_save --model qwen2.5-coder:7b llama3.2 # save several models
    simple_ai_autoresearch_save --no-pull                         # copy local store only, no download
    simple_ai_autoresearch_save --drive-dir /content/drive/MyDrive/ollama_models
"""

import argparse
import logging
import os
import shutil

from .autoresearch import _ollama_reachable, pull_model
from .autoresearch_setup import (
    COLAB_MODELS_DIR,
    DEFAULT_MODEL,
    _ensure_drive_mounted,
    ensure_ollama_running,
    install_ollama,
    resolve_models_dir,
)

logger = logging.getLogger(__name__)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(
        prog="simple_ai_autoresearch_save",
        description="Persist the Ollama models store to Google Drive for reuse across sessions.",
    )
    parser.add_argument(
        "--drive-dir",
        default=COLAB_MODELS_DIR,
        help=f"Destination on Google Drive (default: {COLAB_MODELS_DIR}).",
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Local source Ollama models directory (default: $OLLAMA_MODELS or "
        "~/.ollama/models). Use this to re-save an already-Drive store, or to point "
        "at a custom local location.",
    )
    parser.add_argument(
        "--model",
        nargs="*",
        default=None,
        help=f"Model(s) to pull into the local store before saving (default: "
        f"{DEFAULT_MODEL}). Repeatable, e.g. --model qwen2.5-coder:7b llama3.2.",
    )
    parser.add_argument(
        "--no-pull",
        action="store_true",
        help="Do not pull any model; only copy the existing local store.",
    )
    args = parser.parse_args(argv)

    drive_dir = os.path.expanduser(args.drive_dir)
    # Mount Drive (if needed) and ensure the destination exists. We deliberately
    # do NOT set OLLAMA_MODELS to drive_dir yet, so any pull below goes into the
    # *source* store and gets copied to Drive afterward (avoids a self-copy).
    _ensure_drive_mounted(drive_dir)
    os.makedirs(drive_dir, exist_ok=True)

    # Resolve the *source* store. colab_default=False so we never treat the
    # Drive folder as the source on Colab (that would be a no-op copy).
    src = resolve_models_dir(args.models_dir, colab_default=False)
    logger.info("Source Ollama store: %s", src)

    models = args.model if args.model else [DEFAULT_MODEL]
    if not args.no_pull:
        # Pull into the source store so the subsequent copy moves them to Drive.
        os.environ["OLLAMA_MODELS"] = src
        install_ollama()
        if not _ollama_reachable():
            ensure_ollama_running()
        for model in models:
            pull_model(model)

    if os.path.abspath(src) == os.path.abspath(drive_dir):
        logger.info(
            "Source store is already the Drive folder %s; nothing to copy. "
            "Weights are already persisted on Drive.",
            drive_dir,
        )
    elif os.path.isdir(src) and any(os.scandir(src)):
        logger.info("Copying Ollama store %s -> %s ...", src, drive_dir)
        shutil.copytree(src, drive_dir, dirs_exist_ok=True, symlinks=False)
        logger.info("Saved. Weights are now on Google Drive at %s.", drive_dir)
    else:
        logger.info(
            "No local store found at %s; OLLAMA_MODELS now points at %s so a future "
            "`ollama pull` will land on Drive directly.",
            src,
            drive_dir,
        )
        os.environ["OLLAMA_MODELS"] = drive_dir

    logger.info(
        "Next session, reuse without re-downloading by pointing at the Drive store:"
    )
    logger.info("  simple_ai_autoresearch_setup --models-dir %s", drive_dir)
    logger.info("  simple_ai_autoresearch_train --models-dir %s ...", drive_dir)
    logger.info("  simple_ai_autoresearch_serve  --models-dir %s", drive_dir)


if __name__ == "__main__":
    main()
