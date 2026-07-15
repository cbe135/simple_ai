"""Console-script entry point: ``simple_ai_autoresearch_save``.

Persist the Ollama models store to Google Drive so future Colab sessions reuse
the weights instead of re-downloading them. Behavior (user-selectable):

  - If a local Ollama store already exists (~/.ollama/models or $OLLAMA_MODELS),
    copy it to the Drive folder (default ``/content/drive/MyDrive/ollama_models``).
  - Otherwise, just point ``OLLAMA_MODELS`` at the Drive folder so a subsequent
    ``ollama pull`` lands on Drive. Optionally pull a model first.

The ``--models-dir`` source argument intentionally does NOT auto-default to the
Drive path (unlike setup/serve/train) so the *source* stays the local store.

Examples
--------
    simple_ai_autoresearch_save                                  # copy local store to Drive
    simple_ai_autoresearch_save --drive-dir /content/drive/MyDrive/ollama_models
    simple_ai_autoresearch_save --model qwen2.5-coder:7b         # pull, then save
    simple_ai_autoresearch_save --models-dir /content/drive/MyDrive/ollama_models  # no-op copy
"""

import argparse
import logging
import os
import shutil

from .autoresearch import _ollama_reachable, pull_model
from .autoresearch_setup import (
    COLAB_MODELS_DIR,
    apply_models_dir,
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
        default=None,
        help="Optionally pull this model into the local store before saving it.",
    )
    parser.add_argument(
        "--no-pull",
        action="store_true",
        help="Do not pull a model even if --model is given.",
    )
    args = parser.parse_args(argv)

    drive_dir = os.path.expanduser(args.drive_dir)
    # Ensure Drive is mounted and ready before we read/write under it.
    apply_models_dir(drive_dir, colab_default=False)

    # Optional: make sure the model is present locally before saving.
    if args.model and not args.no_pull:
        install_ollama()
        if not _ollama_reachable():
            ensure_ollama_running()
        pull_model(args.model)

    # Resolve the *source* store. colab_default=False so we never treat the
    # Drive folder as the source on Colab (that would be a no-op copy).
    src = resolve_models_dir(args.models_dir, colab_default=False)
    logger.info("Source Ollama store: %s", src)

    if os.path.isdir(src) and any(os.scandir(src)):
        logger.info("Copying Ollama store %s -> %s ...", src, drive_dir)
        os.makedirs(drive_dir, exist_ok=True)
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
