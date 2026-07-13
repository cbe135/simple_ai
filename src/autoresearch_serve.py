"""Console-script entry point: ``simple_ai_autoresearch_serve``.

Starts the Ollama server in the **background** and returns immediately, so the
calling notebook cell finishes and other cells can run. The autoresearch
training command detects the already-running server (via ``_ollama_reachable``)
and reuses it instead of spawning its own.

This command only starts ``ollama serve``. It does NOT pull or load any model —
``simple_ai_autoresearch_train --local`` does that on demand via
``ensure_ollama_model``.

Examples
--------
    simple_ai_autoresearch_serve                 # start server, return immediately
    simple_ai_autoresearch_serve --models-dir /content/drive/MyDrive/ollama_models
"""

import argparse
import logging
import os
import subprocess
import sys
import time

from .autoresearch import OLLAMA_PORT, _ollama_reachable
from .autoresearch_setup import install_ollama

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = os.path.expanduser("~/.ollama/models")


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(
        prog="simple_ai_autoresearch_serve",
        description="Start the Ollama server in the background and return immediately.",
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Ollama models directory (default: ~/.ollama/models or $OLLAMA_MODELS).",
    )
    args = parser.parse_args(argv)

    models_dir = args.models_dir or os.environ.get("OLLAMA_MODELS") or DEFAULT_MODELS_DIR
    os.makedirs(models_dir, exist_ok=True)
    # ollama serve (and the training command's ollama calls) inherit this.
    os.environ["OLLAMA_MODELS"] = models_dir

    install_ollama()

    if _ollama_reachable():
        logger.info("Ollama server already running on port %d.", OLLAMA_PORT)
        logger.info(
            "Models dir: %s — this cell is done; run training in another cell.",
            models_dir,
        )
        return

    logger.info("Starting Ollama server in the background (ollama serve)...")
    subprocess.Popen(
        ["ollama", "serve"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    for _ in range(60):
        if _ollama_reachable():
            break
        time.sleep(1)
    else:
        raise SystemExit("Ollama server did not become reachable after starting it.")

    logger.info(
        "Ollama serving in background at http://localhost:%d (models: %s). "
        "This cell is done — run `simple_ai_autoresearch_train --local ...` in another cell.",
        OLLAMA_PORT,
        models_dir,
    )


if __name__ == "__main__":
    main()
