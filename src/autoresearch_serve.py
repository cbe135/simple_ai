"""Console-script entry point: ``simple_ai_autoresearch_serve``.

Starts the Ollama server in the **background** and returns immediately, so the
calling notebook cell finishes and other cells can run. The autoresearch
training command detects the already-running server (via ``_ollama_reachable``)
and reuses it instead of spawning its own.

Options:
  - ``--warm-start``: also load the model into GPU memory (blocks until ready).
  - ``--expose {proxy,localtunnel}``: expose the server so another Colab
    instance (or any machine) can reach it; the external URL is printed.

This command only starts ``ollama serve`` (+ optional warm/expose). It does NOT
run training — ``simple_ai_autoresearch_train --local`` loads the model on
demand via ``ensure_ollama_model`` when ``--warm-start`` is not used.

Examples
--------
    simple_ai_autoresearch_serve                      # start server, return now
    simple_ai_autoresearch_serve --warm-start         # also load the model (blocks)
    simple_ai_autoresearch_serve --expose proxy       # Colab account-only URL
    simple_ai_autoresearch_serve --expose localtunnel # public URL
    simple_ai_autoresearch_serve --models-dir /content/drive/MyDrive/ollama_models
"""

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

from src.cli_help import add_default_flag, parse_with_default
from .autoresearch import OLLAMA_PORT, _ollama_reachable, ensure_ollama_model
from .autoresearch_setup import apply_models_dir, install_ollama

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen2.5-coder:7b"


def _start_server():
    """Start ``ollama serve`` detached and wait until it answers."""
    subprocess.Popen(
        ["ollama", "serve"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    for _ in range(60):
        if _ollama_reachable():
            return
        time.sleep(1)
    raise SystemExit("Ollama server did not become reachable after starting it.")


def _expose_proxy():
    """Print a Colab proxy URL for the Ollama port (Method 1).

    ``google.colab`` is only importable inside the notebook kernel, not in a
    ``!`` subprocess, so if it isn't available we print the exact snippet to
    run in a Python cell instead.
    """
    try:
        from google.colab.output import eval_js

        url = eval_js("google.colab.kernel.proxyPort(%d)" % OLLAMA_PORT)
        logger.info("Exposed via Colab proxy: %s", url)
        return
    except Exception:
        pass
    logger.info(
        "Could not auto-generate the Colab proxy URL (google.colab not available "
        "in this process). Run this in a Python cell to get your external URL:\n"
        "    from google.colab.output import eval_js\n"
        "    print(eval_js('google.colab.kernel.proxyPort(%d)'))\n"
        "Open the printed URL in a new tab. Note: the proxy is only reachable by "
        "your Google account.", OLLAMA_PORT,
    )


def _expose_localtunnel():
    """Expose the Ollama port via localtunnel (Method 2, public URL)."""
    if not shutil.which("npx"):
        logger.info("npx not found; attempting to install nodejs/npm...")
        try:
            subprocess.run(["apt-get", "update", "-qq"], check=True)
            subprocess.run(["apt-get", "install", "-y", "nodejs", "npm"], check=True)
        except subprocess.CalledProcessError as e:
            raise SystemExit(
                f"Failed to install node/npm: {e}. Install node manually, then "
                "re-run with --expose localtunnel."
            )

    log_path = tempfile.NamedTemporaryFile(delete=False, suffix=".log").name
    with open(log_path, "w") as logf:
        subprocess.Popen(
            ["npx", "localtunnel", "--port", str(OLLAMA_PORT)],
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    url = None
    for _ in range(90):
        try:
            with open(log_path) as f:
                text = f.read()
        except FileNotFoundError:
            text = ""
        m = re.search(r"https://[^\s'\"<>]+\.loca\.lt", text)
        if m:
            url = m.group(0)
            break
        time.sleep(1)

    if not url:
        raise SystemExit(
            f"localtunnel did not report a URL (see log: {log_path}). "
            "Is node/npx working? The URL is public once available."
        )
    logger.info("Exposed via localtunnel: %s  (tunnel log: %s)", url, log_path)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(
        prog="simple_ai_autoresearch_serve",
        description="Start the Ollama server in the background and return immediately.",
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Ollama models directory. Default on Colab: "
        "/content/drive/MyDrive/ollama_models; otherwise ~/.ollama/models. Can also "
        "be set via $OLLAMA_MODELS.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model to warm (with --warm-start). Default: {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--warm-start",
        action="store_true",
        help="Also load the model into GPU memory (blocks the cell until ready). "
        "Sets OLLAMA_KEEP_ALIVE=-1 so it stays resident for a remote consumer.",
    )
    parser.add_argument(
        "--expose",
        choices=["none", "proxy", "localtunnel"],
        default="localtunnel",
        help="Expose the server to another Colab instance (default: localtunnel=public "
        "URL; proxy=Colab account-only URL; none=don't expose). The external URL is printed.",
    )
    add_default_flag(parser)
    args = parse_with_default(parser, argv)

    models_dir = apply_models_dir(args.models_dir)
    # ollama serve (and the training command's ollama calls) inherit this.

    if args.warm_start:
        # Keep the loaded model resident for a remote consumer (no effect on an
        # already-running server started without this).
        os.environ["OLLAMA_KEEP_ALIVE"] = "-1"

    install_ollama()

    if _ollama_reachable():
        logger.info("Ollama server already running on port %d.", OLLAMA_PORT)
    else:
        logger.info("Starting Ollama server in the background (ollama serve)...")
        _start_server()
        logger.info("Ollama serving in background at http://localhost:%d.", OLLAMA_PORT)

    if args.warm_start:
        logger.info("Warming up model %s (loading into GPU, this may take a while)...", args.model)
        ensure_ollama_model(args.model)
        logger.info("Model %s loaded and ready.", args.model)

    if args.expose == "proxy":
        _expose_proxy()
    elif args.expose == "localtunnel":
        _expose_localtunnel()

    logger.info(
        "This cell is done. Models: %s. Run `simple_ai_autoresearch_train --local ...` "
        "in another cell (or reference this server's URL from another instance).",
        models_dir,
    )


if __name__ == "__main__":
    main()
