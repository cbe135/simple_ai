"""Console-script entry point: ``simple_ai_autoresearch_setup``.

One-shot environment setup for the local-Ollama autoresearch backend. It:

  1. Installs Ollama if the ``ollama`` binary is missing.
  2. Starts the Ollama server (or confirms it is already running).
  3. Verifies a usable NVIDIA GPU / driver (the "driver match" gate Ollama
     needs to run on GPU instead of silently falling back to CPU).
  4. Pre-pulls the default model so the long autoresearch loop starts clean.

It is idempotent and safe to re-run. On Colab a GPU (T4) runtime with a
CUDA >= 12 image must be selected first, otherwise step 3 will tell you.

Examples
--------
    simple_ai_autoresearch_setup                 # install + verify + pull default
    simple_ai_autoresearch_setup --no-pull       # skip the ~5GB model download
    simple_ai_autoresearch_setup --model qwen2.5-coder:7b
"""

import argparse
import logging
import shutil
import subprocess
import sys
import time

from .autoresearch import OLLAMA_PORT, _ollama_reachable

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen2.5-coder:7b"


# --------------------------------------------------------------------------- #
# Ollama installation
# --------------------------------------------------------------------------- #
def _ensure_linux_deps() -> None:
    """Ensure extraction tools the Ollama installer needs are present (e.g. zstd)."""
    if shutil.which("zstd"):
        return
    logger.info("zstd missing; installing it (needed by the Ollama installer)...")

    if shutil.which("apt-get"):
        runner = ["bash", "-c", "apt-get update -qq && apt-get install -y zstd"]
    elif shutil.which("dnf"):
        runner = ["dnf", "install", "-y", "zstd"]
    elif shutil.which("pacman"):
        runner = ["pacman", "-S", "--noconfirm", "zstd"]
    else:
        raise SystemExit(
            "Could not install 'zstd' automatically (no recognized package "
            "manager). Install it manually, then re-run:\n"
            "  Debian/Ubuntu: sudo apt-get install zstd\n"
            "  RHEL/CentOS/Fedora: sudo dnf install zstd\n"
            "  Arch: sudo pacman -S zstd"
        )

    try:
        subprocess.run(runner, check=True)
    except subprocess.CalledProcessError as e:
        raise SystemExit(
            f"Failed to install zstd: {e}\nInstall it manually, then re-run:\n"
            "  Debian/Ubuntu: sudo apt-get install zstd\n"
            "  RHEL/CentOS/Fedora: sudo dnf install zstd\n"
            "  Arch: sudo pacman -S zstd"
        )

    if not shutil.which("zstd"):
        raise SystemExit("zstd still not found after install. Install it manually and re-run.")


def install_ollama(force: bool = False) -> None:
    """Install the Ollama CLI if missing (or when ``force``)."""
    if shutil.which("ollama") and not force:
        logger.info("Ollama already installed; skipping install.")
        return

    if force:
        logger.info("Forcing Ollama reinstall...")

    system = sys.platform
    try:
        if system.startswith("linux"):
            _ensure_linux_deps()
            logger.info("Installing Ollama via official install script...")
            subprocess.run(
                "curl -fsSL https://ollama.com/install.sh | sh",
                shell=True,
                check=True,
            )
        elif system == "darwin":
            if shutil.which("brew"):
                logger.info("Installing Ollama via Homebrew...")
                subprocess.run(["brew", "install", "ollama"], check=True)
            else:
                _ensure_linux_deps()
                logger.info("Installing Ollama via official install script...")
                subprocess.run(
                    "curl -fsSL https://ollama.com/install.sh | sh",
                    shell=True,
                    check=True,
                )
        else:
            raise RuntimeError(
                "Automatic Ollama install is only supported on Linux/macOS. "
                "Install it manually from https://ollama.com/download and re-run."
            )
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"Ollama install failed: {e}\nInstall it manually from https://ollama.com/download")

    if not shutil.which("ollama"):
        raise SystemExit(
            "Ollama was not found on PATH after install. "
            "Install it manually from https://ollama.com/download and re-run."
        )
    logger.info("Ollama installed successfully.")


# --------------------------------------------------------------------------- #
# Server lifecycle
# --------------------------------------------------------------------------- #
def ensure_ollama_running() -> bool:
    """Start ``ollama serve`` if not already up. Returns True if we started it.

    The server is intentionally left running so the subsequent
    ``simple_ai_autoresearch_train --local`` can reuse it without re-spawning.
    """
    if _ollama_reachable():
        logger.info("Ollama server already running on port %d.", OLLAMA_PORT)
        return False

    logger.info("Starting Ollama server (ollama serve)...")
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(60):
        if _ollama_reachable():
            logger.info("Ollama server is up.")
            return True
        time.sleep(1)
    raise SystemExit("Ollama server did not become reachable after starting it.")


# --------------------------------------------------------------------------- #
# GPU / driver verification
# --------------------------------------------------------------------------- #
def check_gpu() -> None:
    """Verify an NVIDIA GPU + driver Ollama can use. Exits if none is usable."""
    if not shutil.which("nvidia-smi"):
        raise SystemExit(
            "No NVIDIA GPU runtime detected (nvidia-smi not found).\n"
            "On Google Colab: Runtime ▸ Change runtime type ▸ Hardware accelerator = GPU "
            "(choose T4) and a CUDA 12.x image. Ollama needs a CUDA >= 12 GPU to run; "
            "without it training falls back to very slow CPU."
        )

    try:
        res = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        info = res.stdout
    except Exception as e:
        raise SystemExit(f"nvidia-smi failed: {e}")

    # Surface the GPU name + the max CUDA version the driver supports.
    name_line = next((l for l in info.splitlines() if "NVIDIA" in l), "")
    cuda_line = next((l for l in info.splitlines() if "CUDA Version" in l), "")
    if name_line:
        logger.info("GPU: %s", name_line.strip())
    if cuda_line:
        logger.info("Driver: %s", cuda_line.strip())

    # Ollama needs the driver to support at least CUDA 12.
    import re

    m = re.search(r"CUDA Version:\s*([0-9]+)\.([0-9]+)", cuda_line)
    if m:
        major = int(m.group(1))
        if major < 12:
            logger.warning(
                "Driver only supports CUDA %s.x. Ollama requires CUDA >= 12 to use the "
                "GPU and will likely fall back to CPU. On Colab pick a CUDA 12.x image "
                "(Runtime ▸ Change runtime type) or use OpenRouter (--remote).",
                major,
            )


def verify_ollama_gpu(model: str) -> None:
    """Confirm Ollama actually runs the model on the GPU, not CPU."""
    logger.info("Warming up model %s to verify GPU usage...", model)
    warm = subprocess.run(
        ["ollama", "run", model, "say hi"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if warm.returncode != 0:
        logger.warning("Model warm-up failed: %s", (warm.stderr or "").strip())

    ps = subprocess.run(["ollama", "ps"], capture_output=True, text=True)
    out = ps.stdout or ""
    if "GPU" in out and "CPU" not in out:
        logger.info("Ollama is using the GPU. ✓")
    elif "CPU" in out:
        logger.warning(
            "Ollama is running the model on CPU only. On Colab you must use a CUDA >= 12 "
            "runtime image (Runtime ▸ Change runtime type ▸ CUDA 12.x) and a T4 GPU, "
            "otherwise Ollama silently falls back to CPU. Consider --remote (OpenRouter) "
            "if the GPU driver cannot be upgraded."
        )
    else:
        logger.info("GPU status could not be confirmed from `ollama ps`; continuing.")


# --------------------------------------------------------------------------- #
# Model pull
# --------------------------------------------------------------------------- #
def pull_model(model: str) -> None:
    res = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if model in (res.stdout or ""):
        logger.info("Model %s already present; skipping pull.", model)
        return
    logger.info("Pulling Ollama model %s (this may take a while, ~5GB)...", model)
    try:
        subprocess.run(["ollama", "pull", model], check=True)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"Failed to pull model {model}: {e}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(
        prog="simple_ai_autoresearch_setup",
        description="Install Ollama, verify GPU/driver, and pull the autoresearch model.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model to pre-pull (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--no-pull",
        action="store_true",
        help="Skip pulling the model (download happens later in the training run).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reinstall Ollama even if already present.",
    )
    args = parser.parse_args(argv)

    install_ollama(force=args.force)
    ensure_ollama_running()
    check_gpu()
    verify_ollama_gpu(args.model)
    if not args.no_pull:
        pull_model(args.model)

    logger.info("Setup complete. Next: simple_ai_autoresearch_train --local ...")


if __name__ == "__main__":
    main()
