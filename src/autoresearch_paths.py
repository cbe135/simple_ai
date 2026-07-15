"""Shared Ollama models-directory resolution (used by setup / serve / train / save).

Kept free of any import from ``autoresearch`` / ``autoresearch_setup`` so it can
be imported from either without creating a circular import.
"""

import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = os.path.expanduser("~/.ollama/models")
COLAB_MODELS_DIR = "/content/drive/MyDrive/ollama_models"


def on_colab() -> bool:
    """Return True if running inside a Google Colab environment.

    Prefers importing ``google.colab`` (works from a Python notebook cell /
    hosted kernel). Falls back to environment/filesystem markers so detection
    also works from a ``!`` shell subprocess, where ``google.colab`` is NOT
    importable but Colab still exposes ``COLAB_*`` env vars and ``/content``.
    """
    try:
        import google.colab  # noqa: F401

        return True
    except Exception:
        pass
    markers = (
        "COLAB_GPU",
        "COLAB_TPU_ADDR",
        "COLAB_RELEASE_TAG",
        "COLAB_NOTEBOOK_METADATA",
    )
    if any(v in os.environ for v in markers) or os.path.isdir("/content"):
        return True
    return False


def _ensure_drive_mounted(drive_dir: str) -> None:
    """Mount Google Drive if ``drive_dir`` lives under /content/drive and isn't mounted.

    Raises a clear error (instead of silently creating a local folder that looks
    like Drive but isn't) when the mount cannot be performed.
    """
    if not drive_dir.startswith("/content/drive"):
        return
    if os.path.ismount("/content/drive"):
        return
    try:
        from google.colab import drive

        logger.info("Mounting Google Drive at /content/drive ...")
        drive.mount("/content/drive")
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "Google Drive is not mounted and `!` commands cannot mount it "
            f"(google.colab is not importable from a shell subprocess: {e}).\n"
            "Mount Drive ONCE in a Python cell (not via !):\n"
            "    from google.colab import drive\n"
            "    drive.mount('/content/drive')\n"
            "then re-run the `!` command. (This is required for "
            "simple_ai_autoresearch_save.)"
        )


def _resolve_shortcut(path: str) -> str:
    """Resolve a Google Drive shortcut (symlink) to its real target.

    On Colab, Drive shortcuts appear as symlinks (e.g. ``MyDrive/ollama_models``
    -> ``drive/.shortcut-targets-by-id/<id>/ollama_models``). ``os.makedirs``
    raises ``FileExistsError`` on a symlink even with ``exist_ok=True`` when the
    target isn't a directory, so callers must resolve before creating.
    """
    if os.path.islink(path):
        return os.path.realpath(path)
    return path


def resolve_models_dir(models_dir_arg=None, colab_default: bool = True) -> str:
    """Resolve the Ollama models directory to use, in priority order:

    1. explicit ``--models-dir`` argument
    2. ``$OLLAMA_MODELS`` environment variable
    3. on Colab (when ``colab_default``): ``COLAB_MODELS_DIR`` (Google Drive)
    4. otherwise: ``DEFAULT_MODELS_DIR`` (``~/.ollama/models``)
    """
    if models_dir_arg:
        return os.path.expanduser(models_dir_arg)
    env = os.environ.get("OLLAMA_MODELS")
    if env:
        return env
    if colab_default and on_colab():
        return COLAB_MODELS_DIR
    return DEFAULT_MODELS_DIR


def apply_models_dir(
    models_dir_arg=None, colab_default: bool = True, require_drive: bool = False
) -> str:
    """Resolve, mount (if on Drive), create, and export ``OLLAMA_MODELS``.

    Returns the resolved path. Child ``ollama`` processes inherit the env var,
    so calling this before ``ollama serve`` / ``ollama pull`` is sufficient.

    When the resolved path lives under ``/content/drive`` but Drive is not
    mounted:
      - if ``require_drive`` is True (e.g. ``save``), attempt to mount and
        hard-fail with a clear message if that's not possible;
      - otherwise (``setup`` / ``train`` / ``serve``), warn and fall back to
        ``DEFAULT_MODELS_DIR`` so the command still runs (weights just won't
        persist across Colab restarts).
    """
    models_dir = resolve_models_dir(models_dir_arg, colab_default=colab_default)
    if models_dir.startswith("/content/drive") and not os.path.ismount("/content/drive"):
        if require_drive:
            _ensure_drive_mounted(models_dir)
        else:
            logger.warning(
                "Google Drive is not mounted, so the auto-default Ollama store "
                "(%s) can't be used. Falling back to %s for this session — weights "
                "will NOT persist across Colab restarts. Mount Drive in a Python "
                "cell (`from google.colab import drive; drive.mount('/content/drive')`) "
                "and re-run to persist.",
                models_dir,
                DEFAULT_MODELS_DIR,
            )
            models_dir = DEFAULT_MODELS_DIR
    # Resolve a Google Drive shortcut (symlink) to its real target so Ollama
    # writes into the actual folder; also creates the target if the shortcut
    # was dangling.
    if os.path.islink(models_dir):
        resolved = os.path.realpath(models_dir)
        if not os.path.exists(resolved):
            logger.warning(
                "Ollama models path %s is a Drive shortcut whose target does not "
                "yet exist; creating the target folder %s so weights can be saved.",
                models_dir,
                resolved,
            )
        else:
            logger.info("Resolved Drive shortcut %s -> %s", models_dir, resolved)
        models_dir = resolved
    try:
        os.makedirs(models_dir, exist_ok=True)
    except OSError as e:
        if models_dir.startswith("/content/drive"):
            raise SystemExit(
                f"Could not create the Ollama models folder at {models_dir} "
                f"({e}). If this path is a Google Drive shortcut to a shared/team "
                "drive, open that target folder once in Google Drive (or re-create "
                "the shortcut so its target is accessible) and re-run. Ollama needs "
                "a writable folder to store weights."
            )
        raise
    os.environ["OLLAMA_MODELS"] = models_dir
    return models_dir
