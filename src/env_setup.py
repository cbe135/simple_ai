"""
Environment detection and data directory discovery.

The pipeline itself never downloads or extracts data. Data preparation
(download via gdown, archive extraction) lives in ``src/prepare_data.py``,
a standalone script the user runs once before ``main.py``.

The data directory passed to ``--data-dir`` must contain ``data_list.yaml``
(or ``data_list.json``) with a top-level ``modality`` key and a ``data`` list.
"""
import os
import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────
# Environment detection
# ──────────────────────────────────────────────────────

def detect_environment():
    """Detect the runtime environment: Colab, Kaggle, or Local."""
    try:
        from google.colab import _reprs
        return "colab"
    except ImportError:
        pass

    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "kaggle"

    return "local"


def default_data_dir():
    """Return the environment-aware base directory for data.

    Colab / Kaggle use ``/content`` (the writable scratch space), while
    local runs default to the current working directory.
    """
    env = detect_environment()
    if env in ("colab", "kaggle"):
        return "/content"
    return os.getcwd()


# ──────────────────────────────────────────────────────
# Data directory validation / discovery
# ──────────────────────────────────────────────────────

def _has_data_list(data_dir):
    """Return True if data_dir contains a data list file."""
    return os.path.exists(os.path.join(data_dir, "data_list.yaml")) or os.path.exists(
        os.path.join(data_dir, "data_list.json")
    )


def setup_data(args, data_dir=None):
    """Validate that the data directory exists and contains a data list.

    No download or extraction is performed. ``--data-dir`` must point at the
    directory containing ``data_list.yaml`` (or ``data_list.json``).
    """
    if data_dir is None:
        data_dir = default_data_dir()
    if _has_data_list(data_dir):
        logger.info("Data ready")
    else:
        logger.error(
            f"Data directory missing data_list.yaml/json: {data_dir}"
        )
    return data_dir


def get_data_count(args, data_dir=None):
    """Log the number of images (and masks if present)."""
    if data_dir is None:
        data_dir = default_data_dir()

    images_dir = os.path.join(data_dir, "images")
    masks_dir = os.path.join(data_dir, "masks")

    if os.path.exists(images_dir):
        logger.info(f"Number of images: {len(os.listdir(images_dir))}")
    if os.path.exists(masks_dir):
        logger.info(f"Number of masks: {len(os.listdir(masks_dir))}")


def find_data_dir(args, data_dir=None):
    """Locate the actual data directory (might be nested)."""
    if data_dir is None:
        data_dir = default_data_dir()

    candidates = [data_dir]
    # Convenience fallback: a directory with the same basename under cwd.
    cwd_candidate = os.path.join(os.getcwd(), os.path.basename(data_dir))
    if cwd_candidate != data_dir:
        candidates.append(cwd_candidate)

    for d in candidates:
        if not os.path.exists(d):
            continue
        if _has_data_list(d):
            return d
        # Check one level deeper
        for sub in os.listdir(d):
            sub_path = os.path.join(d, sub)
            if os.path.isdir(sub_path) and _has_data_list(sub_path):
                return sub_path

    return candidates[0]  # fallback
