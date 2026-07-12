"""
Environment detection and data setup.

Data source configuration lives in args["environ"]["data_source"]:
    file_ids:        list of Google Drive file IDs
    archive_format:  "zip" or "tar.gz"

No hardcoded task names. All source info comes from the config.
"""
import os
import shutil
import tarfile
import zipfile
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


def mount_drive():
    """Mount Google Drive (Colab only)."""
    from google.colab import drive
    drive.mount("drive", force_remount=True)


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
# Data setup
# ──────────────────────────────────────────────────────

def setup_data(args, data_dir=None):
    """Set up data directory based on environment and config."""
    if data_dir is None:
        data_dir = default_data_dir()
    env = detect_environment()
    data_name = args["environ"]["data_name"]
    source = args["environ"].get("data_source", {})
    archive_fmt = source.get("archive_format", "zip")
    file_ids = source.get("file_ids", [])
    group_num = source.get("group_num", None)

    target_dir = os.path.join(data_dir, data_name)
    archive_name = f"{data_name}.{archive_fmt}"

    if env == "colab":
        _setup_colab(data_dir, archive_name, target_dir, file_ids)
    elif env == "kaggle":
        _setup_kaggle(data_dir, archive_name, target_dir, data_name)
    else:
        _setup_local(data_dir, archive_name, target_dir, file_ids, archive_fmt)

    return target_dir


def _setup_colab(data_dir, archive_name, target_dir, file_ids):
    """Copy archive from Google Drive, then extract."""
    drive_path = f"/content/drive/MyDrive/{archive_name}"
    local_archive = os.path.join(data_dir, archive_name)

    if os.path.exists(drive_path):
        shutil.copyfile(drive_path, local_archive)

    if not os.path.exists(local_archive) and file_ids:
        _download_via_gdown(data_dir, archive_name, file_ids)

    _extract_if_needed(data_dir, archive_name, target_dir)

    if os.path.exists(target_dir):
        logger.info("Data ready (Colab)")
    else:
        logger.error("Data directory not found")


def _setup_kaggle(data_dir, archive_name, target_dir, data_name):
    """Check for data in Kaggle input directory."""
    kaggle_input = "/kaggle/input"
    local_archive = os.path.join(data_dir, archive_name)

    if os.path.exists(os.path.join(kaggle_input, data_name)):
        shutil.copytree(os.path.join(kaggle_input, data_name), target_dir)
    elif os.path.exists(local_archive) and not os.path.exists(target_dir):
        _extract(data_dir, archive_name, target_dir)

    if os.path.exists(target_dir):
        logger.info("Data ready (Kaggle)")
    else:
        logger.error("Data not found in Kaggle input directory")


def _setup_local(data_dir, archive_name, target_dir, file_ids, archive_fmt):
    """Download via gdown if needed, then extract."""
    local_archive = os.path.join(data_dir, archive_name)

    if not os.path.exists(target_dir):
        if os.path.exists(local_archive):
            logger.info("Extracting existing data...")
            _extract(data_dir, archive_name, target_dir)
        elif file_ids:
            _download_via_gdown(data_dir, archive_name, file_ids)
            _extract(data_dir, archive_name, target_dir)
        else:
            logger.warning(
                f"No data source configured and no archive found at {local_archive}. "
                f"Place your data in {target_dir} or add file_ids to config."
            )

    if os.path.exists(target_dir):
        logger.info("Data ready (Local)")
    else:
        logger.error(f"Data directory {target_dir} not found")


def _download_via_gdown(data_dir, archive_name, file_ids):
    """Download files from Google Drive via gdown."""
    logger.info("Downloading data via gdown...")
    import gdown

    for fid in file_ids:
        url = f"https://drive.google.com/uc?id={fid}"
        out = os.path.join(data_dir, f"{fid}.zip")
        if not os.path.exists(out):
            gdown.download(url, out, quiet=False)

    # Merge multiple downloads into one archive if needed
    # (for now, treat each file as a separate archive and extract them all)
    for fid in file_ids:
        fpath = os.path.join(data_dir, f"{fid}.zip")
        if os.path.exists(fpath):
            _extract(data_dir, f"{fid}.zip", os.path.join(data_dir, fid))

    logger.info("Download complete")


def _extract(data_dir, archive_name, target_dir):
    """Extract archive if target_dir doesn't exist."""
    archive_path = os.path.join(data_dir, archive_name)
    if not os.path.exists(archive_path):
        return

    if archive_name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(data_dir)
    elif archive_name.endswith(".tar.gz") or archive_name.endswith(".tgz"):
        with tarfile.open(archive_path, "r") as tf:
            tf.extractall(path=data_dir)
    else:
        logger.warning(f"Unknown archive format: {archive_name}")


def _extract_if_needed(data_dir, archive_name, target_dir):
    """Extract archive only if target_dir doesn't exist yet."""
    if not os.path.exists(target_dir):
        _extract(data_dir, archive_name, target_dir)


def get_data_count(args, data_dir=None):
    """Log the number of images (and masks if present)."""
    data_name = args["environ"]["data_name"]
    if data_dir is None:
        data_dir = default_data_dir()
    data_dir = os.path.join(data_dir, data_name)
    if not os.path.exists(data_dir):
        data_dir = os.path.join(os.getcwd(), data_name)

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
    data_name = args["environ"]["data_name"]
    candidates = [
        os.path.join(data_dir, data_name),
        os.path.join(os.getcwd(), data_name),
    ]

    for d in candidates:
        if os.path.exists(d):
            # Check if data_list.yaml exists directly or one level deeper
            if os.path.exists(os.path.join(d, "data_list.yaml")):
                return d
            for sub in os.listdir(d):
                sub_path = os.path.join(d, sub)
                if os.path.isdir(sub_path) and os.path.exists(
                    os.path.join(sub_path, "data_list.yaml")
                ):
                    return sub_path

    return candidates[0]  # fallback
