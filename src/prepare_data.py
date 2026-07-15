"""
Standalone data preparation: download (via gdown) and extract a dataset into
a data directory. Run this ONCE before ``python src/main.py --data-dir <dir>``.

The pipeline (``src/main.py``) never downloads data; it expects the target
directory to already contain ``data_list.yaml`` (or ``data_list.json``) with a
``data`` list of per-patient dicts. The modality is passed to training via the
``--modality`` flag, not stored in the data list.

Idempotent: if a data list already exists in ``--data-dir``, nothing is done.

Examples
--------
    # Local / Colab: download from Google Drive and extract into the data dir
    python src/prepare_data.py \\
        --data-dir /content/liver_data \\
        --file-ids 1LNkF... 1vki3... 1ueP6... \\
        --archive-format zip

    # Re-run safely: if data_list.yaml already exists, it is a no-op.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
import tarfile
import zipfile
import logging
import argparse

from src.env_setup import detect_environment, default_data_dir

logger = logging.getLogger(__name__)


def mount_drive():
    """Mount Google Drive (Colab only)."""
    from google.colab import drive
    drive.mount("drive", force_remount=True)


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
                f"Place your data in {target_dir} or pass --file-ids."
            )

    if os.path.exists(target_dir):
        logger.info("Data ready (Local)")
    else:
        logger.error(f"Data directory {target_dir} not found")


def _download_via_gdown(data_dir, archive_name, file_ids):
    """Download files from Google Drive via gdown.

    Uses ``fuzzy=True`` so gdown can follow redirects / parse the large-file
    "virus scan" confirmation page, and falls back to a manual confirm-token
    extraction for files that still fail.
    """
    logger.info("Downloading data via gdown...")
    import gdown

    os.makedirs(data_dir, exist_ok=True)

    for fid in file_ids:
        url = f"https://drive.google.com/uc?id={fid}"
        out = os.path.join(data_dir, f"{fid}.zip")
        if os.path.exists(out):
            continue
        try:
            gdown.download(url, out, quiet=False, fuzzy=True)
        except gdown.exceptions.FileURLRetrievalError:
            logger.warning(
                f"gdown failed for {fid}; retrying with confirm-token fallback..."
            )
            _download_with_confirm(fid, out)

    # Merge multiple downloads into one archive if needed
    # (for now, treat each file as a separate archive and extract them all)
    for fid in file_ids:
        fpath = os.path.join(data_dir, f"{fid}.zip")
        if os.path.exists(fpath):
            _extract(data_dir, f"{fid}.zip", os.path.join(data_dir, fid))

    logger.info("Download complete")


def _download_with_confirm(file_id, out_path):
    """Fallback downloader for large files behind Google's virus-scan page.

    Fetches the confirmation token from the initial response and re-requests
    with ``confirm=<token>``. Raises on failure with an actionable message.
    """
    import requests

    session = requests.Session()
    base_url = "https://drive.google.com/uc?export=download"
    resp = session.get(base_url, params={"id": file_id}, stream=True)
    token = None
    for key, value in resp.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break
    if token is None:
        # No confirm token -> permission / quota problem, not a large file.
        raise RuntimeError(
            "Could not retrieve a download confirmation token. The Drive item "
            "is likely not shared as 'Anyone with the link' (Viewer), or has hit "
            "a quota. Set sharing to 'Anyone with the link' and retry."
        )
    resp = session.get(
        base_url,
        params={"id": file_id, "confirm": token},
        stream=True,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)


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


def prepare(args):
    """Prepare the data directory: download + extract as needed."""
    data_dir = args.data_dir
    data_name = args.data_name or os.path.basename(data_dir.rstrip("/")) or "dataset"
    archive_fmt = args.archive_format
    file_ids = args.file_ids or []

    # Idempotency: if a data list already exists, skip everything.
    if os.path.exists(os.path.join(data_dir, "data_list.yaml")) or os.path.exists(
        os.path.join(data_dir, "data_list.json")
    ):
        logger.info(f"Data already prepared at {data_dir}; nothing to do.")
        return data_dir

    os.makedirs(data_dir, exist_ok=True)

    target_dir = data_dir
    archive_name = f"{data_name}.{archive_fmt}"

    env = detect_environment()
    if env == "colab":
        _setup_colab(data_dir, archive_name, target_dir, file_ids)
    elif env == "kaggle":
        _setup_kaggle(data_dir, archive_name, target_dir, data_name)
    else:
        _setup_local(data_dir, archive_name, target_dir, file_ids, archive_fmt)

    return target_dir


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare dataset (download + extract) for the pipeline"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Target data directory that will contain data_list.yaml (e.g. /content/dataset). Modality is set at train time via --modality.",
    )
    parser.add_argument(
        "--data-name",
        type=str,
        default=None,
        help="Dataset name used for archive naming (default: basename of --data-dir).",
    )
    parser.add_argument(
        "--file-ids",
        type=str,
        nargs="+",
        default=None,
        help="One or more Google Drive file IDs to download via gdown.",
    )
    parser.add_argument(
        "--gdown-id",
        type=str,
        default=None,
        help="Convenience alias for a single Google Drive file ID (sets --file-ids).",
    )
    parser.add_argument(
        "--archive-format",
        type=str,
        default="zip",
        help="Archive format: zip or tar.gz",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional config YAML providing environ.data_source (file_ids, "
            "archive_format) and environ.data_name as defaults."
        ),
    )
    args = parser.parse_args(argv)

    if args.gdown_id and not args.file_ids:
        args.file_ids = [args.gdown_id]

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    # Optional config supplies defaults for the missing CLI args.
    if args.config and os.path.exists(args.config):
        import yaml

        with open(args.config, "r") as fp:
            cfg = yaml.safe_load(fp) or {}
        source = cfg.get("environ", {}).get("data_source", {})
        if args.file_ids is None:
            args.file_ids = source.get("file_ids")
        if args.archive_format == "zip":
            args.archive_format = source.get("archive_format", "zip")
        if args.data_name is None:
            args.data_name = cfg.get("environ", {}).get("data_name")

    prepare(args)


if __name__ == "__main__":
    main()
