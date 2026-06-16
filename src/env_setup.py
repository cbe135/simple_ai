import os
import shutil
import tarfile
import zipfile
import logging

logger = logging.getLogger(__name__)


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


def setup_data(args, data_dir="/content"):
    """Set up data directory based on environment and task."""
    task = args.get("task", "d3_liver_ct")

    if task == "d4_hackathon":
        return _setup_data_d4(args, data_dir)
    else:
        return _setup_data_d3(args, data_dir)


# ──────────────────────────────────────────────────────
# D3: Liver CT — ZIP from Google Drive
# ──────────────────────────────────────────────────────

def _setup_data_d3(args, data_dir):
    """D3: Copy ZIP from Drive / download, then unzip."""
    env = detect_environment()
    data_name = args["environ"]["data_name"]
    data_zip = f"{data_name}.zip"
    target_dir = os.path.join(data_dir, data_name)

    if env == "colab":
        _d3_colab(args, data_dir, data_zip, target_dir)
    elif env == "kaggle":
        _d3_kaggle(args, data_dir, data_zip, target_dir)
    else:
        _d3_local(args, data_dir, data_zip, target_dir)

    return target_dir


def _d3_colab(args, data_dir, data_zip, target_dir):
    drive_path = f"/content/drive/MyDrive/{data_zip}"
    local_zip = os.path.join(data_dir, data_zip)

    if os.path.exists(drive_path):
        shutil.copyfile(drive_path, local_zip)

    if os.path.exists(local_zip) and not os.path.exists(target_dir):
        logger.info("Unzipping...")
        with zipfile.ZipFile(local_zip, "r") as zip_ref:
            zip_ref.extractall(data_dir)

    if os.path.exists(target_dir):
        logger.info("D3 data ready")
    else:
        logger.error("D3 data directory not found")


def _d3_kaggle(args, data_dir, data_zip, target_dir):
    kaggle_input = "/kaggle/input"
    data_name = args["environ"]["data_name"]
    local_zip = os.path.join(data_dir, data_zip)

    if os.path.exists(os.path.join(kaggle_input, data_name)):
        shutil.copytree(os.path.join(kaggle_input, data_name), target_dir)
    elif os.path.exists(local_zip) and not os.path.exists(target_dir):
        with zipfile.ZipFile(local_zip, "r") as zip_ref:
            zip_ref.extractall(data_dir)

    if os.path.exists(target_dir):
        logger.info("D3 data ready (Kaggle)")
    else:
        logger.error("D3 data not found in Kaggle input directory")


def _d3_local(args, data_dir, data_zip, target_dir):
    local_zip = os.path.join(data_dir, data_zip)

    if not os.path.exists(target_dir):
        if os.path.exists(local_zip):
            logger.info("Unzipping existing D3 data...")
            with zipfile.ZipFile(local_zip, "r") as zip_ref:
                zip_ref.extractall(data_dir)
        else:
            logger.info("Downloading D3 data via gdown...")
            import gdown

            google_drive_ids = [
                "1LNkFfchl4YwKzLJ5SVDovhyvmw6vUUMf",
                "1vki3HykS0akuKoyLQ11yTtmucr-T4leZ",
                "1ueP6RT9NAxMO2khrqFDvIGHyCYglH0eE",
            ]
            for fid in google_drive_ids:
                url = f"https://drive.google.com/uc?id={fid}"
                out = os.path.join(data_dir, f"{fid}.zip")
                gdown.download(url, out, quiet=False)

            for fid in google_drive_ids:
                fpath = os.path.join(data_dir, f"{fid}.zip")
                if os.path.exists(fpath):
                    with zipfile.ZipFile(fpath, "r") as zip_ref:
                        zip_ref.extractall(data_dir)

            logger.info("D3 data downloaded via gdown")

    if os.path.exists(target_dir):
        logger.info("D3 data ready")
    else:
        logger.error(f"D3 data directory {target_dir} not found")


# ──────────────────────────────────────────────────────
# D4: Hackathon — tar.gz via gdown by group number
# ──────────────────────────────────────────────────────

D4_DISEASE_LIST = [
    "Atelectasis",
    "Cardiomegaly",
    "Colon_Polyp",
    "Diabetic_Retinopathy",
    "Melanoma",
]

D4_FILE_IDS = [
    "1w48H3hLAXT7oxQfy1QvMpkVvzVASOFNQ",
    "1kvE5-nqM4Cp7UjmQaSZwhIBtIlhu_bCj",
    "15R1shEqnTF6QJbTy4QJFG6cW0LXTAjVk",
    "1hdH86SIDh0eO64-ArToRhdLxJ6ICXbuc",
    "1X3Fwtk4Q2rUXWbkyJvuTTo5zhxYgsdM0",
    "1GUKdWnZx8KR-XCy8dWIPn_q4h8vVqFYN",
    "1-EBbLErb3f8CD7lznZ4LZVqcBKn5kb6G",
    "17VhHyTOfxadT3EuoqOWfTDMXjM7ix3ZU",
    "1wOZHBzJtbN15qBy2bFrw4CPkmgYspsni",
    "105qkBolkdQdJxkl3jhUJF48ITvBFCpAP",
]


def _setup_data_d4(args, data_dir):
    """D4: Download tar.gz via gdown, then extract."""
    env = detect_environment()
    data_name = args["environ"]["data_name"]
    group_num = args["environ"].get("group_num", 1)
    target_dir = os.path.join(data_dir, data_name)
    tar_file = os.path.join(data_dir, f"{data_name}.tar.gz")

    if not os.path.exists(target_dir):
        if not os.path.exists(tar_file):
            file_id = D4_FILE_IDS[(group_num - 1) % len(D4_FILE_IDS)]
            logger.info(f"Downloading D4 data (group {group_num}, id={file_id})...")
            import gdown
            gdown.download(id=file_id, output=tar_file)
        else:
            logger.info("D4 tar.gz already downloaded")

        if os.path.exists(tar_file) and not os.path.exists(target_dir):
            logger.info("Extracting D4 tar.gz...")
            with tarfile.open(tar_file, "r") as tar:
                tar.extractall(path=data_dir)

    if os.path.exists(target_dir):
        logger.info("D4 data ready")
    else:
        logger.error(f"D4 data directory {target_dir} not found")

    return target_dir


def get_data_count(args):
    """Log the number of images (and masks if applicable)."""
    task = args.get("task", "d3_liver_ct")
    data_name = args["environ"]["data_name"]
    data_dir = os.path.join("/content", data_name)
    if not os.path.exists(data_dir):
        data_dir = os.path.join(os.getcwd(), data_name)

    if task == "d4_hackathon":
        images_dir = os.path.join(data_dir, "images")
        if os.path.exists(images_dir):
            logger.info(f"Number of images: {len(os.listdir(images_dir))}")
        else:
            logger.info(f"Data directory: {data_dir}")
    else:
        images_dir = os.path.join(data_dir, "images")
        masks_dir = os.path.join(data_dir, "masks")
        if os.path.exists(images_dir):
            logger.info(f"Number of images: {len(os.listdir(images_dir))}")
        if os.path.exists(masks_dir):
            logger.info(f"Number of masks: {len(os.listdir(masks_dir))}")
