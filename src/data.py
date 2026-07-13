import os
import json
import yaml
import logging

from sklearn.model_selection import train_test_split

from .model import get_device

logger = logging.getLogger(__name__)


def load_data_list(args, data_dir=None):
    """Load the data list and return the list of data dicts.

    Supports data_list.yaml (preferred) with a data_list.json fallback.
    Both are expected to expose a top-level "data" key.
    """
    if data_dir is None:
        from src.env_setup import default_data_dir

        data_dir = default_data_dir()

    yaml_path = os.path.join(data_dir, "data_list.yaml")
    json_path = os.path.join(data_dir, "data_list.json")

    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as fp:
            return yaml.safe_load(fp)["data"]

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as fp:
            return json.load(fp)["data"]

    raise FileNotFoundError(
        f"No data list found in {data_dir} (expected data_list.yaml or data_list.json)"
    )


def populate_data_lists(args, data_dicts):
    """Split data into training, validation, and testing sets."""
    labels = [a["label"] for a in data_dicts]
    train_dicts, val_test_dicts = train_test_split(
        data_dicts,
        train_size=args["data"]["train_percentage"],
        test_size=args["data"]["val_percentage"] + args["data"]["test_percentage"],
        stratify=labels,
        random_state=args["environ"]["seed"],
        shuffle=True,
    )

    val_test_labels = [a["label"] for a in val_test_dicts]
    val_dicts, test_dicts = train_test_split(
        val_test_dicts,
        train_size=args["data"]["val_percentage"]
        / (args["data"]["val_percentage"] + args["data"]["test_percentage"]),
        test_size=args["data"]["test_percentage"]
        / (args["data"]["val_percentage"] + args["data"]["test_percentage"]),
        stratify=val_test_labels,
        random_state=args["environ"]["seed"],
        shuffle=True,
    )
    return train_dicts, val_dicts, test_dicts


def check_dist(dataset):
    """Check and log the label distribution of a dataset."""
    positive = 0
    negative = 0
    for data in dataset:
        if data["label"].item() == 1:
            positive += 1
        else:
            negative += 1
    logger.info(
        f"number of positive = {positive:.0f}, "
        f"number of negative = {negative:.0f}, "
        f"number of total data = {len(dataset)}"
    )


def generate_dataset(args, datalist, transform):
    """Create a CacheDataset from a data list and transform pipeline."""
    from monai.data import CacheDataset

    cache_rate = args["data"]["cache_rate"]
    logger.info(
        "Creating CacheDataset (cache_rate=%s, num_items=%d)",
        cache_rate, len(datalist),
    )
    dataset = CacheDataset(datalist, transform, cache_rate=cache_rate)
    logger.info(
        "CacheDataset ready (cache_rate=%s); items are cached on first load",
        cache_rate,
    )
    return dataset


def generate_dataloader(args, dataset, shuffle=False, device=None):
    """Create a DataLoader from a dataset.

    Uses parallel workers and pinned memory (when on a GPU) so the device is
    fed continuously instead of waiting on single-process CPU preprocessing.
    """
    from monai.data import DataLoader

    device = device or get_device()
    num_workers = int(args["data"].get("num_workers", 0))

    return DataLoader(
        dataset,
        batch_size=args["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
        persistent_workers=(num_workers > 0),
    )
