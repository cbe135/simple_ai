import os
import hashlib
import json
import yaml
import logging

from sklearn.model_selection import train_test_split

from .model import get_device

logger = logging.getLogger(__name__)


def load_data_list(args, data_dir=None):
    """Load the data list and return the list of per-image data dicts.

    Supports data_list.yaml (preferred) with a data_list.json fallback. Both
    are expected to expose a top-level ``data`` list of dicts (each with at
    least ``image`` and ``label``; ``mask`` optional). The modality is no
    longer read from this file — it is supplied via the ``--modality`` flag.
    """
    if data_dir is None:
        from src.env_setup import default_data_dir

        data_dir = default_data_dir()

    yaml_path = os.path.join(data_dir, "data_list.yaml")
    json_path = os.path.join(data_dir, "data_list.json")

    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as fp:
            raw = yaml.safe_load(fp)
    elif os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as fp:
            raw = json.load(fp)
    else:
        raise FileNotFoundError(
            f"No data list found in {data_dir} (expected data_list.yaml or data_list.json)"
        )

    data_dicts = (raw or {}).get("data")
    if not isinstance(data_dicts, list) or not data_dicts:
        raise SystemExit(
            "The data list must contain a non-empty top-level 'data' list of dicts."
        )

    return data_dicts


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


def resolve_cache_dir(args, data_dir, preprocess_transform):
    """Return the persistent transform-cache directory.

    Defaults to ``<data_dir>/.transform_cache/<hash>``, where the hash isolates
    different preprocessing configs so a changed transform auto-creates a fresh
    cache (the old one is orphaned, never reused). Set ``data.cache_dir`` in the
    config to override.
    """
    cfg_dir = (args.get("data", {}) or {}).get("cache_dir")
    if cfg_dir:
        return cfg_dir
    h = hashlib.md5(repr(preprocess_transform).encode("utf-8")).hexdigest()[:12]
    return os.path.join(str(data_dir), ".transform_cache", h)


def generate_base_cache(args, data_dicts, preprocess_transform, cache_dir):
    """Build ONE persistent ``CacheDataset`` over all patients (preprocessing only).

    Augmentation is intentionally excluded and applied later, per-epoch, by the
    training dataset wrapper. The train/val/test split is an *index* partition
    over this base (see :func:`split_indices`), which keeps the cached items
    valid when the split ratio changes between runs.
    """
    from monai.data import PersistentDataset
    import inspect

    logger.info(
        "Creating persistent CacheDataset (num_items=%d) at %s",
        len(data_dicts), cache_dir,
    )
    # PersistentDataset writes transformed items to disk (cache_dir) and reads
    # them back on later runs/processes, so the heavy NIfTI load+window only
    # happens once. copy_cache=True also keeps an in-RAM mirror for in-run
    # speed (replicates the old cache_rate=1.0 RAM behavior). progress=False
    # keeps the log quiet. Kwargs are filtered so this works across MONAI
    # versions that may lack progress/copy_cache.
    sig = inspect.signature(PersistentDataset.__init__)
    kwargs = {"cache_dir": cache_dir}
    if "progress" in sig.parameters:
        kwargs["progress"] = False
    if "copy_cache" in sig.parameters:
        kwargs["copy_cache"] = True
    dataset = PersistentDataset(data_dicts, preprocess_transform, **kwargs)
    logger.info(
        "CacheDataset ready (num_items=%d) at %s; reused if present",
        len(data_dicts), cache_dir,
    )
    return dataset


def split_indices(args, n, labels):
    """Return ``(train_idx, val_idx, test_idx)`` partitioning ``range(n)``.

    Mirrors :func:`populate_data_lists` but operates on indices so a single
    cached base dataset can be reused across split-ratio changes.
    """
    from sklearn.model_selection import train_test_split

    idx = list(range(n))
    train_p = float(args["data"]["train_percentage"])
    val_p = float(args["data"]["val_percentage"])
    test_p = float(args["data"]["test_percentage"])
    seed = int(args["environ"]["seed"])

    train_idx, val_test_idx = train_test_split(
        idx,
        train_size=train_p,
        test_size=val_p + test_p,
        stratify=labels,
        random_state=seed,
        shuffle=True,
    )
    if val_p + test_p > 0:
        val_idx, test_idx = train_test_split(
            val_test_idx,
            train_size=val_p / (val_p + test_p),
            test_size=test_p / (val_p + test_p),
            stratify=[labels[i] for i in val_test_idx],
            random_state=seed,
            shuffle=True,
        )
    else:
        val_idx, test_idx = [], []
    return train_idx, val_idx, test_idx


class _AugDataset:
    """Wrap a cached base dataset, applying stochastic augmentation per fetch."""

    def __init__(self, base, indices, aug_transform):
        self.base = base
        self.indices = list(indices)
        self.aug = aug_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.aug(self.base[self.indices[i]])


def make_train_dataset(base, idx, aug_transform):
    """Training dataset: cached preprocessing + per-epoch augmentation."""
    return _AugDataset(base, idx, aug_transform)


def make_eval_dataset(base, idx):
    """Validation/test dataset: cached preprocessing only (no augmentation)."""
    from torch.utils.data import Subset

    return Subset(base, list(idx))


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
