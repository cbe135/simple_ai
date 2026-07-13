"""
CNN Classification Pipeline

Entry point for end-to-end training. All behavior is driven by:
  - config YAML (data source, hyperparameters)
  - dataset_info.yaml in the data directory (modality)
  - data_list.yaml structure (has_masks, image format)

Usage:
    python src/main.py
    python src/main.py --config config.yaml
"""
import argparse
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Select a valid matplotlib backend before any module imports pyplot.
# A globally-configured "inline" backend (module://matplotlib_inline...)
# only works inside a Jupyter/IPython kernel. When running headless
# (e.g. `!uv run python src/main.py` from a notebook cell), fall back
# to the non-interactive Agg backend so plotting/saving still works.
import matplotlib

try:
    from IPython import get_ipython

    _IN_NOTEBOOK = get_ipython() is not None
except Exception:
    _IN_NOTEBOOK = False

if not _IN_NOTEBOOK and "inline" in matplotlib.get_backend().lower():
    matplotlib.use("Agg")

import yaml
import numpy as np
from monai.utils.misc import set_determinism

logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "environ": {
        "config_file": "config.yaml",
        "seed": 888,
    },
    "data": {
        "train_percentage": 0.8,
        "val_percentage": 0.1,
        "test_percentage": 0.1,
        "spatial_size": [250, 250],
        "repeats": 3,
        "rotate_range": [
            [np.pi / 18, np.pi / 9],
            [np.pi / 18, np.pi / 9],
        ],
        "shear_range": [[0, 0], [0, 0]],
        "translate_range": [[-60, 60], [0, 0]],
        "scale_range": [[0, 0], [0, 0]],
        "affine_prob": 0,
        "spatial_axis": [0, 1],
        "flip_prob": 0.5,
        "a_min": -125,
        "a_max": 200,
        "cache_rate": 0,
    },
    "img_cnt": 5,
    "training": {
        "num_epoch": 3,
        "batch_size": 128,
        "lr": 1e-3,
        "timm_model": "resnet18",
        "num_classes": 1,
    },
    "threshold": 0.5,
}


def load_config(config_path=None, overrides=None):
    """Load config from YAML, merge with defaults, apply CLI overrides."""
    import copy
    args = copy.deepcopy(DEFAULT_ARGS)

    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as fp:
            file_config = yaml.safe_load(fp) or {}
        args = _deep_merge(args, file_config)
        logger.info(f"Loaded config from {config_path}")

    if overrides:
        args = _deep_merge(args, overrides)

    return args


def _deep_merge(base, override):
    """Recursively merge override dict into base dict."""
    import copy
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def main():
    import subprocess
    import sys
    import traceback

    # Force line-buffered output so logs/errors are never hidden by piping.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    # Catch any uncaught exception and persist the traceback to disk,
    # since `uv run` / Colab can swallow stderr and leave a silent death.
    def _excepthook(exc_type, exc, tb):
        msg = "".join(traceback.format_exception(exc_type, exc, tb))
        print(msg, file=sys.stderr, flush=True)
        try:
            with open(os.path.join(os.getcwd(), "pipeline_errors.log"), "w") as _f:
                _f.write(msg)
        except Exception:
            pass

    sys.excepthook = _excepthook

    # Print the running commit so we can confirm the deployed code is current.
    try:
        _commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        _commit = "unknown"
    print(f">>> pipeline starting — commit={_commit}", flush=True)

    parser = argparse.ArgumentParser(description="CNN Classification Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help=(
            "Directory containing data_list.yaml (or data_list.json) and "
            "dataset_info.yaml, e.g. /content/liver_data. Required."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Parent directory for run outputs. A timestamped subdirectory "
            "is created here holding weights, loss curve, and config. "
            "Defaults to the current working directory."
        ),
    )
    args_cli = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    # Also mirror all logs to a file so output survives even if the cell hides it.
    _log_path = os.path.join(os.getcwd(), "pipeline.log")
    _file_handler = logging.FileHandler(_log_path)
    _file_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(_file_handler)
    logger.info(f"Logging to console and {_log_path}")

    args = load_config(args_cli.config)
    logger.info(f"Config file: {args['environ']['config_file']}")
    set_determinism(args["environ"]["seed"])

    # Resolve data directory
    from src.env_setup import (
        setup_data,
        get_data_count,
        find_data_dir,
    )

    data_dir = args_cli.data_dir
    logger.info(f"Using data directory: {data_dir}")

    # Run output directory: <output_dir>/<YYYYMMDD_HHMMSS>/
    output_base = args_cli.output_dir or os.getcwd()
    run_dir = os.path.join(output_base, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"Run output directory: {run_dir}")

    # Save the resolved config for this run
    from src.config import save_config

    save_config(args, run_dir)

    # Setup data
    setup_data(args, data_dir)
    get_data_count(args, data_dir)

    # Find data directory and load data list
    data_dir = find_data_dir(args, data_dir)
    logger.info(f"Resolved data directory: {data_dir}")

    # Load data list
    with open(os.path.join(data_dir, "data_list.yaml"), "r") as fp:
        data_dicts = yaml.safe_load(fp)["data"]
    logger.info(f"Total data: {len(data_dicts)}")

    # Load dataset_info.yaml (modality, etc.)
    from src.transforms import load_dataset_info

    dataset_info = load_dataset_info(data_dir)
    modality = dataset_info.get("modality", "unknown")
    logger.info(f"Modality: {modality}")

    # Derive properties from data
    from src.transforms import derive_has_masks, derive_reader

    has_masks = derive_has_masks(data_dicts)
    reader_kw = derive_reader(data_dicts)
    logger.info(f"Has masks: {has_masks}")
    logger.info(f"Reader: {reader_kw or 'monai default'}")

    # Split data
    from src.data import populate_data_lists

    train_dicts, val_dicts, test_dicts = populate_data_lists(args, data_dicts)
    logger.info(f"{len(train_dicts)} training, {len(val_dicts)} validation, {len(test_dicts)} testing")

    # Build transforms (pass data_dicts sample for derivation)
    from src.transforms import build_train_transform, build_val_transform

    train_transform = build_train_transform(args, data_dicts, dataset_info)
    val_transform = build_val_transform(args, data_dicts, dataset_info)

    # Build datasets
    from src.data import generate_dataset

    train_set = generate_dataset(args, train_dicts, train_transform)
    val_set = generate_dataset(args, val_dicts, val_transform)
    test_set = generate_dataset(args, test_dicts, val_transform)

    # Train
    logger.info("Starting training...")
    from src.train import train_pipeline

    model, train_loader, val_loader, record = train_pipeline(
        args, train_set, val_set, run_dir
    )

    # Plot loss curves
    from src.utils import plot_loss_curves

    plot_loss_curves(args, record, save_path=os.path.join(run_dir, "loss_curve.png"))

    # Evaluate
    import torch

    from src.evaluate import infer, plot_roc_and_show_result

    best_weights = os.path.join(run_dir, "best_weights.pth")
    if not os.path.exists(best_weights):
        best_weights = "best_weights.pth"

    best_state = torch.load(best_weights, weights_only=True)
    model.load_state_dict(best_state)

    train_true, train_pred = infer(args, model, train_loader, True)
    val_true, val_pred = infer(args, model, val_loader, True)

    from src.data import generate_dataloader

    test_loader = generate_dataloader(args, test_set)
    test_true, test_pred = infer(args, model, test_loader, True)

    plot_roc_and_show_result(
        args, train_true, train_pred, title="Train",
        save_path=os.path.join(run_dir, "roc_train.png"),
    )
    plot_roc_and_show_result(
        args, val_true, val_pred, title="Validation",
        save_path=os.path.join(run_dir, "roc_validation.png"),
    )
    plot_roc_and_show_result(
        args, test_true, test_pred, title="Test",
        save_path=os.path.join(run_dir, "roc_test.png"),
    )

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
