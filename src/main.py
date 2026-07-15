"""
CNN Classification Pipeline

Entry point for end-to-end training. All behavior is driven by:
  - config YAML (data source, hyperparameters)
  - --modality flag (ct | mri | xray | color)
  - data_list.yaml (`data` list of per-patient dicts; drives has_masks / reader)

Usage:
    python src/main.py --data-dir /content/dataset --modality ct
    python src/main.py --config config.yaml --data-dir /content/dataset --modality ct
"""
print(">>> booting pipeline...", flush=True)

import logging
import os
from argparse import ArgumentParser
from logging import getLogger, basicConfig, INFO, Formatter, FileHandler
from os import path, getcwd, makedirs
from sys import stdout, stderr, path as sys_path, excepthook
from datetime import datetime

sys_path.insert(0, path.dirname(path.dirname(path.abspath(__file__))))

# Select a valid matplotlib backend before any module imports pyplot.
# A globally-configured "inline" backend (module://matplotlib_inline...)
# only works inside a Jupyter/IPython kernel. When running headless
# (e.g. `!uv run python src/main.py` from a notebook cell), fall back
# to the non-interactive Agg backend so plotting/saving still works.
from matplotlib import use, get_backend

try:
    from IPython import get_ipython

    _IN_NOTEBOOK = get_ipython() is not None
except Exception:
    _IN_NOTEBOOK = False

if not _IN_NOTEBOOK and "inline" in get_backend().lower():
    use("Agg")

from yaml import safe_load
from numpy import pi
from monai.utils.misc import set_determinism

logger = getLogger(__name__)

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
            [pi / 18, pi / 9],
            [pi / 18, pi / 9],
        ],
        "shear_range": [[0, 0], [0, 0]],
        "translate_range": [[-60, 60], [0, 0]],
        "scale_range": [[0, 0], [0, 0]],
        "affine_prob": 0,
        "spatial_axis": [0, 1],
        "flip_prob": 0.5,
        "a_min": -125,
        "a_max": 200,
        "cache_rate": 1.0,
        "num_workers": 4,
    },
    "img_cnt": 5,
    "training": {
        "num_epoch": 3,
        "batch_size": 16,
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

    if config_path and path.exists(config_path):
        with open(config_path, "r") as fp:
            file_config = safe_load(fp) or {}
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
    import traceback

    # Force line-buffered output so logs/errors are never hidden by piping.
    try:
        stdout.reconfigure(line_buffering=True)
        stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    # Catch any uncaught exception and persist the traceback to disk,
    # since `uv run` / Colab can swallow stderr and leave a silent death.
    def _excepthook(exc_type, exc, tb):
        msg = "".join(traceback.format_exception(exc_type, exc, tb))
        print(msg, file=stderr, flush=True)
        try:
            with open(path.join(getcwd(), "pipeline_errors.log"), "w") as _f:
                _f.write(msg)
        except Exception:
            pass

    excepthook = _excepthook

    # Print the running commit so we can confirm the deployed code is current.
    try:
        _commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        _commit = "unknown"
    print(f">>> pipeline starting — commit={_commit}", flush=True)

    parser = ArgumentParser(description="CNN Classification Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help=(
            "Directory containing data_list.yaml (or data_list.json) with a "
            "`data` list of per-patient dicts, e.g. /content/liver_data. Required."
        ),
    )
    parser.add_argument(
        "--modality",
        type=str,
        required=True,
        choices=["ct", "mri", "xray", "color"],
        help="Imaging modality (ct | mri | xray | color); drives preprocessing.",
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
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Remove the persistent transformed-data cache before training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help=(
            "Force a compute device. Default: auto-detect "
            "(CUDA if available, else MPS on Apple Silicon, else CPU)."
        ),
    )
    args_cli = parser.parse_args()

    # Send the default handler to stdout (not stderr). cli.py pipes stdout to
    # both the cell and run.log, so a single handler covers the console and the
    # log file. This avoids the duplicate output caused by having separate
    # stderr + stdout handlers both reaching the terminal.
    _log_level = getattr(
        logging, os.environ.get("SIMPLE_AI_LOG_LEVEL", "INFO").upper(), logging.INFO
    )
    basicConfig(
        level=_log_level,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=stdout,
        force=True,
    )

    # Also mirror all logs to a file so output survives even if the cell hides it.
    _log_path = path.join(getcwd(), "pipeline.log")
    _file_handler = FileHandler(_log_path)
    _file_handler.setFormatter(
        Formatter(
            "[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
    )
    getLogger().addHandler(_file_handler)
    logger.info(f"Logging to console/run.log and {_log_path}")

    args = load_config(args_cli.config)
    logger.info(f"Config file: {args['environ']['config_file']}")
    logger.info(
        f"Effective settings — cache_rate: {args['data']['cache_rate']}, "
        f"batch_size: {args['training']['batch_size']}, "
        f"num_epoch: {args['training']['num_epoch']}, "
        f"spatial_size: {args['data']['spatial_size']}"
    )
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
    output_base = args_cli.output_dir or getcwd()
    run_dir = path.abspath(
        path.join(output_base, datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    makedirs(run_dir, exist_ok=True)
    logger.info(f"Run output directory: {run_dir}")

    # Mirror all logs into the run directory as well, so each run keeps its
    # own self-contained copy (in addition to pipeline.log in the cwd).
    _run_log_path = path.join(run_dir, "pipeline.log")
    _run_log_handler = FileHandler(_run_log_path)
    _run_log_handler.setFormatter(
        Formatter(
            "[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
    )
    getLogger().addHandler(_run_log_handler)

    # Save the resolved config for this run
    from src.config import save_config

    save_config(args, run_dir)

    # Setup data
    setup_data(args, data_dir)
    get_data_count(args, data_dir)

    # Find data directory and load data list
    data_dir = find_data_dir(args, data_dir)
    logger.info(f"Resolved data directory: {data_dir}")

    # Load the data list; modality comes from the required --modality flag.
    from src.data import load_data_list

    data_dicts = load_data_list(args, data_dir)
    modality = args_cli.modality

    # Derive properties from data
    from src.transforms import derive_has_masks, derive_reader, derive_spatial_dims

    has_masks = derive_has_masks(data_dicts)
    reader_kw = derive_reader(data_dicts)
    spatial_dims = derive_spatial_dims(data_dicts[:1])
    dataset_info = {"modality": modality, "spatial_dims": spatial_dims}
    logger.info(f"Total data: {len(data_dicts)}")
    logger.info(f"Modality: {modality}")
    logger.info(f"Spatial dims: {spatial_dims}")

    logger.info(f"Has masks: {has_masks}")
    logger.info(f"Reader: {reader_kw or 'monai default'}")

    # Split data by INDEX over the full patient list, so a single persistent
    # cache (built below) is reused even when the split ratio changes between
    # runs. Dict subsets are derived from the indices for logging/plots only.
    from src.data import (
        split_indices,
        resolve_cache_dir,
        generate_base_cache,
        make_train_dataset,
        make_eval_dataset,
    )
    from src.transforms import (
        build_train_transform,
        build_val_transform,
        build_preprocess_transform,
        get_augmentation,
        get_augmentation_extra,
    )
    from monai.transforms import Compose

    labels = [a["label"] for a in data_dicts]
    train_idx, val_idx, test_idx = split_indices(args, len(data_dicts), labels)
    train_dicts = [data_dicts[i] for i in train_idx]
    val_dicts = [data_dicts[i] for i in val_idx]
    test_dicts = [data_dicts[i] for i in test_idx]
    logger.info(f"{len(train_dicts)} training, {len(val_dicts)} validation, {len(test_dicts)} testing")

    # Build transforms. Preprocessing is deterministic and cached; augmentation
    # is applied per-epoch on top of the cached items (see make_train_dataset).
    train_transform = build_train_transform(args, data_dicts, dataset_info)
    val_transform = build_val_transform(args, data_dicts, dataset_info)
    preprocess_transform = build_preprocess_transform(args, data_dicts, dataset_info)
    aug_transform = Compose(get_augmentation(args, dataset_info) + get_augmentation_extra(args))

    # Persistent transform cache (auto-isolated by preprocessing-config hash).
    cache_dir = resolve_cache_dir(args, data_dir, preprocess_transform)
    if args_cli.clear_cache:
        import shutil

        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            logger.info(f"Cleared transform cache at {cache_dir}")

    base = generate_base_cache(args, data_dicts, preprocess_transform, cache_dir)
    train_set = make_train_dataset(base, train_idx, aug_transform)
    val_set = make_eval_dataset(base, val_idx)
    test_set = make_eval_dataset(base, test_idx)

    # Save pre/post transformation sample images to run_dir/samples/
    from src.utils import plot_transform_result

    samples_dir = path.join(run_dir, "samples")
    makedirs(samples_dir, exist_ok=True)
    n_samples = min(4, len(data_dicts))
    for _i in range(n_samples):
        _s = data_dicts[_i]
        _pre = val_transform(_s)
        _post = train_transform(_s)
        plot_transform_result(
            _pre, _post, with_mask=has_masks,
            save_path=path.join(samples_dir, f"sample_{_i}.png"),
        )
    logger.info("Saved %d pre/post transformation samples to %s", n_samples, samples_dir)

    # ── Data summary (mirrors the D4 notebook diagnostics) ──────────────────
    # Label distribution per split, counted from the dicts (no image loading).
    def _label_dist(name, dicts):
        pos = sum(1 for a in dicts if a["label"] == 1)
        neg = len(dicts) - pos
        logger.info(
            f"{name} label distribution — total: {len(dicts)}, "
            f"positive: {pos}, negative: {neg}"
        )

    _label_dist("Train", train_dicts)
    _label_dist("Validation", val_dicts)
    _label_dist("Test", test_dicts)

    # Sample image size + intensity range after the transform pipeline.
    sample = train_set[0]
    img = sample["image"]
    logger.info(f"Sample image — shape: {tuple(img.shape)}, dtype: {img.dtype}")
    logger.info(
        f"Sample image intensity — min: {float(img.min()):.4f}, "
        f"max: {float(img.max()):.4f}"
    )

    # Train
    logger.info("Starting training...")
    from src.train import train_pipeline

    model, train_loader, val_loader, record = train_pipeline(
        args, train_set, val_set, run_dir, device=args_cli.device
    )

    # Plot loss curves
    from src.utils import plot_loss_curves

    plot_loss_curves(args, record, save_path=path.join(run_dir, "loss_curve.png"))

    # Evaluate
    import torch

    from src.evaluate import infer, plot_roc_and_show_result

    best_weights = path.join(run_dir, "best_weights.pth")
    if not path.exists(best_weights):
        best_weights = "best_weights.pth"

    best_state = torch.load(best_weights, weights_only=True)
    model.load_state_dict(best_state)

    inference_dir = path.join(run_dir, "inference")
    roc_dir = path.join(run_dir, "roc")
    makedirs(inference_dir, exist_ok=True)
    makedirs(roc_dir, exist_ok=True)

    train_true, train_pred = infer(
        args, model, train_loader, True, device=args_cli.device,
        details_path=path.join(inference_dir, "inference_details_train.log"),
    )
    val_true, val_pred = infer(
        args, model, val_loader, True, device=args_cli.device,
        details_path=path.join(inference_dir, "inference_details_validation.log"),
    )

    from src.data import generate_dataloader

    test_loader = generate_dataloader(args, test_set, device=args_cli.device)
    test_true, test_pred = infer(
        args, model, test_loader, True, device=args_cli.device,
        details_path=path.join(inference_dir, "inference_details_test.log"),
    )

    plot_roc_and_show_result(
        args, train_true, train_pred, title="Train",
        save_path=path.join(roc_dir, "roc_train.png"),
    )
    plot_roc_and_show_result(
        args, val_true, val_pred, title="Validation",
        save_path=path.join(roc_dir, "roc_validation.png"),
    )
    plot_roc_and_show_result(
        args, test_true, test_pred, title="Test",
        save_path=path.join(roc_dir, "roc_test.png"),
    )

    logger.info(
        f"Artifacts saved in {run_dir}:\n"
        f"  best_weights.pth, loss_curve.png, config.yaml\n"
        f"  samples/  (pre/post transformation sample images)\n"
        f"  inference/  (inference_details_train|validation|test.log)\n"
        f"  roc/  (roc_train|validation|test.png)"
    )
    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
