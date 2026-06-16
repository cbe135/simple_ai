"""
D3/D4 Hands-on: CNN Classification Pipeline
Supports both D3 (Liver CT) and D4 (Chest X-ray Hackathon) tasks.

Usage:
    python src/main.py                          # D3 Liver CT (default)
    python src/main.py --task d4_hackathon --group-num 8
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from monai.utils.misc import set_determinism

logger = logging.getLogger(__name__)


def get_default_args(task="d3_liver_ct", group_num=1):
    """Return default args dict for the given task."""
    if task == "d4_hackathon":
        from src.env_setup import D4_DISEASE_LIST
        data_name = D4_DISEASE_LIST[(group_num - 1) % len(D4_DISEASE_LIST)]
        return {
            "task": "d4_hackathon",
            "environ": {
                "config_file": "config.yaml",
                "seed": 42,
                "data_name": data_name,
                "group_num": group_num,
            },
            "data": {
                "train_percentage": 0.7,
                "val_percentage": 0.15,
                "test_percentage": 0.15,
                "spatial_size": [224, 224],
                "repeats": 2,
                "rotate_range": [[np.pi / 18, np.pi / 9]],
                "shear_range": [[0, 0], [0, 0]],
                "translate_range": [[30, 60], [30, 60]],
                "scale_range": [[0.0001, 0.0001], [0.0001, 0.0001]],
                "affine_prob": 0.5,
                "spatial_axis": [0, 1],
                "flip_prob": 0.5,
                "a_min": -125,
                "a_max": 200,
                "cache_rate": 1,
            },
            "img_cnt": 5,
            "training": {
                "num_epoch": 10,
                "batch_size": 16,
                "lr": 1e-3,
                "timm_model": "resnet18",
                "num_classes": 1,
            },
            "threshold": 0.5,
        }
    else:
        # d3_liver_ct
        return {
            "task": "d3_liver_ct",
            "environ": {
                "config_file": "config.yaml",
                "seed": 888,
                "data_name": "liver_data",
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
                "cache_rate": 1,
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


def main():
    parser = argparse.ArgumentParser(description="D3/D4 CNN Classification Pipeline")
    parser.add_argument(
        "--task",
        type=str,
        default="d3_liver_ct",
        choices=["d3_liver_ct", "d4_hackathon"],
        help="Task type: d3_liver_ct or d4_hackathon",
    )
    parser.add_argument(
        "--group-num",
        type=int,
        default=1,
        help="Group number for D4 Hackathon (1-15, determines disease)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file (overrides --task defaults)",
    )
    args_cli = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    # Build args
    args = get_default_args(args_cli.task, args_cli.group_num)

    # Load config file if specified
    if args_cli.config:
        import yaml

        with open(args_cli.config, "r") as fp:
            file_args = yaml.safe_load(fp)
            args.update(file_args)

    logger.info(f"Task: {args['task']}")
    logger.info(f"Data: {args['environ']['data_name']}")
    set_determinism(args["environ"]["seed"])

    # Setup data
    from src.env_setup import setup_data, get_data_count

    setup_data(args)
    get_data_count(args)

    # Load and split data
    from src.data import load_data_list, populate_data_lists, generate_dataset

    data_dicts = load_data_list(args)
    train_dicts, val_dicts, test_dicts = populate_data_lists(args, data_dicts)
    logger.info(f"{len(train_dicts)} data for training")
    logger.info(f"{len(val_dicts)} data for validation")
    logger.info(f"{len(test_dicts)} data for testing")

    # Build transforms and datasets
    from src.transforms import build_train_transform, build_val_transform

    train_transform = build_train_transform(args)
    val_transform = build_val_transform(args)

    train_set = generate_dataset(args, train_dicts, train_transform)
    val_set = generate_dataset(args, val_dicts, val_transform)
    test_set = generate_dataset(args, test_dicts, val_transform)

    # Train
    logger.info("Starting training...")
    from src.train import train_pipeline

    model, train_loader, val_loader, record = train_pipeline(args, train_set, val_set)

    # Plot loss curves
    from src.utils import plot_loss_curves

    plot_loss_curves(args, record)

    # Evaluate
    import torch

    from src.evaluate import infer, plot_roc_and_show_result
    from src.model import get_device

    save_dir = "/content"
    best_weights = os.path.join(save_dir, "best_weights.pth")
    if not os.path.exists(best_weights):
        best_weights = "best_weights.pth"

    best_state = torch.load(best_weights, weights_only=True)
    model.load_state_dict(best_state)

    train_true, train_pred = infer(args, model, train_loader, True)
    val_true, val_pred = infer(args, model, val_loader, True)

    from src.data import generate_dataloader

    test_loader = generate_dataloader(args, test_set)
    test_true, test_pred = infer(args, model, test_loader, True)

    plot_roc_and_show_result(args, train_true, train_pred, title="Train")
    plot_roc_and_show_result(args, val_true, val_pred, title="Validation")
    plot_roc_and_show_result(args, test_true, test_pred, title="Test")

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
