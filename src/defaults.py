"""Lightweight default configuration for the simple_ai pipeline.

Kept separate from ``main.py`` so CLIs (e.g. ``simple_ai_autoresearch_train
--default``) can print the defaults without booting the full pipeline
(matplotlib/monai import banner, etc.). Only ``numpy`` is imported here.
"""
from numpy import pi

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
