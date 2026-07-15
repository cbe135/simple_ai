"""Lightweight default configuration for the simple_ai pipeline.

Kept separate from ``main.py`` so CLIs (e.g. ``simple_ai_autoresearch_train
--default``) can print the defaults without booting the full pipeline
(matplotlib/monai import banner, etc.). Only ``numpy`` is imported here.
"""
from numpy import pi


def _augmentation_pipeline():
    """MONAI bundle augmentation list shared by all modalities (train only)."""
    return [
        {
            "_target_": "monai.transforms.RandAffined",
            "keys": ["image"],
            "rotate_range": "@data::rotate_range",
            "shear_range": "@data::shear_range",
            "translate_range": "@data::translate_range",
            "scale_range": "@data::scale_range",
            "prob": "@data::affine_prob",
            "padding_mode": "border",
        },
        {
            "_target_": "monai.transforms.RandFlipd",
            "keys": ["image"],
            "spatial_axis": "@data::spatial_axis",
            "prob": "@data::flip_prob",
        },
        {
            "_target_": "monai.transforms.RandGaussianNoiseD",
            "keys": ["image"],
        },
    ]


def _preprocess_ct():
    return [
        {
            "_target_": "monai.transforms.Resized",
            "keys": "@data::resize_keys",
            "spatial_size": "@data::spatial_size",
        },
        {
            "_target_": "monai.transforms.ScaleIntensityRanged",
            "keys": ["image"],
            "a_min": "@data::a_min",
            "a_max": "@data::a_max",
            "b_min": 0.0,
            "b_max": 1.0,
            "clip": True,
        },
        {
            "_target_": "monai.transforms.MaskIntensityd",
            "keys": ["image"],
            "mask_key": "mask",
            "_disabled_": "@data::mask_disabled",
        },
        {
            "_target_": "monai.transforms.RepeatChanneld",
            "keys": ["image"],
            "repeats": "@data::repeats",
        },
    ]


def _preprocess_single_channel():
    """Resize + repeat-channel for single-channel volumes (mri, xray)."""
    return [
        {
            "_target_": "monai.transforms.Resized",
            "keys": ["image"],
            "spatial_size": "@data::spatial_size",
        },
        {
            "_target_": "monai.transforms.RepeatChanneld",
            "keys": ["image"],
            "repeats": "@data::repeats",
        },
    ]


def _preprocess_color():
    return [
        {
            "_target_": "monai.transforms.Resized",
            "keys": ["image"],
            "spatial_size": "@data::spatial_size",
        },
    ]


MODALITIES = {
    "ct": {"preprocess": _preprocess_ct(), "augmentation": _augmentation_pipeline()},
    "mri": {"preprocess": _preprocess_single_channel(), "augmentation": _augmentation_pipeline()},
    "xray": {"preprocess": _preprocess_single_channel(), "augmentation": _augmentation_pipeline()},
    "color": {"preprocess": _preprocess_color(), "augmentation": _augmentation_pipeline()},
}


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
    "modalities": MODALITIES,
}
