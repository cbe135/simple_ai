import numpy as np
from monai.transforms import (
    Compose,
    EnsureTyped,
    LoadImaged,
    MaskIntensityd,
    Orientationd,
    RandAffined,
    RandFlipd,
    RandGaussianNoiseD,
    RepeatChanneld,
    Resized,
    ScaleIntensityd,
    ScaleIntensityRanged,
)


def get_loaders(args):
    """Return loader transforms based on task type."""
    task = args.get("task", "d3_liver_ct")

    if task == "d4_hackathon":
        return [
            LoadImaged(keys=["image"], reader="pilreader", ensure_channel_first=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    else:
        # d3_liver_ct: has masks
        return [
            LoadImaged(keys=["image", "mask"], ensure_channel_first=True),
            EnsureTyped(keys=["image", "label"]),
        ]


def get_preprocess(args):
    """Return the preprocessing transform pipeline based on task."""
    task = args.get("task", "d3_liver_ct")

    if task == "d4_hackathon":
        return [
            Resized(keys=["image"], spatial_size=args["data"]["spatial_size"]),
            RepeatChanneld(keys=["image"], repeats=args["data"]["repeats"]),
        ]
    else:
        # d3_liver_ct: resize + CT windowing + mask + repeat
        return [
            Resized(keys=["image", "mask"], spatial_size=args["data"]["spatial_size"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args["data"]["a_min"],
                a_max=args["data"]["a_max"],
                b_min=0,
                b_max=1,
                clip=True,
            ),
            MaskIntensityd(keys="image", mask_key="mask"),
            RepeatChanneld(keys=["image"], repeats=args["data"]["repeats"]),
        ]


def get_augmentation(args):
    """Return the data augmentation transform pipeline based on task."""
    task = args.get("task", "d3_liver_ct")
    aug = [
        RandAffined(
            keys="image",
            rotate_range=args["data"]["rotate_range"],
            shear_range=args["data"]["shear_range"],
            translate_range=args["data"]["translate_range"],
            scale_range=args["data"]["scale_range"],
            prob=args["data"]["affine_prob"],
            padding_mode="border",
        ),
    ]

    if task == "d4_hackathon":
        aug.append(RandGaussianNoiseD(keys="image"))

    return aug


def build_train_transform(args):
    """Build the full training transform (loaders + preprocess + augmentation)."""
    return Compose(get_loaders(args) + get_preprocess(args) + get_augmentation(args))


def build_val_transform(args):
    """Build the validation/test transform (loaders + preprocess only)."""
    return Compose(get_loaders(args) + get_preprocess(args))
