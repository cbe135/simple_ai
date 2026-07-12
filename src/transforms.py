"""
Data-driven transforms. No hardcoded task names.

Transform pipelines are built from:
  - dataset_info.yaml (modality)
  - data_list.yaml structure (has_masks, reader)
  - args (hyperparameters like spatial_size, a_min/a_max, repeats)
"""
import os
from monai.transforms import (
    Compose,
    EnsureTyped,
    LoadImaged,
    MaskIntensityd,
    RandAffined,
    RandFlipd,
    RandGaussianNoiseD,
    RepeatChanneld,
    Resized,
    ScaleIntensityRanged,
)


def load_dataset_info(data_dir):
    """Load dataset_info.yaml from the data directory.

    Returns a dict with at least:
        modality: CT | X-ray | MRI | ...
    """
    import yaml

    info_path = os.path.join(data_dir, "dataset_info.yaml")
    if os.path.exists(info_path):
        with open(info_path, "r") as fp:
            return yaml.safe_load(fp)
    return {}


def derive_reader(data_dicts_sample):
    """Derive the MONAI reader from file extensions in the data list.

    Returns a reader kwarg dict (e.g. {'reader': 'pilreader'} or {}).
    """
    if not data_dicts_sample:
        return {}

    sample = data_dicts_sample[0]
    img_path = str(sample.get("image", ""))

    if img_path.endswith(".nii.gz") or img_path.endswith(".nii"):
        return {}  # MONAI default NIfTI reader
    elif img_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")):
        return {"reader": "pilreader"}
    elif img_path.lower().endswith(".dcm") or ".dcm" in img_path:
        return {"reader": "dcmreader"}

    return {}


def derive_has_masks(data_dicts_sample):
    """Check if the data list contains mask entries."""
    if not data_dicts_sample:
        return False
    return "mask" in data_dicts_sample[0]


def get_loaders(data_dicts_sample):
    """Build loader transforms by deriving keys and reader from the data."""
    has_masks = derive_has_masks(data_dicts_sample)
    reader_kw = derive_reader(data_dicts_sample)

    keys = ["image"]
    if has_masks:
        keys.append("mask")

    return [
        LoadImaged(keys=keys, ensure_channel_first=True, **reader_kw),
        EnsureTyped(keys=["image", "label"]),
    ]


def get_preprocess(args, data_dicts_sample, dataset_info):
    """Build preprocessing pipeline based on modality and data characteristics."""
    modality = dataset_info.get("modality", "")
    has_masks = derive_has_masks(data_dicts_sample)

    steps = []

    if modality == "CT":
        # CT: resize → window → mask → repeat
        keys = ["image"]
        if has_masks:
            keys.append("mask")
        steps.append(Resized(keys=keys, spatial_size=args["data"]["spatial_size"]))
        steps.append(
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args["data"]["a_min"],
                a_max=args["data"]["a_max"],
                b_min=0,
                b_max=1,
                clip=True,
            )
        )
        if has_masks:
            steps.append(MaskIntensityd(keys="image", mask_key="mask"))
    else:
        # Non-CT (X-ray, MRI, etc.): resize only
        steps.append(
            Resized(keys=["image"], spatial_size=args["data"]["spatial_size"])
        )

    steps.append(
        RepeatChanneld(keys=["image"], repeats=args["data"]["repeats"])
    )

    return steps


def get_augmentation(args, dataset_info):
    """Build augmentation pipeline. Modality-agnostic, driven by args."""
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

    # Flip is always applied (controlled by flip_prob)
    aug.append(
        RandFlipd(
            keys="image",
            spatial_axis=args["data"]["spatial_axis"],
            prob=args["data"]["flip_prob"],
        )
    )

    # Gaussian noise is always applied (controlled by prob in augmentation config)
    aug.append(RandGaussianNoiseD(keys="image"))

    return aug


def _instantiate_bundle(args, key):
    """Instantiate an extra transform list from config via MONAI ConfigParser.

    The config section is a list of MONAI bundle `_target_` dicts. `@data.*`
    references resolve against the rest of the config. Returns [] when empty.
    """
    items = (args.get("transforms") or {}).get(key) or []
    if not items:
        return []

    from monai.bundle import ConfigParser

    cp = ConfigParser(config=args)
    return cp.get_parsed_content(f"transforms.{key}")


def get_loaders_extra(args):
    """Extra loader transforms appended after the auto-derived loaders."""
    return _instantiate_bundle(args, "loaders_extra")


def get_preprocess_extra(args):
    """Extra preprocessing transforms appended after the auto-derived preprocess."""
    return _instantiate_bundle(args, "preprocess_extra")


def get_augmentation_extra(args):
    """Extra augmentation transforms appended after the auto-derived augmentation."""
    return _instantiate_bundle(args, "augmentation_extra")


def build_train_transform(args, data_dicts_sample, dataset_info):
    """Build full training transform (presets + user extras from config)."""
    return Compose(
        get_loaders(data_dicts_sample)
        + get_loaders_extra(args)
        + get_preprocess(args, data_dicts_sample, dataset_info)
        + get_preprocess_extra(args)
        + get_augmentation(args, dataset_info)
        + get_augmentation_extra(args)
    )


def build_val_transform(args, data_dicts_sample, dataset_info):
    """Build validation/test transform (no augmentation)."""
    return Compose(
        get_loaders(data_dicts_sample)
        + get_loaders_extra(args)
        + get_preprocess(args, data_dicts_sample, dataset_info)
        + get_preprocess_extra(args)
    )
