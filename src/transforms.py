"""
Data-driven transforms. No hardcoded task names.

Transform pipelines are built from:
  - the `--modality` flag (ct | mri | xray | color)
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


def _adjust_spatial_size(spatial_size, spatial_dims):
    """Pad/truncate a spatial_size list to match the data's spatial rank.

    Short lists are padded by repeating the first element (keeps it roughly
    isotropic); long lists are truncated to the leading axes.
    """
    ss = list(spatial_size)
    if len(ss) == spatial_dims:
        return ss
    if len(ss) < spatial_dims:
        return ss + [ss[0]] * (spatial_dims - len(ss))
    return ss[:spatial_dims]


def _adjust_affine_range(rng, spatial_dims):
    """Pad/truncate a per-axis affine range to match the spatial rank.

    Missing axes are padded with [0, 0] (a no-op for that axis).
    """
    rng = list(rng)
    if len(rng) >= spatial_dims:
        return rng[:spatial_dims]
    return rng + [[0, 0]] * (spatial_dims - len(rng))


def derive_spatial_dims(data_dicts_sample):
    """Load the first image and return its spatial rank (2 or 3).

    Works for both 2D and 3D NIfTI, PIL, and DICOM inputs — the file
    extension alone cannot distinguish a 2D from a 3D volume, so we read the
    actual tensor shape instead.
    """
    from monai.transforms import LoadImaged

    if not data_dicts_sample:
        return 2
    reader_kw = derive_reader(data_dicts_sample)
    img_path = str(data_dicts_sample[0].get("image", ""))
    x = LoadImaged(keys="image", ensure_channel_first=True, **reader_kw)(
        {"image": img_path}
    )["image"]
    return x.ndim - 1


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
    mod = modality.lower()

    spatial_dims = dataset_info.get("spatial_dims") or len(args["data"]["spatial_size"])
    spatial_size = _adjust_spatial_size(args["data"]["spatial_size"], spatial_dims)

    steps = []

    if mod == "ct":
        # CT: resize → window → mask → repeat
        keys = ["image"]
        if has_masks:
            keys.append("mask")
        steps.append(Resized(keys=keys, spatial_size=spatial_size))
        steps.append(
            ScaleIntensityRanged(
                keys=["image"],
                a_min=float(args["data"]["a_min"]),
                a_max=float(args["data"]["a_max"]),
                b_min=0,
                b_max=1,
                clip=True,
            )
        )
        if has_masks:
            steps.append(MaskIntensityd(keys="image", mask_key="mask"))
    else:
        # Non-CT (xray, mri, color, ...): resize only
        steps.append(
            Resized(keys=["image"], spatial_size=spatial_size)
        )

    # Single-channel images are repeated to build a multi-channel input.
    # Multi-channel images (e.g. RGB "color") already have their channels,
    # so we skip the repeat.
    if mod != "color":
        steps.append(
            RepeatChanneld(keys=["image"], repeats=args["data"]["repeats"])
        )

    return steps


def get_augmentation(args, dataset_info):
    """Build augmentation pipeline. Modality-agnostic, driven by args."""
    spatial_dims = dataset_info.get("spatial_dims") or len(args["data"]["rotate_range"])
    rotate_range = _adjust_affine_range(args["data"]["rotate_range"], spatial_dims)
    shear_range = _adjust_affine_range(args["data"]["shear_range"], spatial_dims)
    translate_range = _adjust_affine_range(args["data"]["translate_range"], spatial_dims)
    scale_range = _adjust_affine_range(args["data"]["scale_range"], spatial_dims)
    cfg_axis = args["data"].get("spatial_axis") or []
    spatial_axis = cfg_axis if len(cfg_axis) == spatial_dims else list(range(spatial_dims))

    aug = [
        RandAffined(
            keys="image",
            rotate_range=rotate_range,
            shear_range=shear_range,
            translate_range=translate_range,
            scale_range=scale_range,
            prob=float(args["data"]["affine_prob"]),
            padding_mode="border",
        ),
    ]

    # Flip is always applied (controlled by flip_prob)
    aug.append(
        RandFlipd(
            keys="image",
            spatial_axis=spatial_axis,
            prob=float(args["data"]["flip_prob"]),
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


def build_preprocess_transform(args, data_dicts_sample, dataset_info):
    """Deterministic preprocessing only (no augmentation), for persistent caching.

    The output of this pipeline is what gets cached to disk. Augmentation is
    kept separate (see ``get_augmentation``) and applied per-epoch on top of
    the cached items, so the cache stays valid regardless of split changes or
    augmentation tweaks.
    """
    return Compose(
        get_loaders(data_dicts_sample)
        + get_loaders_extra(args)
        + get_preprocess(args, data_dicts_sample, dataset_info)
        + get_preprocess_extra(args)
    )
