"""
Config-driven transforms. No hardcoded task names.

Per-modality transform pipelines live in ``config.yaml`` under the ``modalities``
section: each modality defines ``preprocess`` and ``augmentation`` as MONAI
bundle ``_target_`` lists. The pipeline is still data-driven at the edges —
``has_masks`` / ``spatial_dims`` are derived from the data at runtime and
injected as ``@data.*`` references so modality transforms stay appropriate to
the imaging type. The CLI ``--modality`` choices are the keys of ``modalities``.
"""
import logging
import os
from monai.transforms import (
    Compose,
    EnsureTyped,
    LoadImaged,
)

logger = logging.getLogger(__name__)


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

    ensure_keys = ["image", "label"] + (["mask"] if has_masks else [])
    return [
        LoadImaged(keys=keys, ensure_channel_first=True, **reader_kw),
        EnsureTyped(keys=ensure_keys, track_meta=False),
    ]


def prepare_transform_config(args, data_dicts_sample, dataset_info):
    """Inject runtime-derived values into ``args['data']`` for MONAI bundle refs.

    The per-modality transform lists in ``config.yaml`` reference ``@data.*``
    placeholders. Some must be derived from the actual data / its spatial rank
    before the bundle config is parsed:
      - ``spatial_dims``                 (2 or 3, from the first image)
      - ``spatial_size``                 (padded/truncated to ``spatial_dims``)
      - ``rotate_range`` etc.            (affine ranges padded to ``spatial_dims``)
      - ``spatial_axis``                 (defaults to all axes when mis-sized)
      - ``has_masks`` / ``mask_disabled``(from the data list)
      - ``resize_keys``                  (include ``mask`` when masks exist)
    """
    data = args.setdefault("data", {})
    spatial_dims = dataset_info.get("spatial_dims") or len(data.get("spatial_size", [0, 0]))
    data["spatial_dims"] = spatial_dims
    data["spatial_size"] = _adjust_spatial_size(data.get("spatial_size", []), spatial_dims)
    for key in ("rotate_range", "shear_range", "translate_range", "scale_range"):
        data[key] = _adjust_affine_range(data.get(key, []), spatial_dims)
    cfg_axis = data.get("spatial_axis") or []
    data["spatial_axis"] = cfg_axis if len(cfg_axis) == spatial_dims else list(range(spatial_dims))

    has_masks = derive_has_masks(data_dicts_sample)
    data["has_masks"] = has_masks
    data["mask_disabled"] = not has_masks
    data["resize_keys"] = ["image", "mask"] if has_masks else ["image"]


def get_modality_pipeline(args, data_dicts_sample, dataset_info):
    """Resolve the modality's ``preprocess`` / ``augmentation`` lists from config.

    Returns ``(preprocess_list, augmentation_list)`` of instantiated MONAI
    transforms, parsed from the MONAI bundle ``modalities.<modality>`` section
    via :class:`monai.bundle.ConfigParser`. Runtime values are injected first by
    :func:`prepare_transform_config`.
    """
    prepare_transform_config(args, data_dicts_sample, dataset_info)
    modality = (dataset_info or {}).get("modality", "")
    modalities = args.get("modalities") or {}
    if not modalities:
        raise ValueError(
            "config has no 'modalities' section; cannot build transforms. "
            "Add a 'modalities' mapping (keys are valid --modality choices)."
        )
    if modality not in modalities:
        # Fallback (e.g. grad_cam invoked without a modality): use the first one.
        fallback = next(iter(modalities))
        logger.warning(
            "Modality %r not declared in config.modalities; falling back to %r",
            modality,
            fallback,
        )
        modality = fallback

    preprocess = _instantiate_bundle(args, f"modalities::{modality}::preprocess")
    augmentation = _instantiate_bundle(args, f"modalities::{modality}::augmentation")
    return preprocess, augmentation


def _instantiate_bundle(args, path):
    """Instantiate a MONAI bundle transform list from config at ``path``.

    ``path`` uses MONAI's ``::`` separator, e.g. ``transforms::preprocess_extra``
    or ``modalities::ct::preprocess``. The list entries are MONAI bundle
    ``_target_`` dicts; ``@data::*`` references resolve against the rest of the
    config. Disabled components (``_disabled_``) resolve to ``None`` and are
    dropped. Returns [] when the section is missing or empty.
    """
    cur = args
    for part in path.split("::"):
        if not isinstance(cur, dict) or part not in cur:
            return []
        cur = cur[part]
    if not cur:
        return []

    from monai.bundle import ConfigParser

    resolved = ConfigParser(config=args).get_parsed_content(path)
    # Components marked ``_disabled_`` resolve to None; drop them.
    return [item for item in resolved if item is not None]


def _strip_meta(data_dicts_sample):
    """Final ``EnsureTyped(track_meta=False)`` that flattens a ``MetaTensor``
    back to a plain ``Tensor`` after all other transforms have run.

    This version of MONAI does not accept ``track_meta`` on spatial/augment
    transforms (they re-wrap into ``MetaTensor``), so we strip meta once, at the
    end of each pipeline, to keep cached/augmented items plain (and thus
    collate-safe). ``mask`` is included when present.
    """
    has_masks = derive_has_masks(data_dicts_sample)
    keys = ["image", "label"] + (["mask"] if has_masks else [])
    return EnsureTyped(keys=keys, track_meta=False)


def strip_image_meta():
    """Image-only variant for the augmentation pipeline (augments ``image``)."""
    return EnsureTyped(keys=["image"], track_meta=False)


def get_loaders_extra(args):
    """Extra loader transforms appended after the auto-derived loaders."""
    return _instantiate_bundle(args, "transforms::loaders_extra")


def get_preprocess_extra(args):
    """Extra preprocessing transforms appended after the modality preprocess."""
    return _instantiate_bundle(args, "transforms::preprocess_extra")


def get_augmentation_extra(args):
    """Extra augmentation transforms appended after the modality augmentation."""
    return _instantiate_bundle(args, "transforms::augmentation_extra")


def build_train_transform(args, data_dicts_sample, dataset_info):
    """Build full training transform (modality preset + user extras from config)."""
    preprocess, augmentation = get_modality_pipeline(args, data_dicts_sample, dataset_info)
    return Compose(
        get_loaders(data_dicts_sample)
        + get_loaders_extra(args)
        + preprocess
        + get_preprocess_extra(args)
        + augmentation
        + get_augmentation_extra(args)
        + [_strip_meta(data_dicts_sample)]
    )


def build_val_transform(args, data_dicts_sample, dataset_info):
    """Build validation/test transform (no augmentation)."""
    preprocess, _ = get_modality_pipeline(args, data_dicts_sample, dataset_info)
    return Compose(
        get_loaders(data_dicts_sample)
        + get_loaders_extra(args)
        + preprocess
        + get_preprocess_extra(args)
        + [_strip_meta(data_dicts_sample)]
    )


def build_preprocess_transform(args, data_dicts_sample, dataset_info):
    """Deterministic preprocessing only (no augmentation), for persistent caching.

    The output of this pipeline is what gets cached to disk. Augmentation is
    kept separate (in ``get_modality_pipeline``) and applied per-epoch on top of
    the cached items, so the cache stays valid regardless of split changes or
    augmentation tweaks.
    """
    preprocess, _ = get_modality_pipeline(args, data_dicts_sample, dataset_info)
    return Compose(
        get_loaders(data_dicts_sample)
        + get_loaders_extra(args)
        + preprocess
        + get_preprocess_extra(args)
        + [_strip_meta(data_dicts_sample)]
    )
