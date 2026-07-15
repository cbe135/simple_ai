import timm
import torch


def create_timm_model(args, in_chans=3):
    """Create a model from timm with pretrained weights.

    ``in_chans`` must match the number of channels the transform pipeline emits
    (single-channel volumes are repeated via ``RepeatChanneld`` to ``data.repeats``
    channels; RGB/color data stays 3). It is derived from the actual preprocessed
    data so the network always matches the transform output.
    """
    return timm.create_model(
        args["training"]["timm_model"],
        pretrained=True,
        num_classes=args["training"]["num_classes"],
        in_chans=in_chans,
    )


def generate_optimizer(args, model):
    """Create an optimizer from the config.

    Supports ``training.optimizer.name`` of: adam, adamw, sgd.
    Extra kwargs (lr, weight_decay, momentum) are read from the config and
    only passed to the optimizers that accept them.
    """
    t = args.get("training", {})
    opt_cfg = t.get("optimizer", {}) or {}
    name = (opt_cfg.get("name") or "adam").lower()
    lr = float(t.get("lr", 0.001))
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))
    momentum = float(opt_cfg.get("momentum", 0.9))

    params = model.parameters()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer name: {name!r}")


def get_device(override=None):
    """Get the best available device: CUDA > MPS (Apple GPU) > CPU.

    An explicit ``override`` (one of "cuda", "mps", "cpu") always wins, which
    is useful for forcing a device on hosts where auto-detection is wrong.
    """
    if override in ("cuda", "mps", "cpu"):
        return override
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
