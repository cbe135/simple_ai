import timm
import torch


def create_timm_model(args):
    """Create a model from timm with pretrained weights."""
    return timm.create_model(
        args["training"]["timm_model"],
        pretrained=True,
        num_classes=args["training"]["num_classes"],
    )


def generate_optimizer(args, model):
    """Create an Adam optimizer."""
    return torch.optim.Adam(model.parameters(), lr=args["training"]["lr"])


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
