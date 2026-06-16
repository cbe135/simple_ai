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


def get_device():
    """Get the available device (CUDA or CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"
