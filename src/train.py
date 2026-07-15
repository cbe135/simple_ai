import logging
import os
import sys
import numpy as np
import torch
from tqdm.auto import tqdm
from monai.utils.misc import set_determinism

from .model import create_timm_model, generate_optimizer, get_device
from .data import generate_dataloader
from .utils import tqdm_disabled

logger = logging.getLogger(__name__)


def build_criterion(args):
    """Create the loss function from ``training.loss.name``.

    ``bce_with_logits`` (default) is for binary classification with
    ``num_classes: 1`` and expects float targets. ``cross_entropy`` is for
    multi-class (``num_classes > 1``) and expects long class-index targets.
    """
    loss_cfg = (args.get("training", {}) or {}).get("loss", {}) or {}
    name = (loss_cfg.get("name") or "bce_with_logits").lower()
    if name == "cross_entropy":
        return torch.nn.CrossEntropyLoss()
    if name == "bce_with_logits":
        return torch.nn.BCEWithLogitsLoss()
    raise ValueError(f"Unsupported loss name: {name!r}")


def _target_for_loss(labels, loss_name):
    """Coerce labels to the dtype/shape the criterion expects."""
    if loss_name == "cross_entropy":
        return labels.long().squeeze(-1)
    return labels.float()


def train_one_epoch(args, model, criterion, optimizer, train_loader, val_loader, device=None):
    """Train for one epoch and return train/val loss."""
    device = device or get_device()
    loss_name = ((args.get("training", {}) or {}).get("loss", {}) or {}).get("name", "bce_with_logits")
    train_loss = 0.0
    val_loss = 0.0

    model.train()
    for data in train_loader:
        images = data["image"].to(device)
        labels = data["label"].to(device)

        optimizer.zero_grad()
        preds = model(images)
        target = _target_for_loss(labels, loss_name)
        loss = criterion(preds, target.reshape(preds.shape) if loss_name != "cross_entropy" else target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    with torch.no_grad():
        for data in val_loader:
            images = data["image"].to(device)
            labels = data["label"].to(device)

            preds = model(images)
            target = _target_for_loss(labels, loss_name)
            loss = criterion(preds, target.reshape(preds.shape) if loss_name != "cross_entropy" else target)
            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    return train_loss, val_loss


def train(args, model, criterion, optimizer, train_loader, val_loader, run_dir=None, device=None):
    """Full training loop. Saves best weights and returns loss record."""
    if run_dir is None:
        from src.env_setup import default_data_dir

        run_dir = default_data_dir()
    os.makedirs(run_dir, exist_ok=True)
    save_path = os.path.join(run_dir, "best_weights.pth")

    record = {"train": [], "val": []}
    best_val_loss = np.inf

    for epoch in tqdm(range(args["training"]["num_epoch"]), file=sys.stderr, disable=tqdm_disabled()):
        train_loss, val_loss = train_one_epoch(
            args, model, criterion, optimizer, train_loader, val_loader, device
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved best weights to {save_path}")

        record["train"].append(train_loss)
        record["val"].append(val_loss)

        logger.info(
            f"[{epoch + 1}/{args['training']['num_epoch']}] "
            f"Train loss: {train_loss:3.3f}, "
            f"Validation loss: {val_loss:3.3f}"
        )

    return record


def train_pipeline(args, train_set, val_set, run_dir=None, device=None, in_chans=3):
    """Complete training pipeline: create model, train, return results."""
    set_determinism(args["environ"]["seed"])

    device = device or get_device()
    if device == "cuda":
        # Small conv models benefit from autotuning kernel selection.
        torch.backends.cudnn.benchmark = True
    if device == "cuda":
        dev_name = torch.cuda.get_device_name(0)
    else:
        dev_name = device
    logger.info("Using device: %s — %s", device, dev_name)
    logger.info(
        "Training config — batch_size: %d, num_workers: %d, cache_rate: %s, num_epoch: %d",
        args["training"]["batch_size"],
        int(args["data"].get("num_workers", 0)),
        args["data"]["cache_rate"],
        args["training"]["num_epoch"],
    )

    model = create_timm_model(args, in_chans=in_chans).to(device)

    train_loader = generate_dataloader(args, train_set, shuffle=True, device=device)
    val_loader = generate_dataloader(args, val_set, device=device)

    criterion = build_criterion(args)
    optimizer = generate_optimizer(args, model)

    record = train(
        args, model, criterion, optimizer, train_loader, val_loader, run_dir
    )

    return model, train_loader, val_loader, record
