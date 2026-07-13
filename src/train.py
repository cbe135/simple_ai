import logging
import os
import numpy as np
import torch
from tqdm.auto import tqdm
from monai.utils.misc import set_determinism

from .model import create_timm_model, generate_optimizer, get_device
from .data import generate_dataloader

logger = logging.getLogger(__name__)


def train_one_epoch(args, model, criterion, optimizer, train_loader, val_loader):
    """Train for one epoch and return train/val loss."""
    device = get_device()
    train_loss = 0.0
    val_loss = 0.0

    model.train()
    for data in train_loader:
        images = data["image"].to(device)
        labels = data["label"].to(device).float()

        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, labels.reshape(preds.shape))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    with torch.no_grad():
        for data in val_loader:
            images = data["image"].to(device)
            labels = data["label"].to(device).float()

            preds = model(images)
            loss = criterion(preds, labels.reshape(preds.shape))
            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    return train_loss, val_loss


def train(args, model, criterion, optimizer, train_loader, val_loader, run_dir=None):
    """Full training loop. Saves best weights and returns loss record."""
    if run_dir is None:
        from src.env_setup import default_data_dir

        run_dir = default_data_dir()
    os.makedirs(run_dir, exist_ok=True)
    save_path = os.path.join(run_dir, "best_weights.pth")

    record = {"train": [], "val": []}
    best_val_loss = np.inf

    for epoch in tqdm(range(args["training"]["num_epoch"])):
        train_loss, val_loss = train_one_epoch(
            args, model, criterion, optimizer, train_loader, val_loader
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        record["train"].append(train_loss)
        record["val"].append(val_loss)

        logger.info(
            f"[{epoch + 1}/{args['training']['num_epoch']}] "
            f"Train loss: {train_loss:3.3f}, "
            f"Validation loss: {val_loss:3.3f}"
        )

    return record


def train_pipeline(args, train_set, val_set, run_dir=None):
    """Complete training pipeline: create model, train, return results."""
    set_determinism(args["environ"]["seed"])

    device = get_device()
    model = create_timm_model(args).to(device)

    train_loader = generate_dataloader(args, train_set, shuffle=True)
    val_loader = generate_dataloader(args, val_set)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = generate_optimizer(args, model)

    record = train(
        args, model, criterion, optimizer, train_loader, val_loader, run_dir
    )

    return model, train_loader, val_loader, record
