"""Fine-tuning pipeline (QLoRA / pure LoRA) for the vlm track."""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

from transformers import TrainerCallback

from .config import save_config
from .data import VLMDataset, VLMCollator, build_samples, load_data_list, split_indices
from .models import load_for_training

logger = logging.getLogger(__name__)


class BestEvalLossCallback(TrainerCallback):
    """Snapshot the LoRA adapter whenever eval (validation) loss reaches a new low."""

    def __init__(self, run_dir, processor):
        self.run_dir = Path(run_dir)
        self.processor = processor
        self.best_loss = float("inf")

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        loss = (metrics or {}).get("eval_loss")
        if loss is None or loss >= self.best_loss:
            return
        self.best_loss = loss
        best_dir = self.run_dir / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(best_dir)
        self.processor.save_pretrained(best_dir)
        (best_dir / "best_eval_loss.txt").write_text(f"{loss:.6f}")
        (best_dir / "best_step.txt").write_text(str(state.global_step))
        logger.info(
            "New best eval_loss=%.4f at step %s -> saved adapter to %s",
            loss, state.global_step, best_dir,
        )


def _save_loss_curve(run_dir, log_history):
    """Plot training (and eval, if present) loss from the Trainer log history."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    steps, train_loss = [], []
    eval_steps, eval_loss = [], []
    for entry in log_history:
        if "loss" in entry:
            steps.append(entry.get("step", len(steps) + 1))
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry.get("step", len(eval_steps) + 1))
            eval_loss.append(entry["eval_loss"])

    fig, ax = plt.subplots(figsize=(8, 5))
    if train_loss:
        ax.plot(steps, train_loss, label="train")
    if eval_loss:
        ax.plot(eval_steps, eval_loss, label="eval")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss")
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = Path(run_dir) / "loss_curve.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved loss curve to %s", path)


def train_pipeline(cfg: dict, data_dir, base_dir=None, device=None, quantize=None):
    from torch.utils.data import DataLoader
    from transformers import Trainer, TrainingArguments

    data_dir = Path(data_dir)
    entries = load_data_list(data_dir)
    labels = [int(e["label"]) for e in entries]
    train_idx, val_idx, test_idx = split_indices(cfg, len(entries), labels, cfg["data"]["seed"])

    image_size = int(cfg["data"]["image_size"])
    train_samples = build_samples(cfg, entries, train_idx, data_dir)
    val_samples = build_samples(cfg, entries, val_idx, data_dir)

    model, processor, device = load_for_training(cfg, base_dir, device, quantize)

    ds_train = VLMDataset(train_samples, image_size)
    ds_val = VLMDataset(val_samples, image_size)
    collator = VLMCollator(processor, max_length=int(cfg["training"].get("max_length", 512)))

    # Output directory: <output_dir>/<task>/<timestamp>/
    out_base = Path(cfg.get("output_dir", "outputs"))
    run_dir = out_base / cfg["task"] / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run output directory: %s", run_dir)

    t = cfg["training"]
    args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=int(t.get("num_epoch", 3)),
        per_device_train_batch_size=int(t.get("batch_size", 4)),
        per_device_eval_batch_size=max(1, int(t.get("batch_size", 4)) // 2),
        gradient_accumulation_steps=int(t.get("gradient_accumulation_steps", 1)),
        learning_rate=float(t.get("lr", 1e-4)),
        warmup_ratio=float(t.get("warmup_ratio", 0.03)),
        weight_decay=float(t.get("weight_decay", 0.0)),
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=1,
        gradient_checkpointing=True,
        report_to="none",
        bf16=device == "cuda",
        fp16=False,
        remove_unused_columns=False,
        dataloader_num_workers=int(cfg["data"].get("num_workers", 4)),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=collator,
        callbacks=[BestEvalLossCallback(run_dir, processor)],
    )
    trainer.train()
    _save_loss_curve(run_dir, trainer.state.log_history)

    # Persist the LoRA adapter (the fine-tuned result) + processor + resolved config.
    adapter_dir = run_dir / "adapter"
    best_dir = run_dir / "best"
    if best_dir.exists():
        shutil.copytree(best_dir, adapter_dir, dirs_exist_ok=True)
        logger.info("Best model (eval_loss=%.4f) copied to %s",
                    float((best_dir / "best_eval_loss.txt").read_text()), adapter_dir)
    else:
        model.save_pretrained(adapter_dir)
        processor.save_pretrained(adapter_dir)
    (adapter_dir / "base_model_id.txt").write_text(cfg["model"]["model_id"])
    save_config(cfg, run_dir / "vlm_config.yaml")

    logger.info("Saved adapter + config to %s", adapter_dir)
    logger.info("To infer: simple_ai_vlm_infer --config <cfg> --data-dir %s --adapter %s",
                data_dir, adapter_dir)
    return run_dir, adapter_dir
