"""Metrics for vlm classification: accuracy, macro-F1, confusion matrix,
and (binary) ROC/AUC. Writes a metrics file + PNGs into the run directory.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def evaluate(rows, label_map: dict, run_dir, scores=None, backend: str = "hf") -> dict:
    import csv

    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    y_true = [r[1] for r in rows]
    y_pred = [r[3] if r[3] is not None else -1 for r in rows]
    classes = sorted(label_map.keys())

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=classes, zero_division=0)

    # Write per-image predictions.
    pred_path = run_dir / "predictions.csv"
    with open(pred_path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["filename", "label_true", "generated_text", "pred_label", "correct"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], "" if r[3] is None else r[3],
                        r[1] == r[3] if r[3] is not None else False])

    # Confusion matrix.
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(max(4, len(classes) + 1), max(4, len(classes) + 1)))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels([label_map[c] for c in classes], rotation=45, ha="right")
    ax.set_yticklabels([label_map[c] for c in classes])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    cm_path = run_dir / "confusion_matrix.png"
    fig.savefig(cm_path)
    plt.close(fig)

    metrics = {
        "backend": backend,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "n": len(rows),
        "confusion_matrix": cm.tolist(),
        "classes": [label_map[c] for c in classes],
    }

    # Binary ROC/AUC when calibrated scores are available.
    if scores is not None and len(classes) == 2:
        try:
            y_score = [s if s is not None else 0.0 for s in scores]
            auc = roc_auc_score(y_true, y_score)
            metrics["auc"] = float(auc)
            fpr, tpr, _ = _roc_curve(np.array(y_true), np.array(y_score), pos_label=classes[1])
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
            ax.plot([0, 1], [0, 1], "--", color="gray")
            ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
            ax.legend()
            fig.tight_layout()
            roc_path = run_dir / "roc.png"
            fig.savefig(roc_path)
            plt.close(fig)
            logger.info("Binary AUC: %.4f", auc)
        except Exception as e:  # noqa: BLE001
            logger.warning("AUC computation skipped: %s", e)

    (run_dir / "metrics.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in metrics.items())
    )
    logger.info("Accuracy: %.4f | Macro-F1: %.4f | n=%d", acc, macro_f1, len(rows))
    logger.info("Artifacts in %s: predictions.csv, confusion_matrix.png, metrics.txt", run_dir)
    return metrics


def _roc_curve(y_true, y_score, pos_label):
    from sklearn.metrics import roc_curve as _rc

    return _rc(y_true, y_score, pos_label=pos_label)
