import logging
import os
import sys
import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.metrics import auc, confusion_matrix, roc_curve

from .model import get_device
from .transforms import build_val_transform
from .utils import tqdm_disabled

logger = logging.getLogger(__name__)


def infer(args, model, data_loader, details=False, device=None, details_path=None):
    """Run inference and return true labels and predictions.

    When ``details`` is True and ``details_path`` is given, the per-image
    match/mismatch lines are written to that file instead of printed to the
    console (which would otherwise flood stdout / run.log across thousands of
    images). If ``details_path`` is None, the lines are printed for backwards
    compatibility.
    """
    device = device or get_device()
    logger.info("Using device for inference: %s", device)
    sigmoid = torch.nn.Sigmoid()
    thres = args["threshold"]

    y_true = []
    y_pred = []

    details_fh = open(details_path, "w") if (details and details_path) else None
    try:
        model.eval()
        with torch.no_grad():
            for data in tqdm(data_loader, file=sys.stderr, disable=tqdm_disabled()):
                images = data["image"].to(device)
                labels = data["label"].to(device)

                preds = sigmoid(model(images))

                for i, (pred, label) in enumerate(zip(preds, labels)):
                    y_pred.append(pred.item())
                    y_true.append(label.item())

                    if details:
                        binary_pred = (pred >= thres).float()
                        filename = images.meta["filename_or_obj"][i]
                        line = (
                            f"{filename} mismatch: pred={binary_pred.item()}, "
                            f"label={label.item()}"
                            if binary_pred != label
                            else f"{filename} match: pred={binary_pred.item()}, "
                            f"label={label.item()}"
                        )
                        if details_fh is not None:
                            details_fh.write(line + "\n")
                        else:
                            print(line)
    finally:
        if details_fh is not None:
            details_fh.close()

    return y_true, y_pred


def plot_roc_and_show_result(args, y_true, y_pred, title="", save_path=None):
    """Plot ROC curve and print confusion matrix metrics."""
    import matplotlib.pyplot as plt

    thres = args["threshold"]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 4))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.ylabel("Sensitivity")
    plt.xlabel("1 - Specificity")
    if save_path:
        plt.savefig(save_path)
    plt.show()

    y_pred_binary = np.where(np.array(y_pred) >= thres, 1, 0)
    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    logger.info(
        f"True positive: {tp}\n"
        f"False positive: {fp}\n"
        f"False negative: {fn}\n"
        f"True negative: {tn}\n"
        f"Sensitivity: {sensitivity:.4f}\n"
        f"Specificity: {specificity:.4f}"
    )


def grad_cam(
    model,
    img_path,
    class_id,
    args,
    data_dicts_sample=None,
    dataset_info=None,
    rgb=True,
    device="cuda",
    alpha=0.4,
    figsize=(16, 16),
    save_path=None,
):
    """Generate and display Grad-CAM visualization."""
    import matplotlib.pyplot as plt
    from torchvision import transforms

    device = get_device()
    if data_dicts_sample is None:
        data_dicts_sample = [
            {
                "image": img_path,
                "mask": img_path.replace("/images/", "/masks/"),
                "label": 0,
            }
        ]
    if dataset_info is None:
        dataset_info = {}
    test_transform = build_val_transform(args, data_dicts_sample, dataset_info)

    acts = [0]
    grads = [0]

    def f_hook(self, input, output):
        acts[0] = output

    def b_hook(self, grad_in, grad_out):
        grads[0] = grad_out

    def find_last_conv(model):
        last_conv = None

        def _find_last_conv(module):
            nonlocal last_conv
            for child in module.children():
                if isinstance(child, torch.nn.Conv2d):
                    last_conv = child
                _find_last_conv(child)

        _find_last_conv(model)
        return last_conv

    module = find_last_conv(model)

    h1 = module.register_forward_hook(f_hook)
    h2 = module.register_backward_hook(b_hook)

    data_item = {
        "image": img_path,
        "mask": img_path.replace("/images/", "/masks/"),
        "label": 0,
    }
    img = test_transform(data_item)["image"]
    img = img.mean(dim=0, keepdim=True)
    img = torch.permute(img, (1, 2, 0))

    img_t = test_transform(data_item)["image"]
    img_t = img_t.unsqueeze(dim=0).to(device)

    model.to(device)
    outs = model(img_t)
    h1.remove()
    h2.remove()
    outs[0, class_id].backward()

    gap = torch.mean(
        grads[0][0].view(grads[0][0].size(0), grads[0][0].size(1), -1), dim=2
    )
    acts = acts[0][0]
    gradcam = torch.nn.ReLU()(
        torch.sum(gap[0].reshape((gap.size()[1], 1, 1)) * acts, dim=0)
    )
    arr = transforms.Resize((img.shape[1], img.shape[0]))(gradcam.unsqueeze(0))
    gradcam_img = arr.detach().cpu().permute((1, 2, 0)).squeeze(-1)

    fig, axs = plt.subplots(1, figsize=figsize)
    axs.imshow(np.asarray(img), cmap="gray")
    axs.imshow(gradcam_img, alpha=alpha, cmap="hot")
    axs.set_xticks([])
    axs.set_yticks([])

    if save_path:
        fig.savefig(save_path)
