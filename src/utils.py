import matplotlib.pyplot as plt
import numpy as np


def plot_samples(samples, with_mask=False):
    """Plot sample images with optional masks."""
    for data in samples:
        column_cnt = 2 if with_mask else 1
        image = data["image"][0]
        mask = data["mask"][0]

        fig, axs = plt.subplots(1, column_cnt, figsize=(5 * column_cnt, 4))
        image_title = f"Label: {data['label'].item()}"

        if with_mask:
            axs[0].imshow(image, cmap="gray")
            axs[0].set_title(image_title)
            axs[1].imshow(mask, cmap="gray")
            axs[1].set_title("Corresponding Mask")
        else:
            axs.imshow(image, cmap="gray")
            axs.set_title(image_title)

        plt.show()


def plot_transform_result(data, trans_data, with_mask=False, with_histogram=False):
    """Plot original vs transformed data with optional mask or histogram."""
    assert not (with_mask and with_histogram), "Cannot plot both histogram and mask"

    if with_mask or with_histogram:
        column_cnt = 4
        axis_for_trans_data = 2
    else:
        column_cnt = 2
        axis_for_trans_data = 1

    fig, axs = plt.subplots(1, column_cnt, figsize=(5 * column_cnt, 4))

    axs[0].imshow(data["image"][0], cmap="gray")
    axs[0].set_title("Original Data")

    axs[axis_for_trans_data].imshow(trans_data["image"][0], cmap="gray")
    axs[axis_for_trans_data].set_title("Transformed Data")

    if with_mask:
        axs[1].imshow(data["mask"][0], cmap="gray")
        axs[1].set_title("Original Mask")
        axs[3].imshow(trans_data["mask"][0], cmap="gray")
        axs[3].set_title("Transformed Mask")

    if with_histogram:
        ori_counts, ori_bins = np.histogram(data["image"][0], bins=256)
        axs[1].hist(ori_bins[:-1], bins=256, weights=ori_counts, log=True)
        trans_counts, trans_bins = np.histogram(trans_data["image"][0], bins=256)
        axs[3].hist(trans_bins[:-1], bins=256, weights=trans_counts, log=True)

    plt.show()


def plot_loss_curves(args, record, save_path=None):
    """Plot training and validation loss curves."""
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    axs.plot(record["train"])
    axs.plot(record["val"])
    axs.set_xticks(range(0, args["training"]["num_epoch"] + 1, 5))
    axs.set_ylabel("Loss")
    axs.set_xlabel("Epoch")
    axs.legend(["train", "val"], loc="lower left")
    if save_path:
        fig.savefig(save_path)
    plt.show()
