# Reproducible pipeline: loads one image, runs the autoencoder, saves a plot.
#
# EXAMPLE USAGE:
#   python generate_plot.py configs/best.yaml checkpoints/best_emb16-epoch=067.ckpt
#
# All .npz files are expected to be somewhere under ../data/
# The target image is found by filename regardless of subfolder.

import sys
import os
import glob
import yaml
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

from autoencoder import Autoencoder

# config 
TARGET_IMAGE   = "O013257.npz"   # image to visualize
PATCH_IDX      = 8               # patch to show
CHANNEL        = 0               # channel to display (0-7)
OUTPUT_PATH    = "../figures/patch_reconstruction.png"



def find_image(filename, search_root="../data"):
    """Search recursively for a .npz file by name under search_root."""
    matches = glob.glob(os.path.join(search_root, "**", filename), recursive=True)
    if len(matches) == 0:
        raise FileNotFoundError(
            f"Could not find {filename} anywhere under {search_root}. "
            "Make sure all .npz files are in ../data/ or a subfolder."
        )
    return matches[0]


def make_data_single(filepath, patch_size=9):
    """
    Load and patch a single .npz image file.
    Normalizes per-image (mean/std per channel).
    Returns:
        image_long: raw numpy array of shape (n_pixels, n_cols)
        patches: list of float32 arrays of shape (n_channels, patch_size, patch_size)
    """
    npz_data = np.load(filepath)
    key = list(npz_data.files)[0]
    data = npz_data[key]
    if data.shape[1] == 11:
        data = data[:, :-1]  # remove expert labels if present

    ys = data[:, 0].astype(int)
    xs = data[:, 1].astype(int)
    miny, minx = ys.min(), xs.min()
    height = int(ys.max() - miny + 1)
    width  = int(xs.max() - minx + 1)

    nchannels = data.shape[1] - 2
    image = np.zeros((nchannels, height, width))
    for c in range(nchannels):
        image[c, ys - miny, xs - minx] = data[:, c + 2]

    # per-image normalization
    means = image.mean(axis=(1, 2), keepdims=True)
    stds  = image.std(axis=(1, 2),  keepdims=True) + 1e-8
    image = (image - means) / stds

    pad_len    = patch_size // 2
    img_mirror = np.pad(
        image,
        ((0, 0), (pad_len, pad_len), (pad_len, pad_len)),
        mode="reflect",
    )

    patches = []
    for y, x in zip(ys, xs):
        y_idx = int(y - miny + pad_len)
        x_idx = int(x - minx + pad_len)
        patch = img_mirror[
            :,
            y_idx - pad_len : y_idx + pad_len + 1,
            x_idx - pad_len : x_idx + pad_len + 1,
        ]
        patches.append(patch.astype(np.float32))

    return data, patches


def main():
    # load config and checkpoint 
    assert len(sys.argv) == 3, "Usage: python generate_plot.py <config.yaml> <checkpoint.ckpt>"
    config_path     = sys.argv[1]
    checkpoint_path = sys.argv[2]

    config = yaml.safe_load(open(config_path, "r"))

    model = Autoencoder(
        patch_size=config["data"]["patch_size"],
        **config["autoencoder"],
    )
    map_location = None if torch.cuda.is_available() else "cpu"
    checkpoint   = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # find and load the target image
    image_path = find_image(TARGET_IMAGE)
    print(f"Found: {image_path}")

    _, patches = make_data_single(image_path, patch_size=config["data"]["patch_size"])

    # run the autoencoder 
    patch = patches[PATCH_IDX]
    patch_tensor = torch.tensor(patch).float().unsqueeze(0)

    with torch.no_grad():
        reconstructed = model(patch_tensor)

    patch_np = patch_tensor.squeeze().numpy()        # (n_channels, 9, 9)
    recon_np = reconstructed.squeeze().numpy()       # (n_channels, 9, 9)

    #  plot 
    vmin = min(patch_np[CHANNEL].min(), recon_np[CHANNEL].min())
    vmax = max(patch_np[CHANNEL].max(), recon_np[CHANNEL].max())

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(patch_np[CHANNEL], vmin=vmin, vmax=vmax)
    axes[0].set_title("Original Patch")

    im = axes[1].imshow(recon_np[CHANNEL], vmin=vmin, vmax=vmax)
    axes[1].set_title("Reconstructed Patch")

    fig.colorbar(im, ax=axes, shrink=0.8)

    # save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=150)
    print(f"Plot saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()