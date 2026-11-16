"""part3_gen_embeddings.py
Generate AE embeddings for the 3 labeled images.
Saves CSV files that part3_modeling.py picks up.

  python part3_gen_embeddings.py --emb_size 16
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
from autoencoder import Autoencoder

# same 3 labeled images used in the modeling script
LABELED_FILES = {
    "O012791": "../data/image_data/O012791.npz",
    "O013257": "../data/image_data/O013257.npz",
    "O013490": "../data/image_data/O013490.npz",
}

EMB_CONFIG = {
    8: {
        "checkpoint": "checkpoints/best_emb8-epoch=072.ckpt",
        "n_input_channels": 8,
        "patch_size": 9,
        "embedding_size": 8,
    },
    16: {
        "checkpoint": "checkpoints/best_emb16-epoch=067.ckpt",
        "n_input_channels": 8,
        "patch_size": 9,
        "embedding_size": 16,
    },
}


def extract_patches_single_image(image_data, patch_size=9):
    """Extract 9x9 patches centered on each pixel, normalized per-image.
    This matches the preprocessing used during AE training so the
    encoder sees the same distribution at inference time."""
    ys = image_data[:, 0].astype(int)
    xs = image_data[:, 1].astype(int)
    features = image_data[:, 2:10]  # columns 2-9 = 8 radiometric features

    n_channels = features.shape[1]
    pad = patch_size // 2  # how much to pad on each side

    # reconstruct the 2D spatial grid from the flat pixel list
    # we need this so we can extract contiguous 9x9 patches
    miny, maxy = ys.min(), ys.max()
    minx, maxx = xs.min(), xs.max()
    height = int(maxy - miny + 1)
    width = int(maxx - minx + 1)

    grid = np.zeros((n_channels, height, width))
    y_rel = ys - miny
    x_rel = xs - minx
    for c in range(n_channels):
        grid[c, y_rel, x_rel] = features[:, c]

    # z-score normalize each channel independently
    for c in range(n_channels):
        mu = grid[c].mean()
        std = grid[c].std()
        if std > 0:
            grid[c] = (grid[c] - mu) / std

    # reflect-pad the edges so border pixels get full patches
    grid_padded = np.pad(
        grid,
        ((0, 0), (pad, pad), (pad, pad)),
        mode="reflect",
    )

    # extract one patch per pixel
    patches = []
    for y, x in zip(ys, xs):
        yi = int(y - miny + pad)
        xi = int(x - minx + pad)
        patch = grid_padded[:, yi - pad : yi + pad + 1, xi - pad : xi + pad + 1]
        patches.append(patch.astype(np.float32))

    return patches, ys, xs


def generate_embeddings(emb_size, output_dir="../data"):
    """Load a trained AE checkpoint, run the encoder on each labeled image,
    and save the resulting embeddings as CSVs with columns (y, x, ae0, ae1, ...)."""
    cfg = EMB_CONFIG[emb_size]
    ckpt_path = cfg["checkpoint"]

    if not os.path.exists(ckpt_path):
        print(f"  [ERROR] Checkpoint not found: {ckpt_path}")
        return False

    print(f"\n--- Generating AE{emb_size} embeddings ---")
    print(f"  Checkpoint: {ckpt_path}")

    # load the autoencoder and put it in eval mode (no dropout etc.)
    model = Autoencoder(
        patch_size=cfg["patch_size"],
        n_input_channels=cfg["n_input_channels"],
        embedding_size=cfg["embedding_size"],
    )
    map_location = None if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    for name, path in LABELED_FILES.items():
        if not os.path.exists(path):
            print(f"  [ERROR] Data file not found: {path}")
            continue

        arr = np.load(path)["arr_0"]
        patches, ys, xs = extract_patches_single_image(arr, cfg["patch_size"])

        # run patches through the encoder in batches to avoid OOM
        batch_size = 4096
        all_emb = []
        with torch.no_grad():
            for start in range(0, len(patches), batch_size):
                batch = torch.tensor(np.array(patches[start : start + batch_size]))
                emb = model.embed(batch).detach().cpu().numpy()
                all_emb.append(emb)
        all_emb = np.concatenate(all_emb, axis=0)

        # build a dataframe with pixel coords + embedding columns
        ae_cols = [f"ae{j}" for j in range(all_emb.shape[1])]
        df = pd.DataFrame(all_emb, columns=ae_cols)
        df.insert(0, "y", ys.astype(float))
        df.insert(1, "x", xs.astype(float))

        # save as CSV; part3_modeling.py will merge on (y, x)
        out_path = os.path.join(output_dir, f"{name}_ae{emb_size}.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved {out_path}  ({len(df)} pixels, {all_emb.shape[1]} dims)")

    return True


def main():
    parser = argparse.ArgumentParser(description="Generate AE embeddings for labeled images")
    parser.add_argument(
        "--emb_size",
        type=str,
        default="all",
        help="Embedding size: 8, 16, or 'all' (default: all)",
    )
    args = parser.parse_args()

    if args.emb_size == "all":
        sizes = [8, 16]
    else:
        sizes = [int(args.emb_size)]

    for sz in sizes:
        ok = generate_embeddings(sz)
        if not ok:
            print(f"  [WARNING] Skipped AE{sz} due to missing checkpoint")

    print("\nDone generating embeddings.")


if __name__ == "__main__":
    main()
