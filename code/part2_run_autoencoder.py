# EXAMPLE USAGE:
# python run_autoencoder.py configs/default.yaml

import numpy as np
import sys
import os
import yaml  # pip install pyyaml
import gc
import torch
import lightning as L

from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
# pip install torchinfo
# from torchinfo import summary

from autoencoder import Autoencoder
from patchdataset import PatchDataset
from data import make_data


print("loading config file")
config_path = sys.argv[1]
assert os.path.exists(config_path), f"Config file {config_path} not found"
config = yaml.safe_load(open(config_path, "r"))

# clean up memory
gc.collect()
torch.cuda.empty_cache()

print("making the patch data")
# get the patches
_, patches = make_data(
    patch_size=config["data"]["patch_size"],
    max_images=config["data"].get("max_images", None)
)
# let's just combine the patches from all images into one list
# all_patches = [patch for image_patches in patches for patch in image_patches]

# get number of images and shuffle indices
n_images = len(patches)
image_indices = np.random.permutation(n_images)

# determine split point (80%)
split_idx = int(0.8 * n_images)

# split by image not all patches
train_image_idx = image_indices[:split_idx]
val_image_idx = image_indices[split_idx:]

# collect patches from those images
train_patches = []
for i in train_image_idx:
    train_patches.extend(patches[i])

val_patches = []
for i in val_image_idx:
    val_patches.extend(patches[i])

train_dataset = PatchDataset(train_patches)
val_dataset = PatchDataset(val_patches)

# create train and val dataloaders
dataloader_train = DataLoader(train_dataset, **config["dataloader_train"])
dataloader_val = DataLoader(val_dataset, **config["dataloader_val"])

print("initializing model")
# Initialize an autoencoder object
model = Autoencoder(
    optimizer_config=config["optimizer"],
    patch_size=config["data"]["patch_size"],
    **config["autoencoder"],
)
print(model)
# print(summary(model, (8, 9, 9)))

print("preparing for training")
# configure the settings for making checkpoints
checkpoint_callback = ModelCheckpoint(**config["checkpoint"])

# if running in slurm, add slurm job id info to the config file
if "SLURM_JOB_ID" in os.environ:
    config["slurm_job_id"] = os.environ["SLURM_JOB_ID"]

# initialize the wandb logger, giving it our config file
# to save, and also configuring the logger itself.
wandb_logger = None
#WandbLogger(config=config, **config["wandb"])

# initialize the trainer
trainer = L.Trainer(
    callbacks=[checkpoint_callback], **config["trainer"]
)

print("training")
trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

# clean up memory
gc.collect()
torch.cuda.empty_cache()
