from __future__ import annotations

import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.pie_dataset import PIEClipDataset, collate_fn
from models.iet_model import IETModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_dataloaders(cfg: dict):
    processed = cfg["data"]["processed_dir"]
    pie_image_root = os.path.join(cfg["data"]["pie_root"], "images")

    train_ds = PIEClipDataset(os.path.join(processed, "train.json"), pie_image_root, cfg)
    val_ds = PIEClipDataset(os.path.join(processed, "val.json"), pie_image_root, cfg)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["data"]["num_workers"], collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=cfg["data"]["num_workers"], collate_fn=collate_fn,
    )
    return train_loader, val_loader