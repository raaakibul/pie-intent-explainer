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


def compute_loss(outputs: dict, batch: dict, cfg: dict):
    l = cfg["loss"]
    ce = nn.CrossEntropyLoss(label_smoothing=l["label_smoothing"])
    intent_loss = ce(outputs["intent_logits"], batch["intent_label"])

    traj_loss = nn.functional.mse_loss(outputs["pred_traj"], batch["traj_future"])

    total = l["intent_weight"] * intent_loss + l["trajectory_weight"] * traj_loss
    return total, {"intent_loss": intent_loss.item(), "traj_loss": traj_loss.item(), "total": total.item()}

def lr_lambda_fn(step: int, total_steps: int, warmup_steps: int):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


def move_batch_to_device(batch: dict, device: str):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

def evaluate_val_loss(model, val_loader, cfg, device):
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch, teacher_forcing=False)
            loss, _ = compute_loss(outputs, batch, cfg)
            total_loss += loss.item() * batch["intent_label"].shape[0]
            n += batch["intent_label"].shape[0]
    model.train()
    return total_loss / max(1, n)

