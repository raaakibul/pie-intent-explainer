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

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["experiment"]["seed"])
    device = cfg["experiment"]["device"] if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["experiment"]["checkpoint_dir"], exist_ok=True)

    train_loader, val_loader = build_dataloaders(cfg)

    model = IETModel(cfg).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed weights from {args.resume}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"],
    )

    total_steps = len(train_loader) * cfg["train"]["epochs"]
    warmup_steps = len(train_loader) * cfg["train"]["warmup_epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: lr_lambda_fn(step, total_steps, warmup_steps)
    )
    

    
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}")
        
        for batch in pbar:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch, teacher_forcing=True)
            loss, loss_dict = compute_loss(outputs, batch, cfg)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip_norm"])
            optimizer.step()
            scheduler.step()

            global_step += 1
            if global_step % cfg["train"]["log_every_n_steps"] == 0:
                pbar.set_postfix(loss_dict)
                
        if (epoch + 1) % cfg["train"]["val_every_n_epochs"] == 0:
            val_loss = evaluate_val_loss(model, val_loader, cfg, device)
            print(f"[epoch {epoch+1}] val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                ckpt_path = os.path.join(cfg["experiment"]["checkpoint_dir"], "best.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"  New best model saved to {ckpt_path}")
            else:
                patience_counter += 1
                if patience_counter >= cfg["train"]["early_stopping_patience"]:
                    print(f"Early stopping at epoch {epoch+1}.")
                    break
    
    last_ckpt = os.path.join(cfg["experiment"]["checkpoint_dir"], "last.pt")
    torch.save(model.state_dict(), last_ckpt)
    print(f"Training complete. Last checkpoint saved to {last_ckpt}")


if __name__ == "__main__":
    main()