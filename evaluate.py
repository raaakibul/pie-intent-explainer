from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.pie_dataset import PIEClipDataset, collate_fn
from models.iet_model import IETModel
from utils import metrics as M


def move_batch_to_device(batch: dict, device: str):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def run_trajectory_and_intent_eval(model, loader, device):
    all_pred_traj, all_gt_traj = [], []
    all_intent_true, all_intent_pred, all_intent_prob = [], [], []
    per_clip_records = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating trajectory/intent"):
            batch_gpu = move_batch_to_device(batch, device)
            outputs = model(batch_gpu, teacher_forcing=False)

            pred_traj = outputs["pred_traj"].cpu().numpy()
            gt_traj = batch["traj_future"].numpy()
            intent_probs = outputs["intent_probs"].cpu().numpy()
            intent_pred = intent_probs.argmax(axis=-1)
            intent_true = batch["intent_label"].numpy()

            all_pred_traj.append(pred_traj)
            all_gt_traj.append(gt_traj)
            all_intent_true.append(intent_true)
            all_intent_pred.append(intent_pred)
            all_intent_prob.append(intent_probs)

            for i, clip_id in enumerate(batch["clip_id"]):
                per_clip_records.append({
                    "clip_id": clip_id,
                    "pred_traj": pred_traj[i].tolist(),
                    "intent_label": "crossing" if intent_pred[i] == 1 else "not-crossing",
                    "intent_confidence": float(intent_probs[i].max()),
                    "ego_speed_mps": float(batch["ego_speed"][i].item()),
                })

    pred_traj = np.concatenate(all_pred_traj, axis=0)
    gt_traj = np.concatenate(all_gt_traj, axis=0)
    intent_true = np.concatenate(all_intent_true, axis=0)
    intent_pred = np.concatenate(all_intent_pred, axis=0)
    intent_prob = np.concatenate(all_intent_prob, axis=0)

    results = {
        "ADE": M.average_displacement_error(pred_traj, gt_traj),
        "FDE": M.final_displacement_error(pred_traj, gt_traj),
        **M.intent_accuracy_f1(intent_true, intent_pred),
        "ECE": M.expected_calibration_error(intent_true, intent_prob),
    }
    return results, per_clip_records