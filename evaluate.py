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

def run_explanation_eval(per_clip_records, human_explanations_path, cfg):
    from explain.llm_explainer import build_explainer

    if not os.path.exists(human_explanations_path):
        print(f"[warn] {human_explanations_path} not found — skipping BLEU-4 explanation eval. "
              f"See docs/annotation_guidelines.md to produce this file.")
        return None

    with open(human_explanations_path) as f:
        human_refs = json.load(f)

    eval_clip_ids = [c for c in per_clip_records if c["clip_id"] in human_refs]
    if not eval_clip_ids:
        print("[warn] No overlap between predictions and human_explanations.json clip_ids.")
        return None

    explainer = build_explainer(cfg)
    frame_dt_s = cfg["data"]["frame_stride"] / 30.0

    candidates, references = [], []
    for rec in tqdm(eval_clip_ids, desc="Generating explanations"):
        result = explainer.explain_from_model_output(
            intent_label=rec["intent_label"],
            intent_confidence=rec["intent_confidence"],
            pred_traj_xy=rec["pred_traj"],
            ego_speed_mps=rec["ego_speed_mps"],
            frame_dt_s=frame_dt_s,
            cfg=cfg,
        )
        candidates.append(result["explanation"])
        references.append(human_refs[rec["clip_id"]])

    return {"BLEU-4": M.bleu4(candidates, references), "n_examples": len(candidates)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--skip_explanations", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = cfg["experiment"]["device"] if torch.cuda.is_available() else "cpu"

    pie_image_root = os.path.join(cfg["data"]["pie_root"], "images")
    split_json = os.path.join(cfg["data"]["processed_dir"], f"{args.split}.json")
    ds = PIEClipDataset(split_json, pie_image_root, cfg)
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                         num_workers=cfg["data"]["num_workers"], collate_fn=collate_fn)

    model = IETModel(cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    traj_intent_results, per_clip_records = run_trajectory_and_intent_eval(model, loader, device)
    print("\n=== Trajectory & Intent metrics ===")
    for k, v in traj_intent_results.items():
        print(f"  {k}: {v:.4f}")

    explanation_results = None
    if not args.skip_explanations:
        explanation_results = run_explanation_eval(
            per_clip_records, cfg["evaluate"]["human_explanations_path"], cfg
        )
        if explanation_results:
            print("\n=== Explanation metrics ===")
            for k, v in explanation_results.items():
                print(f"  {k}: {v}")

    os.makedirs(cfg["experiment"]["output_dir"], exist_ok=True)
    out_path = os.path.join(cfg["experiment"]["output_dir"], f"eval_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump({"trajectory_intent": traj_intent_results, "explanation": explanation_results}, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()