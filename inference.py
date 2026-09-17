from __future__ import annotations

import argparse
import json
import os

import torch
import yaml

from data.pie_dataset import PIEClipDataset
from models.iet_model import IETModel
from explain.llm_explainer import build_explainer
from utils.visualization import plot_trajectory


def find_record(processed_dir: str, clip_id: str, splits=("test", "val", "train")):
    for split in splits:
        path = os.path.join(processed_dir, f"{split}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            records = json.load(f)
        for i, rec in enumerate(records):
            if rec["clip_id"] == clip_id:
                return split, i
    raise ValueError(f"clip_id {clip_id!r} not found in any split under {processed_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--clip_id", type=str, required=True)
    parser.add_argument("--llm_backend", type=str, choices=["hf", "openai"], default=None,
                         help="Overrides configs/default.yaml -> explain.backend")
    parser.add_argument("--llm_model", type=str, default=None,
                         help="Overrides the model name for the chosen backend")
    parser.add_argument("--save_plot", type=str, default=None,
                         help="If set, save the trajectory visualization to this path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.llm_backend:
        cfg["explain"]["backend"] = args.llm_backend
    if args.llm_model:
        if cfg["explain"]["backend"] == "hf":
            cfg["explain"]["hf_model"] = args.llm_model
        else:
            cfg["explain"]["openai_model"] = args.llm_model

    device = cfg["experiment"]["device"] if torch.cuda.is_available() else "cpu"
    pie_image_root = os.path.join(cfg["data"]["pie_root"], "images")
    
    split, _ = find_record(cfg["data"]["processed_dir"], args.clip_id)
    split_json = os.path.join(cfg["data"]["processed_dir"], f"{split}.json")
    ds = PIEClipDataset(split_json, pie_image_root, cfg)
    clip_id_to_idx = {r["clip_id"]: i for i, r in enumerate(ds.records)}
    item = ds[clip_id_to_idx[args.clip_id]]

    model = IETModel(cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    batch = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else [v]) for k, v in item.items()}
    with torch.no_grad():
        outputs = model(batch, teacher_forcing=False)

    intent_probs = outputs["intent_probs"][0].cpu().numpy()
    intent_pred = int(intent_probs.argmax())
    intent_label = "crossing" if intent_pred == 1 else "not-crossing"
    pred_traj = outputs["pred_traj"][0].cpu().numpy()

    print(f"\nClip: {args.clip_id}")
    print(f"Predicted intent: {intent_label} ({intent_probs[intent_pred]*100:.1f}% confidence)")
    print(f"Predicted trajectory (ego-frame meters, first/last points): "
          f"{pred_traj[0].round(2).tolist()} -> {pred_traj[-1].round(2).tolist()}")
    
    print(f"\nLoading LLM explainer (backend={cfg['explain']['backend']})...")
    explainer = build_explainer(cfg)
    frame_dt_s = cfg["data"]["frame_stride"] / 30.0
    result = explainer.explain_from_model_output(
        intent_label=intent_label,
        intent_confidence=float(intent_probs[intent_pred]),
        pred_traj_xy=pred_traj.tolist(),
        ego_speed_mps=float(item["ego_speed"].item()),
        frame_dt_s=frame_dt_s,
        cfg=cfg,
    )

    print(f"\nManeuver: {result['maneuver']}")
    print(f"\nExplanation:\n  {result['explanation']}")

    if args.save_plot:
        gt_future = item["traj_future"].numpy()
        obs = item["traj_obs"].numpy()
        plot_trajectory(obs, pred_traj, gt_future,
                         title=f"{args.clip_id} — {intent_label}", save_path=args.save_plot)
        print(f"\nSaved trajectory plot to {args.save_plot}")


if __name__ == "__main__":
    main()