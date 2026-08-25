from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import yaml

from utils.coordinate_transform import bounding_box_sequence_to_ego_xy


def _load_pie_toolkit(pie_root: str):
    pie_repo = os.path.join(pie_root, "PIE")
    if pie_repo not in sys.path:
        sys.path.insert(0, pie_repo)
    try:
        from pie_data import PIE  # type: ignore
    except ImportError as e:
        raise ImportError(
            f"Could not import pie_data.PIE from {pie_repo}. "
            "Run `bash data/download_pie.sh <pie_root>` first."
        ) from e
    return PIE(data_path=pie_root)

def extract_pedestrian_tracks(pie, cfg: dict) -> List[Dict]:
    obs_len = cfg["data"]["observe_frames"]
    pred_len = cfg["data"]["predict_frames"]
    stride = cfg["data"]["frame_stride"]
    min_area = cfg["data"]["min_bbox_area"]
    window = obs_len + pred_len

    database = pie.generate_database()

    records: List[Dict] = []
    for set_id, videos in database.items():
        for video_id, vdata in videos.items():
            ped_annotations = vdata.get("ped_annotations", {})
            for ped_id, pdata in ped_annotations.items():
                frames = pdata["frames"]
                bboxes = pdata["bbox"]
                intent = pdata.get("attributes", {}).get("crossing", None)
                if intent is None or intent < 0:
                    continue

                sampled_idx = list(range(0, len(frames), stride))
                if len(sampled_idx) < window:
                    continue

                for start in range(0, len(sampled_idx) - window + 1, window):
                    idx = sampled_idx[start:start + window]
                    obs_idx, fut_idx = idx[:obs_len], idx[obs_len:]

                    obs_boxes = np.array([bboxes[i] for i in obs_idx], dtype=np.float32)
                    fut_boxes = np.array([bboxes[i] for i in fut_idx], dtype=np.float32)

                    areas = (obs_boxes[:, 2] - obs_boxes[:, 0]) * (obs_boxes[:, 3] - obs_boxes[:, 1])
                    if np.any(areas < min_area):
                        continue

                    records.append({
                        "clip_id": f"{set_id}_{video_id}_{ped_id}_{start}",
                        "set_id": set_id,
                        "video_id": video_id,
                        "ped_id": str(ped_id),
                        "frames": [frames[i] for i in obs_idx],
                        "bboxes_obs": obs_boxes.tolist(),
                        "bboxes_future": fut_boxes.tolist(),
                        "intent_label": int(intent),
                    })
    return records


def split_by_set(records: List[Dict], cfg: dict) -> Dict[str, List[Dict]]:
    splits = {"train": cfg["data"]["train_split"],
              "val": cfg["data"]["val_split"],
              "test": cfg["data"]["test_split"]}
    out = {k: [] for k in splits}
    for rec in records:
        for split_name, set_ids in splits.items():
            if rec["set_id"] in set_ids:
                out[split_name].append(rec)
                break
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rng = np.random.default_rng(cfg["experiment"]["seed"])

    pie = _load_pie_toolkit(cfg["data"]["pie_root"])
    print("Extracting fixed-length pedestrian track windows from PIE annotations...")
    records = extract_pedestrian_tracks(pie, cfg)
    print(f"  -> {len(records)} raw windows extracted.")

    records = augment_with_ego_speed_and_metric_traj(records, cfg, rng)

    splits = split_by_set(records, cfg)
    os.makedirs(cfg["data"]["processed_dir"], exist_ok=True)
    for split_name, split_records in splits.items():
        out_path = os.path.join(cfg["data"]["processed_dir"], f"{split_name}.json")
        with open(out_path, "w") as f:
            json.dump(split_records, f)
        print(f"  {split_name}: {len(split_records)} clips -> {out_path}")


if __name__ == "__main__":
    main()
    

