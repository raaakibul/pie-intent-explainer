from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PIEClipDataset(Dataset):

    def __init__(self, split_json: str, pie_image_root: str, cfg: dict,
                 transform: Optional[transforms.Compose] = None):
        with open(split_json) as f:
            self.records: List[Dict] = json.load(f)
        self.pie_image_root = pie_image_root
        self.image_size = tuple(cfg["data"]["image_size"])
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.records)
    

    def _frame_path(self, rec: Dict, frame_idx: int) -> str:
        return os.path.join(self.pie_image_root, rec["set_id"], rec["video_id"],
                             f"{frame_idx:05d}.png")

    def _load_crop(self, rec: Dict, frame_idx: int, bbox: List[float]) -> np.ndarray:
        path = self._frame_path(rec, frame_idx)
        img = cv2.imread(path)
        if img is None:
            return np.full((*self.image_size, 3), 127, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return np.full((*self.image_size, 3), 127, dtype=np.uint8)
        return img[y1:y2, x1:x2]
    
    @staticmethod
    def _bbox_to_feat(bbox: List[float], img_w: float = 1920.0, img_h: float = 1080.0) -> List[float]:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = x2 - x1, y2 - y1
        return [cx / img_w, cy / img_h, w / img_w, h / img_h]
    
    def __getitem__(self, idx: int):
        rec = self.records[idx]

        frame_tensors = []
        bbox_feats = []
        for frame_idx, bbox in zip(rec["frames"], rec["bboxes_obs"]):
            crop = self._load_crop(rec, frame_idx, bbox)
            frame_tensors.append(self.transform(crop))
            bbox_feats.append(self._bbox_to_feat(bbox))

        frames = torch.stack(frame_tensors, dim=0)
        bbox_feats = torch.tensor(bbox_feats, dtype=torch.float32)
        traj_obs = torch.tensor(rec["traj_obs_xy"], dtype=torch.float32)
        traj_future = torch.tensor(rec["traj_future_xy"], dtype=torch.float32)
        intent_label = torch.tensor(rec["intent_label"], dtype=torch.long)
        ego_speed = torch.tensor(rec["ego_speed_kmh"] / 3.6, dtype=torch.float32)

        return {
            "frames": frames,
            "bbox_feats": bbox_feats,
            "traj_obs": traj_obs,
            "traj_future": traj_future,
            "intent_label": intent_label,
            "ego_speed": ego_speed,
            "clip_id": rec["clip_id"],
        }
        
def collate_fn(batch: List[Dict]) -> Dict:
    """Default collate works for everything except the string clip_id list."""
    clip_ids = [b.pop("clip_id") for b in batch]
    out = torch.utils.data.dataloader.default_collate(batch)
    out["clip_id"] = clip_ids
    return out