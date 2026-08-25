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