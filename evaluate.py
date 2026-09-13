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

