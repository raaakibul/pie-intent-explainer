from __future__ import annotations

from typing import List, Sequence

import numpy as np


def average_displacement_error(pred: np.ndarray, gt: np.ndarray) -> float:
    dist = np.linalg.norm(pred - gt, axis=-1)  # [N, T]
    return float(dist.mean())


def final_displacement_error(pred: np.ndarray, gt: np.ndarray) -> float:
    dist = np.linalg.norm(pred[:, -1, :] - gt[:, -1, :], axis=-1)  # [N]
    return float(dist.mean())

