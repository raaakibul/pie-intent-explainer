from __future__ import annotations
from sklearn.metrics import accuracy_score, f1_score
from typing import List, Sequence

import numpy as np


def average_displacement_error(pred: np.ndarray, gt: np.ndarray) -> float:
    dist = np.linalg.norm(pred - gt, axis=-1)  # [N, T]
    return float(dist.mean())


def final_displacement_error(pred: np.ndarray, gt: np.ndarray) -> float:
    dist = np.linalg.norm(pred[:, -1, :] - gt[:, -1, :], axis=-1)  # [N]
    return float(dist.mean())


def intent_accuracy_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
    }