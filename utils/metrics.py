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
    
def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, num_bins: int = 10) -> float:
    confidences = y_prob.max(axis=-1)
    predictions = y_prob.argmax(axis=-1)
    accuracies = (predictions == y_true).astype(np.float32)

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        bin_count = in_bin.sum()
        if bin_count == 0:
            continue
        bin_acc = accuracies[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece += (bin_count / n) * abs(bin_acc - bin_conf)
    return float(ece)

def _tokenize(text: str) -> List[str]:
    return text.lower().strip().split()


def bleu4(candidates: Sequence[str], references: Sequence[str]) -> float:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    assert len(candidates) == len(references), "candidates/references length mismatch"
    list_of_references = [[_tokenize(r)] for r in references]
    hypotheses = [_tokenize(c) for c in candidates]
    smoothie = SmoothingFunction().method1
    return float(corpus_bleu(list_of_references, hypotheses,
                              weights=(0.25, 0.25, 0.25, 0.25),
                              smoothing_function=smoothie))
    
