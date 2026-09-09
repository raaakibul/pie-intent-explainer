from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(obs_xy: np.ndarray, pred_xy: np.ndarray,
                     gt_future_xy: Optional[np.ndarray] = None,
                     title: str = "", save_path: Optional[str] = None):
    """
    obs_xy:       [T_obs, 2]   observed ego-frame trajectory (meters)
    pred_xy:      [T_pred, 2]  predicted future trajectory
    gt_future_xy: [T_pred, 2]  ground-truth future trajectory (optional)
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(obs_xy[:, 1], obs_xy[:, 0], "o-", color="tab:blue", label="observed")
    ax.plot(pred_xy[:, 1], pred_xy[:, 0], "o--", color="tab:red", label="predicted")
    if gt_future_xy is not None:
        ax.plot(gt_future_xy[:, 1], gt_future_xy[:, 0], "o--", color="tab:green",
                 label="ground truth", alpha=0.7)

    ax.plot([obs_xy[-1, 1], pred_xy[0, 1]], [obs_xy[-1, 0], pred_xy[0, 0]],
             "--", color="tab:red", alpha=0.5)

    ax.scatter([0], [0], marker="^", s=120, color="black", label="ego vehicle")
    ax.set_xlabel("lateral y (m, + = left)")
    ax.set_ylabel("forward x (m)")
    ax.set_title(title)
    ax.legend()
    ax.axis("equal")
    ax.invert_xaxis()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return fig