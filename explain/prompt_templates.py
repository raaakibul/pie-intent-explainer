from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class ManeuverContext:
    intent_label: str
    intent_confidence: float
    pred_traj_xy: List[Tuple[float, float]]
    ego_speed_mps: float
    frame_dt_s: float
    pedestrian_distance_m: float = field(init=False)
    lateral_velocity_mps: float = field(init=False)
    time_to_collision_s: float = field(init=False)

    def __post_init__(self):
        traj = np.array(self.pred_traj_xy, dtype=np.float32)
        self.pedestrian_distance_m = float(np.linalg.norm(traj[0]))
        if len(traj) >= 2:
            disp = traj[1] - traj[0]
            self.lateral_velocity_mps = float(disp[1] / self.frame_dt_s)
        else:
            self.lateral_velocity_mps = 0.0
        forward_dist = max(traj[0][0], 0.1)
        self.time_to_collision_s = float(forward_dist / max(self.ego_speed_mps, 0.1))
        
