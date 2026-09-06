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
        
def recommend_maneuver(ctx: ManeuverContext, cfg: dict) -> dict:
    e = cfg["explain"]
    recommend_decel = (
        ctx.intent_label == "crossing"
        and ctx.time_to_collision_s < e["decel_trigger_ttc_s"]
    )

    if recommend_decel:
        target_speed = max(0.0, ctx.ego_speed_mps - e["max_comfortable_decel_mps2"])
        yield_clearance = e["min_yield_clearance_m"]
        action = "decelerate_and_yield"
    else:
        target_speed = ctx.ego_speed_mps
        yield_clearance = 0.0
        action = "maintain_speed"

    return {
        "action": action,
        "target_speed_mps": round(target_speed, 2),
        "yield_clearance_m": yield_clearance,
    }
    

SYSTEM_PROMPT = (
    "You are the explanation module of an autonomous vehicle's pedestrian-safety "
    "system. You are given the perception system's numeric outputs (pedestrian "
    "crossing-intent probability, predicted trajectory, ego-vehicle state) and a "
    "maneuver that has already been decided by a separate, rule-based planner. "
    "Your ONLY job is to explain, in 2-3 short sentences, why that maneuver is "
    "appropriate given the numbers. Be concrete: cite the confidence percentage, "
    "approximate pedestrian speed/direction, and the target speed / clearance. "
    "Do not invent sensor readings you were not given. Do not change the "
    "recommended action."
)


def build_user_prompt(ctx: ManeuverContext, maneuver: dict) -> str:
    direction = "toward the curb" if ctx.lateral_velocity_mps > 0.15 else (
        "away from the curb" if ctx.lateral_velocity_mps < -0.15 else "roughly parallel to the road"
    )
    ped_speed = float(np.linalg.norm(
        np.array(ctx.pred_traj_xy[1]) - np.array(ctx.pred_traj_xy[0])
    ) / ctx.frame_dt_s) if len(ctx.pred_traj_xy) >= 2 else 0.0

    return (
        f"Intent: {ctx.intent_label} (confidence {ctx.intent_confidence * 100:.0f}%)\n"
        f"Predicted pedestrian motion: moving {direction} at ~{ped_speed:.1f} m/s, "
        f"currently ~{ctx.pedestrian_distance_m:.1f} m ahead.\n"
        f"Ego-vehicle speed: {ctx.ego_speed_mps:.1f} m/s "
        f"(time-to-reach-pedestrian ≈ {ctx.time_to_collision_s:.1f}s).\n"
        f"Planner decision: {maneuver['action']} "
        f"(target speed {maneuver['target_speed_mps']:.1f} m/s, "
        f"yield clearance {maneuver['yield_clearance_m']:.1f} m).\n\n"
        f"Explain this decision to a passenger in 2-3 sentences."
    )
    
def build_messages(ctx: ManeuverContext, maneuver: dict) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(ctx, maneuver)},
    ]