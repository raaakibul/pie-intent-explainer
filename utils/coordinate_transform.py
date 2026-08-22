from __future__ import annotations
import numpy as np

#1920×1080 resolution (HD video)
IMAGE_WIDTH_PX = 1920
IMAGE_HEIGHT_PX = 1080

#55° horizontal view angle (how wide the lens sees)

HORIZONTAL_FOV_DEG = 55.0

#1.5 m above ground (typical dashcam height)
CAMERA_HEIGHT_M = 1.5

#0° pitch (camera is level, not pointing up or down)
CAMERA_PITCH_DEG = 0.0


def focal_length_px(image_width_px: int = IMAGE_WIDTH_PX,
                      hfov_deg: float = HORIZONTAL_FOV_DEG) -> float:
    return (image_width_px / 2.0) / np.tan(np.radians(hfov_deg / 2.0))


_FX = focal_length_px()
_FY = _FX
_CX = IMAGE_WIDTH_PX / 2.0
_CY = IMAGE_HEIGHT_PX / 2.0

def bounding_box_to_feet_point(bounding_box_xyxy: np.ndarray) -> np.ndarray:
    x1,y1, x2, y2 = np.split(bounding_box_xyxy, 4, axis=-1)
    u = (x1 + x2) / 2.0
    v = y2
    return np.concatenate([u, v], axis=-1)

def pixel_to_ground_plane(uv: np.ndarray,
                           camera_height_m: float = CAMERA_HEIGHT_M,
                           pitch_deg: float = CAMERA_PITCH_DEG) -> np.ndarray:
    
    u, v = uv[..., 0], uv[..., 1]
    pitch = np.radians(pitch_deg)

    x_cam = (u - _CX) / _FX
    y_cam = (v - _CY) / _FY
    z_cam = np.ones_like(x_cam)

    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    y_rot = y_cam * cos_p - z_cam * sin_p
    z_rot = y_cam * sin_p + z_cam * cos_p

    eps = 1e-6
    t = -camera_height_m / np.clip(y_rot, eps, None)

    x_forward = t * z_rot
    y_lateral = -t * x_cam

    return np.stack([x_forward, y_lateral], axis=-1)

def bounding_box_sequence_to_ego_xy(bounding_box_seq_xyxy: np.ndarray) -> np.ndarray:
    feet = bounding_box_to_feet_point(bounding_box_seq_xyxy)
    return pixel_to_ground_plane(feet)


def normalize_trajectory(xy: np.ndarray) -> np.ndarray:
    return xy - xy[0:1]
