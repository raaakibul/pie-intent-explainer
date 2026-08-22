import numpy as np
from utils.coordinate_transform import bounding_box_sequence_to_ego_xy, normalize_trajectory

T_OBS = 10

def test_coordinate_transform_shapes():
    bounding_boxes = np.array([[900, 500, 1000, 700]] * T_OBS, dtype=np.float32)
    xy = bounding_box_sequence_to_ego_xy(bounding_boxes)
    assert xy.shape == (T_OBS, 2)
    normed = normalize_trajectory(xy)
    assert np.allclose(normed[0], 0.0)