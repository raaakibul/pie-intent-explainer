import numpy as np
import torch

from models.backbone import ResNetBackbone
from utils.coordinate_transform import bounding_box_sequence_to_ego_xy, normalize_trajectory

T_OBS = 10

def test_coordinate_transform_shapes():
    bounding_boxes = np.array([[900, 500, 1000, 700]] * T_OBS, dtype=np.float32)
    xy = bounding_box_sequence_to_ego_xy(bounding_boxes)
    assert xy.shape == (T_OBS, 2)
    normed = normalize_trajectory(xy)
    assert np.allclose(normed[0], 0.0)

B, T_OBS = 2, 10


def test_backbone_output_shape():
    backbone = ResNetBackbone(pretrained=False, freeze_stages=0, feature_dim=2048)
    x = torch.randn(B, T_OBS, 3, 224, 224)
    out = backbone(x)
    assert out.shape == (B, T_OBS, 2048)