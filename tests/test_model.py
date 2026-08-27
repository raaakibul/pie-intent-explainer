import numpy as np
import torch

from models.backbone import ResNetBackbone
from utils.coordinate_transform import bounding_box_sequence_to_ego_xy, normalize_trajectory
from models.temporal_encoder import TemporalEncoder

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

def test_temporal_encoder_output_shape():
    enc = TemporalEncoder(feature_dim=2048, bbox_feat_dim=32, hidden_dim=256,
                           num_layers=2, bidirectional=True)
    cnn_feats = torch.randn(B, T_OBS, 2048)
    bbox_feats = torch.randn(B, T_OBS, 4)
    seq_out, summary = enc(cnn_feats, bbox_feats)
    assert seq_out.shape == (B, T_OBS, 512)
    assert summary.shape == (B, 512)