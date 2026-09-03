import numpy as np
import torch

from models.backbone import ResNetBackbone
from utils.coordinate_transform import bounding_box_sequence_to_ego_xy, normalize_trajectory
from models.temporal_encoder import TemporalEncoder
from models.intent_head import IntentHead
from models.trajectory_decoder import TrajectoryDecoder

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
    

def test_intent_head_output_shape():
    head = IntentHead(in_dim=512, hidden_dim=128, num_classes=2, embed_dim=64)
    summary = torch.randn(B, 512)
    logits, embedding = head(summary)
    assert logits.shape == (B, 2)
    assert embedding.shape == (B, 64)
    
    
T_PRED = 15

def test_trajectory_decoder_output_shape_with_and_without_teacher_forcing():
    dec = TrajectoryDecoder(context_dim=512, intent_embed_dim=64, hidden_dim=256,
                             num_layers=1, pred_len=T_PRED)
    context = torch.randn(B, 512)
    intent_emb = torch.randn(B, 64)
    last_obs = torch.randn(B, 2)
    gt_future = torch.randn(B, T_PRED, 2)

    dec.train()
    out_tf = dec(context, intent_emb, last_obs, teacher_forcing_target=gt_future)
    assert out_tf.shape == (B, T_PRED, 2)

    dec.eval()
    out_no_tf = dec(context, intent_emb, last_obs, teacher_forcing_target=None)
    assert out_no_tf.shape == (B, T_PRED, 2)

def test_intent_conditioning_changes_trajectory():
    dec = TrajectoryDecoder(context_dim=64, intent_embed_dim=64, hidden_dim=64,
                             num_layers=1, pred_len=T_PRED)
    dec.eval()
    context = torch.randn(1, 64)
    last_obs = torch.zeros(1, 2)
    intent_a = torch.zeros(1, 64)
    intent_b = torch.ones(1, 64)

    traj_a = dec(context, intent_a, last_obs)
    traj_b = dec(context, intent_b, last_obs)
    assert not torch.allclose(traj_a, traj_b), "Trajectory should depend on intent embedding"