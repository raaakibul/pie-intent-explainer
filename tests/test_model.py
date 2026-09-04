import numpy as np
import torch

from models.backbone import ResNetBackbone
from utils.coordinate_transform import bounding_box_sequence_to_ego_xy, normalize_trajectory
from models.temporal_encoder import TemporalEncoder
from models.intent_head import IntentHead
from models.trajectory_decoder import TrajectoryDecoder

from utils import metrics as M

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
    
import yaml

from models.iet_model import IETModel
import pytest

@pytest.fixture(scope="module")
def cfg():
    with open("configs/default.yaml") as f:
        c = yaml.safe_load(f)
    c["model"]["backbone_pretrained"] = False
    return c


def test_full_model_forward(cfg):
    model = IETModel(cfg)
    model.eval()
    batch = {
        "frames": torch.randn(B, T_OBS, 3, *cfg["data"]["image_size"]),
        "bbox_feats": torch.randn(B, T_OBS, 4),
        "traj_obs": torch.randn(B, T_OBS, 2),
        "traj_future": torch.randn(B, T_PRED, 2),
        "intent_label": torch.randint(0, 2, (B,)),
        "ego_speed": torch.rand(B) * 8.0,
    }
    with torch.no_grad():
        out = model(batch, teacher_forcing=False)

    assert out["intent_logits"].shape == (B, 2)
    assert out["pred_traj"].shape == (B, T_PRED, 2)
    assert torch.allclose(out["intent_probs"].sum(dim=-1), torch.ones(B), atol=1e-4)
    

def test_ade_fde_zero_for_perfect_prediction():
    pred = np.random.randn(4, T_PRED, 2).astype(np.float32)
    gt = pred.copy()
    assert M.average_displacement_error(pred, gt) == pytest.approx(0.0, abs=1e-6)
    assert M.final_displacement_error(pred, gt) == pytest.approx(0.0, abs=1e-6)


def test_ade_matches_manual_computation():
    pred = np.array([[[0.0, 0.0], [1.0, 0.0]]], dtype=np.float32)
    gt = np.array([[[0.0, 0.0], [0.0, 0.0]]], dtype=np.float32)
    assert M.average_displacement_error(pred, gt) == pytest.approx(0.5, abs=1e-6)
    assert M.final_displacement_error(pred, gt) == pytest.approx(1.0, abs=1e-6)
    
def test_ece_zero_when_confidence_matches_accuracy():
    rng = np.random.default_rng(0)
    n = 1000
    y_true = (rng.random(n) < 0.9).astype(int)
    y_prob = np.zeros((n, 2), dtype=np.float32)
    y_prob[:, 1] = 0.9
    y_prob[:, 0] = 0.1
    ece = M.expected_calibration_error(y_true, y_prob, num_bins=10)
    assert ece < 0.05