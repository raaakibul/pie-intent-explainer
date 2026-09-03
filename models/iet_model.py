from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from models.backbone import ResNetBackbone
from models.temporal_encoder import TemporalEncoder
from models.intent_head import IntentHead
from models.trajectory_decoder import TrajectoryDecoder


class IETModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]

        self.backbone = ResNetBackbone(
            pretrained=m["backbone_pretrained"],
            freeze_stages=m["freeze_backbone_stages"],
            feature_dim=m["feature_dim"],
        )
        self.temporal_encoder = TemporalEncoder(
            feature_dim=m["feature_dim"],
            bbox_feat_dim=m["bbox_feat_dim"],
            hidden_dim=m["lstm_hidden_dim"],
            num_layers=m["lstm_num_layers"],
            bidirectional=m["lstm_bidirectional"],
            dropout=m["dropout"],
        )
        self.intent_head = IntentHead(
            in_dim=self.temporal_encoder.output_dim,
            hidden_dim=m["intent_hidden_dim"],
            num_classes=m["intent_num_classes"],
            embed_dim=m["intent_embed_dim"],
            dropout=m["dropout"],
        )
        self.trajectory_decoder = TrajectoryDecoder(
            context_dim=self.temporal_encoder.output_dim,
            intent_embed_dim=m["intent_embed_dim"],
            hidden_dim=m["traj_decoder_hidden_dim"],
            num_layers=m["traj_decoder_num_layers"],
            pred_len=cfg["data"]["predict_frames"],
            dropout=m["dropout"],
        )
        
    def forward(self, batch: Dict[str, torch.Tensor], teacher_forcing: bool = True) -> Dict[str, torch.Tensor]:
        cnn_feats = self.backbone(batch["frames"])
        seq_out, summary = self.temporal_encoder(cnn_feats, batch["bbox_feats"])
        intent_logits, intent_embedding = self.intent_head(summary)

        last_obs_xy = batch["traj_obs"][:, -1, :]
        tf_target = batch.get("traj_future") if teacher_forcing else None
        pred_traj = self.trajectory_decoder(
            context=summary,
            intent_embedding=intent_embedding,
            last_obs_xy=last_obs_xy,
            teacher_forcing_target=tf_target,
        )

        return {
            "intent_logits": intent_logits,
            "intent_probs": torch.softmax(intent_logits, dim=-1),
            "pred_traj": pred_traj,
            "intent_embedding": intent_embedding,
            "temporal_summary": summary,
        }
        