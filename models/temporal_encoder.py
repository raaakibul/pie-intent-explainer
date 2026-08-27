from __future__ import annotations

import torch
import torch.nn as nn

class BBoxEncoder(nn.Module):
    def __init__(self, in_dim: int = 4, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class TemporalEncoder(nn.Module):
    def __init__(self, feature_dim: int = 2048, bbox_feat_dim: int = 32,
                 hidden_dim: int = 256, num_layers: int = 2,
                 bidirectional: bool = True, dropout: float = 0.3):
        
        super().__init__()
        self.bbox_encoder = BBoxEncoder(out_dim=bbox_feat_dim)
        self.input_proj = nn.Linear(feature_dim + bbox_feat_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.num_directions = 2 if bidirectional else 1
        self.output_dim = hidden_dim * self.num_directions
        
    def forward(self, cnn_feats: torch.Tensor, bbox_feats: torch.Tensor):
        bbox_emb = self.bbox_encoder(bbox_feats)
        fused = torch.cat([cnn_feats, bbox_emb], dim=-1)
        fused = self.input_proj(fused)

        seq_out, (h_n, c_n) = self.lstm(fused)
        h_n = h_n.view(self.lstm.num_layers, self.num_directions, -1, self.lstm.hidden_size)
        last_layer = h_n[-1]
        summary = torch.cat([last_layer[d] for d in range(self.num_directions)], dim=-1)
        return seq_out, summary
    
