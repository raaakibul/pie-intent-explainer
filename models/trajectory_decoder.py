from __future__ import annotations

import torch
import torch.nn as nn

class TrajectoryDecoder(nn.Module):
    def __init__(self, context_dim: int, intent_embed_dim: int,
                 hidden_dim: int = 256, num_layers: int = 1,
                 pred_len: int = 15, dropout: float = 0.3):
        super().__init__()
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        cond_dim = context_dim + intent_embed_dim
        self.init_hidden_proj = nn.Linear(cond_dim, hidden_dim * num_layers)
        self.init_cell_proj = nn.Linear(cond_dim, hidden_dim * num_layers)

        self.cell = nn.LSTM(
            input_size=2 + cond_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2),
        )
        