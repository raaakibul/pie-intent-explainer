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
    
