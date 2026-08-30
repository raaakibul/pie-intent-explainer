from __future__ import annotations

import torch
import torch.nn as nn


class IntentHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_classes: int = 2,
                 embed_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.embed_proj = nn.Linear(num_classes, embed_dim)

    def forward(self, summary: torch.Tensor):
        logits = self.classifier(summary)
        probs = torch.softmax(logits, dim=-1)
        intent_embedding = self.embed_proj(probs)
        return logits, intent_embedding