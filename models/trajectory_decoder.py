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
        
    def forward(self, context: torch.Tensor, intent_embedding: torch.Tensor,
                last_obs_xy: torch.Tensor, teacher_forcing_target: torch.Tensor = None):
        
        b = context.shape[0]
        cond = torch.cat([context, intent_embedding], dim=-1)

        h0 = self.init_hidden_proj(cond).view(b, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        c0 = self.init_cell_proj(cond).view(b, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()

        hidden = (h0, c0)
        cur_pos = last_obs_xy
        prev_disp = torch.zeros_like(last_obs_xy)

        outputs = []
        for t in range(self.pred_len):
            step_input = torch.cat([prev_disp, cond], dim=-1).unsqueeze(1)
            out, hidden = self.cell(step_input, hidden)
            disp = self.output_proj(out.squeeze(1))
            cur_pos = cur_pos + disp
            outputs.append(cur_pos)

            if teacher_forcing_target is not None and self.training:
                gt_pos = teacher_forcing_target[:, t, :]
                prev_disp = gt_pos - (cur_pos - disp)
                cur_pos = gt_pos
            else:
                prev_disp = disp

        return torch.stack(outputs, dim=1)
        