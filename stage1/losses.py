from __future__ import annotations

import torch
from torch import nn


class BCEWithLogitsLoss2D(nn.Module):
    def __init__(self, pos_weight: float | None = None) -> None:
        super().__init__()
        pw = None if pos_weight is None else torch.tensor([float(pos_weight)], dtype=torch.float32)
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits/target: (B, 1, H, W)
        return self.loss(logits, target)

