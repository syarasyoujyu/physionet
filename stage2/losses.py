from __future__ import annotations

import torch
from torch import nn


class IoULoss(nn.Module):
    """
    Differentiable IoU loss for binary segmentation.
    logits/target: (B, 1, H, W), target in {0,1}.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        t = target
        inter = (p * t).sum(dim=(1, 2, 3))
        union = (p + t - p * t).sum(dim=(1, 2, 3))
        iou = (inter + self.eps) / (union + self.eps)
        return 1.0 - iou.mean()

