from __future__ import annotations

import torch


@torch.no_grad()
def iou_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    pred = (torch.sigmoid(logits) > threshold).to(target.dtype)
    inter = (pred * target).sum().item()
    union = (pred + target - pred * target).sum().item()
    return float((inter + eps) / (union + eps))

