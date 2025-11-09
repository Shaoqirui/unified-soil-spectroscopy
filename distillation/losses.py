from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch


def normalized_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2, dim=0)


def mixup(alpha: float, *tensors: torch.Tensor) -> Tuple[List[torch.Tensor], float]:
    if alpha <= 0.0:
        return list(tensors), 1.0
    lam = np.random.beta(alpha, alpha)
    batch = tensors[0].size(0)
    index = torch.randperm(batch, device=tensors[0].device)
    mixed = [lam * t + (1.0 - lam) * t[index] for t in tensors]
    return mixed, float(lam)


def blend_weight(epoch: int, total: int, alpha_start: float, alpha_end: float, turn: float) -> float:
    pivot = int(total * turn)
    if epoch <= pivot:
        return alpha_start
    ratio = (epoch - pivot) / max(1, total - pivot)
    return float(alpha_start - (alpha_start - alpha_end) * ratio)
