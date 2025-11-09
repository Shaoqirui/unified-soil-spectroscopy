from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataset import TASKS


def _prior_logits(prior: Dict[str, Sequence[float]]) -> torch.Tensor:
    arr = np.log(np.asarray([prior[t] for t in TASKS], dtype=np.float32))
    return torch.tensor(arr, dtype=torch.float32)


class TaskGate(nn.Module):
    def __init__(self, prior: Dict[str, Sequence[float]]) -> None:
        super().__init__()
        self.logits = nn.Parameter(_prior_logits(prior))

    def forward(self) -> torch.Tensor:
        return F.softmax(self.logits, dim=1)


class MoESelector(nn.Module):
    def __init__(self, hidden: int, activation: str, tau: float) -> None:
        super().__init__()
        act = nn.GELU() if activation.lower() == "gelu" else nn.ReLU()
        tasks = len(TASKS)
        self.logits = nn.Parameter(torch.zeros(tasks, 2, dtype=torch.float32))
        self.branch_a = nn.Sequential(nn.Linear(256, hidden), act, nn.Linear(hidden, 256))
        self.branch_b = nn.Sequential(nn.Linear(256, hidden), act, nn.Linear(hidden, 256))
        self.tau = float(tau)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        tasks = fused.size(1)
        weight = F.softmax(self.logits / self.tau, dim=1)[:, 0].view(1, tasks, 1)
        a = self.branch_a(fused.reshape(-1, 256)).reshape_as(fused)
        b = self.branch_b(fused.reshape(-1, 256)).reshape_as(fused)
        return weight * a + (1.0 - weight) * b
