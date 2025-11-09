from __future__ import annotations

from typing import Dict, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataset import TASKS
from .fusion import MoESelector, TaskGate


class Norm1d(nn.Module):
    def __init__(self, channels: int, kind: str) -> None:
        super().__init__()
        kind = kind.lower()
        if kind == "bn":
            self.mod = nn.BatchNorm1d(channels)
        elif kind == "ln":
            self.mod = nn.LayerNorm(channels)
        elif kind == "gn":
            groups = 8 if channels % 8 == 0 else 1
            self.mod = nn.GroupNorm(num_groups=groups, num_channels=channels)
        else:
            raise ValueError("Unsupported norm kind.")
        self.kind = kind

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind in {"bn", "gn"}:
            return self.mod(x)
        if x.dim() == 3:
            x = self.mod(x.transpose(1, 2))
            return x.transpose(1, 2)
        return self.mod(x)


class GeMTopK(nn.Module):
    def __init__(self, p: float, local_kernel: int, mix: float) -> None:
        super().__init__()
        self.p = float(p)
        self.local_kernel = int(local_kernel)
        self.mix = float(mix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(2)
        k = min(32, max(1, length // 200))
        gem = torch.clamp(x, min=1e-6).pow(self.p).mean(dim=2).pow(1.0 / self.p)
        pad = self.local_kernel // 2
        pooled = F.max_pool1d(x, kernel_size=self.local_kernel, stride=1, padding=pad)
        topk, _ = torch.topk(pooled, k=k, dim=2)
        return self.mix * gem + (1.0 - self.mix) * topk.mean(dim=2)


def _conv1d(in_channels: int, out_channels: int, k: int, separable: bool) -> nn.Module:
    if not separable:
        return nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k // 2, bias=False)
    return nn.Sequential(
        nn.Conv1d(in_channels, in_channels, kernel_size=k, padding=k // 2, groups=in_channels, bias=False),
        nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
    )


class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, separable: bool, norm: str) -> None:
        super().__init__()
        branch_channels = max(8, out_channels // 4)

        def branch(kernel: int) -> nn.Module:
            return nn.Sequential(
                _conv1d(in_channels, branch_channels, kernel, separable=separable),
                nn.GELU(),
                Norm1d(branch_channels, norm),
            )

        self.block = nn.Sequential(
            nn.Conv1d(branch_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.GELU(),
            Norm1d(out_channels, norm),
        )
        self.branches = nn.ModuleList([branch(k) for k in (1, 3, 5, 7)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.block(feats)


def _make_kernel(k: int, kind: str) -> torch.Tensor:
    grid = torch.arange(k, dtype=torch.float32) - (k - 1) / 2
    sigma = k / 6.0
    if kind == "gauss":
        g = torch.exp(-(grid**2) / (2 * sigma**2))
        return (g / g.sum()).view(1, 1, k)
    if kind == "log":
        g = torch.exp(-(grid**2) / (2 * sigma**2))
        log = ((grid**2 - sigma**2) / (sigma**4)) * g
        log = log - log.mean()
        return (log / (log.abs().sum() + 1e-8)).view(1, 1, k)
    if kind == "diff":
        g = torch.exp(-(grid**2) / (2 * sigma**2))
        diff = (-grid / (sigma**2)) * g
        diff = diff - diff.mean()
        return (diff / (diff.abs().sum() + 1e-8)).view(1, 1, k)
    raise ValueError("Unknown kernel kind.")


class SpectralBranch(nn.Module):
    def __init__(
        self,
        stem_channels: int,
        widths: Sequence[int],
        separable: bool,
        cap: int,
        enable_peak: bool,
        norm: str,
        gem_p: float,
        gem_mix: float,
        peak_kset: Sequence[int],
    ) -> None:
        super().__init__()
        widths = tuple(min(width, cap) for width in widths)
        self.enable_peak = enable_peak
        self.stem = nn.Sequential(
            nn.Conv1d(1, stem_channels, kernel_size=9, stride=4, padding=4, bias=False),
            nn.GELU(),
            Norm1d(stem_channels, norm),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )
        self.inc1 = InceptionBlock(stem_channels, widths[0], separable=separable, norm=norm)
        self.inc2 = InceptionBlock(widths[0], widths[1], separable=separable, norm=norm)
        self.mid_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.inc3 = InceptionBlock(widths[1], widths[2], separable=separable, norm=norm)
        self.collector = GeMTopK(p=gem_p, local_kernel=9, mix=gem_mix)
        if self.enable_peak:
            k1, k2, k3 = peak_kset
            self.register_buffer("k_gauss", _make_kernel(k1, "gauss"))
            self.register_buffer("k_log", _make_kernel(k2, "log"))
            self.register_buffer("k_diff", _make_kernel(k3, "diff"))
            self.reducer = nn.Linear(widths[2] * 4, 256)
            self.post = nn.Sequential(nn.GELU(), nn.LayerNorm(256))
        else:
            self.head = nn.Sequential(nn.Linear(widths[2], 256), nn.GELU(), nn.LayerNorm(256))

    def _depthwise(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        channels = x.size(1)
        kernel = kernel.expand(channels, 1, kernel.size(-1))
        return F.conv1d(x, kernel, padding=kernel.size(-1) // 2, groups=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.inc1(x)
        x = self.inc2(x)
        x = self.mid_pool(x)
        x = self.inc3(x)
        core = self.collector(x)
        if not self.enable_peak:
            return self.head(core)
        g1 = self.collector(self._depthwise(x, self.k_gauss))
        g2 = self.collector(self._depthwise(x, self.k_log))
        g3 = self.collector(self._depthwise(x, self.k_diff))
        cat = torch.cat([core, g1, g2, g3], dim=1)
        return self.post(self.reducer(cat))


def build_branch(cfg: Dict) -> SpectralBranch:
    size = cfg["size"].lower()
    if size == "s":
        base = dict(stem_channels=48, widths=(96, 144, 192))
    elif size == "m":
        base = dict(stem_channels=64, widths=(128, 192, 256))
    else:
        raise ValueError("branch size must be 's' or 'm'")
    return SpectralBranch(
        stem_channels=base["stem_channels"],
        widths=base["widths"],
        separable=cfg.get("separable", False),
        cap=cfg.get("cap", max(base["widths"])),
        enable_peak=cfg.get("enable_peak", False),
        norm=cfg.get("norm", "bn"),
        gem_p=cfg.get("gem_p", 3.0),
        gem_mix=cfg.get("gem_mix", 0.5),
        peak_kset=cfg.get("peak_kset", (7, 11, 15)),
    )


class StudentNet(nn.Module):
    def __init__(
        self,
        branch_v_cfg: Dict,
        branch_x_cfg: Dict,
        branch_l_cfg: Dict,
        prior: Dict[str, Sequence[float]],
        moe_cfg: Dict[str, float],
    ) -> None:
        super().__init__()
        self.branch_v = build_branch(branch_v_cfg)
        self.branch_x = build_branch(branch_x_cfg)
        self.branch_l = build_branch(branch_l_cfg)
        self.gate = TaskGate(prior)
        self.moe = MoESelector(
            hidden=int(moe_cfg.get("hidden", 128)),
            activation=str(moe_cfg.get("act", "relu")),
            tau=float(moe_cfg.get("tau", 1.0)),
        )
        self.out_heads = nn.ModuleDict(
            {task: nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1)) for task in TASKS}
        )
        self.feature_heads = nn.ModuleDict(
            {
                "V": nn.ModuleDict({task: nn.Linear(256, 1) for task in TASKS}),
                "X": nn.ModuleDict({task: nn.Linear(256, 1) for task in TASKS}),
                "L": nn.ModuleDict({task: nn.Linear(256, 1) for task in TASKS}),
            }
        )
        self.log_sigma2 = nn.Parameter(torch.zeros(len(TASKS), dtype=torch.float32))

    def forward(
        self,
        x_vnir: torch.Tensor,
        x_xrf: torch.Tensor,
        x_libs: torch.Tensor,
        return_aux: bool = False,
    ):
        feats_v = self.branch_v(x_vnir)
        feats_x = self.branch_x(x_xrf)
        feats_l = self.branch_l(x_libs)
        gates = self.gate()
        batch_size = feats_v.size(0)
        stacked = torch.stack([feats_v, feats_x, feats_l], dim=1)
        fused = (stacked.unsqueeze(1) * gates.unsqueeze(0).unsqueeze(-1)).sum(dim=2)
        fused = self.moe(fused)
        outputs = torch.cat([self.out_heads[task](fused[:, idx, :]) for idx, task in enumerate(TASKS)], dim=1)
        if not return_aux:
            return outputs
        aux = {
            "V": torch.cat([self.feature_heads["V"][task](feats_v) for task in TASKS], dim=1),
            "X": torch.cat([self.feature_heads["X"][task](feats_x) for task in TASKS], dim=1),
            "L": torch.cat([self.feature_heads["L"][task](feats_l) for task in TASKS], dim=1),
        }
        return outputs, aux
