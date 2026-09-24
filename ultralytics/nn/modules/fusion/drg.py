"""Depth reliability gated fusion for RGB + pseudo-depth features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DepthReliabilityBase(nn.Module):
    """Shared utilities for reliability-aware RGB-depth fusion blocks."""

    def __init__(
        self,
        dim: int | None = None,
        reduction: int = 8,
        use_spatial: bool = True,
        residual: bool = True,
        init_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.reduction = max(int(reduction), 1)
        self.use_spatial = bool(use_spatial)
        self.residual = bool(residual)
        self.init_scale = float(init_scale)
        self._built = False
        self._c: int | None = None
        self.channel_gate: nn.Module | None = None
        self.spatial_gate: nn.Module | None = None
        self.depth_scale: nn.Parameter | None = None

    @staticmethod
    def _gradient_energy(x: torch.Tensor) -> torch.Tensor:
        gx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        gy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        gx = F.pad(gx.mean(1, keepdim=True), (0, 1, 0, 0))
        gy = F.pad(gy.mean(1, keepdim=True), (0, 0, 0, 1))
        return gx + gy

    @staticmethod
    def _local_variance(x: torch.Tensor) -> torch.Tensor:
        x_mean = x.mean(1, keepdim=True)
        avg = F.avg_pool2d(x_mean, kernel_size=3, stride=1, padding=1)
        avg_sq = F.avg_pool2d(x_mean * x_mean, kernel_size=3, stride=1, padding=1)
        return (avg_sq - avg * avg).clamp_min(0.0).sqrt()

    @staticmethod
    def _validate_inputs(x_rgb, x_depth, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        if x_depth is None and isinstance(x_rgb, (list, tuple)):
            x_rgb, x_depth = x_rgb
        if not isinstance(x_rgb, torch.Tensor) or not isinstance(x_depth, torch.Tensor):
            raise TypeError(f"{name} expects two tensor inputs")
        if x_rgb.shape != x_depth.shape:
            raise ValueError(f"{name} requires equal shapes, got {x_rgb.shape} vs {x_depth.shape}")
        return x_rgb, x_depth


class DepthReliabilityGate(_DepthReliabilityBase):
    """Fuse RGB and pseudo-depth features with a learned depth reliability gate."""

    def _build_if_needed(self, c: int, device: torch.device) -> None:
        if self._built and self._c == c:
            return
        hidden = max(c // self.reduction, 4)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c * 3, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c, 1, bias=True),
            nn.Sigmoid(),
        )
        if self.use_spatial:
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(4, 1, 3, padding=1, bias=True),
                nn.Sigmoid(),
            )
        else:
            self.spatial_gate = None
        self.depth_scale = nn.Parameter(torch.tensor(self.init_scale, dtype=torch.float32, device=device))
        self.to(device=device)
        self._built = True
        self._c = c

    def _compute_gate(self, x_rgb: torch.Tensor, x_depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, c, _, _ = x_rgb.shape
        self._build_if_needed(c, x_rgb.device)
        assert self.channel_gate is not None and self.depth_scale is not None

        diff = torch.abs(x_rgb - x_depth)
        gate = self.channel_gate(torch.cat([x_rgb, x_depth, diff], dim=1))

        if self.spatial_gate is not None:
            rgb_mean = x_rgb.mean(1, keepdim=True)
            depth_mean = x_depth.mean(1, keepdim=True)
            diff_mean = diff.mean(1, keepdim=True)
            depth_grad = self._gradient_energy(x_depth)
            gate = gate * self.spatial_gate(torch.cat([rgb_mean, depth_mean, diff_mean, depth_grad], dim=1))
        return gate, diff

    def forward(self, x_rgb, x_depth=None) -> torch.Tensor:
        x_rgb, x_depth = self._validate_inputs(x_rgb, x_depth, "DepthReliabilityGate")
        gate, _ = self._compute_gate(x_rgb, x_depth)
        assert self.depth_scale is not None
        scale = torch.clamp(self.depth_scale, 0.0, 1.0)
        if self.residual:
            return x_rgb + scale * gate * x_depth
        return x_rgb * (1.0 - gate) + x_depth * gate


class DepthReliabilityConcatGate(DepthReliabilityGate):
    """Gate pseudo-depth reliability, then concatenate RGB and gated depth."""

    def forward(self, x_rgb, x_depth=None) -> torch.Tensor:
        x_rgb, x_depth = self._validate_inputs(x_rgb, x_depth, "DepthReliabilityConcatGate")
        gate, _ = self._compute_gate(x_rgb, x_depth)
        assert self.depth_scale is not None
        gated_depth = torch.clamp(self.depth_scale, 0.0, 1.0) * gate * x_depth
        return torch.cat([x_rgb, gated_depth], dim=1)


class DepthConcatGateNoDiff(_DepthReliabilityBase):
    """Concat-preserving RGB-depth gate that does not use an absolute difference term."""

    def _build_if_needed(self, c: int, device: torch.device) -> None:
        if self._built and self._c == c:
            return
        hidden = max(c // self.reduction, 4)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c * 2, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c, 1, bias=True),
            nn.Sigmoid(),
        )
        if self.use_spatial:
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(3, 1, 3, padding=1, bias=True),
                nn.Sigmoid(),
            )
        else:
            self.spatial_gate = None
        self.depth_scale = nn.Parameter(torch.tensor(self.init_scale, dtype=torch.float32, device=device))
        self.to(device=device)
        self._built = True
        self._c = c

    def _compute_gate(self, x_rgb: torch.Tensor, x_depth: torch.Tensor) -> torch.Tensor:
        _, c, _, _ = x_rgb.shape
        self._build_if_needed(c, x_rgb.device)
        assert self.channel_gate is not None and self.depth_scale is not None

        gate = self.channel_gate(torch.cat([x_rgb, x_depth], dim=1))
        if self.spatial_gate is not None:
            rgb_mean = x_rgb.mean(1, keepdim=True)
            depth_mean = x_depth.mean(1, keepdim=True)
            depth_grad = self._gradient_energy(x_depth)
            gate = gate * self.spatial_gate(torch.cat([rgb_mean, depth_mean, depth_grad], dim=1))
        return gate

    def forward(self, x_rgb, x_depth=None) -> torch.Tensor:
        x_rgb, x_depth = self._validate_inputs(x_rgb, x_depth, "DepthConcatGateNoDiff")
        gate = self._compute_gate(x_rgb, x_depth)
        assert self.depth_scale is not None
        gated_depth = torch.clamp(self.depth_scale, 0.0, 1.0) * gate * x_depth
        return torch.cat([x_rgb, gated_depth], dim=1)


class DepthReliabilityGateV2(_DepthReliabilityBase):
    """Depth reliability fusion with local statistics and RGB-conditioned proxy depth."""

    def __init__(
        self,
        dim: int | None = None,
        reduction: int = 8,
        use_spatial: bool = True,
        residual: bool = True,
        init_scale: float = 0.7,
        min_depth_keep: float = 0.1,
        use_proxy: bool = True,
    ) -> None:
        super().__init__(dim=dim, reduction=reduction, use_spatial=use_spatial, residual=residual, init_scale=init_scale)
        self.min_depth_keep = float(min_depth_keep)
        self.use_proxy = bool(use_proxy)
        self.global_gate: nn.Module | None = None
        self.depth_proxy: nn.Module | None = None

    def _build_if_needed(self, c: int, device: torch.device) -> None:
        if self._built and self._c == c:
            return
        hidden = max(c // self.reduction, 8)
        spatial_hidden = max(c // (self.reduction * 2), 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c * 4, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c, 1, bias=True),
        )
        if self.use_spatial:
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(6, spatial_hidden, 3, padding=1, bias=True),
                nn.SiLU(inplace=True),
                nn.Conv2d(spatial_hidden, 1, 1, bias=True),
            )
        else:
            self.spatial_gate = None
        self.global_gate = nn.Sequential(
            nn.Conv2d(6, spatial_hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(spatial_hidden, 1, 1, bias=True),
        )
        if self.use_proxy:
            self.depth_proxy = nn.Sequential(
                nn.Conv2d(c, c, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.SiLU(inplace=True),
            )
        else:
            self.depth_proxy = None
        self.depth_scale = nn.Parameter(torch.tensor(self.init_scale, dtype=torch.float32, device=device))
        self.to(device=device)
        self._built = True
        self._c = c

    def _estimate_reliability(self, x_rgb: torch.Tensor, x_depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, c, _, _ = x_rgb.shape
        self._build_if_needed(c, x_rgb.device)
        assert self.channel_gate is not None and self.global_gate is not None and self.depth_scale is not None

        diff = torch.abs(x_rgb - x_depth)
        prod = x_rgb * x_depth
        stats = torch.cat(
            [
                x_rgb.mean(1, keepdim=True),
                x_depth.mean(1, keepdim=True),
                diff.mean(1, keepdim=True),
                prod.mean(1, keepdim=True),
                self._gradient_energy(x_depth),
                self._local_variance(x_depth),
            ],
            dim=1,
        )
        channel_gate = torch.sigmoid(self.channel_gate(torch.cat([x_rgb, x_depth, diff, prod], dim=1)))
        global_gate = torch.sigmoid(self.global_gate(F.adaptive_avg_pool2d(stats, 1)))
        reliability = channel_gate * global_gate
        if self.spatial_gate is not None:
            reliability = reliability * torch.sigmoid(self.spatial_gate(stats))
        if self.min_depth_keep > 0.0:
            reliability = self.min_depth_keep + (1.0 - self.min_depth_keep) * reliability
        return reliability, diff

    def _refined_depth(self, x_rgb: torch.Tensor, x_depth: torch.Tensor) -> torch.Tensor:
        reliability, _ = self._estimate_reliability(x_rgb, x_depth)
        proxy = x_rgb if self.depth_proxy is None else self.depth_proxy(x_rgb)
        return reliability * x_depth + (1.0 - reliability) * proxy

    def forward(self, x_rgb, x_depth=None) -> torch.Tensor:
        x_rgb, x_depth = self._validate_inputs(x_rgb, x_depth, "DepthReliabilityGateV2")
        refined_depth = self._refined_depth(x_rgb, x_depth)
        assert self.depth_scale is not None
        scale = torch.clamp(self.depth_scale, 0.0, 1.5)
        if self.residual:
            return x_rgb + scale * refined_depth
        return x_rgb * (1.0 - scale) + scale * refined_depth


class DepthReliabilityConcatGateV2(DepthReliabilityGateV2):
    """Concat-preserving DRGv2 for drop-in replacement of RGB/X concatenation."""

    def forward(self, x_rgb, x_depth=None) -> torch.Tensor:
        x_rgb, x_depth = self._validate_inputs(x_rgb, x_depth, "DepthReliabilityConcatGateV2")
        refined_depth = self._refined_depth(x_rgb, x_depth)
        assert self.depth_scale is not None
        gated_depth = torch.clamp(self.depth_scale, 0.0, 1.5) * refined_depth
        return torch.cat([x_rgb, gated_depth], dim=1)


__all__ = (
    "DepthReliabilityGate",
    "DepthReliabilityConcatGate",
    "DepthReliabilityGateV2",
    "DepthReliabilityConcatGateV2",
    "DepthConcatGateNoDiff",
)
