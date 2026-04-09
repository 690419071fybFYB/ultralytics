# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Adaptive feature enhancement modules."""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv import autopad

__all__ = ("AFEM",)


class _AFEMConv(nn.Module):
    """Conv-BN-(ReLU) block used by AFEM branches and fusion layers."""

    def __init__(self, c1: int, c2: int, k=1, s: int = 1, p=None, d: int = 1, act: bool = True):
        """Initialize a Conv-BN-(ReLU) block with explicit ReLU activation."""
        super().__init__()
        padding = autopad(k, p, d)
        if isinstance(padding, list):
            padding = tuple(padding)
        self.conv = nn.Conv2d(c1, c2, k, s, padding, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, batch normalization, and optional ReLU activation."""
        return self.act(self.bn(self.conv(x)))


class AdaptiveFeatureFusion(nn.Module):
    """Adaptive weighted fusion over concatenated AFEM branch features."""

    def __init__(self, channels: int):
        """Initialize the FC and AG subpaths used in AFEM fusion."""
        super().__init__()
        self.fc = _AFEMConv(channels, channels, k=1, act=True)
        self.ag = _AFEMConv(channels, channels, k=1, act=False)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhance concatenated features with a learned attention gate."""
        feat = self.fc(x)
        return feat * self.gate(self.ag(feat))


class AFEM(nn.Module):
    """Adaptive Feature Enhancement Module from LSDN for shallow feature refinement."""

    def __init__(self, c1: int, c2: int, dilation: int = 2, branch_channels: int | None = None):
        """Initialize AFEM with paper-aligned heterogeneous branches and adaptive fusion.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            dilation (int): Dilation rate for the contextual 3x3 branch tails.
            branch_channels (int | None): Width of each feature branch before fusion. Defaults to ``c2``.
        """
        super().__init__()
        if dilation < 1:
            raise ValueError(f"AFEM dilation must be >= 1, but got {dilation}.")
        if c2 < 1:
            raise ValueError(f"AFEM requires c2 >= 1, but got c2={c2}.")

        self.c1 = c1
        self.c2 = c2
        self.dilation = dilation
        self.branch_channels = c2 if branch_channels is None else branch_channels
        if self.branch_channels < 1:
            raise ValueError(f"AFEM branch_channels must be >= 1, but got {self.branch_channels}.")

        bc = self.branch_channels
        self.branch1 = nn.Sequential(
            _AFEMConv(c1, bc, k=1),
            _AFEMConv(bc, bc, k=3),
        )
        self.branch2 = nn.Sequential(
            _AFEMConv(c1, bc, k=1),
            _AFEMConv(bc, bc, k=(1, 3)),
            _AFEMConv(bc, bc, k=(3, 1)),
            _AFEMConv(bc, bc, k=3, d=dilation),
        )
        self.branch3 = nn.Sequential(
            _AFEMConv(c1, bc, k=1),
            _AFEMConv(bc, bc, k=(3, 1)),
            _AFEMConv(bc, bc, k=(1, 3)),
            _AFEMConv(bc, bc, k=3, d=dilation),
        )
        self.residual = _AFEMConv(c1, c2, k=1, act=False)
        self.fusion = AdaptiveFeatureFusion(bc * 3)
        self.project = _AFEMConv(bc * 3, c2, k=1, act=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multibranch enhancement, adaptive fusion, and residual refinement."""
        branches = (self.branch1(x), self.branch2(x), self.branch3(x))
        fused = self.project(self.fusion(torch.cat(branches, 1)))
        return self.act(fused + self.residual(x))
