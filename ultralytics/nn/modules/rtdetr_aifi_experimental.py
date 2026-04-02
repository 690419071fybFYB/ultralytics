# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import AIFI, TransformerEncoderLayer


class SparseGatedAIFI(TransformerEncoderLayer):
    """Sparse-gated AIFI with local/global attention branches and token-wise gating."""

    def __init__(
        self,
        c1: int,
        cm: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.0,
        act: nn.Module = nn.GELU(),
        normalize_before: bool = False,
        local_window: int = 3,
        global_pool: int = 2,
        gate_hidden_ratio: float = 0.25,
        enable_local: bool = True,
        enable_global: bool = True,
        enable_gate: bool = True,
    ):
        """Initialize sparse-gated AIFI.

        Args:
            c1 (int): Input dimension.
            cm (int): Hidden dimension in the feedforward network.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
            act (nn.Module): Activation function.
            normalize_before (bool): Whether to apply normalization before attention and feedforward.
            local_window (int): Odd local attention window size.
            global_pool (int): Spatial downsampling factor for global tokens.
            gate_hidden_ratio (float): Hidden ratio for the gate MLP.
            enable_local (bool): Whether to enable the local attention branch.
            enable_global (bool): Whether to enable the pooled-global attention branch.
            enable_gate (bool): Whether to predict token-wise local/global fusion weights.
        """
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)
        if not enable_local and not enable_global:
            raise ValueError("SparseGatedAIFI requires at least one attention branch to be enabled.")
        if local_window < 1 or local_window % 2 == 0:
            raise ValueError("local_window must be a positive odd integer.")
        if global_pool < 1:
            raise ValueError("global_pool must be >= 1.")
        if gate_hidden_ratio <= 0:
            raise ValueError("gate_hidden_ratio must be > 0.")

        self.local_window = local_window
        self.global_pool = global_pool
        self.gate_hidden_ratio = gate_hidden_ratio
        self.enable_local = enable_local
        self.enable_global = enable_global
        # 只有同时启用 local/global 两条分支时，gate 才有实际意义。
        self.enable_gate = enable_gate and enable_local and enable_global

        # 两个注意力分支各自独立：local 负责邻域细节，global 负责压缩后的场景级上下文。
        self.local_ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        self.global_ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)

        gate_hidden = max(1, int(round(c1 * gate_hidden_ratio)))
        # 轻量 gate 网络，为每个 token 预测 local/global 的融合系数。
        self.gate = nn.Sequential(
            nn.Linear(c1, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def build_local_attention_mask(h: int, w: int, window: int, device: torch.device) -> torch.Tensor:
        """Build a square local-neighborhood attention mask for HxW tokens."""
        radius = window // 2
        yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        coords = torch.stack((yy.reshape(-1), xx.reshape(-1)), dim=1)
        delta = (coords[:, None, :] - coords[None, :, :]).abs()
        within = (delta[:, :, 0] <= radius) & (delta[:, :, 1] <= radius)
        # MultiheadAttention 的 attn_mask 里 True 表示“不允许注意到这个位置”。
        return ~within

    def _pool_global_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Average-pool 2D features into compact global tokens and their positional embeddings."""
        _, c, h, w = x.shape
        gh = max(1, math.ceil(h / self.global_pool))
        gw = max(1, math.ceil(w / self.global_pool))
        # 先把高层特征压成更少的全局 token，避免恢复成 full attention 的高开销。
        pooled = F.adaptive_avg_pool2d(x, (gh, gw))
        tokens = pooled.flatten(2).permute(0, 2, 1)
        pos = AIFI.build_2d_sincos_position_embedding(gw, gh, c).to(device=x.device, dtype=x.dtype)
        return tokens, pos

    def _compute_gate(self, tokens: torch.Tensor) -> torch.Tensor:
        """Predict a token-wise local/global fusion weight in [0, 1]."""
        return self.gate(tokens)

    def _compute_attention(
        self,
        tokens: torch.Tensor,
        x2d: torch.Tensor,
        pos: torch.Tensor,
        h: int,
        w: int,
    ) -> torch.Tensor:
        """Compute sparse-gated attention from local/global branches."""
        q = self.with_pos_embed(tokens, pos)
        local_out = global_out = None

        if self.enable_local:
            # local 分支只在固定二维邻域内做注意力，更偏向小目标和密集局部结构。
            local_mask = self.build_local_attention_mask(h, w, self.local_window, tokens.device)
            local_out = self.local_ma(q, q, value=tokens, attn_mask=local_mask)[0]

        if self.enable_global:
            # global 分支不再看全部 token，而是看压缩后的全局 token，保留场景语义同时控制复杂度。
            global_tokens, global_pos = self._pool_global_tokens(x2d)
            global_k = self.with_pos_embed(global_tokens, global_pos)
            global_out = self.global_ma(q, global_k, value=global_tokens)[0]

        if local_out is None:
            return global_out
        if global_out is None:
            return local_out
        if self.enable_gate:
            gate = self._compute_gate(tokens)
            # gate 越接近 1，越偏向 local；越接近 0，越偏向 global。
            return gate * local_out + (1.0 - gate) * global_out
        # 不启用 gate 时退化为固定 0.5 融合，便于做消融。
        return 0.5 * (local_out + global_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for sparse-gated AIFI on [B, C, H, W] inputs."""
        b, c, h, w = x.shape
        pos = AIFI.build_2d_sincos_position_embedding(w, h, c).to(device=x.device, dtype=x.dtype)
        tokens = x.flatten(2).permute(0, 2, 1)

        if self.normalize_before:
            norm_tokens = self.norm1(tokens)
            norm_x = norm_tokens.permute(0, 2, 1).view(b, c, h, w).contiguous()
            attn_out = self._compute_attention(norm_tokens, norm_x, pos, h, w)
            tokens = tokens + self.dropout1(attn_out)
            # 注意力结构变了，但 FFN/残差/归一化仍沿用标准 TransformerEncoderLayer。
            ff_tokens = self.norm2(tokens)
            ff_tokens = self.fc2(self.dropout(self.act(self.fc1(ff_tokens))))
            tokens = tokens + self.dropout2(ff_tokens)
        else:
            attn_out = self._compute_attention(tokens, x, pos, h, w)
            tokens = tokens + self.dropout1(attn_out)
            tokens = self.norm1(tokens)
            ff_tokens = self.fc2(self.dropout(self.act(self.fc1(tokens))))
            tokens = tokens + self.dropout2(ff_tokens)
            tokens = self.norm2(tokens)

        return tokens.permute(0, 2, 1).view(b, c, h, w).contiguous()
