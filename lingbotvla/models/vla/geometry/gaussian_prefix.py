"""Single-step static Gaussian prefix adapter.

This follows GaussianDream's static encoder path: preserve the 16x16 spatial
token grid and apply only a learned linear projection into the VLA hidden size.
The first LingBot stage intentionally consumes one agent/top camera view.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class Gaussian3DPrefixAdapter(nn.Module):
    """Project one-view VGGT spatial tokens into the LingBot prefix width."""

    def __init__(
        self,
        feature_dim: int,
        adapter_dim: int = 512,
        num_tokens: int = 256,
        llm_dim: int = 2560,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # Keep legacy constructor arguments for checkpoint/config compatibility.
        del adapter_dim, num_tokens, num_heads, dropout
        self.feature_dim = int(feature_dim)
        self.to_llm = nn.Linear(self.feature_dim, int(llm_dim))

    def forward(self, view_features: Tensor, key_padding_mask=None) -> Tensor:
        """Return [B,256,llm_dim] from [B,V,256,feature_dim] or [B,256,D]."""
        del key_padding_mask
        if view_features.ndim == 4:
            _, views, _, dim = view_features.shape
            if views < 1:
                raise ValueError("GaussianDream static prefix received no views")
            # Official static path: camera_top/agent is view zero. Any wrist
            # tensors supplied by the legacy batch are intentionally ignored.
            features = view_features[:, 0]
        elif view_features.ndim == 3:
            features = view_features
            _, _, dim = features.shape
        else:
            raise ValueError(
                "Expected [B,V,N,D] or [B,N,D], "
                f"got {tuple(view_features.shape)}"
            )
        if dim != self.feature_dim:
            raise ValueError(
                f"Expected VGGT feature dim {self.feature_dim}, got {dim}"
            )
        return self.to_llm(features.to(dtype=self.to_llm.weight.dtype))
