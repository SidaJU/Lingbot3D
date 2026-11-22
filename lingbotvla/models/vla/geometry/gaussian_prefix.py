"""Single-step multi-view adapter for the optional 3D prefix branch."""

# [SIDA++] add by SidaZhu for the Gaussian 3D prefix extension.
# Purpose: compress multi-view visual features into the dedicated gs_3d VLM
# prefix segment, separate from LingBot official model components. Original
# additions retain SidaZhu attribution and applicable rights, subject to the
# repository license and third-party notices.
# [SIDA--]

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


# [SIDA++] add by SidaZhu for the first single-step 3D prefix adapter
class Gaussian3DPrefixAdapter(nn.Module):
    """Turn frozen multi-view features into fixed-length LingBot prefix tokens.

    This module intentionally has no temporal/TGE path. The upstream feature
    extractor owns camera/view processing; this adapter only performs learned
    query pooling and projects into the Qwen3-VL hidden width.
    """

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
        if adapter_dim % num_heads != 0:
            raise ValueError("adapter_dim must be divisible by num_heads")
        self.num_tokens = int(num_tokens)
        self.feature_proj = nn.Linear(feature_dim, adapter_dim)
        self.query_tokens = nn.Parameter(torch.randn(1, self.num_tokens, adapter_dim) * 0.02)
        self.cross_attention = nn.MultiheadAttention(
            adapter_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(adapter_dim)
        self.to_llm = nn.Linear(adapter_dim, llm_dim)

    def forward(self, view_features: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Return ``[B,num_tokens,llm_dim]`` from [B,V,N,D] or [B,N,D]."""

        if view_features.ndim == 4:
            batch, views, tokens, dim = view_features.shape
            memory = view_features.reshape(batch, views * tokens, dim)
        elif view_features.ndim == 3:
            batch, _, dim = view_features.shape
            memory = view_features
        else:
            raise ValueError(f"Expected [B,N,D] or [B,V,N,D], got {tuple(view_features.shape)}")
        adapter_dtype = self.feature_proj.weight.dtype
        memory = self.feature_proj(memory.to(dtype=adapter_dtype))
        queries = self.query_tokens.expand(batch, -1, -1).to(dtype=adapter_dtype)
        pooled, _ = self.cross_attention(
            queries,
            memory,
            memory,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.to_llm(self.norm(pooled)).to(dtype=view_features.dtype)
# [SIDA--]
