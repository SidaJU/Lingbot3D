"""Static Gaussian prefix adapter aligned with GaussianDream."""

# [SIDA++] add by SidaZhu for the Gaussian 3D prefix extension.
# Purpose: compress multi-view visual features into the dedicated gs_3d VLM
# prefix segment, separate from LingBot official model components. Original
# additions retain SidaZhu attribution and applicable rights, subject to the
# repository license and third-party notices.
# [SIDA--]

from __future__ import annotations

from torch import Tensor, nn


# [SIDA++] add by SidaZhu for the first single-step 3D prefix adapter
class Gaussian3DPrefixAdapter(nn.Module):
    """Project one agent/top-view VGGT spatial grid into VLA width."""

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
        # Keep legacy arguments so existing configs and checkpoints still load.
        del adapter_dim, num_tokens, num_heads, dropout
        self.feature_dim = int(feature_dim)
        self.to_llm = nn.Linear(self.feature_dim, int(llm_dim))

    def forward(self, view_features: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Return [B,256,llm_dim] from [B,V,256,D] or [B,256,D]."""
        del key_padding_mask
        if view_features.ndim == 4:
            _, views, _, dim = view_features.shape
            if views < 1:
                raise ValueError("GaussianDream static prefix received no views")
            # Official static path: current agent/top view only.
            features = view_features[:, 0]
        elif view_features.ndim == 3:
            features = view_features
            _, _, dim = features.shape
        else:
            raise ValueError(f"Expected [B,V,N,D] or [B,N,D], got {tuple(view_features.shape)}")
        if dim != self.feature_dim:
            raise ValueError(f"Expected VGGT feature dim {self.feature_dim}, got {dim}")
        return self.to_llm(features.to(dtype=self.to_llm.weight.dtype))
# [SIDA--]
