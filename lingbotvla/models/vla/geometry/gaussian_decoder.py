"""Token-to-Gaussian parameter decoder for the first RGB reconstruction stage."""

# [SIDA++] add by SidaZhu for the Gaussian parameter decoder extension.
# Purpose: decode post-VLM gs_3d tokens into world-space Gaussian attributes,
# separately from LingBot official code. Original additions retain SidaZhu
# attribution and applicable rights, subject to the repository license and
# third-party notices.
# [SIDA--]

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


# [SIDA++] add by SidaZhu for GaussianDream-style token decoding
@dataclass
class GaussianParameters:
    means: Tensor
    scales: Tensor
    rotations: Tensor
    colors: Tensor
    opacities: Tensor
    sh: Tensor


class GaussianParameterDecoder(nn.Module):
    """Decode post-VLM 3D tokens into a compact world-space Gaussian set.

    The decoder emits the standard Gaussian attributes consumed by the fused
    CUDA rasterizer: means, scales, quaternion rotations, opacity and degree-0
    spherical-harmonic RGB coefficients.
    """

    def __init__(self, hidden_dim: int, num_gaussians: int = 256) -> None:
        super().__init__()
        self.num_gaussians = int(num_gaussians)
        self.norm = nn.LayerNorm(hidden_dim)
        self.to_params = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 14),
        )

    def forward(self, tokens: Tensor) -> GaussianParameters:
        if tokens.ndim != 3 or tokens.shape[1] != self.num_gaussians:
            raise ValueError(
                f"expected [B,{self.num_gaussians},D] Gaussian tokens, got {tuple(tokens.shape)}"
            )
        decoder_dtype = self.norm.weight.dtype
        raw = self.to_params(self.norm(tokens.to(dtype=decoder_dtype)))
        means = torch.tanh(raw[..., 0:3])
        scales = torch.nn.functional.softplus(raw[..., 3:6]) + 1e-3
        rotations = raw[..., 6:10]
        rotations = rotations / rotations.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        colors = torch.sigmoid(raw[..., 10:13])
        opacities = torch.sigmoid(raw[..., 13:14])
        return GaussianParameters(
            means=means,
            scales=scales,
            rotations=rotations,
            colors=colors,
            opacities=opacities,
            sh=colors.unsqueeze(-2),
        )
# [SIDA--]
