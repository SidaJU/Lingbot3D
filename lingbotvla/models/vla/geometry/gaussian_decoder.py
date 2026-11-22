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
    colors: Tensor
    opacities: Tensor


class GaussianParameterDecoder(nn.Module):
    """Decode post-VLM 3D tokens into a compact world-space Gaussian set.

    The decoder deliberately uses a minimal isotropic screen-space renderer in
    the first stage.  Rotation and spherical harmonics can be added later
    without changing the token contract.
    """

    def __init__(self, hidden_dim: int, num_gaussians: int = 256) -> None:
        super().__init__()
        self.num_gaussians = int(num_gaussians)
        self.norm = nn.LayerNorm(hidden_dim)
        self.to_params = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 10),
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
        colors = torch.sigmoid(raw[..., 6:9])
        opacities = torch.sigmoid(raw[..., 9:10])
        return GaussianParameters(means=means, scales=scales, colors=colors, opacities=opacities)
# [SIDA--]
