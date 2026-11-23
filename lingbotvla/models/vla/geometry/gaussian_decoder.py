"""Token-to-Gaussian parameter decoder for the first RGB reconstruction stage."""

# [SIDA++] add by SidaZhu for the Gaussian parameter decoder extension.
# Purpose: decode post-VLM gs_3d tokens into world-space Gaussian attributes,
# separately from LingBot official code. Original additions retain SidaZhu
# attribution and applicable rights, subject to the repository license and
# third-party notices.
# [SIDA--]

from __future__ import annotations

from dataclasses import dataclass
import math

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

    def forward(
        self,
        tokens: Tensor,
        intrinsics: Tensor | None = None,
        world_to_cam: Tensor | None = None,
        target_hw: tuple[int, int] | None = None,
        view_index: int = 0,
    ) -> GaussianParameters:
        if tokens.ndim != 3 or tokens.shape[1] != self.num_gaussians:
            raise ValueError(
                f"expected [B,{self.num_gaussians},D] Gaussian tokens, got {tuple(tokens.shape)}"
            )
        decoder_dtype = self.norm.weight.dtype
        raw = self.to_params(self.norm(tokens.to(dtype=decoder_dtype)))
        # GaussianDream predicts a depth map on a square token grid and
        # unprojects it with the calibrated camera.  Keep a fallback free-space
        # parameterization for standalone decoder tests, but use calibrated
        # unprojection whenever the renderer supplies K and world-to-camera T.
        depth = 0.05 + 2.0 * torch.sigmoid(raw[..., 0:1])
        scales = torch.nn.functional.softplus(raw[..., 1:4]) * 0.01
        rotations = raw[..., 4:8]
        rotations = rotations / rotations.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        colors = torch.sigmoid(raw[..., 8:11])
        opacities = torch.sigmoid(raw[..., 11:12])

        if intrinsics is not None or world_to_cam is not None:
            if intrinsics is None or world_to_cam is None or target_hw is None:
                raise ValueError("intrinsics, world_to_cam, and target_hw are required together")
            if intrinsics.ndim != 4 or world_to_cam.ndim != 4:
                raise ValueError("camera tensors must be [B,V,3,3] and [B,V,4,4]")
            if not 0 <= int(view_index) < intrinsics.shape[1]:
                raise IndexError(f"view_index={view_index} outside {intrinsics.shape[1]} camera views")
            side = int(math.isqrt(self.num_gaussians))
            if side * side != self.num_gaussians:
                raise ValueError("calibrated Gaussian decoding requires a square token grid")
            target_h, target_w = map(int, target_hw)
            k = intrinsics[:, int(view_index)].to(device=tokens.device, dtype=decoder_dtype)
            t_world_cam = world_to_cam[:, int(view_index)].to(device=tokens.device, dtype=decoder_dtype)
            ys, xs = torch.meshgrid(
                torch.arange(side, device=tokens.device, dtype=decoder_dtype),
                torch.arange(side, device=tokens.device, dtype=decoder_dtype),
                indexing="ij",
            )
            u = (xs.reshape(1, -1) + 0.5) * (target_w / side)
            v = (ys.reshape(1, -1) + 0.5) * (target_h / side)
            z = depth.squeeze(-1)
            x = (u - k[:, None, 0, 2]) / k[:, None, 0, 0] * z
            y = (v - k[:, None, 1, 2]) / k[:, None, 1, 1] * z
            xyz_cam = torch.stack((x, y, z), dim=-1)
            cam_to_world = torch.linalg.inv(t_world_cam)
            means = torch.einsum("bij,bnj->bni", cam_to_world[:, :3, :3], xyz_cam)
            means = means + cam_to_world[:, None, :3, 3]
        else:
            means = torch.tanh(raw[..., 0:3])

        return GaussianParameters(
            means=means,
            scales=scales,
            rotations=rotations,
            colors=colors,
            opacities=opacities,
            sh=colors.unsqueeze(-2),
        )
# [SIDA--]
