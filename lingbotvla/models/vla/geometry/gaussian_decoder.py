"""Token-to-Gaussian decoder for the first RGB reconstruction stage."""

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
from torch.nn import functional as F


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
    """Decode 256 latent tokens into the full 256x256 Gaussian grid.

    The 256 tokens are the compressed prefix representation. Following the
    static GaussianDream path, they are normalized to a 32x32 feature grid and
    decoded with three 2x upsampling stages to 65,536 Gaussian primitives.
    """

    def __init__(
        self,
        hidden_dim: int,
        input_num_tokens: int = 256,
        output_grid_size: int = 256,
    ) -> None:
        super().__init__()
        self.input_num_tokens = int(input_num_tokens)
        self.output_grid_size = int(output_grid_size)
        self.num_gaussians = self.output_grid_size * self.output_grid_size
        input_grid_size = int(math.isqrt(self.input_num_tokens))
        if input_grid_size * input_grid_size != self.input_num_tokens:
            raise ValueError("Gaussian input tokens must form a square grid")
        if self.output_grid_size != 256:
            raise ValueError("GaussianDream static decoding uses a 256x256 Gaussian grid")
        self.input_grid_size = input_grid_size

        self.norm = nn.LayerNorm(hidden_dim)
        # [SIDA++] Match GaussianDream's static decoder: canonical 32x32
        # features followed by 64x64, 128x128 and 256x256 upsampling.
        self.input_proj = nn.Conv2d(hidden_dim, 512, kernel_size=1)
        self.backbone = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GELU(),
        )
        # rot(4) + scale(3) + opacity(1) + first-order SH(12).
        self.param_head = nn.Conv2d(128, 20, kernel_size=3, padding=1)
        self.depth_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )

    def forward(
        self,
        tokens: Tensor,
        intrinsics: Tensor | None = None,
        world_to_cam: Tensor | None = None,
        target_hw: tuple[int, int] | None = None,
        view_index: int = 0,
    ) -> GaussianParameters:
        if tokens.ndim != 3 or tokens.shape[1] != self.input_num_tokens:
            raise ValueError(
                f"expected [B,{self.input_num_tokens},D] Gaussian tokens, got {tuple(tokens.shape)}"
            )

        decoder_dtype = self.norm.weight.dtype
        token_grid = self.norm(tokens.to(dtype=decoder_dtype)).transpose(1, 2)
        token_grid = token_grid.reshape(
            tokens.shape[0], tokens.shape[2], self.input_grid_size, self.input_grid_size
        )
        token_grid = F.interpolate(token_grid, size=(32, 32), mode="bilinear", align_corners=False)
        features = self.backbone(self.input_proj(token_grid))
        raw = self.param_head(features).permute(0, 2, 3, 1)
        depth_raw = self.depth_head(features).squeeze(1)

        # GaussianDream's static decoder uses an 0..8m depth range before
        # calibrated unprojection. The fused rasterizer clips everything closer
        # than its 0.01m near plane; keeping the floor below that lets the
        # optimizer push all Gaussians behind the near plane, which yields a
        # fully black render with zero gradient. Keep a safe margin above near.
        depth = (0.5 + 7.5 * torch.sigmoid(depth_raw)).unsqueeze(-1)
        rotations = raw[..., 0:4]
        rotations = rotations / rotations.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        scales = F.softplus(raw[..., 4:7]) * 0.01
        opacities = torch.sigmoid(raw[..., 7:8])

        # Four SH coefficients per channel: DC + first-order SH.
        sh = raw[..., 8:20].reshape(tokens.shape[0], self.num_gaussians, 4, 3)
        sh_mask = sh.new_tensor([1.0, 1.0, 1.0, 0.025]).view(1, 1, 4, 1)
        sh = sh * sh_mask
        colors = torch.sigmoid(sh[..., 0, :])

        if intrinsics is not None or world_to_cam is not None:
            if intrinsics is None or world_to_cam is None or target_hw is None:
                raise ValueError("intrinsics, world_to_cam, and target_hw are required together")
            if intrinsics.ndim != 4 or world_to_cam.ndim != 4:
                raise ValueError("camera tensors must be [B,V,3,3] and [B,V,4,4]")
            if not 0 <= int(view_index) < intrinsics.shape[1]:
                raise IndexError(f"view_index={view_index} outside {intrinsics.shape[1]} camera views")

            target_h, target_w = map(int, target_hw)
            k = intrinsics[:, int(view_index)].to(device=tokens.device, dtype=decoder_dtype)
            t_world_cam = world_to_cam[:, int(view_index)].to(device=tokens.device, dtype=decoder_dtype)
            ys, xs = torch.meshgrid(
                torch.arange(self.output_grid_size, device=tokens.device, dtype=decoder_dtype),
                torch.arange(self.output_grid_size, device=tokens.device, dtype=decoder_dtype),
                indexing="ij",
            )
            u = (xs.reshape(1, -1) + 0.5) * (target_w / self.output_grid_size)
            v = (ys.reshape(1, -1) + 0.5) * (target_h / self.output_grid_size)
            z = depth.reshape(tokens.shape[0], -1)
            x = (u - k[:, None, 0, 2]) / k[:, None, 0, 0] * z
            y = (v - k[:, None, 1, 2]) / k[:, None, 1, 1] * z
            xyz_cam = torch.stack((x, y, z), dim=-1)
            # [SIDA++] PyTorch does not implement matrix inverse for BF16;
            # keep camera calibration numerically stable in FP32 and cast the
            # resulting transform back to the decoder dtype.
            cam_to_world = torch.linalg.inv(t_world_cam.float()).to(decoder_dtype)
            means = torch.einsum("bij,bnj->bni", cam_to_world[:, :3, :3], xyz_cam)
            means = means + cam_to_world[:, None, :3, 3]
        else:
            # Standalone decoder tests still receive a finite tensor, while all
            # training renders use calibrated K/T above.
            means = torch.zeros(
                (tokens.shape[0], self.num_gaussians, 3),
                device=tokens.device,
                dtype=decoder_dtype,
            )

        return GaussianParameters(
            means=means,
            scales=scales.reshape(tokens.shape[0], self.num_gaussians, 3),
            rotations=rotations.reshape(tokens.shape[0], self.num_gaussians, 4),
            colors=colors,
            opacities=opacities.reshape(tokens.shape[0], self.num_gaussians, 1),
            sh=sh,
        )
# [SIDA--]
