"""Small differentiable Gaussian RGB renderer used for the first training stage."""

# [SIDA++] add by SidaZhu for the differentiable Gaussian renderer extension.
# Purpose: provide the first low-resolution RGB reconstruction path, separate
# from LingBot official rendering/model code. Original additions retain SidaZhu
# attribution and applicable rights, subject to the repository license and
# third-party notices.
# [SIDA--]

from __future__ import annotations

import torch
from torch import Tensor, nn

from .gaussian_decoder import GaussianParameters


# [SIDA++] add by SidaZhu for differentiable Gaussian RGB supervision
class DifferentiableGaussianRenderer(nn.Module):
    """Project decoded Gaussians and alpha-composite them onto a low-res grid."""

    def __init__(self, render_size: int = 64, background: float = 0.0) -> None:
        super().__init__()
        if render_size < 8:
            raise ValueError("render_size must be at least 8")
        self.render_size = int(render_size)
        self.background = float(background)

    def forward(
        self,
        gaussians: GaussianParameters,
        intrinsics: Tensor,
        world_to_cam: Tensor,
        target_hw: tuple[int, int],
        view_index: int = 0,
    ) -> Tensor:
        if intrinsics.ndim != 4 or world_to_cam.ndim != 4:
            raise ValueError("camera tensors must be [B,V,3,3] and [B,V,4,4]")
        batch, views = intrinsics.shape[:2]
        if view_index < 0 or view_index >= views:
            raise IndexError(f"view_index={view_index} outside {views} camera views")
        if gaussians.means.shape[0] != batch:
            raise ValueError("Gaussian batch and camera batch differ")
        target_h, target_w = map(int, target_hw)
        render_h = render_w = self.render_size
        device = gaussians.means.device
        dtype = gaussians.means.dtype

        k = intrinsics[:, view_index].to(device=device, dtype=dtype).clone()
        k[:, 0, :] *= render_w / max(target_w, 1)
        k[:, 1, :] *= render_h / max(target_h, 1)
        t_world_cam = world_to_cam[:, view_index].to(device=device, dtype=dtype)
        points_cam = torch.einsum("bij,bnj->bni", t_world_cam[:, :3, :3], gaussians.means)
        points_cam = points_cam + t_world_cam[:, None, :3, 3]
        depth = points_cam[..., 2].clamp_min(1e-3)
        pixels = torch.einsum("bij,bnj->bni", k, points_cam / depth.unsqueeze(-1))
        centers_x, centers_y = pixels[..., 0], pixels[..., 1]
        valid = (points_cam[..., 2] > 1e-3) & (centers_x > -render_w) & (centers_x < 2 * render_w)
        valid = valid & (centers_y > -render_h) & (centers_y < 2 * render_h)

        y, x = torch.meshgrid(
            torch.arange(render_h, device=device, dtype=dtype),
            torch.arange(render_w, device=device, dtype=dtype),
            indexing="ij",
        )
        x = x.view(1, 1, render_h, render_w)
        y = y.view(1, 1, render_h, render_w)
        radius = (gaussians.scales[..., :2] * k[:, None, (0, 1), (0, 1)] / depth.unsqueeze(-1)).clamp(0.75, render_w)
        # Front-to-back ordering makes the result stable while preserving gradients.
        order = torch.argsort(depth, dim=1)
        centers_x = torch.gather(centers_x, 1, order)
        centers_y = torch.gather(centers_y, 1, order)
        radius = torch.gather(radius, 1, order[..., None].expand_as(radius))
        valid = torch.gather(valid, 1, order)
        colors = torch.gather(gaussians.colors, 1, order[..., None].expand_as(gaussians.colors))
        opacity = torch.gather(
            gaussians.opacities.squeeze(-1), 1, order
        )
        transmittance = torch.ones((batch, render_h, render_w), device=device, dtype=dtype)
        rendered = torch.zeros((batch, 3, render_h, render_w), device=device, dtype=dtype)
        for start in range(0, centers_x.shape[1], 32):
            stop = min(start + 32, centers_x.shape[1])
            dx = (x[:, : stop - start] - centers_x[:, start:stop, None, None]) / radius[:, start:stop, 0, None, None]
            dy = (y[:, : stop - start] - centers_y[:, start:stop, None, None]) / radius[:, start:stop, 1, None, None]
            alpha = opacity[:, start:stop, None, None] * torch.exp(-0.5 * (dx.square() + dy.square()))
            alpha = (alpha * valid[:, start:stop, None, None]).clamp(0.0, 0.98)
            weights = alpha * transmittance[:, None]
            rendered = rendered + (weights[..., None] * colors[:, start:stop, None, None, :]).sum(dim=1).permute(0, 3, 1, 2)
            transmittance = transmittance * (1.0 - alpha + 1e-4).prod(dim=1)
        return rendered + transmittance[:, None] * self.background
# [SIDA--]
