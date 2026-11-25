"""Image-space losses for the Gaussian rendering branch."""

# [SIDA++] add by SidaZhu for Gaussian RGB reconstruction supervision.
# Purpose: define the new render loss independently from LingBot official loss
# terms. Original additions retain SidaZhu attribution and applicable rights,
# subject to the repository license and third-party notices.
# [SIDA--]

from __future__ import annotations

import logging
import os
from typing import Callable

import torch
import torch.nn.functional as F


# [SIDA++] add by SidaZhu for Gaussian RGB render supervision
def build_lpips_fn(net: str = "vgg", cache_dir: str | None = None) -> Callable | None:
    """Build a frozen LPIPS module, or return None when lpips is unavailable.

    ``cache_dir`` is used as ``TORCH_HOME`` before creating LPIPS, so pretrained
    VGG/AlexNet backbone weights placed under ``<cache_dir>/hub/checkpoints``
    are loaded directly without runtime downloads.
    """
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "hub", "checkpoints"), exist_ok=True)
        os.environ["TORCH_HOME"] = cache_dir
        expected_checkpoint = {
            "vgg": "vgg16-397923af.pth",
            "alex": "alexnet-owt-7be5be79.pth",
        }.get(str(net).lower())
        if expected_checkpoint:
            logging.info(
                "LPIPS backbone cache: %s/hub/checkpoints/%s",
                cache_dir,
                expected_checkpoint,
            )
    try:
        import lpips
    except ImportError:
        logging.warning("lpips package not installed; Gaussian render LPIPS is disabled. Install with: pip install lpips")
        return None
    try:
        lpips_fn = lpips.LPIPS(net=net)
    except Exception as exc:  # missing/download-failed VGG weights should not break training
        logging.warning(f"Failed to initialize LPIPS ({net}): {exc}; Gaussian render LPIPS is disabled")
        return None
    lpips_fn.eval()
    for param in lpips_fn.parameters():
        param.requires_grad_(False)
    return lpips_fn


def _ssim_loss(prediction: torch.Tensor, target: torch.Tensor, window_size: int = 3) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError(f"Expected matching [B,C,H,W] tensors, got {prediction.shape} and {target.shape}")
    padding = window_size // 2
    reflect_pad = torch.nn.ReflectionPad2d(padding)
    prediction = reflect_pad(prediction)
    target = reflect_pad(target)
    mu_pred = F.avg_pool2d(prediction, window_size, stride=1)
    mu_target = F.avg_pool2d(target, window_size, stride=1)
    sigma_pred = F.avg_pool2d(prediction * prediction, window_size, stride=1) - mu_pred.square()
    sigma_target = F.avg_pool2d(target * target, window_size, stride=1) - mu_target.square()
    sigma_cross = F.avg_pool2d(prediction * target, window_size, stride=1) - mu_pred * mu_target
    c1, c2 = 0.01**2, 0.03**2
    numerator = (2 * mu_pred * mu_target + c1) * (2 * sigma_cross + c2)
    denominator = (mu_pred.square() + mu_target.square() + c1) * (sigma_pred + sigma_target + c2) + 1e-8
    ssim_map = numerator / denominator
    # Match GaussianDream: D-SSIM is scaled to [0, 1] before weighting.
    return torch.clamp((1.0 - ssim_map) / 2.0, 0.0, 1.0).mean()


def compute_rgb_render_loss(
    rendered_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    lpips_fn: Callable | None = None,
    lpips_weight: float = 0.0,
) -> torch.Tensor:
    """Compute GaussianDream-style RGB reconstruction loss.

    Both tensors must be float BCHW in the same value range. The caller owns
    image normalization and camera rendering; this function has no depth path.
    LPIPS is optional and follows GaussianDream: clamp to [0,1], map to [-1,1].
    """

    if rendered_rgb.shape != target_rgb.shape:
        raise ValueError(f"RGB shape mismatch: {rendered_rgb.shape} vs {target_rgb.shape}")
    rendered_rgb = rendered_rgb.float()
    target_rgb = target_rgb.float().detach()
    ssim = _ssim_loss(rendered_rgb, target_rgb)
    l1 = F.l1_loss(rendered_rgb, target_rgb)
    base_loss = 0.85 * ssim + 0.15 * l1
    if lpips_fn is None or lpips_weight <= 0.0:
        return base_loss
    lpips_pred = (rendered_rgb.clamp(0.0, 1.0) * 2.0 - 1.0).contiguous()
    lpips_target = (target_rgb.clamp(0.0, 1.0) * 2.0 - 1.0).contiguous()
    return base_loss + float(lpips_weight) * lpips_fn(lpips_pred, lpips_target).mean()


def compute_gaussian_regularization(
    scales: torch.Tensor,
    opacities: torch.Tensor,
    lambda_scale: float = 0.001,
    lambda_opacity: float = 0.01,
) -> torch.Tensor:
    """GaussianDream regularization: encourage compact, sparse Gaussians."""
    reg = scales.new_zeros(())
    if scales.numel() > 0:
        reg = reg + float(lambda_scale) * scales.norm(dim=-1).mean()
    if opacities.numel() > 0:
        reg = reg + float(lambda_opacity) * opacities.abs().mean()
    return reg


def compute_edge_smooth_loss(rgb: torch.Tensor, depth_map: torch.Tensor) -> torch.Tensor:
    """GaussianDream edge-aware disparity smoothness.

    Args:
        rgb: [B,3,H,W]
        depth_map: [B,1,H,W] or [B,H,W,1] positive depth.
    """
    if depth_map.ndim == 4 and depth_map.shape[-1] == 1:
        depth_map = depth_map.permute(0, 3, 1, 2)
    if depth_map.ndim != 4 or depth_map.shape[1] != 1:
        raise ValueError(f"depth_map must be [B,1,H,W], got {tuple(depth_map.shape)}")
    if depth_map.shape[2:] != rgb.shape[2:]:
        depth_map = F.interpolate(depth_map, size=rgb.shape[2:], mode="bilinear", align_corners=False)
    disp = 1.0 / (depth_map.float() + 1e-6)
    disp = disp / (disp.mean() + 1e-6)
    grad_rgb_x = (rgb[:, :, :, :-1] - rgb[:, :, :, 1:]).abs().mean(1, True)
    grad_rgb_y = (rgb[:, :, :-1, :] - rgb[:, :, 1:, :]).abs().mean(1, True)
    grad_disp_x = (disp[:, :, :, :-1] - disp[:, :, :, 1:]).abs()
    grad_disp_y = (disp[:, :, :-1, :] - disp[:, :, 1:, :]).abs()
    grad_disp_x = grad_disp_x * (-1.0 * grad_rgb_x).exp()
    grad_disp_y = grad_disp_y * (-1.0 * grad_rgb_y).exp()
    return grad_disp_x.mean() + grad_disp_y.mean()
# [SIDA--]

