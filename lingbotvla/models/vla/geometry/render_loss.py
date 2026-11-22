"""Image-space losses for the Gaussian rendering branch."""

# [SIDA++] add by SidaZhu for Gaussian RGB reconstruction supervision.
# Purpose: define the new render loss independently from LingBot official loss
# terms. Original additions retain SidaZhu attribution and applicable rights,
# subject to the repository license and third-party notices.
# [SIDA--]

from __future__ import annotations

import torch
import torch.nn.functional as F


# [SIDA++] add by SidaZhu for Gaussian RGB render supervision
def _ssim_loss(prediction: torch.Tensor, target: torch.Tensor, window_size: int = 3) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError(f"Expected matching [B,C,H,W] tensors, got {prediction.shape} and {target.shape}")
    padding = window_size // 2
    mu_pred = F.avg_pool2d(prediction, window_size, stride=1, padding=padding)
    mu_target = F.avg_pool2d(target, window_size, stride=1, padding=padding)
    sigma_pred = F.avg_pool2d(prediction * prediction, window_size, stride=1, padding=padding) - mu_pred.square()
    sigma_target = F.avg_pool2d(target * target, window_size, stride=1, padding=padding) - mu_target.square()
    sigma_cross = F.avg_pool2d(prediction * target, window_size, stride=1, padding=padding) - mu_pred * mu_target
    c1, c2 = 0.01**2, 0.03**2
    numerator = (2 * mu_pred * mu_target + c1) * (2 * sigma_cross + c2)
    denominator = (mu_pred.square() + mu_target.square() + c1) * (sigma_pred + sigma_target + c2)
    return (1.0 - numerator / denominator.clamp_min(1e-6)).mean()


def compute_rgb_render_loss(rendered_rgb: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
    """Compute GaussianDream-style RGB reconstruction loss.

    Both tensors must be float BCHW in the same value range. The caller owns
    image normalization and camera rendering; this function has no depth path.
    """

    if rendered_rgb.shape != target_rgb.shape:
        raise ValueError(f"RGB shape mismatch: {rendered_rgb.shape} vs {target_rgb.shape}")
    rendered_rgb = rendered_rgb.float()
    target_rgb = target_rgb.float().detach()
    ssim = _ssim_loss(rendered_rgb, target_rgb)
    l1 = F.l1_loss(rendered_rgb, target_rgb)
    return 0.85 * ssim + 0.15 * l1
# [SIDA--]

