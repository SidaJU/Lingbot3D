"""VGGT aggregator adapter for the SidaZhu Gaussian 3D extension."""

# [SIDA++] add by SidaZhu for the actual VGGT-to-VLM 3D feature path.
# Purpose: consume raw multi-view RGB, run a frozen VGGT aggregator, and expose
# spatially pooled 3D tokens separately from LingBot official vision code.
# Original additions retain SidaZhu attribution and applicable rights, subject
# to the repository license and third-party notices.
# [SIDA--]

from __future__ import annotations

import math
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# [SIDA++] add by SidaZhu for optional server-side VGGT dependency loading
class VGGTFeatureEncoder(nn.Module):
    """Run the real VGGT aggregator and return ``[B,V,256,D]`` patch tokens."""

    def __init__(
        self,
        model_path: str,
        image_size: int = 518,
        feature_dim: int = 2048,
        freeze: bool = True,
        output_grid: int = 16,
    ) -> None:
        super().__init__()
        try:
            from vggt.models.vggt import VGGT
        except ImportError as exc:
            raise ImportError(
                "VGGT is required when enable_gaussian_3d=true. Install the pinned VGGT package "
                "or expose its source package on PYTHONPATH."
            ) from exc
        self.image_size = int(image_size)
        self.output_grid = int(output_grid)
        self.feature_dim = int(feature_dim)
        self.encoder = VGGT(
            img_size=self.image_size,
            embed_dim=self.feature_dim // 2,
            enable_camera=False,
            enable_point=False,
            enable_depth=False,
            enable_track=False,
            feature_only=True,
        )
        checkpoint = Path(model_path).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"VGGT checkpoint does not exist: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu")
        state = payload.get("state_dict", payload.get("model", payload)) if isinstance(payload, dict) else payload
        # The public VGGT-1B checkpoint contains optional camera/point/depth/track
        # heads. This first LingBot stage instantiates only the aggregator, so
        # discard checkpoint entries for disabled heads before strict validation.
        encoder_keys = set(self.encoder.state_dict().keys())
        state = {key: value for key, value in state.items() if key in encoder_keys}
        missing, unexpected = self.encoder.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"VGGT checkpoint is missing {len(missing)} parameters")
        self.requires_grad_(not freeze)
        if freeze:
            self.eval()

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError(f"VGGT expects [B,V,3,H,W] RGB images, got {tuple(images.shape)}")
        images = images.float()
        # Raw RoboTwin frames arrive as uint8-like 0..255 tensors. Accept both
        # that representation and already-normalized float RGB inputs.
        if images.numel() > 0 and images.detach().amax() > 1.0:
            images = images / 255.0
        images = images.clamp(0.0, 1.0)
        batch, views = images.shape[:2]
        images = F.interpolate(
            images.reshape(batch * views, 3, images.shape[-2], images.shape[-1]),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, views, 3, self.image_size, self.image_size)
        # [SIDA++] The frozen VGGT may be materialized in BF16 by the 6-GPU
        # training launcher. Match the input dtype to the encoder parameters;
        # feeding FP32 images into a BF16 patch-embed convolution fails before
        # the Gaussian branch can produce tokens.
        patch_embed = self.encoder.aggregator.patch_embed
        patch_proj = getattr(patch_embed, "proj", None)
        if patch_proj is None:
            patch_proj = patch_embed.patch_embed.proj
        patch_dtype = patch_proj.bias.dtype if patch_proj.bias is not None else patch_proj.weight.dtype
        images = images.to(device=patch_proj.weight.device, dtype=patch_dtype)
        context = torch.no_grad() if not self.training or not any(p.requires_grad for p in self.parameters()) else torch.enable_grad()
        amp_context = (
            torch.autocast(device_type=images.device.type, dtype=patch_dtype)
            if images.is_cuda and patch_dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        with context, amp_context:
            outputs = self.encoder(images)
        features = outputs["features"]
        patch_start = int(outputs.get("patch_start_idx", getattr(self.encoder.aggregator, "patch_start_idx", 0)))
        if isinstance(features, (list, tuple)):
            features = features[-1]
        if features.ndim != 4:
            raise ValueError(f"VGGT feature tensor must be [B,V,P,D], got {tuple(features.shape)}")
        features = features[:, :, patch_start:]
        patches = features.shape[2]
        side = int(math.sqrt(patches))
        if side * side != patches:
            raise ValueError(f"VGGT patch count must form a square grid, got {patches}")
        features = features.reshape(batch * views, side, side, features.shape[-1]).permute(0, 3, 1, 2)
        features = F.adaptive_avg_pool2d(features, (self.output_grid, self.output_grid))
        return features.flatten(2).transpose(1, 2).reshape(batch, views, self.output_grid**2, -1)
# [SIDA--]
