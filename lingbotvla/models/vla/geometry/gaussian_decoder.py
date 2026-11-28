"""Official GaussianDream static decoder replicated for LingBot 3D prefix.

Future Gaussian / velocity branches are intentionally omitted. Everything in
the static reconstruction path follows the official PyTorch implementation:
ConvTranspose upsample blocks, DPT-style feature fusion, GeometryHead,
AppearanceHead with RGB fusion, SH masking and Gaussian parameter conversion.
"""

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
    depth_map: Tensor | None = None


def _reset_child_parameters(module: nn.Module) -> None:
    """Re-run each child's own initialization.

    Layers and heads that define ``reset_parameters`` are reset through it;
    plain containers are traversed. This reproduces the construction-time
    initialization of a normally built module, which is what the official
    GaussianDream decoder relies on, after a ``to_empty()`` materialization.
    """
    for child in module.children():
        reset = getattr(child, "reset_parameters", None)
        if callable(reset):
            child.reset_parameters()
        else:
            _reset_child_parameters(child)


class _UpsampleBlock(nn.Module):
    """GaussianDream upsample block: ConvTranspose + GroupNorm + residual."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        skip = F.interpolate(self.skip(x), scale_factor=2, mode="bilinear", align_corners=False)
        h = F.gelu(self.norm1(self.up(x)))
        h = F.gelu(self.norm2(self.conv(h)))
        return h + skip


class _FeatureFusionBlock(nn.Module):
    """GaussianDream DPT-style multi-scale feature fusion block."""

    def __init__(self, features: int, has_residual: bool = True) -> None:
        super().__init__()
        self.has_residual = bool(has_residual)
        if self.has_residual:
            self.residual_conv = nn.Sequential(
                nn.Conv2d(features, features, 3, padding=1, bias=True),
                nn.GroupNorm(min(32, features), features),
                nn.GELU(),
                nn.Conv2d(features, features, 3, padding=1, bias=True),
            )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1, bias=True),
            nn.GroupNorm(min(32, features), features),
            nn.GELU(),
            nn.Conv2d(features, features, 1, bias=True),
        )

    def forward(self, x: Tensor, residual: Tensor | None = None, size=None) -> Tensor:
        if self.has_residual and residual is not None:
            if residual.shape[2:] != x.shape[2:]:
                residual = F.interpolate(
                    residual, size=x.shape[2:], mode="bilinear", align_corners=True
                )
            x = x + self.residual_conv(residual)
        x = self.fusion_conv(x)
        if size is not None and x.shape[2:] != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
        return x


class SharedGaussianBackbone(nn.Module):
    """Official GaussianDream shared token-to-feature backbone (static path)."""

    def __init__(self, token_dim: int = 2048) -> None:
        super().__init__()
        self.layer1 = _UpsampleBlock(token_dim, 512)
        self.layer2 = _UpsampleBlock(512, 256)
        self.layer3 = _UpsampleBlock(256, 128)

        self.fusion1 = _FeatureFusionBlock(128, has_residual=False)
        self.fusion2 = _FeatureFusionBlock(128, has_residual=True)
        self.fusion3 = _FeatureFusionBlock(128, has_residual=True)

        self.proj_feat2 = nn.Conv2d(256, 128, 1)
        self.proj_feat1 = nn.Conv2d(512, 128, 1)

    def forward(self, token_grid: Tensor) -> Tensor:
        feat1 = self.layer1(token_grid)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)

        fused = self.fusion1(feat3)
        fused = self.fusion2(fused, residual=self.proj_feat2(feat2))
        fused = self.fusion3(fused, residual=self.proj_feat1(feat1))
        return fused


class GeometryHead(nn.Module):
    """Official GaussianDream geometry-only head."""

    def __init__(self, predict_depth: bool = True, feature_dim: int = 128) -> None:
        super().__init__()
        self.predict_depth = bool(predict_depth)
        self.feature_dim = int(feature_dim)
        self.param_head = nn.Conv2d(self.feature_dim, 8, 3, padding=1)

        if self.predict_depth:
            self.depth_refine = nn.Sequential(
                nn.Conv2d(feature_dim, 64, 3, padding=1),
                nn.GroupNorm(min(32, 64), 64),
                nn.GELU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.GroupNorm(min(32, 64), 64),
                nn.GELU(),
                nn.Conv2d(64, 1, 3, padding=1),
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Official GaussianDream head init: default layers, then small outputs."""
        _reset_child_parameters(self)
        if self.predict_depth:
            for module in self.depth_refine.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.xavier_uniform_(module.weight, gain=0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        nn.init.xavier_uniform_(self.param_head.weight, gain=0.01)
        nn.init.zeros_(self.param_head.bias)

    def forward(self, shared_features: Tensor) -> dict[str, Tensor]:
        result = {"geom_params": self.param_head(shared_features)}
        if self.predict_depth:
            result["depth"] = self.depth_refine(shared_features)
        return result


class AppearanceHead(nn.Module):
    """Official GaussianDream appearance/SH head with RGB image fusion."""

    def __init__(
        self,
        use_image_fusion: bool = True,
        img_dim: int = 3,
        feature_dim: int = 128,
        image_fusion_alpha: float = 1.0,
        stop_grad_geometry: bool = True,
    ) -> None:
        super().__init__()
        self.use_image_fusion = bool(use_image_fusion)
        self.feature_dim = int(feature_dim)
        self.image_fusion_alpha = float(image_fusion_alpha)
        self.stop_grad_geometry = bool(stop_grad_geometry)

        if self.use_image_fusion:
            self.img_merger = nn.Sequential(
                nn.Conv2d(img_dim, feature_dim, 7, padding=3),
                nn.GELU(),
            )
        self.sh_head = nn.Conv2d(feature_dim, 9, 3, padding=1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Official GaussianDream appearance init: default img_merger, small SH head."""
        _reset_child_parameters(self)
        nn.init.xavier_uniform_(self.sh_head.weight, gain=0.01)
        nn.init.zeros_(self.sh_head.bias)

    def set_image_fusion_enabled(self, enabled: bool) -> None:
        self.use_image_fusion = bool(enabled)

    def forward(self, shared_features: Tensor, images: Tensor | None = None) -> Tensor:
        geom_context = shared_features.detach() if self.stop_grad_geometry else shared_features
        appearance_features = geom_context
        if self.use_image_fusion and images is not None:
            if images.shape[2:] != geom_context.shape[2:]:
                images = F.interpolate(
                    images, size=geom_context.shape[2:], mode="bilinear", align_corners=True
                )
            appearance_features = appearance_features + self.image_fusion_alpha * self.img_merger(images)
        return self.sh_head(appearance_features)


class StaticGaussianHead(nn.Module):
    """Official GaussianDream static reconstruction head, no future branch."""

    def __init__(
        self,
        use_image_fusion: bool = True,
        img_dim: int = 3,
        predict_depth: bool = True,
        image_fusion_alpha: float = 1.0,
        stop_grad_geometry_for_appearance: bool = True,
    ) -> None:
        super().__init__()
        self.use_image_fusion = bool(use_image_fusion)
        self.predict_depth = bool(predict_depth)
        self.geometry_head = GeometryHead(predict_depth=predict_depth)
        self.appearance_head = AppearanceHead(
            use_image_fusion=use_image_fusion,
            img_dim=img_dim,
            image_fusion_alpha=image_fusion_alpha,
            stop_grad_geometry=stop_grad_geometry_for_appearance,
        )
        self.register_buffer(
            "sh_mask",
            torch.tensor([1.0, 1.0, 1.0, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025], dtype=torch.float32),
            persistent=False,
        )

    def reset_parameters(self) -> None:
        _reset_child_parameters(self)

    def set_image_fusion_enabled(self, enabled: bool) -> None:
        self.use_image_fusion = bool(enabled)
        self.appearance_head.set_image_fusion_enabled(enabled)

    def forward(self, shared_features: Tensor, images: Tensor | None = None) -> dict[str, Tensor]:
        geometry_output = self.geometry_head(shared_features)
        sh_logits = self.appearance_head(shared_features, images=images)
        result = {
            "gaussian_params": torch.cat([geometry_output["geom_params"], sh_logits], dim=1),
        }
        if self.predict_depth:
            result["depth"] = geometry_output["depth"]
        return result


class GaussianParameterDecoder(nn.Module):
    """LingBot adapter around the official GaussianDream static decoder.

    LingBot's VLM hidden width is usually 2560 while GaussianDream's official
    backbone expects 2048 channels, so the only non-official layer is the
    documented 1x1 input projection. After that projection, the backbone and
    static heads match the official implementation.
    """

    def __init__(
        self,
        hidden_dim: int,
        input_num_tokens: int = 256,
        output_grid_size: int = 256,
        token_dim: int = 2048,
        use_image_fusion: bool = True,
        image_fusion_alpha: float = 1.0,
        stop_grad_geometry_for_appearance: bool = True,
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
        self.token_dim = int(token_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        # LingBot-only 2560 -> 2048 interface from the architecture plan.
        self.input_proj = nn.Conv2d(hidden_dim, self.token_dim, kernel_size=1)
        self.shared_backbone = SharedGaussianBackbone(token_dim=self.token_dim)
        self.static_head = StaticGaussianHead(
            use_image_fusion=bool(use_image_fusion),
            img_dim=3,
            predict_depth=True,
            image_fusion_alpha=float(image_fusion_alpha),
            stop_grad_geometry_for_appearance=bool(stop_grad_geometry_for_appearance),
        )

    def reset_parameters(self) -> None:
        """Restore construction-time init (LingBot input projection + official decoder)."""
        _reset_child_parameters(self)

    def set_image_fusion_enabled(self, enabled: bool) -> None:
        self.static_head.set_image_fusion_enabled(enabled)

    def forward(
        self,
        tokens: Tensor,
        intrinsics: Tensor | None = None,
        world_to_cam: Tensor | None = None,
        target_hw: tuple[int, int] | None = None,
        view_index: int = 0,
        images: Tensor | None = None,
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
        token_grid = self.input_proj(token_grid)

        # Empty cache in case a future branch is later re-enabled.
        fusion_images = None
        if images is not None:
            fusion_images = images.to(device=token_grid.device, dtype=decoder_dtype)
            if fusion_images.ndim == 5:
                view = int(view_index)
                if not 0 <= view < fusion_images.shape[1]:
                    raise IndexError(f"view_index={view} outside image views={fusion_images.shape[1]}")
                fusion_images = fusion_images[:, view]
            if fusion_images.ndim != 4 or fusion_images.shape[1] != 3:
                raise ValueError(f"appearance fusion images must be [B,3,H,W], got {tuple(fusion_images.shape)}")
            if fusion_images.detach().amax() > 1.5:
                fusion_images = fusion_images / 255.0

        shared_features = self.shared_backbone(token_grid)
        decoder_output = self.static_head(shared_features, images=fusion_images)
        raw = decoder_output["gaussian_params"]
        rot_raw, scale_raw, opa_raw, sh_raw = raw.split([4, 3, 1, 9], dim=1)

        depth_logits = decoder_output["depth"].squeeze(1)
        # Official GaussianDream static depth parameterization: 0..8m.
        depth_map = 8.0 * torch.sigmoid(depth_logits)

        H_dec, W_dec = depth_map.shape[-2:]
        batch = depth_map.shape[0]
        rot_maps = rot_raw.permute(0, 2, 3, 1)
        rot_maps = rot_maps / (rot_maps.norm(dim=-1, keepdim=True) + 1e-8)
        scale_maps = F.softplus(scale_raw.permute(0, 2, 3, 1), beta=1) * 0.01
        opacity_maps = torch.sigmoid(opa_raw.permute(0, 2, 3, 1))

        sh_maps = sh_raw.permute(0, 2, 3, 1)
        sh_mask = self.static_head.sh_mask.view(1, 1, 1, 9)
        sh_maps = sh_maps * sh_mask
        sh_flat = sh_maps.reshape(batch, self.num_gaussians, 9)
        # Official renderer pads 9 SH coefficients to 12 and reshapes to [N,4,3].
        sh_padded = sh_flat.new_zeros(batch, self.num_gaussians, 12)
        sh_padded[..., :9] = sh_flat
        sh = sh_padded.reshape(batch, self.num_gaussians, 4, 3)
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
            z = depth_map.reshape(batch, -1)
            x = (u - k[:, None, 0, 2]) / k[:, None, 0, 0] * z
            y = (v - k[:, None, 1, 2]) / k[:, None, 1, 1] * z
            xyz_cam = torch.stack((x, y, z), dim=-1)
            cam_to_world = torch.linalg.inv(t_world_cam.float()).to(decoder_dtype)
            means = torch.einsum("bij,bnj->bni", cam_to_world[:, :3, :3], xyz_cam)
            means = means + cam_to_world[:, None, :3, 3]
            means = torch.clamp(means, min=-100.0, max=100.0)
            means = torch.where(torch.isnan(means) | torch.isinf(means), torch.zeros_like(means), means)
        else:
            means = torch.zeros((batch, self.num_gaussians, 3), device=tokens.device, dtype=decoder_dtype)

        scales = scale_maps.reshape(batch, self.num_gaussians, 3).clamp(min=1e-7, max=10.0)
        return GaussianParameters(
            means=means,
            scales=scales,
            rotations=rot_maps.reshape(batch, self.num_gaussians, 4),
            colors=colors,
            opacities=opacity_maps.reshape(batch, self.num_gaussians, 1),
            sh=sh,
            depth_map=depth_map.unsqueeze(1),
        )
# [SIDA--]
