"""VGGT aggregator adapter matching GaussianDream's static encoder path."""

# [SIDA++] add by SidaZhu for the actual VGGT-to-VLM 3D feature path.
# Purpose: consume raw multi-view RGB, run GaussianDream's VGGT3DGSModel-style
# LoRA-wrapped aggregator, and expose spatially pooled 3D tokens separately
# from LingBot official vision code. Original additions retain SidaZhu
# attribution and applicable rights, subject to the repository license and
# third-party notices.
# [SIDA--]

from __future__ import annotations

import math
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# [SIDA++] add by SidaZhu for optional server-side VGGT dependency loading
# VGGT with the ``feature_only`` option is vendored under ``third_party/VGGT``.
# Upstream VGGT does not return aggregator features, so the vendored copy is
# imported directly; a ``vggt`` installed with pip must never shadow it.
_VENDORED_VGGT_SRC = Path(__file__).resolve().parents[4] / "third_party" / "VGGT" / "src"


def _import_vggt() -> "nn.Module":
    """Return the vendored ``vggt.models.vggt.VGGT`` class."""
    package = _VENDORED_VGGT_SRC / "vggt"
    if not (package / "models" / "vggt.py").is_file():
        raise ImportError(
            f"Vendored VGGT not found at {package}. It is required when enable_gaussian_3d=true "
            "and ships with this repository under third_party/VGGT."
        )
    loaded = sys.modules.get("vggt")
    if loaded is not None and Path(loaded.__file__).resolve().parent != package:
        for name in [name for name in sys.modules if name == "vggt" or name.startswith("vggt.")]:
            del sys.modules[name]
    source = str(_VENDORED_VGGT_SRC)
    if source in sys.path:
        sys.path.remove(source)
    sys.path.insert(0, source)
    from vggt.models.vggt import VGGT

    return VGGT
# [SIDA--]


# [SIDA++] GaussianDream's VGGT3DGSModel wraps aggregator linears with this
# exact LoRALinear form. Keep the parameter names (lora_down/lora_up) so the
# official LoRA state dict layout is preserved.
class OfficialVGGTLoRALinear(nn.Module):
    """GaussianDream-style LoRA wrapper for a frozen VGGT aggregator Linear."""

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 32.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"VGGT LoRA rank must be positive, got {rank}")
        rank = int(rank)
        self.linear = linear
        self.scaling = float(alpha) / rank
        factory = {"device": linear.weight.device, "dtype": linear.weight.dtype}
        self.lora_down = nn.Linear(linear.in_features, rank, bias=False, **factory)
        self.lora_up = nn.Linear(rank, linear.out_features, bias=False, **factory)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.reset_lora_parameters()
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: Tensor) -> Tensor:
        base = self.linear(x)
        lora = self.lora_up(self.dropout(self.lora_down(x)))
        return base + lora * self.scaling
# [SIDA--]


def _apply_official_vggt_lora(
    aggregator: nn.Module,
    target_names: tuple[str, ...],
    rank: int,
    alpha: float,
    dropout: float,
) -> int:
    """Inject GaussianDream's qkv/proj/fc1/fc2 LoRA into an VGGT aggregator."""

    count = 0
    for parent in list(aggregator.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, OfficialVGGTLoRALinear):
                continue
            if isinstance(child, nn.Linear) and any(target in name for target in target_names):
                setattr(
                    parent,
                    name,
                    OfficialVGGTLoRALinear(child, rank=rank, alpha=alpha, dropout=dropout),
                )
                count += 1
    return count


def _is_vggt_lora_parameter(name: str) -> bool:
    return "lora_down" in name or "lora_up" in name


class VGGTFeatureEncoder(nn.Module):
    """Run the real VGGT aggregator and return ``[B,V,256,D]`` patch tokens."""

    def __init__(
        self,
        model_path: str,
        image_size: int = 518,
        feature_dim: int = 2048,
        freeze: bool = True,
        output_grid: int = 16,
        use_depth: bool = True,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05,
        lora_targets: tuple[str, ...] = ("qkv", "proj", "fc1", "fc2"),
    ) -> None:
        super().__init__()
        VGGT = _import_vggt()
        self.image_size = int(image_size)
        self.output_grid = int(output_grid)
        self.feature_dim = int(feature_dim)
        self.use_depth = bool(use_depth)
        self.freeze = bool(freeze)
        self.use_lora = bool(use_lora)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.lora_targets = tuple(lora_targets)
        self._lora_injected = False
        self.lora_injected_layers = 0
        self.encoder = VGGT(
            img_size=self.image_size,
            embed_dim=self.feature_dim // 2,
            enable_camera=False,
            enable_point=False,
            enable_depth=self.use_depth,
            enable_track=False,
            feature_only=True,
        )
        checkpoint = Path(model_path).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"VGGT checkpoint does not exist: {checkpoint}")
        self.checkpoint_path = str(checkpoint)
        self.checkpoint_loaded = False
        self.load_check: dict = {}
        self._load_checkpoint()
        # Materialized models can be wrapped immediately. Meta-device models
        # inject LoRA in load_pretrained_weights(), after to_empty(), so the
        # new parameters are real tensors before optimizer construction.
        if self.use_lora and not any(value.is_meta for value in self.encoder.state_dict().values()):
            self._inject_lora()
        self._apply_trainability()

    def _inject_lora(self) -> None:
        if self._lora_injected:
            return
        if any(value.is_meta for value in self.encoder.state_dict().values()):
            return
        count = _apply_official_vggt_lora(
            self.encoder.aggregator,
            self.lora_targets,
            self.lora_rank,
            self.lora_alpha,
            self.lora_dropout,
        )
        if count == 0:
            raise RuntimeError(
                f"VGGT LoRA targets {self.lora_targets} matched no aggregator Linear layers"
            )
        self.lora_injected_layers = count
        self._lora_injected = True

    def _reset_lora_parameters(self) -> None:
        for module in self.encoder.modules():
            if isinstance(module, OfficialVGGTLoRALinear):
                module.reset_lora_parameters()

    def _apply_trainability(self) -> None:
        if self.use_lora and self._lora_injected:
            for param in self.encoder.parameters():
                param.requires_grad_(False)
            for name, param in self.encoder.named_parameters():
                if _is_vggt_lora_parameter(name):
                    param.requires_grad_(True)
        else:
            self.encoder.requires_grad_(not self.freeze)
        self.encoder.train(not self.freeze or self._any_trainable_lora())

    def _any_trainable_lora(self) -> bool:
        return any(
            _is_vggt_lora_parameter(name) and param.requires_grad
            for name, param in self.encoder.named_parameters()
        )

    def _read_checkpoint(self) -> dict:
        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        state = payload.get("state_dict", payload.get("model", payload)) if isinstance(payload, dict) else payload
        # The public VGGT-1B checkpoint contains optional camera/point/depth/track
        # heads. This first LingBot stage instantiates only the aggregator, so
        # discard checkpoint entries for disabled heads before strict validation.
        # After official LoRA wrapping, original Linear weights live at
        # ``*.linear.weight``; map them back from the unwrapped checkpoint keys.
        encoder_keys = set(self.encoder.state_dict().keys())
        mapped: dict = {}
        for key, value in state.items():
            if key in encoder_keys:
                mapped[key] = value
                continue
            for suffix in (".weight", ".bias"):
                if key.endswith(suffix):
                    wrapped_key = f"{key[: -len(suffix)]}.linear{suffix}"
                    if wrapped_key in encoder_keys:
                        mapped[wrapped_key] = value
                        break
        return mapped

    def _load_checkpoint(self) -> tuple[int, int]:
        """Load the VGGT checkpoint; returns ``(loaded_tensors, total_tensors)``.

        The VLA model may be constructed under ``init_empty_weights()``, where
        every parameter is on the meta device and ``load_state_dict`` is a
        no-op. Loading is then deferred to ``load_pretrained_weights``, which
        the weight loader calls after the model has been materialized.
        """
        tensors = self.encoder.state_dict()
        if any(value.is_meta for value in tensors.values()):
            return 0, len(tensors)
        state = self._read_checkpoint()
        missing, unexpected = self.encoder.load_state_dict(state, strict=False)
        missing = [name for name in missing if not _is_vggt_lora_parameter(name)]
        if missing:
            raise RuntimeError(f"VGGT checkpoint is missing {len(missing)} parameters: {missing[:5]}")
        if unexpected:
            raise RuntimeError(f"VGGT checkpoint has {len(unexpected)} unexpected parameters: {unexpected[:5]}")
        self.checkpoint_loaded = True
        # Verify while the reference tensors are still in memory; compare a
        # deterministic sample so the check stays cheap for the 1B encoder.
        keys = sorted(key for key, value in state.items() if torch.is_floating_point(value))
        stride = max(1, len(keys) // 24)
        live = self.encoder.state_dict()
        max_rel_err, checked = 0.0, 0
        for key in keys[::stride]:
            source = state[key].detach().float()
            current = live[key].detach().float().cpu()
            error = (current - source).abs().sum()
            scale = source.abs().sum().clamp_min(1e-12)
            max_rel_err = max(max_rel_err, float((error / scale).item()))
            checked += 1
        expected_original = len(tensors) - sum(1 for key in tensors if _is_vggt_lora_parameter(key))
        self.load_check = {
            "loaded": len(state),
            "total": len(tensors),
            "expected_original": expected_original,
            "lora_layers": self.lora_injected_layers,
            "checked": checked,
            "max_rel_err": max_rel_err,
            "ok": len(state) == expected_original and expected_original > 0 and max_rel_err < 0.05,
            "source": self.checkpoint_path,
        }
        return len(state), len(tensors)

    def load_pretrained_weights(self) -> dict:
        """Reload and verify VGGT weights after the model left the meta device."""
        if any(value.is_meta for value in self.encoder.state_dict().values()):
            raise RuntimeError("VGGT encoder still has meta tensors; load weights after to_empty().")
        if self.use_lora:
            self._inject_lora()
            self._reset_lora_parameters()
        self._load_checkpoint()
        self._apply_trainability()
        return dict(self.load_check)

    def forward(self, images: Tensor, return_depth: bool = False):
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError(f"VGGT expects [B,V,3,H,W] RGB images, got {tuple(images.shape)}")
        images = images.float()
        # Raw RoboTwin frames arrive as uint8-like 0..255 tensors. Accept both
        # that representation and already-normalized float RGB inputs.
        if images.numel() > 0 and images.detach().amax() > 1.0:
            images = images / 255.0
        images = images.clamp(0.0, 1.0)
        batch, views = images.shape[:2]
        # Official GaussianDream static path: use only current agent/top view.
        if views > 1:
            images = images[:, :1]
            views = 1
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
        # GaussianDream calls the VGGT3DGSModel aggregator directly; its
        # unused camera/point/depth/gs heads are intentionally skipped.
        encoder_trainable = self.training and any(
            param.requires_grad for param in self.encoder.parameters()
        )
        self.encoder.train(encoder_trainable)

        def grad_context():
            return torch.enable_grad() if encoder_trainable else torch.no_grad()

        amp_context = (
            torch.autocast(device_type=images.device.type, dtype=patch_dtype)
            if images.is_cuda and patch_dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        with grad_context(), amp_context:
            aggregated_tokens, patch_start = self.encoder.aggregator(images)
        features = aggregated_tokens[-1] if isinstance(aggregated_tokens, (list, tuple)) else aggregated_tokens
        patch_start = int(patch_start or getattr(self.encoder.aggregator, "patch_start_idx", 0))
        if features.ndim != 4:
            raise ValueError(f"VGGT feature tensor must be [B,V,P,D], got {tuple(features.shape)}")
        features = features[:, :, patch_start:]
        patches = features.shape[2]
        side = int(math.sqrt(patches))
        if side * side != patches:
            raise ValueError(f"VGGT patch count must form a square grid, got {patches}")
        features = features.reshape(batch * views, side, side, features.shape[-1]).permute(0, 3, 1, 2)
        features = F.adaptive_avg_pool2d(features, (self.output_grid, self.output_grid))
        pooled_features = features.flatten(2).transpose(1, 2).reshape(batch, views, self.output_grid**2, -1)
        if not return_depth:
            return pooled_features
        if not self.use_depth:
            return pooled_features, None, None

        # FSDP mixed precision can leave DPTHead parameters split between FP32
        # and BF16. Keep the whole depth head in FP32; the aggregator remains
        # BF16 so patch embedding/conv inputs match.
        depth_head = getattr(self.encoder, "depth_head", None)
        if depth_head is None:
            return pooled_features, None, None
        for param in depth_head.parameters():
            if param.dtype != torch.float32:
                param.data = param.data.float()
        for buffer in depth_head.buffers():
            if buffer.dtype != torch.float32:
                buffer.data = buffer.data.float()
        with grad_context(), torch.autocast(device_type=images.device.type, enabled=False):
            depth, depth_conf = depth_head(
                aggregated_tokens, images=images, patch_start_idx=patch_start
            )
        return pooled_features, depth, depth_conf
# [SIDA--]
