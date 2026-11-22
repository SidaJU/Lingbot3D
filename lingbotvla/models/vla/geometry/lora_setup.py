"""Explicit LoRA setup for the VLM-only Gaussian 3D experiment."""

# [SIDA++] add by SidaZhu for the VLM-only LoRA extension.
# Purpose: keep LoRA injection/freeze policy separate from LingBot official
# optimizer/model code. Original additions retain SidaZhu attribution and
# applicable rights, subject to the repository license and third-party notices.
# [SIDA--]

from __future__ import annotations

from typing import Iterable

from lingbotvla.utils.lora_utils import add_lora_to_model


# [SIDA++] add by SidaZhu for explicit VLM-only LoRA freezing and injection
def inject_vlm_lora(
    vlm_wrapper,
    rank: int = 8,
    alpha: int | None = None,
    target_modules: Iterable[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
) -> None:
    """Freeze VLM/action-expert weights and inject LoRA only into ``qwenvl``.

    This function intentionally fails early when a target suffix is not found,
    instead of silently training the wrong submodule.
    """

    target_modules = tuple(target_modules)
    if not target_modules:
        raise ValueError("At least one VLM LoRA target module is required")
    vlm = vlm_wrapper.qwenvl
    vlm.requires_grad_(False)
    if hasattr(vlm, "visual"):
        vlm.visual.requires_grad_(False)
    if hasattr(vlm_wrapper, "qwen_expert"):
        vlm_wrapper.qwen_expert.requires_grad_(False)

    supported = {name.rsplit(".", 1)[-1] for name, _ in vlm.named_modules()}
    missing = [name for name in target_modules if name not in supported]
    if missing:
        raise ValueError(f"VLM LoRA targets not found: {missing}; available suffixes include {sorted(supported)}")

    add_lora_to_model(
        vlm,
        lora_rank=rank,
        lora_alpha=alpha if alpha is not None else 2 * rank,
        lora_target_modules=",".join(target_modules),
        lora_target_modules_support=supported,
    )

    trainable = [name for name, param in vlm.named_parameters() if param.requires_grad]
    if not trainable or not all("lora" in name.lower() for name in trainable):
        raise RuntimeError("VLM LoRA injection left non-LoRA VLM parameters trainable")
# [SIDA--]
