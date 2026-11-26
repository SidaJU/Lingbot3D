# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import os
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Literal, Optional, Sequence, Tuple, Union

import torch
try:
    from diffusers.utils import SAFE_WEIGHTS_INDEX_NAME as DIFFUSERS_SAFE_WEIGHTS_INDEX_NAME
    from diffusers.utils import SAFETENSORS_WEIGHTS_NAME as DIFFUSERS_SAFETENSORS_WEIGHTS_NAME
except Exception:
    DIFFUSERS_SAFE_WEIGHTS_INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"
    DIFFUSERS_SAFETENSORS_WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"
from torch import distributed as dist
from torch import nn
from tqdm import tqdm
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME
from transformers.utils.hub import cached_file, get_checkpoint_shard_files
from transformers.utils.import_utils import is_safetensors_available

from ..utils import logging
from ..utils.helper import empty_cache, get_dtype_size


if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.torch import save_file


if TYPE_CHECKING:
    from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    ModelAssets = Union[GenerationConfig, PretrainedConfig, PreTrainedTokenizer, ProcessorMixin]


logger = logging.get_logger(__name__)


@contextmanager
def init_empty_weights():
    """
    A context manager under which models are initialized with all parameters on the meta device.

    Borrowed from: https://github.com/huggingface/accelerate/blob/v1.0.0rc1/src/accelerate/big_modeling.py#L57
    """
    old_register_parameter = nn.Module.register_parameter

    def register_empty_parameter(module: "nn.Module", name: str, param: "nn.Parameter"):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to("meta"), **kwargs)

    try:
        nn.Module.register_parameter = register_empty_parameter
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter


@dataclass
class StateDictIterator:
    filepath: str
    prefix: str = ''

    def __iter__(self) -> Generator[Tuple[str, "torch.Tensor"], None, None]:
        if self.filepath.endswith(".safetensors"):
            with safe_open(self.filepath, framework="pt", device="cpu") as f:
                for key in f.keys():
                    yield key, f.get_tensor(key)

        else:
            state_dict = torch.load(self.filepath, map_location="cpu", weights_only=True, mmap=True)
            for key in state_dict.keys():
                yield key, state_dict[key]


def _load_state_dict(weights_path: str, **kwargs) -> List["StateDictIterator"]:
    """
    Loads (sharded) state dict in transformers' format.
    """
    cache_kwargs = {"_raise_exceptions_for_missing_entries": False, **kwargs}
    resolved_weight_file = cached_file(weights_path, SAFE_WEIGHTS_NAME, **cache_kwargs)
    if resolved_weight_file:
        return [StateDictIterator(resolved_weight_file)]

    resolved_weight_file = cached_file(weights_path, SAFE_WEIGHTS_INDEX_NAME, **cache_kwargs)
    if resolved_weight_file:
        shard_files, _ = get_checkpoint_shard_files(weights_path, resolved_weight_file, **kwargs)
        return [StateDictIterator(shard_file) for shard_file in shard_files]

    resolved_weight_file = cached_file(weights_path, DIFFUSERS_SAFETENSORS_WEIGHTS_NAME, **cache_kwargs)
    if resolved_weight_file:
        return [StateDictIterator(resolved_weight_file)]

    resolved_weight_file = cached_file(weights_path, DIFFUSERS_SAFE_WEIGHTS_INDEX_NAME, **cache_kwargs)
    if resolved_weight_file:
        shard_files, _ = get_checkpoint_shard_files(weights_path, resolved_weight_file, **kwargs)
        return [StateDictIterator(shard_file) for shard_file in shard_files]

    resolved_weight_file = cached_file(weights_path, WEIGHTS_NAME, **cache_kwargs)
    if resolved_weight_file:
        return [StateDictIterator(resolved_weight_file)]

    resolved_weight_file = cached_file(weights_path, WEIGHTS_INDEX_NAME, **cache_kwargs)
    if resolved_weight_file:
        shard_files, _ = get_checkpoint_shard_files(weights_path, resolved_weight_file, **kwargs)
        return [StateDictIterator(shard_file) for shard_file in shard_files]

    raise ValueError(f"Cannot find checkpoint files in {weights_path}.")


def _find_submodule(module: "nn.Module", name: str) -> Tuple["nn.Module", str]:
    """
    Finds the leaf module according to the name.
    """
    pieces = name.split(".")
    for piece in pieces[:-1]:
        if not hasattr(module, piece):
            raise ValueError(f"Cannot find {piece} in {module}.")

        module = getattr(module, piece)

    return module, pieces[-1]


def _dispatch_parameter(
    module: "nn.Module",
    name: str,
    tensor: "torch.Tensor",
    dtensor_factory: Optional[Callable[["torch.Tensor", Any, Any], "torch.Tensor"]] = None,
) -> None:
    """
    Assigns parameter to an empty model.

    NOTE: FSDP module must use in-place operators.
    """
    module, name = _find_submodule(module, name)
    orig_tensor = module._parameters[name].data
    tensor = tensor.to(orig_tensor)
    if hasattr(orig_tensor, "device_mesh"):  # dtensor
        if orig_tensor.device.type == "cpu":
            raise ValueError("Cannot load dtensor on CPU.")

        device_mesh = getattr(orig_tensor, "device_mesh")
        placements = getattr(orig_tensor, "placements")
        module._parameters[name].data.copy_(dtensor_factory(tensor, device_mesh, placements))
    else:  # not dtensor
        module._parameters[name].data.copy_(tensor)


def _dispatch_buffer(
    module: "nn.Module",
    name: str,
    buffer: "torch.Tensor",
) -> None:
    """
    Assigns buffer to an empty model.
    """
    module, name = _find_submodule(module, name)
    orig_tensor = module._buffers[name].data
    module._buffers[name] = buffer.to(orig_tensor)


def _init_parameter(
    module: "nn.Module",
    name: str,
    init_func: Optional[Callable[["nn.Module"], None]] = None,
) -> None:
    """
    Initializes parameter in model.
    """
    pieces = name.split(".")
    found_init_func = init_func

    for piece in pieces[:-1]:
        if not hasattr(module, piece):
            raise ValueError(f"Cannot find {piece} in {module}.")

        if found_init_func is None and hasattr(module, "_init_weights"):
            found_init_func = getattr(module, "_init_weights")

        module = getattr(module, piece)

    if found_init_func is None:
        print(module)
        raise ValueError(f"Cannot retrieve `_init_weights` function in the parents of {module}.")

    module.apply(found_init_func)


def _reset_gaussian_decoder(model: "nn.Module", pending_names: Sequence[str]) -> None:
    """Re-apply the GaussianDream decoder's own construction-time initialization.

    The generic ``_init_gaussian_weights`` kaiming init overrode the official
    small ``xavier(gain=0.01)`` head init and never touched ``nn.GroupNorm``,
    leaving those affine parameters at whatever ``to_empty()`` produced.
    """
    for module_name, module in model.named_modules():
        is_decoder = module_name.split(".")[-1] == "gaussian_decoder" or type(module).__name__ == "GaussianParameterDecoder"
        if not is_decoder:
            continue
        reset = getattr(module, "reset_parameters", None)
        if not callable(reset):
            raise TypeError(f"{module_name} must implement reset_parameters()")
        reset()
        prefix = f"{module_name}." if module_name else "gaussian_decoder."
        tensors = sum(name.startswith(prefix) for name in pending_names)
        logger.info_rank0(
            "%s Gaussian decoder %s: official construction-time init re-applied (%d tensors)",
            WEIGHT_CHECK_TAG, module_name, tensors,
        )


def _init_gaussian_weights(module: "nn.Module") -> None:
    """Default-style initialization for newly added Gaussian modules."""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.MultiheadAttention):
        if module.in_proj_weight is not None:
            nn.init.xavier_uniform_(module.in_proj_weight)
        if module.in_proj_bias is not None:
            nn.init.zeros_(module.in_proj_bias)
    if isinstance(getattr(module, "query_tokens", None), nn.Parameter):
        nn.init.normal_(module.query_tokens, mean=0.0, std=0.02)


def _get_weight_loader(model) -> "VLAWeightLoader":
    from .loader import VLAWeightLoader
    loader_factory = getattr(type(model), "get_weight_loader", None)
    if loader_factory is None:
        raise TypeError(
            f"{type(model).__name__} must implement the classmethod `get_weight_loader()` "
            f"that returns a VLAWeightLoader instance."
        )
    return loader_factory()


# [SIDA++] Weight-loading verification shared by the base VLA and the Gaussian extension.
WEIGHT_CHECK_TAG = "[WEIGHT-LOAD-CHECK]"
_OFFICIAL_WEIGHTS_LABEL = "LingBot official checkpoint"
_PRETRAINED_EXTERNAL_MARKERS = (("gaussian_vggt.", "VGGT (pretrained, frozen)"),)
_NEW_PARAM_MARKERS = (
    ("gaussian_3d_adapter.", "Gaussian 3D adapter (new)"),
    ("gaussian_decoder.", "Gaussian decoder (new)"),
    (".lora_A.", "LoRA adapters (new)"),
    (".lora_B.", "LoRA adapters (new)"),
)


def _is_main_process() -> bool:
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def _classify_param(name: str) -> str:
    for marker, label in _PRETRAINED_EXTERNAL_MARKERS + _NEW_PARAM_MARKERS:
        if marker in name:
            return label
    return _OFFICIAL_WEIGHTS_LABEL


def _relative_error(current: "torch.Tensor", reference: "torch.Tensor") -> float:
    with torch.no_grad():
        current = current.detach().float()
        reference = reference.detach().to(device=current.device, dtype=torch.float32)
        scale = reference.abs().sum().clamp_min(1e-12)
        return float(((current - reference).abs().sum() / scale).item())


def _dispatch_relative_error(model: "nn.Module", name: str, reference: "torch.Tensor") -> Optional[float]:
    """Compare a checkpoint tensor with the parameter it was just copied into."""
    try:
        module, leaf = _find_submodule(model, name)
        current = module._parameters[leaf]
    except Exception:
        return None
    if current is None or current.is_meta or hasattr(current.data, "device_mesh"):
        return None
    return _relative_error(current, reference)


def _reload_pretrained_external_modules(model: "nn.Module", pending_names: set) -> Dict[str, dict]:
    """Load pretrained frozen modules once the model has left the meta device.

    ``to_empty()`` discards the weights such modules loaded in ``__init__`` under
    ``init_empty_weights()``, where ``load_state_dict`` is a no-op. Modules expose
    ``load_pretrained_weights()`` to reload and self-verify after materialization.
    """
    checks: Dict[str, dict] = {}
    for module_name, module in model.named_modules():
        reload_fn = getattr(module, "load_pretrained_weights", None)
        if not callable(reload_fn):
            continue
        result = reload_fn()
        prefix = f"{module_name}." if module_name else ""
        for name in list(pending_names):
            if name.startswith(prefix):
                pending_names.remove(name)
        checks[module_name or type(module).__name__] = result
        logger.info_rank0(
            "%s module %s: loaded %s/%s tensors, sampled %s, max_rel_err=%.3e, source=%s, result=%s",
            WEIGHT_CHECK_TAG,
            module_name or type(module).__name__,
            result.get("loaded"),
            result.get("total"),
            result.get("checked"),
            float(result.get("max_rel_err", 1.0)),
            result.get("source"),
            "OK" if result.get("ok") else "FAIL",
        )
    return checks


def _log_weight_load_report(
    model: "nn.Module",
    weights_path: str,
    loaded_names: Sequence[str],
    pending_names: Sequence[str],
    dispatch_errors: Dict[str, float],
    module_checks: Dict[str, dict],
) -> bool:
    """Summarize which weights were loaded, re-initialized, or left unloaded."""
    loaded, pending = set(loaded_names), set(pending_names)
    groups: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    check_enabled = _is_main_process()
    for name, param in model.named_parameters():
        stats = groups.setdefault(
            _classify_param(name),
            {"params": 0, "tensors": 0, "from_checkpoint": 0, "missing": 0, "meta": 0, "nonfinite": 0, "all_zero": 0, "checked": 0, "max_rel_err": 0.0},
        )
        stats["params"] += param.numel()
        stats["tensors"] += 1
        if name in loaded:
            stats["from_checkpoint"] += 1
        if name in pending:
            stats["missing"] += 1
        if param.is_meta:
            stats["meta"] += 1
        elif check_enabled:
            with torch.no_grad():
                if not bool(torch.isfinite(param).all()):
                    stats["nonfinite"] += 1
                elif bool((param == 0).all()):
                    stats["all_zero"] += 1
        error = dispatch_errors.get(name)
        if error is not None:
            stats["checked"] += 1
            stats["max_rel_err"] = max(stats["max_rel_err"], error)

    external_labels = {label for _, label in _PRETRAINED_EXTERNAL_MARKERS}
    overall = True
    lines = [
        "=" * 100,
        f"{WEIGHT_CHECK_TAG} base checkpoint: {weights_path}",
        f"{WEIGHT_CHECK_TAG} {'component':<42}{'params':>14}{'tensors':>9}{'loaded':>8}{'missing':>8}{'chk':>5}{'zero':>6}{'bad':>5}  max_err  status",
    ]
    for label, stats in groups.items():
        is_pretrained = label == _OFFICIAL_WEIGHTS_LABEL or label in external_labels
        if is_pretrained:
            ok = stats["missing"] == 0 and stats["meta"] == 0 and stats["nonfinite"] == 0 and stats["max_rel_err"] < 0.05
            if label in external_labels:
                ok = ok and any(result.get("ok") for result in module_checks.values())
            status = "all from checkpoint" if ok else "NOT fully loaded"
        else:
            ok = stats["meta"] == 0 and stats["nonfinite"] == 0 and stats["all_zero"] < stats["tensors"] and stats["max_rel_err"] < 0.05
            status = "fresh init (by design)" if ok else "INVALID init"
        overall = overall and ok
        bad = stats["meta"] + stats["nonfinite"]
        lines.append(
            f"{WEIGHT_CHECK_TAG} {label:<42}{stats['params']:>14,}{stats['tensors']:>9}{stats['from_checkpoint']:>8}"
            f"{stats['missing']:>8}{stats['checked']:>5}{stats['all_zero']:>6}{bad:>5}  {stats['max_rel_err']:.1e}  {'OK' if ok else 'FAIL'} - {status}"
        )
    lines.append(f"{WEIGHT_CHECK_TAG} RESULT: {'PASS' if overall else 'FAIL'}")
    lines.append("=" * 100)
    logger.info_rank0("\n".join(lines))
    return overall


def log_external_model_check(label: str, module: Optional["nn.Module"], source: Optional[str] = None) -> bool:
    """Verify a standalone frozen teacher (MoGe/MoRGBD/DINO) holds real weights."""
    if module is None:
        logger.info_rank0("%s %s: DISABLED (not built in this run)", WEIGHT_CHECK_TAG, label)
        return True
    params = list(module.parameters())
    meta = sum(1 for param in params if param.is_meta)
    nonfinite = all_zero = 0
    l1_sum = 0.0
    for param in params:
        if param.is_meta:
            continue
        with torch.no_grad():
            if not bool(torch.isfinite(param).all()):
                nonfinite += 1
            elif bool((param == 0).all()):
                all_zero += 1
            else:
                l1_sum += float(param.detach().float().abs().sum().item())
    ok = bool(params) and meta == 0 and nonfinite == 0 and all_zero < len(params) and l1_sum > 0.0
    logger.info_rank0(
        "%s external %s: tensors=%d meta=%d nonfinite=%d all_zero=%d l1_checksum=%.4e source=%s result=%s",
        WEIGHT_CHECK_TAG, label, len(params), meta, nonfinite, all_zero, l1_sum, source, "OK" if ok else "FAIL",
    )
    return ok
# [SIDA--]


@torch.no_grad()
def load_model_weights(
    model: Union["nn.Module", "PreTrainedModel"],
    weights_path: str,
    init_device: Literal["cpu", "cuda"] = "cuda",
    dtensor_factory: Optional[Callable[["torch.Tensor", Any, Any], "torch.Tensor"]] = None,
    load_vlm_only: bool = False,
    post_training: bool = False,
    incremental_training: bool = False,
    adanorm_time: bool = False,
) -> None:
    """
    Loads pre-trained model states in transformers' format.
    """
    weight_loader = _get_weight_loader(model)

    buffer_dict     = {name: buffer.clone() for name, buffer in model.named_buffers()}
    parameter_names = {name for name, _ in model.named_parameters()}

    # Get model parameter names
    vlm_param_names  = weight_loader.get_vlm_para_fullnames(model)
    print(f'====vlm contains {len(vlm_param_names)} paras=====')

    # [SIDA++] Track what was actually written from the checkpoint.
    check_enabled = _is_main_process()
    loaded_names: List[str] = []
    dispatch_errors: Dict[str, float] = {}
    # [SIDA--]

    model.to_empty(device=init_device)

    if post_training:
        logger.info_rank0(">>> Doing Post-Training now.")
    elif load_vlm_only:
        logger.info_rank0(">>> Doing Pre-Training now.")

    # Load checkpoint weight iterators
    state_dict_iterators = _load_state_dict(weights_path)
    for state_dict_iterator in tqdm(
        state_dict_iterators, desc="Loading checkpoint shards", disable=int(os.getenv("LOCAL_RANK", "-1")) > 0
    ):
        for ckpt_key, tensor in state_dict_iterator:
            name = weight_loader.map_ckpt_key(ckpt_key, load_vlm_only=load_vlm_only, post_training=post_training)
            if name is None:
                continue
            # [SIDA++] PEFT wraps LoRA-targeted Linear layers and moves their
            # pretrained weights under ``base_layer``.  Keep the official
            # checkpoint key format loadable after VLM-only LoRA injection.
            if name not in buffer_dict and name not in parameter_names:
                for suffix in (".weight", ".bias"):
                    if name.endswith(suffix):
                        base_name = name[: -len(suffix)] + ".base_layer" + suffix
                        if base_name in parameter_names:
                            name = base_name
                            break
            # [SIDA--]
            if name in buffer_dict:
                buffer_dict[name] = tensor.clone()
            elif name in parameter_names:
                if incremental_training:
                    try:
                        _dispatch_parameter(model, name, tensor, dtensor_factory)
                        parameter_names.remove(name)
                    except Exception as exc:
                        logger.info_rank0(f">>>The {name} weight need to be reinitialized: {exc}")
                else:
                    parameter_names.remove(name)
                    _dispatch_parameter(model, name, tensor, dtensor_factory)
                    # [SIDA++] Confirm the copy landed in live model memory.
                    loaded_names.append(name)
                    # Sample one tensor in twenty: the comparison allocates FP32
                    # temporaries on the device and load-time VRAM is tight.
                    if check_enabled and len(loaded_names) % 20 == 0:
                        error = _dispatch_relative_error(model, name, tensor)
                        if error is not None:
                            dispatch_errors[name] = error
                    # [SIDA--]
            else:
                if post_training:
                    # [SIDA++] A smoke run may intentionally disable the
                    # depth/video alignment heads while reusing a checkpoint
                    # trained with them. Their checkpoint tensors are expected
                    # extras; keep strict validation for every other key.
                    if any(
                        marker in name
                        for marker in ("depth_align", "current_video_align", "future_video_align", "shared_task_proj")
                    ):
                        logger.info_rank0(f"Ignoring disabled alignment head checkpoint key: {name}")
                        continue
                    error_msg = f"Unexpected key '{name}' found in state dict during Post-Training. This is not allowed!!!"
                    logger.info_rank0(error_msg)
                    raise KeyError(error_msg)
                logger.info_rank0(f"Unexpected key in state dict: {name}.")

        del state_dict_iterator
        empty_cache()

    for name, buffer in buffer_dict.items():
        _dispatch_buffer(model, name, buffer)

    # [SIDA++] Frozen pretrained modules (VGGT) must be reloaded after
    # to_empty(); otherwise they are randomly re-initialized below.
    module_checks = _reload_pretrained_external_modules(model, parameter_names)
    # The Gaussian decoder replicates GaussianDream, so re-run its own
    # construction-time initialization rather than the generic one below.
    _reset_gaussian_decoder(model, parameter_names)
    # [SIDA--]

    if post_training:
        # [SIDA++] Newly added Gaussian/LoRA modules intentionally have no
        # tensors in the official LingBot checkpoint; initialize those modules
        # locally while retaining strict validation for all original weights.
        allowed_missing_markers = (
            "gaussian_vggt.",
            "gaussian_3d_adapter.",
            "gaussian_decoder.",
            "gaussian_renderer.",
            ".lora_A.",
            ".lora_B.",
        )
        strict_missing = {
            name for name in parameter_names
            if not any(marker in name for marker in allowed_missing_markers)
        }
        if parameter_names - strict_missing:
            logger.info_rank0(
                "Initializing new Gaussian/LoRA parameters absent from the "
                "official checkpoint."
            )
        assert len(strict_missing) == 0, f"Missing {strict_missing} during Post-Training. This is not allowed!!!"
        # [SIDA--]
        if adanorm_time:
            logger.info_rank0(">>> Parameters in AdaNorm has been ZERO initialized.")
            exclude_keywords = [
                "input_layernorm.gamma_beta_gate",
                "post_attention_layernorm.gamma_beta_gate",
                "norm.gamma_beta_gate",
                "input_layernorm.gamma",
                "post_attention_layernorm.gamma",
                "norm.gamma",
                "input_layernorm.beta",
                "post_attention_layernorm.beta",
                "norm.beta",
                "input_layernorm.gate",
                "post_attention_layernorm.gate",
                "norm.gate",
                "depth_align_embs",
                "current_video_align_embs",
                "future_video_align_embs",
            ]
        gaussian_missing_markers = (
            "gaussian_3d_adapter.",
            "gaussian_decoder.",
        )
        for name in parameter_names:
            # Newly added Gaussian modules are absent from the official
            # checkpoint. They must be re-initialized with normal random
            # weights; leaving them after to_empty() keeps them at zero and
            # freezes the renderer at constant output.
            if "gaussian_decoder." in name:
                # Initialized above through the decoder's reset_parameters().
                continue
            if any(marker in name for marker in gaussian_missing_markers):
                _init_parameter(model, name, init_func=_init_gaussian_weights)
            elif not adanorm_time:
                _init_parameter(model, name)
            elif not any(keyword in name for keyword in exclude_keywords):
                _init_parameter(model, name)

    # we should tie embeddings after loading weights because to_empty() leads to untied weights,
    # except for fsdp1 (custom init) and fsdp2 (swap tensor) contexts.
    if getattr(model.config, "tie_word_embeddings", True):
        try:
            input_embeddings = model.get_input_embeddings()
            output_embeddings = model.get_output_embeddings()
            output_embeddings._parameters["weight"] = input_embeddings._parameters["weight"]
        except Exception as e:
            logger.info_rank0(f"Failed to tie embeddings: {e}")

    # [SIDA++] One grep-able summary of what is loaded, fresh, or missing.
    _log_weight_load_report(model, weights_path, loaded_names, parameter_names, dispatch_errors, module_checks)
    # [SIDA--]


def _get_shard_info(
    state_dict: Dict[str, "torch.Tensor"],
    save_dtype: Optional[Union[str, "torch.dtype"]],
    shard_size: int,
    safe_serialization: bool,
) -> Tuple[bool, int, Dict[str, str]]:
    """
    Gets the shard information, should be executed at rank 0.
    """
    current_size, total_size = 0, 0
    current_shard, shard_list = [], []
    for name, tensor in state_dict.items():
        if isinstance(save_dtype, str):
            dtype = getattr(torch, save_dtype)
        elif isinstance(save_dtype, torch.dtype):
            dtype = save_dtype
        else:
            dtype = tensor.dtype
        tensor_size = tensor.numel() * get_dtype_size(dtype)  # dtensor's numel == tensor's numel
        if current_size != 0 and current_size + tensor_size > shard_size:
            total_size += current_size
            shard_list.append(current_shard)
            current_size = 0
            current_shard = []

        current_size += tensor_size
        current_shard.append(name)

    if current_size != 0:
        total_size += current_size
        shard_list.append(current_shard)

    weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
    num_shards = len(shard_list)
    weight_map = OrderedDict()
    is_sharded = None
    if num_shards == 1:
        is_sharded = False
        for name in shard_list[0]:
            weight_map[name] = weights_name
    else:
        is_sharded = True
        for shard_idx, shard in enumerate(shard_list):
            prefix, extension = weights_name.rsplit(".", maxsplit=1)
            file_name = f"{prefix}-{shard_idx + 1:05d}-of-{num_shards:05d}.{extension}"
            for name in shard:
                weight_map[name] = file_name

    return is_sharded, total_size, weight_map


def _save_state_dict(
    state_dict: Dict[str, "torch.Tensor"],
    path_to_save: "os.PathLike",
    safe_serialization: bool,
) -> None:
    """
    Save function.
    """
    if os.path.exists(path_to_save):
        os.remove(path_to_save)
    if safe_serialization:
        save_file(state_dict, path_to_save, metadata={"format": "pt"})
    else:
        torch.save(state_dict, path_to_save)


@torch.no_grad()
def save_model_weights(
    output_dir: Union[str, "os.PathLike"],
    state_dict: Dict[str, "torch.Tensor"],
    global_rank: Optional[int] = None,
    save_dtype: Optional[Union[str, "torch.dtype"]] = "bfloat16",
    shard_size: int = 5_000_000_000,
    safe_serialization: bool = True,
    model_assets: Optional[Sequence["ModelAssets"]] = None,
) -> None:
    """
    Saves full model weights. The model parameters should be either tensor or dtensor.

    If global_rank is given, it will assume it is executed on all ranks.
    """

    os.makedirs(output_dir, exist_ok=True)
    is_sharded, total_size, weight_map = _get_shard_info(state_dict, save_dtype, shard_size, safe_serialization)
    full_state_dict = OrderedDict()
    prev_file_name = None
    for name, tensor in state_dict.items():
        if hasattr(tensor.data, "full_tensor"):  # dtensor
            tensor = tensor.data.full_tensor()
        else:
            tensor = tensor.data

        if save_dtype:
            tensor = tensor.to(dtype=getattr(torch, save_dtype) if isinstance(save_dtype, str) else save_dtype)

        if prev_file_name is not None and weight_map[name] != prev_file_name:
            if global_rank is None or global_rank == 0:
                _save_state_dict(full_state_dict, os.path.join(output_dir, prev_file_name), safe_serialization)
                full_state_dict = OrderedDict()

            empty_cache()
            if global_rank is not None and dist.is_initialized():  # avoid process hanging
                torch.cuda.synchronize()
                dist.barrier()

        if global_rank is None or global_rank == 0:
            full_state_dict[name] = tensor.detach().cpu()

        prev_file_name = weight_map[name]
        del tensor

    if global_rank is None or global_rank == 0:
        if len(full_state_dict):
            _save_state_dict(full_state_dict, os.path.join(output_dir, prev_file_name), safe_serialization)

        if is_sharded:
            index = {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map,
            }

            index_file = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME
            with open(os.path.join(output_dir, index_file), "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)

            logger.info(f"Model weight splits saved in {output_dir}.")
        else:
            logger.info(f"Model weights saved at {os.path.join(output_dir, prev_file_name)}.")

        if model_assets is not None:
            for model_asset in model_assets:
                if hasattr(model_asset, "save_pretrained"):
                    model_asset.save_pretrained(output_dir)
                else:
                    logger.warning(f"Model asset {model_asset} should implement `save_pretrained`.")


def save_model_assets(output_dir: Union[str, "os.PathLike"], model_assets: Sequence["ModelAssets"]):
    for model_asset in model_assets:
        if hasattr(model_asset, "save_pretrained"):
            model_asset.save_pretrained(output_dir)
        else:
            logger.warning(f"Model asset {model_asset} should implement `save_pretrained`.")
