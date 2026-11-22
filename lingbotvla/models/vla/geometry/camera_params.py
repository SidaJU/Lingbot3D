"""Camera configuration and projection matrices for Gaussian rendering.

The numeric RobotWin calibration is intentionally kept outside Python code.
Populate the JSON from the pinned RoboTwin checkout before enabling rendering.
"""

# [SIDA++] add by SidaZhu for camera lookup/projection in the Gaussian extension.
# Purpose: adapt verified RoboTwin camera metadata to renderer matrices while
# remaining separate from LingBot official camera/model code. Original
# additions retain SidaZhu attribution and applicable rights, subject to the
# repository license and third-party notices.
# [SIDA--]

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch


# [SIDA++] add by SidaZhu for camera parameter lookup and validation
_DEFAULT_CONFIG = Path(__file__).resolve().parents[4] / "configs" / "robot_configs" / "robotwin_camera.json"
_CONFIG_CACHE: Dict[str, Mapping[str, Any]] = {}


def _config_path(config_path: Optional[str | os.PathLike[str]]) -> Path:
    configured = config_path or os.environ.get("LINGBOT_CAMERA_CONFIG")
    return Path(configured).expanduser().resolve() if configured else _DEFAULT_CONFIG


def _load_config(config_path: Optional[str | os.PathLike[str]]) -> Mapping[str, Any]:
    path = _config_path(config_path)
    key = str(path)
    if key not in _CONFIG_CACHE:
        if not path.exists():
            raise FileNotFoundError(
                f"Camera config not found: {path}. Populate it from the pinned RoboTwin revision."
            )
        with path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        if config.get("status") == "pending_source_verification":
            raise RuntimeError(
                f"Camera config {path} is a placeholder. Do not train before RoboTwin K/T values are verified."
            )
        _CONFIG_CACHE[key] = config
    return _CONFIG_CACHE[key]


def _projection_matrix(znear: float, zfar: float, fov_x: float, fov_y: float, device, dtype):
    tan_half_x = math.tan(fov_x / 2.0)
    tan_half_y = math.tan(fov_y / 2.0)
    matrix = torch.zeros((4, 4), device=device, dtype=dtype)
    matrix[0, 0] = 1.0 / tan_half_x
    matrix[1, 1] = 1.0 / tan_half_y
    matrix[3, 2] = 1.0
    matrix[2, 2] = zfar / (zfar - znear)
    matrix[2, 3] = -(zfar * znear) / (zfar - znear)
    return matrix


def get_camera_params_for_view(
    view_name: str,
    device: torch.device | str,
    batch_size: int,
    target_size: Optional[Tuple[int, int]] = None,
    world_to_cam: Optional[torch.Tensor] = None,
    config_path: Optional[str | os.PathLike[str]] = None,
) -> Dict[str, Any]:
    """Load one static RobotWin camera and build rasterizer matrices.

    The JSON convention is explicit: ``T_world_cam`` maps world coordinates
    into camera coordinates. For dynamic wrist cameras, pass ``world_to_cam``
    per batch; a static JSON extrinsic is intentionally not required.
    """

    config = _load_config(config_path)
    cameras = config.get("cameras", config)
    if view_name not in cameras:
        raise KeyError(f"Unknown camera {view_name!r}; available={sorted(cameras)}")
    spec = cameras[view_name]
    required = ("width", "height", "fx", "fy", "cx", "cy")
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(f"Camera {view_name!r} is missing fields: {missing}")

    width = float(spec["width"])
    height = float(spec["height"])
    fx, fy = float(spec["fx"]), float(spec["fy"])
    cx, cy = float(spec["cx"]), float(spec["cy"])
    if target_size is not None:
        target_width, target_height = map(float, target_size)
        sx, sy = target_width / width, target_height / height
        fx, cx = fx * sx, cx * sx
        fy, cy = fy * sy, cy * sy
        width, height = target_width, target_height

    dtype = torch.float32
    intrinsics = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy

    # [SIDA++] add by SidaZhu for static head / dynamic wrist camera poses
    raw_viewmatrix = world_to_cam if world_to_cam is not None else spec.get("T_world_cam")
    if raw_viewmatrix is None:
        raise ValueError(
            f"Camera {view_name!r} has no static T_world_cam. Pass a per-sample world_to_cam tensor "
            "for dynamic wrist cameras."
        )
    viewmatrix = torch.as_tensor(raw_viewmatrix, device=device, dtype=dtype)
    if viewmatrix.ndim == 2:
        if tuple(viewmatrix.shape) != (4, 4):
            raise ValueError(f"world_to_cam must have shape [4,4] or [B,4,4], got {tuple(viewmatrix.shape)}")
        viewmatrix = viewmatrix.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    elif viewmatrix.ndim == 3:
        if tuple(viewmatrix.shape[1:]) != (4, 4) or viewmatrix.shape[0] != batch_size:
            raise ValueError(f"world_to_cam must have shape [B,4,4], got {tuple(viewmatrix.shape)}")
    else:
        raise ValueError(f"world_to_cam must have shape [4,4] or [B,4,4], got {tuple(viewmatrix.shape)}")
    # [SIDA--]

    fov_x = 2.0 * math.atan(width / (2.0 * fx))
    fov_y = 2.0 * math.atan(height / (2.0 * fy))
    projection = _projection_matrix(0.01, 100.0, fov_x, fov_y, device, dtype)
    projmatrix = torch.matmul(projection.unsqueeze(0).expand(batch_size, -1, -1), viewmatrix)
    campos = torch.linalg.inv(viewmatrix)[:, :3, 3]

    return {
        "intrinsics": intrinsics,
        "viewmatrix": viewmatrix,
        "projmatrix": projmatrix,
        "tanfovx": math.tan(fov_x / 2.0),
        "tanfovy": math.tan(fov_y / 2.0),
        "campos": campos,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": int(width),
        "height": int(height),
    }
# [SIDA--]

