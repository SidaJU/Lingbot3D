"""Geometry utilities for the optional Gaussian 3D training branch."""

# [SIDA++] add by SidaZhu for the LingBot Gaussian 3D extension namespace.
# Purpose: export SidaZhu-authored geometry/data interfaces separately from
# LingBot official model implementations. Original additions retain SidaZhu
# attribution and applicable rights, subject to the repository license and
# third-party notices.
# [SIDA--]

from .camera_params import get_camera_params_for_view
from .camera_metadata import (
    CameraFrameMetadata,
    CameraMetadataIndex,
    canonical_camera_name,
    load_camera_metadata,
    resolve_camera_metadata_root,
    sidecar_path_for_episode,
)
from .gaussian_prefix import Gaussian3DPrefixAdapter
from .gaussian_decoder import GaussianParameterDecoder, GaussianParameters
from .gaussian_renderer import DifferentiableGaussianRenderer
from .render_loss import compute_rgb_render_loss

__all__ = [
    "CameraFrameMetadata",
    "CameraMetadataIndex",
    "Gaussian3DPrefixAdapter",
    "GaussianParameterDecoder",
    "GaussianParameters",
    "DifferentiableGaussianRenderer",
    "canonical_camera_name",
    "compute_rgb_render_loss",
    "get_camera_params_for_view",
    "load_camera_metadata",
    "resolve_camera_metadata_root",
    "sidecar_path_for_episode",
]
