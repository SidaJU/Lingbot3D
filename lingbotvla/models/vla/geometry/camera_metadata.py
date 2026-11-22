"""RoboTwin per-frame camera metadata sidecar reader.

The raw RoboTwin pkl/HDF5 path stores camera calibration under the
``observation`` tree.  The official HDF5-to-LeRobot converter currently keeps
RGB/state/action fields only, so this module reads a small ``*.camera.npz``
sidecar produced by :mod:`experiment.robotwin.extract_camera_metadata`.
"""

# [SIDA++] add by SidaZhu for the RoboTwin camera sidecar extension.
# Purpose: preserve and validate per-frame K/T metadata outside LingBot's
# official dataset implementation. Original additions retain SidaZhu
# attribution and applicable rights, subject to the repository license and
# third-party notices.
# [SIDA--]

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import json
from pathlib import Path
from typing import Mapping

import numpy as np


# [SIDA++] add by SidaZhu for RoboTwin camera metadata canonicalization
CAMERA_NAME_ALIASES: dict[str, tuple[str, ...]] = {
    "camera_top": ("camera_top", "head_camera", "cam_high"),
    "camera_wrist_left": ("camera_wrist_left", "left_camera", "cam_left_wrist"),
    "camera_wrist_right": ("camera_wrist_right", "right_camera", "cam_right_wrist"),
}
_FIELDS = ("intrinsic_cv", "extrinsic_cv", "cam2world_gl")
# [SIDA--]


def _as_float_array(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    expected = (3, 3) if name.endswith("intrinsic_cv") else (4, 4)
    if array.ndim not in (2, 3) or array.shape[-2:] != expected or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite [T,{expected[0]},{expected[1]}] or {expected}, got {array.shape}")
    return array


# [SIDA++] add by SidaZhu for loading validated per-episode sidecars
@dataclass(frozen=True)
class CameraFrameMetadata:
    """Calibration matrices for one RoboTwin episode."""

    cameras: Mapping[str, Mapping[str, np.ndarray]]
    frame_count: int | None = None
    source_hdf5: str | None = None
    image_sizes: Mapping[str, tuple[int, int]] | None = None

    def __post_init__(self) -> None:
        if not self.cameras:
            raise ValueError("camera sidecar contains no cameras")
        inferred_count: int | None = None
        for camera_name, fields in self.cameras.items():
            missing = set(_FIELDS) - set(fields)
            if missing:
                raise ValueError(f"{camera_name} is missing camera fields: {sorted(missing)}")
            for field_name in _FIELDS:
                value = _as_float_array(fields[field_name], name=f"{camera_name}.{field_name}")
                if value.ndim == 3:
                    count = int(value.shape[0])
                    if inferred_count is None:
                        inferred_count = count
                    elif inferred_count != count:
                        raise ValueError(
                            f"camera frame counts disagree: {camera_name}.{field_name} has {count}, "
                            f"expected {inferred_count}"
                        )
        if self.frame_count is not None and inferred_count is not None and self.frame_count != inferred_count:
            raise ValueError(f"sidecar frame_count={self.frame_count} disagrees with matrices={inferred_count}")

    @property
    def num_frames(self) -> int | None:
        if self.frame_count is not None:
            return self.frame_count
        for fields in self.cameras.values():
            matrix = fields["intrinsic_cv"]
            if matrix.ndim == 3:
                return int(matrix.shape[0])
        return None

    def for_frame(self, frame_index: int, camera_name: str) -> dict[str, np.ndarray]:
        """Return matrices for ``frame_index``; static matrices are broadcast."""

        if camera_name not in self.cameras:
            raise KeyError(f"unknown camera {camera_name!r}; available={sorted(self.cameras)}")
        if frame_index < 0:
            raise IndexError(f"frame_index must be non-negative, got {frame_index}")
        if self.num_frames is not None and frame_index >= self.num_frames:
            raise IndexError(f"frame_index={frame_index} outside episode of length {self.num_frames}")
        result: dict[str, np.ndarray] = {}
        for field_name in _FIELDS:
            value = np.asarray(self.cameras[camera_name][field_name], dtype=np.float32)
            if value.ndim == 3:
                if frame_index >= value.shape[0]:
                    raise IndexError(
                        f"frame_index={frame_index} is outside {camera_name} ({value.shape[0]} frames)"
                    )
                value = value[frame_index]
            result[field_name] = value.copy()
        # RoboTwin's extrinsic_cv is the OpenCV world-to-camera matrix.  Keep
        # this explicit alias so the Gaussian renderer never guesses a pose
        # convention from cam2world_gl.
        result["world_to_cam"] = result["extrinsic_cv"]
        return result


def _metadata_from_npz(npz_path: Path) -> CameraFrameMetadata:
    with np.load(npz_path, allow_pickle=False) as archive:
        meta_json = archive.get("__meta_json__")
        metadata = json.loads(str(meta_json.item())) if meta_json is not None else {}
        cameras: dict[str, dict[str, np.ndarray]] = {}
        for key in archive.files:
            if key == "__meta_json__":
                continue
            if "__" not in key:
                raise ValueError(f"invalid camera sidecar key {key!r} in {npz_path}")
            camera_name, field_name = key.split("__", 1)
            if field_name not in _FIELDS:
                raise ValueError(f"unsupported camera field {field_name!r} in {npz_path}")
            cameras.setdefault(camera_name, {})[field_name] = archive[key]
    return CameraFrameMetadata(
        cameras=cameras,
        frame_count=metadata.get("frame_count"),
        source_hdf5=metadata.get("source_hdf5"),
        image_sizes=metadata.get("image_sizes"),
    )


def load_camera_metadata(sidecar_path: str | Path) -> CameraFrameMetadata:
    """Load and validate one ``*.camera.npz`` sidecar."""

    path = Path(sidecar_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"camera metadata sidecar does not exist: {path}")
    if path.suffix != ".npz":
        raise ValueError(f"camera metadata sidecar must be .npz, got {path}")
    return _metadata_from_npz(path)


def sidecar_path_for_episode(sidecar_root: str | Path, episode_path: str | Path) -> Path:
    """Resolve an HDF5 episode path to the matching sidecar path."""

    root = Path(sidecar_root).expanduser()
    episode = Path(episode_path)
    return root / f"{episode.stem}.camera.npz"


def resolve_camera_metadata_root(sidecar_root: str | Path, dataset_id: str | Path | None = None) -> Path:
    """Resolve a single-dataset root or a multi-dataset subdirectory."""

    root = Path(sidecar_root).expanduser()
    if dataset_id is not None:
        candidate = root / Path(dataset_id).name
        if (candidate / "manifest.json").is_file():
            return candidate
    if (root / "manifest.json").is_file():
        return root
    dataset_hint = f" for dataset {dataset_id!s}" if dataset_id is not None else ""
    raise FileNotFoundError(
        f"camera metadata manifest not found in {root}{dataset_hint}; "
        "expected manifest.json at the dataset root or its named subdirectory"
    )


class CameraMetadataIndex:
    """Resolve sidecars by LeRobot episode index using ``manifest.json``."""

    def __init__(self, root: str | Path, manifest_name: str = "manifest.json") -> None:
        self.root = Path(root).expanduser()
        manifest_path = self.root / manifest_name
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"camera metadata manifest does not exist: {manifest_path}; "
                "run extract_camera_metadata.py first"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != "lingbot.robottwin.camera.manifest.v1":
            raise ValueError(f"unsupported camera metadata manifest format in {manifest_path}")
        if manifest.get("training_aligned") is not True:
            raise ValueError("sidecars are not training aligned; regenerate with --episode-map")
        entries = manifest.get("episodes")
        if not isinstance(entries, list):
            raise ValueError(f"camera metadata manifest has no episode list: {manifest_path}")
        self._sidecars = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict) or "sidecar" not in entry:
                raise ValueError(f"invalid episode entry {index} in {manifest_path}")
            if entry.get("episode_index", index) != index:
                raise ValueError(
                    f"camera metadata manifest episode_index mismatch at position {index}: "
                    f"{entry.get('episode_index')}"
                )
            self._sidecars.append(self.root / entry["sidecar"])
        self._cache: OrderedDict[int, CameraFrameMetadata] = OrderedDict()

    def episode(self, episode_index: int) -> CameraFrameMetadata:
        if episode_index < 0 or episode_index >= len(self._sidecars):
            raise IndexError(f"episode_index={episode_index} outside sidecar index of size {len(self._sidecars)}")
        if episode_index not in self._cache:
            self._cache[episode_index] = load_camera_metadata(self._sidecars[episode_index])
            if len(self._cache) > 4:
                self._cache.popitem(last=False)
        self._cache.move_to_end(episode_index)
        return self._cache[episode_index]

    def for_frame(self, episode_index: int, frame_index: int, camera_name: str) -> dict[str, np.ndarray]:
        return self.episode(episode_index).for_frame(frame_index, camera_name)

    def for_resized_frame(self, episode_index: int, frame_index: int, camera_name: str,
                          target_hw: tuple[int, int]) -> dict[str, np.ndarray]:
        episode = self.episode(episode_index)
        result = episode.for_frame(frame_index, camera_name)
        if episode.image_sizes is None or camera_name not in episode.image_sizes:
            raise ValueError("sidecar has no RGB image size; regenerate it before resizing intrinsics")
        height, width = episode.image_sizes[camera_name]
        target_h, target_w = target_hw
        if min(height, width, target_h, target_w) <= 0:
            raise ValueError("source and target image sizes must be positive")
        result["intrinsic_cv"][0, :] *= target_w / width
        result["intrinsic_cv"][1, :] *= target_h / height
        return result


def canonical_camera_name(raw_name: str) -> str:
    """Map a RoboTwin raw camera key to LingBot's canonical name."""

    for canonical, aliases in CAMERA_NAME_ALIASES.items():
        if raw_name in aliases:
            return canonical
    raise KeyError(f"unsupported RoboTwin camera name {raw_name!r}")
# [SIDA--]
