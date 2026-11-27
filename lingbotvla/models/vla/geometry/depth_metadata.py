"""RoboTwin Depth Anything V2 sidecar reader.

Depth targets are generated offline from LeRobot RGB frames and stored next to
the camera metadata sidecars.  Training reads them directly by
``(episode_index, frame_index, camera_name)`` so no depth teacher runs in the
training loop.
"""

# [SIDA++] add by SidaZhu for offline DA2 Gaussian depth targets.
# Original additions retain SidaZhu attribution and applicable rights,
# subject to the repository license and third-party notices.
# [SIDA--]

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path

import numpy as np


DEPTH_MANIFEST_FORMAT = "lingbot.robottwin.depth.manifest.v1"
DEPTH_SIDECAR_FORMAT = "lingbot.robottwin.depth.v1"


def resolve_depth_metadata_root(sidecar_root: str | Path, dataset_id: str | Path | None = None) -> Path:
    """Resolve a shared depth root or a single-dataset depth root."""

    root = Path(sidecar_root).expanduser()
    if dataset_id is not None:
        candidate = root / Path(dataset_id).name
        if (candidate / "manifest.json").is_file():
            return candidate
    if (root / "manifest.json").is_file():
        return root
    dataset_hint = f" for dataset {dataset_id!s}" if dataset_id is not None else ""
    raise FileNotFoundError(
        f"depth metadata manifest not found in {root}{dataset_hint}; "
        "expected manifest.json at the dataset root or its named subdirectory"
    )


class DepthSidecarIndex:
    """Read validated per-episode DA2 depth maps from sidecars."""

    def __init__(self, root: str | Path, manifest_name: str = "manifest.json", cache_size: int = 2) -> None:
        self.root = Path(root).expanduser()
        manifest_path = self.root / manifest_name
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"depth metadata manifest does not exist: {manifest_path}; "
                "run tools/generate_depth_for_robotwin.py first"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != DEPTH_MANIFEST_FORMAT:
            raise ValueError(f"unsupported depth metadata manifest format in {manifest_path}")
        if manifest.get("training_aligned") is not True:
            raise ValueError("depth sidecars are not training aligned; regenerate them")
        entries = manifest.get("episodes")
        if not isinstance(entries, list):
            raise ValueError(f"depth metadata manifest has no episode list: {manifest_path}")

        self._sidecars: list[Path] = []
        self.frame_counts: list[int] = []
        for position, entry in enumerate(entries):
            if not isinstance(entry, dict) or "sidecar" not in entry:
                raise ValueError(f"invalid episode entry {position} in {manifest_path}")
            episode_index = int(entry.get("episode_index", position))
            if episode_index != position:
                raise ValueError(
                    f"depth manifest episode_index mismatch at position {position}: {episode_index}"
                )
            self._sidecars.append(self.root / entry["sidecar"])
            self.frame_counts.append(int(entry.get("frame_count", 0)))

        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self._cache_size = max(1, int(cache_size))

    def _load_episode(self, episode_index: int) -> dict[str, np.ndarray]:
        if episode_index < 0 or episode_index >= len(self._sidecars):
            raise IndexError(
                f"episode_index={episode_index} outside depth sidecar index of size {len(self._sidecars)}"
            )
        path = self._sidecars[episode_index]
        if not path.is_file():
            raise FileNotFoundError(f"depth sidecar does not exist: {path}")
        with np.load(path, allow_pickle=False) as archive:
            meta_value = archive.get("__meta_json__")
            meta = json.loads(str(meta_value.item())) if meta_value is not None else {}
            if meta.get("format") != DEPTH_SIDECAR_FORMAT:
                raise ValueError(f"unsupported depth sidecar format in {path}")
            expected_count = self.frame_counts[episode_index]
            result: dict[str, np.ndarray] = {}
            for key in archive.files:
                if key == "__meta_json__":
                    continue
                if "__" not in key:
                    raise ValueError(f"invalid depth sidecar key {key!r} in {path}")
                camera_name, field_name = key.split("__", 1)
                if field_name != "depth":
                    raise ValueError(f"unsupported depth field {field_name!r} in {path}")
                depth = np.asarray(archive[key], dtype=np.float32)
                if depth.ndim != 3:
                    raise ValueError(f"{key} in {path} must be [N,H,W], got {depth.shape}")
                if expected_count and depth.shape[0] != expected_count:
                    raise ValueError(
                        f"{key} in {path} has {depth.shape[0]} frames, expected {expected_count}"
                    )
                if not np.isfinite(depth).all():
                    raise ValueError(f"{key} in {path} contains non-finite depth values")
                result[camera_name] = depth
        if not result:
            raise ValueError(f"depth sidecar contains no cameras: {path}")
        return result

    def episode(self, episode_index: int) -> dict[str, np.ndarray]:
        if episode_index not in self._cache:
            self._cache[episode_index] = self._load_episode(episode_index)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        self._cache.move_to_end(episode_index)
        return self._cache[episode_index]

    def for_frame(self, episode_index: int, frame_index: int, camera_name: str) -> np.ndarray:
        episode = self.episode(episode_index)
        if camera_name not in episode:
            raise KeyError(
                f"unknown depth camera {camera_name!r}; available={sorted(episode)}"
            )
        depth = episode[camera_name]
        if frame_index < 0 or frame_index >= depth.shape[0]:
            raise IndexError(
                f"frame_index={frame_index} outside {camera_name} depth sidecar of length {depth.shape[0]}"
            )
        return depth[frame_index]
