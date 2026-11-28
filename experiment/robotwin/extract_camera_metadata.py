"""Extract RoboTwin per-frame camera calibration into compact sidecars.

This operates on the already generated RoboTwin raw HDF5 files.  It does not
need embodiment assets: the recorded observation tree contains the exact
intrinsic and pose matrices used for each frame.
"""

# [SIDA++] add by SidaZhu for RoboTwin camera metadata sidecar extraction.
# Purpose: preserve verified HDF5 K/T data for the Gaussian extension, separate
# from LingBot official dataset conversion. Original additions retain SidaZhu
# attribution and applicable rights, subject to the repository license and
# third-party notices.
# [SIDA--]

from __future__ import annotations

import argparse
from pathlib import Path
import json
import fnmatch
import os
import io

import h5py
import numpy as np
from PIL import Image


# [SIDA++] add by SidaZhu for RoboTwin HDF5 camera metadata extraction
CAMERA_CANDIDATES: dict[str, tuple[str, ...]] = {
    "camera_top": ("head_camera", "camera_top", "cam_high"),
    "camera_wrist_left": ("left_camera", "camera_wrist_left", "cam_left_wrist"),
    "camera_wrist_right": ("right_camera", "camera_wrist_right", "cam_right_wrist"),
}
FIELDS = ("intrinsic_cv", "extrinsic_cv", "cam2world_gl")
# [SIDA--]


def _find_group(handle: h5py.File, raw_names: tuple[str, ...]) -> h5py.Group | None:
    for parent in ("observation", "observations"):
        for raw_name in raw_names:
            path = f"/{parent}/{raw_name}"
            if path in handle and isinstance(handle[path], h5py.Group):
                return handle[path]
    return None


def _read_matrix(group: h5py.Group, field_name: str, source: Path) -> np.ndarray:
    if field_name not in group:
        raise KeyError(f"{source}: camera group {group.name} is missing {field_name}")
    array = np.asarray(group[field_name][()], dtype=np.float32)
    if array.shape[-2:] not in ((3, 3), (3, 4), (4, 4)):
        raise ValueError(f"{source}: {group.name}/{field_name} has invalid shape {array.shape}")
    if array.ndim not in (2, 3):
        raise ValueError(f"{source}: {group.name}/{field_name} must be [M,M] or [T,M,M], got {array.shape}")
    # RoboTwin stores OpenCV extrinsics as [T,3,4]. Promote them to the
    # homogeneous [T,4,4] convention consumed by the renderer.
    if array.shape[-2:] == (3, 4):
        padded = np.zeros(array.shape[:-2] + (4, 4), dtype=array.dtype)
        padded[..., :3, :4] = array
        padded[..., 3, 3] = 1.0
        return padded
    return array


def extract_episode(hdf5_path: str | Path, sidecar_path: str | Path, *, overwrite: bool = False,
                    frame_indices: list[int] | None = None) -> Path:
    """Extract one raw RoboTwin HDF5 episode into ``*.camera.npz``."""

    source = Path(hdf5_path).expanduser()
    destination = Path(sidecar_path).expanduser()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"sidecar already exists: {destination}; pass --overwrite to replace it")
    destination.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    frame_counts: set[int] = set()
    cameras_found: list[str] = []
    image_sizes = {}
    with h5py.File(source, "r") as handle:
        for canonical_name, candidates in CAMERA_CANDIDATES.items():
            group = _find_group(handle, candidates)
            if group is None:
                continue
            camera_arrays = {field: _read_matrix(group, field, source) for field in FIELDS}
            if "rgb" not in group:
                raise ValueError(f"{source}: {group.name} has no RGB to verify frame alignment")
            rgb = group["rgb"]
            frame_counts.add(len(rgb))
            if len(rgb) == 0:
                raise ValueError(f"{source}: empty RGB episode")
            if rgb.ndim == 4:
                image_sizes[canonical_name] = list(rgb.shape[1:3])
            else:
                encoded = rgb[0]
                content = (encoded.tobytes() if hasattr(encoded, "tobytes") else bytes(encoded)).rstrip(b"\0")
                with Image.open(io.BytesIO(content)) as image:
                    image_sizes[canonical_name] = [image.height, image.width]
            for field_name, array in camera_arrays.items():
                arrays[f"{canonical_name}__{field_name}"] = array
                if array.ndim == 3:
                    frame_counts.add(int(array.shape[0]))
            cameras_found.append(canonical_name)

    if not cameras_found:
        raise ValueError(f"{source}: no supported camera metadata groups found")
    if len(frame_counts) > 1:
        raise ValueError(f"{source}: camera frame counts disagree: {sorted(frame_counts)}")
    frame_count = next(iter(frame_counts))
    if frame_indices is not None:
        if not frame_indices or any(not isinstance(i, int) or i < 0 or i >= frame_count for i in frame_indices):
            raise ValueError(f"{source}: invalid source frame_indices")
        for key, array in arrays.items():
            if array.ndim == 3:
                arrays[key] = array[frame_indices]
        frame_count = len(frame_indices)
    metadata = {
        "format": "lingbot.robottwin.camera.v1",
        "source_hdf5": str(source.resolve()),
        "frame_count": frame_count,
        "image_sizes": image_sizes,
        "source_frame_indices": frame_indices,
        "cameras": cameras_found,
        "matrix_convention": {
            "intrinsic_cv": "OpenCV K matrix",
            "extrinsic_cv": "world_to_camera (OpenCV)",
            "cam2world_gl": "camera_to_world (SAPIEN/OpenGL)",
        },
    }
    arrays["__meta_json__"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(destination, **arrays)
    return destination


def _discover_hdf5(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        # Keep the same os.walk traversal used by RoboTwin's official
        # converter so manifest episode_index matches the generated LeRobot
        # episode_index when both commands see the same input tree.
        files: list[Path] = []
        for root, _, names in os.walk(input_path):
            files.extend(Path(root) / name for name in fnmatch.filter(names, "*.hdf5"))
        return files
    raise FileNotFoundError(f"input path does not exist: {input_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="raw HDF5 inspection/export only (not training aligned)")
    parser.add_argument("--episode-map", type=Path, help="JSON episode list in actual LeRobot conversion order")
    parser.add_argument("--output-root", required=True, type=Path, help="directory for *.camera.npz sidecars")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.episode_map:
        extract_mapped_dataset(args.episode_map, args.output_root, overwrite=args.overwrite)
        return
    if args.input is None:
        parser.error("provide --episode-map for training, or --input for inspection")

    inputs = _discover_hdf5(args.input.expanduser())
    if not inputs:
        raise FileNotFoundError(f"no .hdf5 files found under {args.input}")
    input_root = args.input.expanduser() if args.input.expanduser().is_dir() else args.input.expanduser().parent
    manifest_entries = []
    for episode in inputs:
        relative = episode.relative_to(input_root)
        sidecar = args.output_root.expanduser() / relative.parent / f"{relative.stem}.camera.npz"
        output_path = extract_episode(episode, sidecar, overwrite=args.overwrite)
        manifest_entries.append(
            {
                "episode_index": len(manifest_entries),
                "source_hdf5": relative.as_posix(),
                "sidecar": output_path.relative_to(args.output_root.expanduser()).as_posix(),
            }
        )
        print(output_path)
    manifest = {
        "format": "lingbot.robottwin.camera.manifest.v1",
        "source_root": str(input_root.resolve()),
        "episodes": manifest_entries,
        "training_aligned": False,
    }
    args.output_root.expanduser().mkdir(parents=True, exist_ok=True)
    (args.output_root.expanduser() / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(args.output_root.expanduser() / "manifest.json")


def extract_mapped_dataset(map_path: Path, output_root: Path, *, overwrite: bool = False) -> None:
    """Join raw camera data to the explicitly recorded conversion episode order.

    Paths in the map are relative to the map file. Official pi0 preprocessing
    keeps observations [0, T-1), resizes RGB, and shifts actions by one frame.
    Other preprocessing must supply its exact source_frame_indices; when those
    indices are present, ``processed_hdf5`` is optional.
    """
    map_path = map_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    entries = json.loads(map_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("episode map must be a non-empty JSON list in conversion order")
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"manifest already exists: {manifest_path}")
    manifest_entries = []
    for index, entry in enumerate(entries):
        if entry.get("episode_index") != index:
            raise ValueError(f"map must specify consecutive episode_index; invalid entry {index}")
        source = Path(entry["source_hdf5"])
        source = source if source.is_absolute() else map_path.parent / source
        processed_value = entry.get("processed_hdf5")
        processed = Path(processed_value) if processed_value else None
        if processed is not None:
            processed = processed if processed.is_absolute() else map_path.parent / processed
        indices = entry.get("source_frame_indices")

        if processed is not None:
            with h5py.File(processed, "r") as handle:
                count = len(handle["observations/qpos"])
                if len(handle["action"]) != count:
                    raise ValueError(f"{processed}: state/action lengths disagree")
                for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                    if len(handle[f"observations/images/{camera}"]) != count:
                        raise ValueError(f"{processed}: RGB/state lengths disagree")
        elif indices is not None:
            count = len(indices)
        else:
            raise ValueError(
                f"episode {index}: provide processed_hdf5 or source_frame_indices in the episode map"
            )

        if indices is None:
            with h5py.File(source, "r") as handle:
                raw_count = len(handle["observation/head_camera/rgb"])
            if raw_count != count + 1:
                raise ValueError("expected official pi0 T-1 preprocessing; provide source_frame_indices explicitly")
            indices = list(range(count))
        if len(indices) != count:
            raise ValueError(f"episode {index}: source frame map does not match processed frame count")
        name = f"episode_{index:06d}.camera.npz"
        extract_episode(source, output_root / name, overwrite=overwrite, frame_indices=indices)
        manifest_entries.append({"episode_index": index, "sidecar": name, "frame_count": count,
                                 "source_hdf5": str(source),
                                 "processed_hdf5": str(processed) if processed is not None else ""})
    manifest = {"format": "lingbot.robottwin.camera.manifest.v1", "training_aligned": True,
                "episodes": manifest_entries}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
