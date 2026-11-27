#!/usr/bin/env python
"""Generate offline Depth Anything V2 targets for RoboTwin LeRobot datasets.

This follows GaussianDream's data preparation: depth is predicted once from
RGB, stored with the dataset, and loaded directly as a Gaussian depth target
during training.  No depth teacher is run inside the training loop.

Example (from the repository root):
    python tools/generate_depth_for_robotwin.py \
        --dataset-root ./data/datasets/robotwin_smoke_v2 \
        --limit-episodes 2 --batch-size 8
"""

# [SIDA++] add by SidaZhu for GaussianDream-style offline DA2 depth targets.
# Original additions retain SidaZhu attribution and applicable rights,
# subject to the repository license and third-party notices.
# [SIDA--]

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DA2_ROOT = PROJECT_ROOT / "third_party" / "DepthAnythingV2"
sys.path.insert(0, str(DA2_ROOT))

from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402

from lingbotvla.models.vla.geometry.depth_metadata import (  # noqa: E402
    DEPTH_MANIFEST_FORMAT,
    DEPTH_SIDECAR_FORMAT,
)


RAW_CAMERA_COLUMNS = {
    "camera_top": "observation.images.cam_high",
    "camera_wrist_left": "observation.images.cam_left_wrist",
    "camera_wrist_right": "observation.images.cam_right_wrist",
}

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-manifest",
        default=str(PROJECT_ROOT / "data/training_data/robotwin_a10080.txt"),
        help="Training manifest with one '<name> <path>' dataset per line.",
    )
    parser.add_argument(
        "--dataset-root",
        action="append",
        default=[],
        help="Process this dataset root instead of the manifest. Can be repeated.",
    )
    parser.add_argument(
        "--dataset-name-filter",
        default=None,
        help="Only process manifest entries whose name contains this substring.",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "data/depth_sidecars"),
        help="Root directory for depth sidecars.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DA2_ROOT / "checkpoints/depth_anything_v2_vits.pth"),
        help="Depth Anything V2 checkpoint path.",
    )
    parser.add_argument("--model-size", choices=("small", "base", "large"), default="small")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=list(RAW_CAMERA_COLUMNS),
        choices=list(RAW_CAMERA_COLUMNS),
        help="Canonical cameras to process.",
    )
    parser.add_argument("--input-size", type=int, default=518, help="DA2 long-side input size.")
    parser.add_argument("--target-size", type=int, default=256, help="Stored depth map size.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-episodes", type=int, default=0, help="Only process first N episodes per dataset.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing sidecars.")
    return parser.parse_args()


def build_model(args: argparse.Namespace) -> DepthAnythingV2:
    encoder = {"small": "vits", "base": "vitb", "large": "vitl"}[args.model_size]
    if encoder not in MODEL_CONFIGS:
        raise ValueError(f"unsupported DA2 encoder: {encoder}")
    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DA2 checkpoint does not exist: {checkpoint}")
    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(args.device)
    model.eval()
    return model


def prepare_image(image: Image.Image, input_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    width, height = image.size
    scale = input_size / max(width, height)
    new_w = int(width * scale) // 14 * 14
    new_h = int(height * scale) // 14 * 14
    new_w, new_h = max(new_w, 14), max(new_h, 14)
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0)
    tensor = (tensor - MEAN[:, None, None]) / STD[:, None, None]
    return tensor, (new_h, new_w)


def image_bytes(value) -> bytes:
    if isinstance(value, dict):
        value = value.get("bytes")
    if isinstance(value, memoryview):
        value = value.tobytes()
    if not isinstance(value, (bytes, bytearray)):
        raise TypeError(f"unsupported image payload: {type(value)}")
    return bytes(value)


def predict_depth_batch(
    model: DepthAnythingV2,
    images: list[Image.Image],
    args: argparse.Namespace,
    target_hw: tuple[int, int],
) -> np.ndarray:
    prepared = [prepare_image(image, args.input_size) for image in images]
    groups: dict[tuple[int, int], list[tuple[int, torch.Tensor]]] = {}
    for index, (tensor, size) in enumerate(prepared):
        groups.setdefault(size, []).append((index, tensor))

    output = np.empty((len(images), target_hw[0], target_hw[1]), dtype=np.float32)
    for size, items in groups.items():
        for start in range(0, len(items), args.batch_size):
            chunk = items[start : start + args.batch_size]
            batch = torch.stack([tensor for _, tensor in chunk]).to(args.device)
            with torch.no_grad():
                prediction = model(batch)
            prediction = prediction[:, None]
            prediction = F.interpolate(prediction, size=target_hw, mode="bilinear", align_corners=False)
            prediction = prediction[:, 0].detach().float().cpu().numpy()
            for local_index, (global_index, _) in enumerate(chunk):
                output[global_index] = prediction[local_index]
    return output


def episode_files(dataset_root: Path) -> list[Path]:
    files = sorted(dataset_root.glob("data/chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {dataset_root / 'data'}")
    return files


def sidecar_is_complete(path: Path, cameras: list[str], frame_count: int) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            for camera in cameras:
                key = f"{camera}__depth"
                if key not in archive or archive[key].shape[0] != frame_count:
                    return False
    except Exception:
        return False
    return True


def process_dataset(model: DepthAnythingV2, dataset_root: Path, args: argparse.Namespace) -> None:
    root_name = dataset_root.name
    output_root = Path(args.output_root).expanduser() / root_name
    output_root.mkdir(parents=True, exist_ok=True)
    files = episode_files(dataset_root)

    grouped: dict[int, list[tuple[Path, int, int]]] = {}
    for parquet_path in files:
        table = pq.read_table(parquet_path, columns=["episode_index", "frame_index"])
        episode_indices = table["episode_index"].to_pylist()
        frame_indices = table["frame_index"].to_pylist()
        for row, (episode_index, frame_index) in enumerate(zip(episode_indices, frame_indices)):
            grouped.setdefault(int(episode_index), []).append(
                (parquet_path, row, int(frame_index))
            )

    episodes = sorted(grouped)
    if args.limit_episodes:
        episodes = episodes[: args.limit_episodes]

    table_cache: dict[Path, Any] = {}

    def cached_table(parquet_path: Path):
        if parquet_path not in table_cache:
            table_cache[parquet_path] = pq.read_table(
                parquet_path,
                columns=[RAW_CAMERA_COLUMNS[c] for c in args.cameras],
            )
        return table_cache[parquet_path]

    manifest_entries = []
    started = time.time()
    for episode_position, episode_index in enumerate(episodes):
        rows = grouped[episode_index]
        frame_indices = [frame_index for _, _, frame_index in rows]
        if len(set(frame_indices)) != len(frame_indices):
            raise ValueError(f"episode {episode_index} has duplicate frame_index values")
        frame_count = max(frame_indices) + 1
        sidecar_path = output_root / f"episode_{episode_index:06d}.depth.npz"
        if not args.overwrite and sidecar_is_complete(sidecar_path, args.cameras, frame_count):
            manifest_entries.append(
                {
                    "episode_index": episode_index,
                    "sidecar": sidecar_path.name,
                    "frame_count": frame_count,
                }
            )
            continue

        arrays = {
            f"{camera}__depth": np.zeros(
                (frame_count, args.target_size, args.target_size), dtype=np.float32
            )
            for camera in args.cameras
        }
        for camera in args.cameras:
            column_name = RAW_CAMERA_COLUMNS[camera]
            values = [
                cached_table(parquet_path)[column_name][row].as_py()
                for parquet_path, row, _ in rows
            ]
            images = [
                Image.open(io.BytesIO(image_bytes(value))).convert("RGB") for value in values
            ]
            predicted = predict_depth_batch(
                model, images, args, (args.target_size, args.target_size)
            )
            arrays[f"{camera}__depth"][frame_indices] = predicted

        metadata = {
            "format": DEPTH_SIDECAR_FORMAT,
            "cameras": args.cameras,
            "frame_count": frame_count,
            "image_size": [args.target_size, args.target_size],
            "model_size": args.model_size,
            "source_frame_indices": frame_indices,
        }
        payload = {key: value for key, value in arrays.items()}
        payload["__meta_json__"] = np.asarray(json.dumps(metadata))
        np.savez_compressed(sidecar_path, **payload)
        manifest_entries.append(
            {
                "episode_index": episode_index,
                "sidecar": sidecar_path.name,
                "frame_count": frame_count,
            }
        )
        if (episode_position + 1) % 10 == 0 or episode_position + 1 == len(episodes):
            elapsed = time.time() - started
            print(
                f"[{root_name}] {episode_position + 1}/{len(episodes)} episodes, "
                f"{elapsed:.1f}s elapsed",
                flush=True,
            )

    manifest = {
        "format": DEPTH_MANIFEST_FORMAT,
        "training_aligned": True,
        "dataset": root_name,
        "cameras": args.cameras,
        "target_size": args.target_size,
        "model_size": args.model_size,
        "episodes": manifest_entries,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[{root_name}] manifest written to {manifest_path}", flush=True)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    datasets: list[Path] = []
    for root in args.dataset_root:
        datasets.append(Path(root).expanduser().resolve())
    if not datasets:
        manifest_path = Path(args.dataset_manifest).expanduser()
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 2 or parts[0].startswith("#"):
                continue
            name, path = parts[0], parts[1]
            if args.dataset_name_filter and args.dataset_name_filter not in name:
                continue
            datasets.append(Path(path).expanduser().resolve())

    if not datasets:
        raise RuntimeError("no datasets selected")

    model = build_model(args)
    print(
        f"DA2 model={args.model_size} device={args.device} cameras={args.cameras} "
        f"datasets={len(datasets)}",
        flush=True,
    )
    for dataset_root in datasets:
        process_dataset(model, dataset_root, args)


if __name__ == "__main__":
    main()
