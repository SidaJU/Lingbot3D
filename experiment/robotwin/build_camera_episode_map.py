"""Build an explicit raw-to-processed RoboTwin episode map.

The map is consumed by ``extract_camera_metadata.py --episode-map``.  It
records the exact source and processed HDF5 episode order instead of relying
on directory enumeration at training time.
"""

# [SIDA++] add by SidaZhu for explicit RoboTwin episode alignment.
# Purpose: generate the sidecar-to-LeRobot episode contract outside LingBot
# official preprocessing code. Original additions retain SidaZhu attribution
# and applicable rights, subject to the repository license and third-party
# notices.
# [SIDA--]

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import h5py
import numpy as np
import pyarrow.parquet as pq


# [SIDA++] add by SidaZhu for explicit RoboTwin episode alignment
def _episodes(root: Path) -> list[Path]:
    def order(path: Path) -> tuple[int, str]:
        matches = re.findall(r"\d+", path.stem)
        return (int(matches[-1]) if matches else 2**63 - 1, str(path))

    paths = sorted(root.rglob("*.hdf5"), key=order)
    if not paths:
        raise FileNotFoundError(f"no HDF5 episodes found under {root}")
    return paths


def _state_key(state: np.ndarray) -> tuple[float, ...]:
    return tuple(np.round(np.asarray(state, dtype=np.float64), 6))


def _lerobot_states(root: Path) -> dict[int, list[tuple[int, np.ndarray]]]:
    episodes: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    parquet_files = sorted((root / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no LeRobot data parquet files found under {root / 'data'}")
    for path in parquet_files:
        table = pq.read_table(path, columns=["episode_index", "frame_index", "observation.state"])
        for episode_index, frame_index, state in zip(
            table["episode_index"].to_pylist(),
            table["frame_index"].to_pylist(),
            table["observation.state"].to_pylist(),
        ):
            episodes[int(episode_index)].append((int(frame_index), np.asarray(state)))
    for frames in episodes.values():
        frames.sort(key=lambda item: item[0])
    return episodes


def build_map_from_lerobot(lerobot_root: Path, raw_root: Path, output: Path) -> None:
    """Match LeRobot states to raw HDF5 trajectories and emit a camera episode map.

    The LeRobot dataset is the training-time source of truth.  Its state vectors
    are written from the processed HDF5 ``qpos`` stream, so exact state matching
    recovers the raw episode and the frame order even when processed files have
    been regenerated or reordered after the LeRobot conversion.
    """
    lerobot_episodes = _lerobot_states(lerobot_root)
    raw_files = _episodes(raw_root)

    state_index: dict[tuple[float, ...], list[tuple[int, int]]] = defaultdict(list)
    raw_lengths: list[int] = []
    for episode_index, path in enumerate(raw_files):
        with h5py.File(path, "r") as handle:
            if "joint_action/vector" in handle:
                key = "joint_action/vector"
            elif "observations/qpos" in handle:
                key = "observations/qpos"
            elif "observation/qpos" in handle:
                key = "observation/qpos"
            else:
                raise ValueError(f"{path}: no supported state dataset found")
            states = np.asarray(handle[key][:], dtype=np.float64)
        raw_lengths.append(len(states))
        for frame_index, state in enumerate(states):
            state_index[_state_key(state)].append((episode_index, frame_index))

    def monotonic_path(
        candidates: list[list[tuple[int, int]]], raw_episode: int
    ) -> tuple[list[int], int]:
        path: list[int] = []
        gaps = 0
        last = -1
        raw_length = raw_lengths[raw_episode]
        for candidates_for_frame in candidates:
            options = sorted(
                frame for episode, frame in candidates_for_frame
                if episode == raw_episode and frame > last
            )
            if options:
                chosen = options[0]
            else:
                gaps += 1
                chosen = min(last + 1, raw_length - 1)
            path.append(chosen)
            last = chosen
        return path, gaps

    entries = []
    for episode_index in sorted(lerobot_episodes):
        frames = lerobot_episodes[episode_index]
        candidates = [state_index.get(_state_key(state), []) for _, state in frames]
        votes = Counter(
            raw_episode
            for candidates_for_frame in candidates
            for raw_episode, _ in candidates_for_frame
        )
        if not votes:
            raise ValueError(f"LeRobot episode {episode_index}: no matching raw states")

        max_gaps = max(2, len(frames) // 100)
        best = None
        for raw_episode, vote_count in votes.most_common(10):
            path, gaps = monotonic_path(candidates, raw_episode)
            if gaps > max_gaps:
                continue
            score = (-gaps, len(path), vote_count)
            if best is None or score > best[0]:
                best = (score, raw_episode, path, gaps)
        if best is None:
            raise ValueError(f"LeRobot episode {episode_index}: no monotonic raw trajectory match")
        _, raw_episode, frame_indices, gaps = best
        if gaps > max_gaps:
            raise ValueError(
                f"LeRobot episode {episode_index}: {gaps} unmatched frames exceed limit {max_gaps}"
            )
        if len(frame_indices) != len(frames):
            raise ValueError(
                f"LeRobot episode {episode_index}: matched {len(frame_indices)} frames, "
                f"expected {len(frames)}"
            )
        entries.append({
            "episode_index": episode_index,
            "source_hdf5": str(raw_files[raw_episode].resolve()),
            "source_frame_indices": frame_indices,
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(output)


def build_map(raw_root: Path, processed_root: Path, output: Path) -> None:
    raw = _episodes(raw_root)
    processed = _episodes(processed_root)
    if len(raw) != len(processed):
        raise ValueError(f"raw/processed episode counts differ: {len(raw)} vs {len(processed)}")
    entries = []
    for index, (source, target) in enumerate(zip(raw, processed)):
        with h5py.File(source, "r") as source_file, h5py.File(target, "r") as target_file:
            source_count = len(source_file["observation/head_camera/rgb"])
            target_count = len(target_file["observations/qpos"])
            if source_count != target_count + 1:
                raise ValueError(
                    f"episode {index}: expected source T={target_count + 1}, got {source_count}; "
                    "supply a custom source_frame_indices map for another preprocessing rule"
                )
        entries.append({
            "episode_index": index,
            "source_hdf5": str(source.resolve()),
            "processed_hdf5": str(target.resolve()),
            "source_frame_indices": list(range(target_count)),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(output)
# [SIDA--]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, help="raw RoboTwin HDF5 root")
    parser.add_argument("--processed-root", type=Path, help="processed HDF5 root (legacy mode)")
    parser.add_argument("--lerobot-root", type=Path, help="generate a training-truth map from a LeRobot dataset")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.lerobot_root is not None:
        if args.raw_root is None:
            parser.error("--raw-root is required with --lerobot-root")
        build_map_from_lerobot(
            args.lerobot_root.expanduser(), args.raw_root.expanduser(), args.output.expanduser()
        )
        return
    if args.raw_root is None or args.processed_root is None:
        parser.error("provide --lerobot-root --raw-root, or --raw-root --processed-root")
    build_map(args.raw_root.expanduser(), args.processed_root.expanduser(), args.output.expanduser())


if __name__ == "__main__":
    main()
