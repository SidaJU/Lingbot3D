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
import json
from pathlib import Path
import re

import h5py


# [SIDA++] add by SidaZhu for explicit RoboTwin episode alignment
def _episodes(root: Path) -> list[Path]:
    def order(path: Path) -> tuple[int, str]:
        matches = re.findall(r"\d+", path.stem)
        return (int(matches[-1]) if matches else 2**63 - 1, str(path))

    paths = sorted(root.rglob("*.hdf5"), key=order)
    if not paths:
        raise FileNotFoundError(f"no HDF5 episodes found under {root}")
    return paths


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
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--processed-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build_map(args.raw_root.expanduser(), args.processed_root.expanduser(), args.output.expanduser())


if __name__ == "__main__":
    main()
