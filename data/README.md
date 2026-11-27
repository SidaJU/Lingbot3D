# Project data root

`data/` holds **datasets only**: converted LeRobot datasets, raw RoboTwin
downloads, per-episode camera/depth sidecars, and the small text manifests that
list which datasets to train on.

Pretrained weights live in the repository-root `models/` directory; third-party
models and their source code live in `third_party/`; training checkpoints and
logs live in `runs/`. Every path in `configs/vla/robotwin/*.yaml`, in the
manifests under `data/training_data/`, and in the preparation tools is relative
to the repository root. Run all commands from the repository root after cloning,
on any machine and at any filesystem location. No machine-specific absolute
paths are required or allowed.

Large files (model weights, datasets, camera/depth sidecars, checkpoints,
TensorBoard runs, debug images) stay local. Git only tracks the directory
skeletons (`.gitkeep`), the manifests and the small third-party source that is
vendored in `third_party/`.

## Layout — what belongs where

```text
data/                                # datasets and dataset metadata only
├── datasets/                        # LeRobot datasets consumed by training
│   └── <repo_id>/                   # one directory per entry in a manifest
├── lerobot/                         # optional HF_LEROBOT_HOME cache
├── robotwin2_raw/                   # raw RoboTwin downloads and HDF5 (by_task/)
├── camera_sidecars/                 # per-episode camera intrinsics/extrinsics
│   └── <repo_id>/
│       ├── manifest.json
│       └── episode_*.camera.npz
├── depth_sidecars/                  # offline Depth Anything V2 targets
│   └── <repo_id>/
│       ├── manifest.json
│       └── episode_*.depth.npz
└── training_data/                   # dataset manifests read by data.train_path
    ├── robotwin.txt                 # minimal single-dataset example
    └── robotwin_a10080.txt          # full RoboTwin list used by the 4D configs

models/                              # all pretrained weights and caches
├── lingbot-vla-v2-6b/               # VLA backbone + depth/ + dino_video/ teachers
├── Qwen3-VL-4B-Instruct/            # tokenizer / VLM backbone
├── moge-2-vitb-normal/
│   └── model.pt                     # MoGe normal teacher
├── VGGT-1B/                         # model.pt + config.json (VGGT aggregator)
└── lpips_cache/                     # optional offline LPIPS VGG16 cache
    └── hub/checkpoints/vgg16-397923af.pth

third_party/                         # third-party models and source code
├── DepthAnythingV2/                 # DA2 code + checkpoints/ (see section 4)
├── VGGT/                            # vendored feature_only VGGT fork
├── GaussianDream/                   # reference implementation
├── RoboTwin/                        # RoboTwin 2.0 source
└── diff-gaussian-rasterization/     # vendored CUDA rasterizer

runs/                                # checkpoints, TensorBoard, Gaussian debug
└── <run_name>/
```

`data/camera_sidecars/<name>` and `data/depth_sidecars/<name>` use the exact
dataset directory name (the last path component of the manifest entry). The
loader resolves them automatically via `data.camera_metadata_root` and
`data.depth_metadata_root`.

## What the training configs read

| Config key | Resolves to |
|---|---|
| `data.train_path` | `./data/training_data/<manifest>.txt` |
| `data.robot_config_root` | `./configs/robot_configs` |
| `data.camera_metadata_root` | `./data/camera_sidecars` |
| `data.depth_metadata_root` | `./data/depth_sidecars` |
| `model.model_path` | `./models/lingbot-vla-v2-6b` |
| `model.tokenizer_path` | `./models/Qwen3-VL-4B-Instruct` |
| `train.output_dir` | `./runs/<run_name>` |
| `train.gaussian_lpips_cache_dir` | `./models/lpips_cache` |
| `train.gaussian_vggt_model_path` | `./models/VGGT-1B/model.pt` |
| `train.*.moge_path` | `./models/moge-2-vitb-normal/model.pt` |
| `train.*.morgbd_path` | `./models/lingbot-vla-v2-6b/depth/model.pt` |
| `train.*.ckpt_path` / `config_path` | `./models/lingbot-vla-v2-6b/dino_video/` |

A manifest line is `<data_name> <path relative to the repository root>`:

```text
robotwin ./data/datasets/robotwin_smoke_v2
```

## 1. Download pretrained models into `models/`

Run from the repository root after `bash tools/create_train_env.sh`:

```bash
mkdir -p models

hf download robbyant/lingbot-vla-v2-6b \
  --repo-type model --local-dir models/lingbot-vla-v2-6b

hf download Qwen/Qwen3-VL-4B-Instruct \
  --repo-type model --local-dir models/Qwen3-VL-4B-Instruct

hf download Ruicheng/moge-2-vitb-normal \
  --repo-type model --local-dir models/moge-2-vitb-normal

hf download facebook/VGGT-1B model.pt config.json \
  --revision 769a73bd2e0a03215e13f53f06943702d5396c7b \
  --local-dir models/VGGT-1B
```

If LPIPS must run offline, put the VGG16 backbone at
`models/lpips_cache/hub/checkpoints/vgg16-397923af.pth`, or let `lpips`
download it on first use into that cache root.

Verify:

```bash
test -f models/VGGT-1B/model.pt
test -f models/lingbot-vla-v2-6b/depth/model.pt
test -f models/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth
test -f models/moge-2-vitb-normal/model.pt
```

Third-party source and its weights go in `third_party/`, never in `models/` or
`data/`. The setup script installs the vendored `VGGT/` and
`diff-gaussian-rasterization/`; section 4 explains the DA2 code and checkpoint.

## 2. Raw RoboTwin data and LeRobot conversion

Follow `experiment/robotwin/README.md` to download RoboTwin 2.0, run
`process_data_pi0.sh` and `generate.sh`, and place the resulting LeRobot
datasets under `data/datasets/<repo_id>/`. Set the converter cache to the
project-local folder so nothing lands in a home directory:

```bash
export HF_LEROBOT_HOME=./data/lerobot
export HF_HOME=./data/lerobot
```

Add one line per converted dataset to `data/training_data/robotwin_a10080.txt`:

```text
robotwin ./data/datasets/<repo_id>
```

Keep the path relative to the repository root — the loader resolves it with
`Path(path).exists()`, so a relative path only works when training is launched
from the repository root (as all provided commands do).

## 3. Camera calibration sidecars

The official LeRobot converter drops camera calibration. Extract it from the
raw HDF5 into `data/camera_sidecars/` (see `experiment/robotwin/README.md`):

```bash
python experiment/robotwin/build_camera_episode_map.py \
  --raw-root ./data/robotwin2_raw/by_task/<task>/<setting>/data \
  --processed-root ./data/robotwin2_raw/by_task/<task>/<setting> \
  --output ./data/camera_sidecars/episodes.json

python experiment/robotwin/extract_camera_metadata.py \
  --episode-map ./data/camera_sidecars/episodes.json \
  --output-root ./data/camera_sidecars
```

Move the generated per-dataset folders under
`data/camera_sidecars/<repo_id>/` so their names match the manifest entries.
These sidecars are optional until `enable_gaussian_3d: true` is set.

## 4. Offline Depth Anything V2 targets

The Gaussian depth loss follows GaussianDream: depth is predicted once from
RGB, stored beside the dataset, and read directly as the training target. No
depth teacher runs inside the training loop.

Fetch the official DA2 code and Small checkpoint into `third_party/` (not
committed):

```bash
git clone --depth 1 https://github.com/DepthAnything/Depth-Anything-V2.git \
  third_party/DepthAnythingV2
mkdir -p third_party/DepthAnythingV2/checkpoints

hf download depth-anything/Depth-Anything-V2-Small \
  depth_anything_v2_vits.pth \
  --local-dir third_party/DepthAnythingV2/checkpoints
```

If Hugging Face is unreachable, prefix the command with
`HF_ENDPOINT=https://hf-mirror.com`.

Generate sidecars into `data/depth_sidecars/`:

```bash
python tools/generate_depth_for_robotwin.py \
  --dataset-manifest ./data/training_data/robotwin_a10080.txt \
  --output-root ./data/depth_sidecars \
  --checkpoint ./third_party/DepthAnythingV2/checkpoints/depth_anything_v2_vits.pth \
  --model-size small --batch-size 16 --device cuda
```

Use `--limit-episodes 1` for a smoke run, or `--dataset-root
./data/datasets/<repo_id>` to process a single dataset. Each sidecar contains
`float32` depth maps `[N, 256, 256]` for `camera_top`, `camera_wrist_left` and
`camera_wrist_right`.

The 4D configs enable this path with:

```yaml
gaussian_use_vggt_depth_target: false
gaussian_use_offline_depth_target: true
```

Keep `gaussian_render_views: [camera_top]`. The VGGT depth head is disabled;
VGGT is still used as the frozen 3D feature aggregator.

## 5. Train

```bash
bash train.sh tasks/vla/train_lingbotvla.py \
  configs/vla/robotwin/robotwin_4d_topview.yaml \
  --data.data_name multi
```

Outputs (checkpoints, TensorBoard events, Gaussian debug images) are written
to `train.output_dir` under `./runs/` and are never committed.
