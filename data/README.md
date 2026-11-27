# Project-local data root

This directory is the only place for external assets needed by the portable
Gaussian 3D configs. Paths in `configs/vla/robotwin/*.yaml` are relative to the
repository root, so the same checkout works on any machine and any filesystem
path.

Keep large files out of Git. Only `.gitkeep` files and this README are tracked.

## Layout

```text
data/
├── models/
│   ├── lingbot-vla-v2-6b/            # model weights, depth/, dino_video/
│   ├── Qwen3-VL-4B-Instruct/
│   ├── moge-2-vitb-normal/
│   ├── VGGT-1B/
│   └── lpips_cache/                  # optional offline VGG16 backbone cache
├── datasets/
│   └── robotwin_subset/              # LeRobot v2.1/v3.0 dataset directories
├── camera_sidecars/                  # episode maps and *.camera.npz files
├── training_data/
│   └── robotwin.txt                  # dataset manifest, paths relative to repo root
└── outputs/                          # checkpoints, runs, TensorBoard, debug images
```

## One-time model download

Run from the repository root after creating the environment:

```bash
mkdir -p data/models

hf download robbyant/lingbot-vla-v2-6b \
  --repo-type model --local-dir data/models/lingbot-vla-v2-6b

hf download Qwen/Qwen3-VL-4B-Instruct \
  --repo-type model --local-dir data/models/Qwen3-VL-4B-Instruct

hf download Ruicheng/moge-2-vitb-normal \
  --repo-type model --local-dir data/models/moge-2-vitb-normal

hf download facebook/VGGT-1B model.pt config.json \
  --revision 769a73bd2e0a03215e13f53f06943702d5396c7b \
  --local-dir data/models/VGGT-1B
```

Verify before training:

```bash
test -f data/models/VGGT-1B/model.pt
test -f data/models/lingbot-vla-v2-6b/depth/model.pt
test -f data/models/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth
```

If LPIPS must run offline, place the VGG16 backbone at:

```text
data/models/lpips_cache/hub/checkpoints/vgg16-397923af.pth
```

## Dataset and camera metadata

Prepare LeRobot datasets with the RoboTwin guide in
`experiment/robotwin/README.md`, then place them under `data/datasets/`.

`data/training_data/robotwin.txt` uses the format:

```text
<data_name> <path relative to the repository root>
```

Generate camera sidecars into `data/camera_sidecars/` with the scripts in
`experiment/robotwin/`. The config reads them through
`data.camera_metadata_root: ./data/camera_sidecars`.

## Training output

`train.output_dir` points to `./data/outputs/robotwin_gaussian_stage1`.
Checkpoints,TensorBoard runs and Gaussian debug images stay here and should not
be committed.
