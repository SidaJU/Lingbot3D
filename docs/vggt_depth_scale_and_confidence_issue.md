# VGGT 伪深度目标的尺度与置信度问题

## 结论

当前 Gaussian 分支的 VGGT 深度监督存在两个独立问题：

1. **尺度问题**：项目直接使用 VGGT depth head 的原始输出作为“米制深度”监督，但这些数值不是稳定的度量深度；同一场景/同一相机架在不同帧上的尺度会漂移。
2. **置信度问题**：VGGT 的 confidence 激活是 `1 + exp(x)`，恒大于等于 1；当前代码用 `conf >= conf_threshold` 过滤，而配置阈值是 0.5，因此门限永远为真，等于没有过滤。

这两个问题会让 Gaussian depth 被错误拉向一个近乎常数、尺度不稳定的伪深度面。但它们**不是前一次渲染退化的主因**：单视角 + 深度的对照实验显示，修复多视角监督错配后，保留 VGGT 深度目标不会破坏渲染，指标反而略好。

---

## 当前代码如何处理

### 1. VGGT 以 feature-only 方式运行，depth head 打开但 camera head 关闭

文件：`lingbotvla/models/vla/geometry/vggt_adapter.py`

```python
self.encoder = VGGT(
    img_size=self.image_size,
    embed_dim=self.feature_dim // 2,
    enable_camera=False,
    enable_point=False,
    enable_depth=self.use_depth,
    enable_track=False,
    feature_only=True,
)
```

项目用 VGGT 同时产出：

- 聚合特征：送入 `gaussian_3d_adapter`，生成 Gaussian prefix tokens；
- depth / depth_conf：作为 Gaussian depth map 的伪监督。

但 `enable_camera=False` 意味着 VGGT 的 camera head 没有参与，原始 depth 没有经过相机 head 的尺度反归一化。

### 2. depth 目标直接取 `return_depth=True` 的原始输出

文件：`lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py`

训练时：

```python
want_depth = bool(getattr(self.config, "gaussian_use_vggt_depth_target", False)) and self.training
if want_depth:
    vggt_features, vggt_depth, vggt_depth_conf = self.gaussian_vggt(
        gaussian_rgb_images, return_depth=True
    )
    self._gaussian_vggt_depth = vggt_depth.detach()
    self._gaussian_vggt_depth_conf = vggt_depth_conf.detach()
```

随后在 Gaussian render loss 中直接用作 target：

```python
depth_loss = compute_gaussian_depth_loss(
    depth_map,
    pseudo_depth,
    confidence=pseudo_conf,
    conf_threshold=0.5,
    min_depth=0.05,
    max_depth=8.0,
    scale_align=False,
)
```

### 3. 默认用线性 L1，不做尺度对齐

文件：`lingbotvla/models/vla/geometry/render_loss.py`

```python
if not scale_align:
    return torch.where(valid, diff, torch.zeros_like(diff)).sum() / denom
```

配置中：

```yaml
gaussian_depth_scale_align: false
gaussian_depth_conf_threshold: 0.5
```

这意味着：

- VGGT depth 被当作米制 target；
- confidence 门限用 `conf >= 0.5`；
- 没有 per-sample median/log alignment。

---

## 观测到的证据

### 1. VGGT depth 几乎是常数

用真实 RoboTwin 帧 + 项目自己的 `VGGTFeatureEncoder` 测试：

| 视角 | depth min | depth max | mean | std |
|---|---:|---:|---:|---:|
| camera_top | 0.902 | 1.027 | 1.003 | 0.019 |
| camera_wrist_left | 0.749 | 1.130 | 1.061 | 0.029 |
| camera_wrist_right | 1.116 | 1.303 | 1.217 | 0.025 |

top 视角深度动态范围只有约 12%，std 约 0.02，基本是一张平面。真实桌面场景中，桌面、物体、机械臂应该有明显深度层次。

### 2. 尺度不稳定

真实相机架中，top 到左右腕部相机的物理距离约 0.49 m，基本固定；但 VGGT 内部估计的相机间距离在 0.16–0.73 之间波动。

同一个静态场景，不同帧的腕部 depth 均值也从 1.22 → 0.34 → 0.40 大幅变化。说明 VGGT 原始 depth 的单位不是稳定米制，不能直接拿来做线性 L1。

### 3. confidence 门限失效

VGGT 的 DPT head 使用：

```python
conf_out = 1 + conf.exp()
```

所以 confidence 恒 >= 1。配置阈值是 0.5，因此：

```python
valid = conf >= 0.5
```

永远为真。

实测腕部视角的 `depth_conf` 均值恰好是 1.000，对应 `exp(logit) ≈ 0`，即模型本身认为这些视角的 depth 极不可信；但当前代码仍然全像素监督。

---

## 与 GaussianDream 官方对比

官方 GaussianDream 的深度监督链路是：

1. **离线生成伪深度**：官方脚本 `scripts/generate_depth_for_libero.py` / `generate_depth_for_robocasa.py` 使用 **Depth Anything V2** 生成伪深度图。
2. **深度进入数据集**：生成的 depth 被写入 LeRobot 数据的 `observation.depth` / `observation.wrist_depth`。
3. **训练时直接读取 target depth**：官方 policy 中：

```python
gt_depth = target_observation.depth
if depth_map.shape != gt_depth.shape:
    depth_map = F.interpolate(depth_map, size=gt_depth.shape[-2:], ...)
depth_loss = F.l1_loss(depth_map, gt_depth)
```

4. **官方线性 L1 成立的前提**：target depth 是 Depth Anything V2 生成的、相对稳定且近似度量/一致的伪深度，不是 VGGT 未反归一化的原始 depth。

本项目把 target 来源从 Depth Anything V2 换成了 VGGT 原始 depth，但保留了官方的线性 L1 和 `scale_align=False` 假设，因此“官方公式”本身没错，错在输入深度的分布和尺度不满足这个假设。

---

## 与渲染退化的关系

三次 300 步对照：

| run | 渲染视角 | VGGT depth | top loss 最后窗口 | rendered std | rendered max | opacity mean |
|---|---|---|---:|---:|---:|---:|
| `recon_depthfix_300` | 三视角 | 开 | 0.241 | 0.043 | 0.496 | 0.062 |
| `recon_nodepth_300` | 三视角 | 关 | 0.236 | 0.049 | 0.552 | 0.079 |
| `recon_topview_300` | 单视角 | 关 | 0.155 | 0.158 | 0.821 | 0.222 |
| `recon_topview_depth_300` | 单视角 | 开 | **0.151** | **0.170** | **0.902** | **0.285** |

结论：

- 多视角监督错配才是渲染退化的主因；
- VGGT depth 的问题是独立的；
- 在单视角监督修复后，保留 VGGT depth 不会破坏渲染，反而略好；
- 但深度本身的尺度/置信度问题仍应修，否则 depth map 的物理意义和跨场景稳定性不可靠。

---

## 未来改进方案

### 方案 A：最接近官方，推荐

离线生成 Depth Anything V2 伪深度，写入训练数据，训练时读取 `observation.depth` 做线性 L1。

优点：

- 与官方 GaussianDream 数据链路一致；
- 深度来源稳定、结构更清晰；
- 不需要 VGGT camera head 或复杂的尺度反归一化。

缺点：

- 需要新增数据预处理步骤；
- 数据集要重新写入 depth 字段或 sidecar。

### 方案 B：继续用 VGGT，但修尺度

不要直接用原始 depth。需要：

- 启用 VGGT camera head；
- 用 camera head 的 pose encoding 对 depth 做反归一化；
- 确保 depth 转到与 Gaussian 世界坐标一致的尺度。

当前 adapter 中 `enable_camera=False`，该信息缺失。

### 方案 C：继续用 VGGT，但改成尺度无关监督

把：

```yaml
gaussian_depth_scale_align: false
```

改为：

```yaml
gaussian_depth_scale_align: true
```

当前代码已实现 per-sample median-aligned log-L1，可缓解尺度漂移，但无法修复 depth 近乎常数、结构弱的问题。

### 方案 D：修 confidence 门限

当前 `conf = 1 + exp(x)`，所以阈值必须相对于 1 设置，例如：

```yaml
gaussian_depth_conf_threshold: 1.2
```

或改代码把 confidence 归一化为 `[0,1]`：

```python
conf_norm = conf - 1.0
valid = conf_norm >= threshold
```

推荐后者，语义更清晰。

---

## 已落地：方案 A（离线 DA2 深度 target）

项目代码已切换到方案 A，和 GaussianDream 官方保持同一数据链路语义：

1. `tools/generate_depth_for_robotwin.py` 用本地 `third_party/DepthAnythingV2` 对小/base/large 模型进行离线推理，逐 episode 逐相机生成 `256×256` 的 `float32` 深度。
2. 深度写入 `data/depth_sidecars/<dataset>/episode_*.depth.npz`，并由 `manifest.json` 按 LeRobot 训练顺序对齐；相机顺序与 `data.cameras` 一致：`camera_top`、`camera_wrist_left`、`camera_wrist_right`。
3. `VLADataset` 在取帧时直接读取 sidecar，输出 batch 字段 `gaussian_depth_targets`，形状 `[B,V,1,H,W]`。
4. `LingbotVlaV2Policy.forward` / `FlowMatchingV2.forward` 显式接收该字段，并传给 `_gaussian_rgb_render_loss`；主视角深度直接作为 Gaussian depth target。
5. `gaussian_use_vggt_depth_target: false`，VGGT 不再输出 depth head；`gaussian_use_offline_depth_target: true`；depth loss 保持官方线性 L1，无 confidence mask、无 scale align、无有效 clamp。
6. `gaussian_render_views` 仍保持 `[camera_top]`，避免回退到此前发现的多视角 target/render 错配问题。

运行方式：

```bash
# 全量 10 个数据集、3 个相机
python tools/generate_depth_for_robotwin.py --batch-size 16 --device cuda:0

# 单个数据集 / smoke 验证
python tools/generate_depth_for_robotwin.py \
  --dataset-root cache/huggingface/lerobot/robotwin_smoke_v2 \
  --limit-episodes 1 --cameras camera_top
```

## 当前建议

- **渲染问题**：保持 `gaussian_render_views: [camera_top]`。
- **深度问题**：使用离线 DA2 depth sidecar，不再依赖 VGGT depth head；训练时直接加载 `gaussian_depth_targets`。
- **置信度**：离线 DA2 路径没有 confidence 权重，彻底绕开 `1 + exp(x)` 恒真的门限问题。
