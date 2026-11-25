# Gaussian RGB Render Loss 在第 260 步附近崩塌的根因分析

## 结论

第 260 步附近的 Gaussian RGB render loss 崩塌不是动作专家、LingBot 原始 depth/video 分支、NaN 权重或预训练权重的问题。

根因是 Gaussian decoder 的深度参数化允许 depth 被优化到接近 0：

```python
depth = (8.0 * torch.sigmoid(depth_raw)).clamp_min(1e-3)
```

RGB render loss 在优化过程中会把 Gaussian 的深度持续压低。当所有 Gaussian 都接近相机时，fused rasterizer 的投影和协方差进入数值退化区，渲染结果变成全背景（黑图），对 Gaussian 参数的梯度随之消失。模型进入一个无法自行恢复的吸收态：`rendered ≈ 0`、`raw loss ≈ 0.976`、`GradNorm ≈ 0`。

修复方式是把深度限制在有物理意义的 `0.5–8m` 区间：

```python
depth = (0.5 + 7.5 * torch.sigmoid(depth_raw)).unsqueeze(-1)
```

同时修复了新增 Gaussian 模块被错误零初始化的问题，并增加了按 optimizer step 周期性保存 rendered/target 对比图的 debug 开关。

---

## 1. 现象

出问题的 run 使用 6 张 A100 40G，只开启 Gaussian 渲染分支和动作分支：

```yaml
enable_depth_align: false
enable_gaussian_3d: true
gaussian_render_loss_weight: 0.2
max_steps: 2000
```

### 崩溃前

```text
step 1–100:   GaussianRGB_Raw 均值 0.225
step 101–200: GaussianRGB_Raw 均值 0.158
step 240:     GaussianRGB_Raw ≈ 0.12–0.17
```

这一阶段渲染 loss 正常下降，说明初始化和总体训练方向是对的。

### 崩溃瞬间

```text
step 260: raw=0.174,  GradNorm=0.78
step 261: raw=0.194,  GradNorm=4.72
step 262: raw=0.328,  GradNorm=9.43
step 264: raw=0.527,  GradNorm=6.11
step 266: raw=0.587,  GradNorm=18.81
step 267: raw=0.837,  GradNorm=10.66
step 275: raw=0.945,  GradNorm=17.27
step 276: raw=0.975,  GradNorm=4.51
```

### 崩溃后

```text
step 280–1777:
GaussianRGB_Raw  ≈ 0.976
GaussianRGB_Loss ≈ 0.195
GradNorm         ≈ 0.05–0.4
VLA_Loss         持续下降，最终约 0.005–0.015
```

总 loss 几乎完全被 Gaussian 项主导，但 Gaussian 分支已经不再更新。

---

## 2. 为什么 `raw ≈ 0.976` 说明渲染成了全黑背景

`GaussianRGB_Raw` 的定义是：

```python
0.85 * ssim_loss + 0.15 * l1_loss
```

如果渲染结果接近全黑背景 `0`，而 target 图像均值约为 `0.85`：

```text
ssim_loss ≈ 1
l1_loss   ≈ 0.85

raw ≈ 0.85 * 1 + 0.15 * 0.85
    ≈ 0.9775
```

这和实际观测到的 `0.976` 几乎一致。

所以 `raw ≈ 0.976` 不是普通的高 loss，而是渲染输出已经接近全背景的强信号。

---

## 3. 最小复现过程

为了排除 VLA、动作专家、MoE、depth/video 教师等变量，我用以下组件做了单分支复现：

- `GaussianParameterDecoder`
- `DifferentiableGaussianRenderer`
- `compute_rgb_render_loss`
- 一张 RoboTwin smoke 数据中的真实 target 图像
- `adjust_bottle` 任务的真实 camera K/T sidecar
- `AdamW(lr=1e-4)`
- `gaussian_render_loss_weight=0.2`

复现结果（修复前）：

```text
step 1:  raw=0.70, depth_mean=3.13, rendered_mean=0.49
step 10: raw=0.15, depth_mean=0.51, rendered_mean=0.98
step 15: raw=0.13, depth_mean=0.17
step 20: raw=0.34, depth_mean=0.026
step 25: raw=0.89, depth_mean=0.003, rendered_mean=0.08
step 30: raw=0.99, depth_mean=0.001, rendered_mean≈0, GradNorm≈0
step 40+: raw=0.989 锁死, rendered=0, GradNorm=0
```

这个实验完整复现了正式训练中的现象。

关键观察是：**崩塌前先发生的是 depth 持续下降**，然后渲染输出才变成 0。

---

## 4. 根因链条

### 4.1 原深度参数化没有安全下界

原代码：

```python
depth = (8.0 * torch.sigmoid(depth_raw)).clamp_min(1e-3)
```

理论范围是：

```text
depth ∈ [0.001, 8.0]
```

但 fused rasterizer 的近裁剪面是 `0.01`。即使不考虑裁剪面，`0.001m` 对机器人桌面场景也不是合理的物理深度。

### 4.2 render loss 会奖励“更容易覆盖图像”的浅深度

在早期优化中，把 Gaussian 拉近相机可以更容易覆盖像素、降低 SSIM/L1 loss。因此梯度会持续推动 `depth_raw` 变小。

由于：

```text
gaussian_render_loss_weight = 0.2
VLA_Loss ≈ 0.01
```

Gaussian 项在总 loss 中占主导，浅深度方向会被强烈优化。

### 4.3 深度接近 0 时 rasterizer 退化

当 depth 降到 `0.001` 或类似量级时：

- Gaussian 离相机极近；
- 投影后的协方差/半径进入数值退化区；
- rasterizer 不再产生有效覆盖；
- rendered 变成 background=0；
- 对 means / scales / opacities / SH 的梯度变成 0。

于是模型进入吸收态：

```text
rendered = 0
raw loss ≈ 0.976
GradNorm ≈ 0
```

这和正式 run 中 step 280 之后的状态完全一致。

### 4.4 不是权重变成 NaN 或 0

检查 step-1000 checkpoint 后：

```text
gaussian_decoder.* 权重 finite、非 0
gaussian_3d_adapter.* 权重 finite、非 0
```

例如：

```text
param_head.weight absmean ≈ 0.033
adapter.to_llm.weight absmean ≈ 0.050
query_tokens absmean ≈ 0.016
```

因此问题不是参数爆炸、NaN 或零权重，而是**激活/几何状态进入 rasterizer 退化区后梯度链断开**。

---

## 5. 另一个独立问题：新增 Gaussian 模块曾被零初始化

早期 run 中还发现 Gaussian 参数完全是常数：

```text
colors    = 0.5, std=0
opacities = 0.5, std=0
scales    = 0.00692749, std=0
rotations = 0
```

原因是官方 checkpoint 不包含 Gaussian 模块：

```text
gaussian keys in pretrain checkpoint: 0
```

模型加载时先执行 `to_empty()`，然后对缺失参数初始化。由于 `adanorm_time=true`，新增 Gaussian 参数也被错误地零初始化。所有层都是 0 时，decoder 输出恒定，梯度也无法恢复。

修复见 `lingbotvla/models/module_utils.py`：

- 新增 `_init_gaussian_weights()`
- 对 `gaussian_3d_adapter.*` 和 `gaussian_decoder.*` 使用显式随机初始化
- 不再让这些模块走 AdaNorm 的零初始化路径

---

## 6. 最终修复

### 6.1 深度范围修复

文件：

```text
lingbotvla/models/vla/geometry/gaussian_decoder.py
```

修改：

```python
depth = (0.5 + 7.5 * torch.sigmoid(depth_raw)).unsqueeze(-1)
```

范围变成：

```text
depth ∈ [0.5, 8.0]
```

这样优化器无法把 Gaussian 推到近裁剪面或数值退化区。

### 6.2 Gaussian 缺失模块初始化修复

文件：

```text
lingbotvla/models/module_utils.py
```

新增 `_init_gaussian_weights()`，并对缺失的 Gaussian 模块显式初始化：

```python
gaussian_missing_markers = (
    "gaussian_3d_adapter.",
    "gaussian_decoder.",
)
```

### 6.3 周期性渲染图对比

文件：

```text
lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py
tasks/vla/train_lingbotvla.py
```

新增环境变量：

```bash
LINGBOT_GAUSSIAN_DEBUG=1
LINGBOT_GAUSSIAN_DEBUG_LIMIT=3
LINGBOT_GAUSSIAN_SAVE_INTERVAL=300
LINGBOT_GAUSSIAN_DEBUG_DIR=/path/to/gaussian_debug
```

行为：

- 前 3 个 microbatch 保存 debug 图；
- 每 300 个 optimizer step 保存一次：
  - `step000300_idxNNN_rendered.png`
  - `step000300_idxNNN_target.png`
  - `step000300_idxNNN_compare.png`

`compare.png` 左边是 target，右边是 rendered。

训练循环会把 `global_step` 写入外层和内层模型，确保 hook 能读到正确步数。

---

## 7. 修复验证

### 7.1 最小复现验证

使用同样的 decoder / renderer / loss / target / camera，修复后：

```text
step 1:   raw=0.666, depth_mean=4.35, grad=0.50
step 10:  raw=0.195, depth_mean=2.51, grad=0.67
step 20:  raw=0.094, depth_mean=1.87, grad=0.70
step 40:  raw=0.075, depth_mean=1.75, grad=0.63
step 80:  raw=0.068, depth_mean=2.11, grad=0.43
step 200: raw=0.054, depth_mean=2.99, grad=0.29
step 400: raw=0.049, depth_mean=4.92, grad=0.31
```

结果：

- loss 持续下降；
- 没有反弹；
- rendered_mean 稳定在 `0.94–0.96`；
- GradNorm 始终非零；
- depth 在合理物理范围内移动。

### 7.2 正式训练验证

修复后的 1000 步 run 使用：

```text
output: /pfs/slfan/SIDAZHU/robotwin_gaussian_1000_depthfix_debug2
TensorBoard run: G_DEPTHFIX_DEBUG2
```

早期状态：

```text
step 3:
Loss      = 0.1330
VLA_Loss  = 0.0355
Gauss_raw = 0.4821
GradNorm  = 2.57
```

debug 日志中 step 正常传递：

```text
step=1 ...
step=2 ...
step=3 ...
```

不再出现 `step=-1`。周期图像保存在：

```text
/pfs/slfan/SIDAZHU/robotwin_gaussian_1000_depthfix_debug2/gaussian_debug
```

---

## 8. 经验总结

1. 新加入的 Gaussian 模块不在官方 checkpoint 中，必须显式随机初始化；`to_empty()` 后跳过初始化会留下近似全零参数。
2. 可学习的 3D depth 必须有物理合理下界；不能让优化器把 Gaussian 推到相机近裁剪面。
3. `rendered` 变成背景且 `GradNorm≈0` 是 rasterizer 梯度断掉的典型信号。
4. 单分支最小复现是定位该问题最有效的手段：它排除了 VLA / MoE / depth teacher 等所有干扰项。
5. 建议长期保留周期性 rendered/target 对比图，作为 Gaussian 分支健康监控手段。
