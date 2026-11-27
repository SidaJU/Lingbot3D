# Gaussian 渲染偏差根因：多视角渲染监督与单参考视角深度图表示错配

## 结论

Gaussian RGB 渲染长期停留在“灰褐色模糊团块”，根因不是 rasterizer、相机内参、权重加载或深度下界，而是：

> 项目用 **top 参考相机的一张深度图** 反投影生成整片 Gaussian，却同时用 **top + 左右腕部三路相机** 做 RGB 渲染监督。
>
> 腕部相机看到的并不是同一张几何正确的平面，它们在三路平均 loss 中把 Gaussian 往“降低透明度、抹平颜色”的方向推，压制了 top 参考视角本身的上色学习。

官方 GaussianDream 在有第三人称/主视角时，渲染监督只使用主视角（`agent`），不是三路视角平均。本项目配置成三路，造成了表示与监督目标不一致。

---

## 问题图像

对比图中左侧是 target，右侧是 rendered。

### 三视角监督：step 300 仍是灰褐模糊团块

![三视角 step300](../runs/recon_depthfix_300/gaussian_debug/step000300_idx003_camera_top_compare.png)

即使关闭 VGGT 深度目标，三视角监督下也没有恢复：

![三视角关闭 VGGT 深度 step300](../runs/recon_nodepth_300/gaussian_debug/step000300_idx003_camera_top_compare.png)

### 单视角监督：step 200 / 300 开始恢复结构

![单视角 step200](../runs/recon_topview_300/gaussian_debug/step000200_idx003_camera_top_compare.png)

![单视角 step300](../runs/recon_topview_300/gaussian_debug/step000300_idx003_camera_top_compare.png)

---

## 典型特征

- target 有清晰桌面纹理、物体轮廓和颜色。
- rendered 长期是低对比度灰褐色，几乎无物体边界。
- rendered 的颜色范围卡在 0.5 附近，无法学出白色/亮色区域。
- opacity 均值持续下降，Gaussian 变得几乎透明。
- top 参考视角 loss 在高位平台不降；腕部视角 loss 反而更低。
- 改为只渲染 top 后，颜色、对比度、loss 立刻明显改善。

---

## 对照实验与指标

三次 run 均基于代码 `a05ea1d`，同样 300 步、同样学习率、同样 batch size；只改变深度目标和渲染视角。

| Run | 渲染视角 | VGGT 深度目标 | top loss 最后窗口 | colors max（step300） | rendered std（step300） | rendered max（step300） | opacity mean（step300） |
|---|---|---|---:|---:|---:|---:|---:|
| `recon_depthfix_300` | top + 左右腕部 | 开 | 0.241 | 0.563 | 0.043 | 0.496 | 0.062 |
| `recon_nodepth_300` | top + 左右腕部 | 关 | 0.236 | 0.558 | 0.049 | 0.552 | 0.079 |
| `recon_topview_300` | **仅 top** | 关 | **0.155** | **0.756** | **0.158** | **0.821** | **0.222** |

分视角 loss 的关键差异：

| 50 步窗口 | 1–50 | 51–100 | 101–150 | 151–200 | 201–250 | 251–300 |
|---|---:|---:|---:|---:|---:|---:|
| 三视角 top loss | 0.262 | 0.243 | 0.229 | 0.242 | 0.227 | 0.236 |
| 单视角 top loss | 0.247 | 0.246 | 0.193 | 0.162 | 0.153 | **0.155** |

三视角下参考视角 top 的 loss 卡在 0.23 左右；单视角下持续下降。

---

## 原理分析

### 1. 当前 Gaussian 表示本质是“单参考视角深度片”

`GaussianParameterDecoder` 用一张 depth map 反投影得到 Gaussian 位置：

- depth map 是 256×256；
- 使用主视角 `camera_top` 的 K/T；
- 每个 depth 像素反投影成一个世界坐标 Gaussian；
- 所有视角共用这一套 Gaussian。

这套表示天然适合重建参考视角：depth 像素与 top 图像像素一一对应，模型只要学习的颜色和 opacity 就能拟合 top target。

但它不是完整的多视图一致几何。左右腕部相机看到的是这张深度片在另一个视角下的投影，位置天然不对，尤其初始深度接近 4.25 m 时，腕部看到的是一张远离真实桌面的平面。

### 2. 三视角监督产生错误梯度方向

对 top 视角，降低 loss 的正确方式是：

- 学出正确颜色；
- 学出局部 opacity；
- 调整 depth/scale 使纹理对齐。

对腕部视角，由于几何本身不对，最低成本的降 loss 方式是：

- 把 Gaussian opacity 降低，让渲染接近黑色背景；
- 把颜色推向均值，降低局部误差；
- 不需要真正学出正确几何。

三路 loss 平均后，两个腕部视角与一个 top 视角共同竞争。腕部视角更容易降 loss，于是梯度主导结果：

- opacity 从 0.5 掉到 0.06–0.08；
- `colors max` 卡在 0.56 左右；
- rendered 保持灰褐色；
- top 视角无法学会亮色和结构。

### 3. 官方实现只监督主视角

官方 GaussianDream 在选择渲染监督视角时，如果 observation 中存在第三人称/主视角（`agent`），只用该视角：

```python
render_views = ["agent"] if "agent" in valid_views else valid_views
```

本项目原来配置为：

```yaml
gaussian_render_views:
  - camera_top
  - camera_wrist_left
  - camera_wrist_right
```

这就是表示与监督目标错配的地方。

---

## 解决办法（已验证）

把渲染监督视角改成仅 top：

```yaml
gaussian_render_views:
  - camera_top
```

已生成配置：

```text
configs/vla/robotwin/robotwin_4d_topview.yaml
```

相对早期多视角版本，它不再启用：

```yaml
    - camera_wrist_left
    - camera_wrist_right
```

验证脚本：

```text
tools/run_topview_300.sh
```

核心代码未修改，代码版本保持：

```text
a05ea1d Restore the physical lower bound on Gaussian depth
```

---

## 后续建议

1. **短期正式配置**：使用 `gaussian_render_views: [camera_top]`，与官方单主视角监督一致。
2. **如果需要腕部视角监督**：不能直接把腕部 RGB loss 加到单参考视角深度片上；需要改成真正的多视图一致 Gaussian 表示，例如多视角 depth map、逐视角 Gaussian 场，或显式世界几何重建。
3. **VGGT 深度目标是另一个独立问题**：本次对照证明它不是渲染差的主因。若后续启用，仍需修复其“原始深度非稳定度量”和 `conf >= 0.5` 门限失效问题。
