# Codex Model Architecture Check

本文档记录当前 SpecFlow 模型从输入到输出路径中的主要架构疑问与后续建议。

## 当前输入到输出路径

训练时 batch 主要包含：

```text
ctrl_expr: (B, G)
pert_expr: (B, G)
pert_mask: (B, G)
spectral: {"go": (B, G, Kgo), "coexp": (B, G, Kcoexp)}
time: (B, 1)
x_t: (B, G)
```

模型路径：

```text
GO/coexp 谱嵌入 + pert_mask
    -> DualGraphSpectralFusion
    -> fused spectral embedding: (B, G, spectral_dim)

ctrl_expr + fused spectral + pert_mask
    -> GeneTokenEncoder
    -> gene_tokens: (B, G, d_model)

gene_tokens
    -> AttentivePooling
    -> cell_condition: (B, d_model)

x_t + ctrl_expr + fused spectral + pert_mask + cell_condition + time
    -> VelocityField
    -> velocity: (B, G)
```

训练目标：

```text
x_0 = ctrl_expr + sigma * noise
x_t = (1 - t) * x_0 + t * pert_expr
target_velocity = pert_expr - x_0
loss = MSE(predicted_velocity, target_velocity)
```

推理时从 `x_0 = ctrl_expr + sigma * noise` 出发，用 Euler ODE 积分速度场，输出预测扰动表达 `x_1`。

## 主要疑问

### 1. 输出没有非负约束

表达数据通常是 log-normalized 非负值，但当前模型在无约束实数空间中积分：

```text
x_0 = ctrl_expr + sigma * noise
state = state + step_size * velocity
```

因此生成的表达值可能为负。

这不一定导致训练失败，但如果后续评估或生物解释默认表达非负，需要明确处理方式。可选方案：

- 明确模型工作在标准化实数空间，允许负值。
- 推理输出后 `clamp_min(0)`。
- 使用 `softplus` 或其他非负参数化。
- 对输入表达做 z-score 标准化，并在输出后反标准化。

### 2. 控制细胞和扰动细胞是随机配对

当前 Dataset 从控制池随机采样 `ctrl_expr`，从扰动池随机采样 `pert_expr`。训练目标中的：

```text
target_velocity = pert_expr - ctrl_expr
```

并不对应同一个细胞的真实转变，而是随机 coupling。

这符合未配对单细胞扰动数据的实际情况，但目标速度噪声较大。潜在改进：

- 使用 minibatch optimal transport pairing。
- 使用条件均值或近邻匹配降低 coupling 噪声。
- 加强分布级目标，例如 MMD 或 energy distance。
- 对比随机 pairing 与 OT pairing 的消融。

### 3. 速度场的跨基因交互能力偏弱

当前 `VelocityField` 是逐基因局部 MLP 加一个全局 `cell_condition`。每个基因的局部输入为：

```text
x_t(g_i), ctrl_expr(g_i), spectral_i, pert_mask_i
```

基因间交互主要通过：

- 谱嵌入提供结构坐标；
- attentive pooling 提供全局条件向量。

速度场内部没有 Transformer、cross-gene attention 或 message passing。因此复杂的基因间依赖可能表达不足。

潜在改进：

- 增加轻量 Transformer / Performer / Set Transformer 速度场版本。
- 在速度场中加入 gene-token cross attention。
- 做 MLP 速度场 vs attention 速度场消融。

### 4. 可解释性信号没有从 forward 正常返回

`SpecFlow.encode_condition()` 已能拿到：

```text
gene_attention
cross_graph
go_scale
coexp_scale
spectral_embedding
```

但 `SpecFlow.forward()` 目前只返回：

```text
velocity, attention
```

训练没有问题，但论文中的跨图融合权重、多尺度权重和谱偏移分析需要额外调用 `encode_condition()`，否则不会自然进入评估或可视化输出。

建议：

- 增加单独的 analysis/visualization 脚本。
- 或让 forward 支持 `return_aux=True`。
- 在 evaluation 阶段按扰动条件导出 attention 和 fusion weights。

### 5. ODE 推理目前只有固定步长 Euler

当前推理为：

```text
state = state + step_size * velocity
```

Euler 实现简单、速度快，但在速度场较尖锐或步数较小时误差可能偏大。

建议：

- 增加 midpoint / RK4。
- 可选接入 `torchdiffeq` 的 `dopri5`。
- 做 ODE solver 与步数敏感性分析。

## 优先级建议

短期优先处理：

1. 明确表达空间：是否允许负值。
2. 改善未配对 control/perturbation 的 coupling。
3. 给速度场增加一个具备跨基因交互能力的版本或消融。

中期处理：

1. 补可解释性导出脚本。
2. 增加 ODE solver 选项。
3. 对图融合、尺度融合和动态图修正做完整消融。
