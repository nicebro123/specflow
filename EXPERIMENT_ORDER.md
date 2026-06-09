# SpecFlow 实验补齐顺序

本文档用于安排 `code_0602_opo` 版本后续实验。目标不是复刻 scDFM，而是在同一评测协议下补齐必要证据，同时突出 SpecFlow 的核心动机：扰动信号如何沿基因结构传播并导致表达状态变化。

## 总原则

1. 主结果以 `code_0602_opo` 为基准，不混用 0529/0531/0603 代码结果。
2. 训练、测试、baseline 和消融必须使用同一 split、同一 gene set、同一 cell_eval 协议。
3. 每个实验目录必须保留 `run_config.yaml`、`train.log`、`training_summary.json`、`agg_results.csv`、`results.csv`、`scdfm_evaluation_summary.json`。
4. 主表优先报告 mean ± std，不只报告单次最好结果。
5. 和 scDFM 对比时，必须标清楚结果来源：同代码复现、同协议重跑，或 scDFM paper reported。

## 阶段 0：锁定评测协议

目标：先固定所有实验使用的协议，避免后续结果不可比。

需要确认：

| 项目 | 要求 |
|---|---|
| 数据集 | Norman 作为主实验；ComboSciPlex 作为后续扩展实验 |
| split | 使用 scDFM 对齐 split；如果使用 `split_results.pkl`，优先跑 fold 0-4 |
| 评测 | 使用 cell_eval |
| 主指标 | Pearson Delta、MSE、MAE、DE Spearman、DS |
| 需补指标 | L2、Pearson Delta-hat、Pearson Delta-hat20 |
| 主代码 | `code_0602_opo` |

阶段完成标准：

```text
所有后续实验都能说明：训练 split、测试 split、gene set、评测脚本完全一致。
```

## 阶段 1：Norman Additive 主结果多 fold

这是最高优先级。当前 `output_0602` 只能作为单次主结果，不能替代稳定性结果。

建议实验：

| 实验 | fold | seed | 目的 |
|---|---:|---:|---|
| full_additive_f0 | 0 | 42 | 主模型第 1 个 split |
| full_additive_f1 | 1 | 42 | 主模型第 2 个 split |
| full_additive_f2 | 2 | 42 | 主模型第 3 个 split |
| full_additive_f3 | 3 | 42 | 主模型第 4 个 split |
| full_additive_f4 | 4 | 42 | 主模型第 5 个 split |

输出表格：

| Method | Pearson Delta | MSE | MAE | DE Spearman | DS |
|---|---:|---:|---:|---:|---:|
| SpecFlow | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std |

阶段完成标准：

```text
得到 SpecFlow 在 Norman additive 上的 mean ± std，并确认 0602 单次最好结果不是偶然 seed。
```

## 阶段 2：Control 和 Additive Baseline

这一步必须补。它们是最基础的非训练 baseline，也是证明 SpecFlow 超越简单表达叠加的关键。

建议 baseline：

| Baseline | 定义 | 是否训练 |
|---|---|---|
| Control | 直接用 control 表达作为预测 | 否 |
| Additive | double perturbation = control + delta(A) + delta(B) | 否 |

要求：

1. 使用和 SpecFlow 完全相同的 test split。
2. 使用和 SpecFlow 完全相同的评测基因集合。
3. 使用 cell_eval 输出同一套指标。
4. 每个 fold 都要算，最后也报告 mean ± std。

输出表格：

| Method | Pearson Delta | MSE | MAE | DE Spearman | DS |
|---|---:|---:|---:|---:|---:|
| Control | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std |
| Additive | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std |
| SpecFlow | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std |

阶段完成标准：

```text
SpecFlow 相比 Control 和 Additive 的提升明确，尤其是 Pearson Delta 和 DE Spearman。
```

## 阶段 3：Norman Holdout 泛化实验

这是和 scDFM 对齐时最重要的泛化实验。Additive split 主要证明插值能力，holdout split 才能证明未见扰动泛化。

建议实验：

| 实验 | 配置 | 目的 |
|---|---|---|
| holdout_full | `configs/norman_holdout.yaml` | 未见扰动主结果 |
| holdout_single | holdout single perturbations | 未见单扰动泛化 |
| holdout_double | holdout double perturbations | 未见双扰动组合泛化 |

输出表格：

| Setting | Method | Pearson Delta | MSE | MAE | DE Spearman | DS |
|---|---|---:|---:|---:|---:|---:|
| Holdout Single | SpecFlow | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std |
| Holdout Double | SpecFlow | mean ± std | mean ± std | mean ± std | mean ± std | mean ± std |

阶段完成标准：

```text
SpecFlow 在 unseen single 和 unseen double 上都有结果，并且能与 scDFM 的 holdout 表格对齐。
```

## 阶段 4：核心消融实验

消融要服务于论文贡献，不要无节制增加。主消融建议优先验证双图结构、谱传播、OT coupling 和 delta correlation loss。

当前已规划的消融：

| 实验 | 改动 | 验证问题 |
|---|---|---|
| full | 无 | 完整 SpecFlow |
| no_spectral_propagation | `model.spectral_propagation=false` | 谱传播是否有效 |
| no_ot_coupling | `flow.ot_coupling=false` | OT 配对是否有效 |
| no_delta_corr | `flow.delta_corr_weight=0.0` | delta correlation loss 是否提升 Pearson Delta |
| go_only | `model.graph_mode=go` | GO 图单独是否足够 |
| coexp_only | `model.graph_mode=coexp` | 共表达图单独是否足够 |
| graph_none | `model.graph_mode=none` | 无图结构时性能下降多少 |
| mmd_on | `flow.mmd_weight=0.05` | MMD 是否有额外收益 |

建议顺序：

1. 先在 Norman additive fold 1 上跑完整消融。
2. 再把最关键的 4 个消融放到 holdout 上验证：

```text
full
graph_none
no_spectral_propagation
no_delta_corr
```

输出表格：

| Variant | Pearson Delta | MSE | MAE | DE Spearman | DS |
|---|---:|---:|---:|---:|---:|
| Full SpecFlow | value | value | value | value | value |
| w/o Graph | value | value | value | value | value |
| w/o Spectral Propagation | value | value | value | value | value |
| w/o Delta Corr | value | value | value | value | value |

阶段完成标准：

```text
能清楚证明：双图结构和图上传播不是装饰，而是 SpecFlow 的主要性能来源。
```

## 阶段 5：补齐 scDFM 指标

当前结果已经有大部分 cell_eval 指标，但为了严格对齐 scDFM 表格，需要额外补：

| 指标 | 用途 |
|---|---|
| L2 | scDFM 主表常用误差指标 |
| Pearson Delta-hat | 预测扰动方向的相关性扩展指标 |
| Pearson Delta-hat20 | top 20 differential genes 上的扰动相关性 |

建议：

1. 如果 cell_eval 已经能直接输出这些指标，优先直接读取。
2. 如果当前 summary 没有这些字段，则单独补一个指标统计步骤。
3. 不要改变训练结果，只对已有预测输出重新计算指标。

阶段完成标准：

```text
论文主表可以做到指标维度和 scDFM 基本一致。
```

## 阶段 6：可视化实验

可视化不是装饰，它要解释 SpecFlow 的机制。

建议图：

| 图 | 目的 |
|---|---|
| predicted delta vs true delta scatter | 展示 Pearson Delta 的直观含义 |
| UMAP: control / true perturbed / SpecFlow prediction / baseline | 展示表达状态迁移 |
| top DE genes boxplot | 展示关键差异基因预测是否准确 |
| graph propagation heatmap | 展示扰动信号在 GO/co-expression 图上的传播 |
| target-to-response gene path | 展示从扰动靶基因到响应基因的结构路径 |

优先 case：

```text
优先选择 scDFM 论文中出现过的 Norman perturbation case，
这样可视化可以和 scDFM 的叙事对齐，同时突出我们的结构传播解释。
```

阶段完成标准：

```text
至少有 1 张 UMAP、1 张 top DE gene boxplot、1 张结构传播可视化。
```

## 阶段 7：ComboSciPlex 扩展实验

这个阶段放在 Norman 主实验之后。ComboSciPlex 可以增强论文完整性，但工程成本更高。

建议实验：

| 实验 | 目的 |
|---|---|
| combosciplex_full | 药物组合扰动主结果 |
| combosciplex_control | control baseline |
| combosciplex_additive | additive baseline |

需要注意：

1. drug-to-target 映射必须可靠。
2. split 必须和 scDFM 对齐。
3. 不要在 Norman 结果还没稳定前投入过多时间。

阶段完成标准：

```text
如果 ComboSciPlex 结果稳定，可以作为主文第二数据集；否则放 appendix 或后续版本。
```

## 阶段 8：超参数实验

超参数实验放在 appendix，目的是证明模型稳定，而不是调参刷榜。

当前建议 sweep：

| 参数 | 候选值 |
|---|---|
| `flow.delta_corr_weight` | 0.01, 0.03, 0.05 |
| `flow.sigma` | 0.15, 0.2, 0.25 |
| `model.propagation_channels` | 4, 8, 16 |
| `inference.n_control_cells` | 128, 256, 512 |

输出表格：

| Parameter | Value | Pearson Delta | MSE | MAE | DE Spearman |
|---|---:|---:|---:|---:|---:|

阶段完成标准：

```text
证明 0602 的主设置不是偶然超参数，而是在合理范围内稳定有效。
```

## 推荐总执行顺序

最小强实验版本：

```text
1. Norman additive full model, fold 0-4
2. Control / Additive baseline, fold 0-4
3. Norman holdout full model
4. Norman holdout single / double 结果拆分
5. 核心消融：full, graph_none, no_spectral_propagation, no_delta_corr
6. 补齐 L2 / Delta-hat / Delta-hat20
7. 可视化：UMAP, top DE genes, graph propagation
```

完整论文版本：

```text
1. 最小强实验版本全部完成
2. 完整 additive 消融
3. holdout 核心消融
4. ComboSciPlex full + baseline
5. 超参数 sweep
6. 训练效率和资源对比
```

## 4 GPU 排队建议

第一批：Additive 主结果多 fold。

```text
GPU0: full_additive_f0 -> full_additive_f4
GPU1: full_additive_f1
GPU2: full_additive_f2
GPU3: full_additive_f3
```

第二批：核心消融。

```text
GPU0: full -> go_only
GPU1: no_spectral_propagation -> coexp_only
GPU2: no_ot_coupling -> graph_none
GPU3: no_delta_corr -> mmd_on
```

第三批：Holdout。

```text
GPU0: holdout_full_f0
GPU1: holdout_full_f1
GPU2: holdout_full_f2
GPU3: holdout_full_f3
```

第四批：超参数和可视化补充。

```text
GPU0: delta sweep
GPU1: sigma sweep
GPU2: propagation channel sweep
GPU3: inference control cell sweep
```

## 论文表格安排

主文建议放：

| 表格 | 内容 |
|---|---|
| Table 1 | Norman additive main comparison |
| Table 2 | Norman holdout single/double comparison |
| Table 3 | Core ablation |
| Figure 1 | Method overview |
| Figure 2 | UMAP / prediction visualization |
| Figure 3 | Graph propagation mechanism |

Appendix 建议放：

| 表格 | 内容 |
|---|---|
| Appendix Table A1 | All fold results |
| Appendix Table A2 | Hyperparameter sweep |
| Appendix Table A3 | Extra ablations |
| Appendix Table A4 | Runtime and resource usage |
| Appendix Figure A1 | More perturbation case studies |

## 当前最高优先级

下一步最应该先做：

```text
Norman additive full model fold 0-4
Control baseline fold 0-4
Additive baseline fold 0-4
```

只有这三项完成后，SpecFlow 的主结果才适合写成和 scDFM 严格对齐的论文主表。
