# SpecFlow / code_0602_opo 最小补实验计划

更新时间：2026-06-22
代码目录：`/mnt/infini-data/test/quan_space/codespace/aidd_0604/code_0602_opo`
推荐环境：`/mnt/infini-data/test/quan_space/envs/aidd/bin/python`

## 0. 当前判断

当前 `configs/experiments/*.yaml` 计划内共有 73 个 run。按实验记录和指标文件检查，65 个 run 完整，8 个 run 不完整。

完整实验记录至少应包含：

```text
run_config.yaml
train.log
training_summary.json      # baseline 可没有
agg_results.csv
results.csv
scdfm_evaluation_summary.json
```

当前真正影响论文主线的缺口是：

1. Norman additive baseline 的 4 个 additive fold 缺失。
2. holdout 结果需要拆成 Single / Double 子集，才能和 scDFM reported table 对齐。
3. L2 / Pearson Delta-hat / Pearson Delta-hat20 / DS 等 scDFM 表格指标需要补齐或确认。
4. graph mechanism 的消融和解释性图需要整理成 paper-facing evidence。
5. `mmd_on` 消融失败，但它不是主贡献，可放在第二优先级。

## 1. 论文创新点应如何收缩

不建议把故事写成“我们提出一个更强的 flow matching 模型并全面超过 scDFM”。当前证据更适合下面这个版本：

> Existing distributional perturbation models predict population shifts but provide limited mechanistic grounding in gene regulatory structure. SpecFlow injects GO/co-expression spectral structure into control-anchored flow matching, enabling competitive prediction and interpretable perturbation signal propagation.

主贡献建议写成三点：

1. **Control-anchored flow matching**：从 control cell state 出发生成扰动响应。
2. **Gene-structure conditioning**：使用 GO + co-expression 双图谱嵌入，为扰动响应提供结构先验。
3. **Mechanistic interpretability**：通过传播热图、target-to-response path、fusion weights 解释扰动信号如何沿基因结构扩散。

目前最强的定量证据是 graph 相关消融：

```text
Norman additive fold1:
full       Pearson Delta = 0.8804
graph_none Pearson Delta = 0.1310

Norman holdout fold1:
full       Pearson Delta = 0.4059
graph_none Pearson Delta = 0.0944

ComboSciPlex:
full       Pearson Delta = 0.8272
graph_none Pearson Delta = 0.5092
```

需要注意：`no_spectral_propagation` 在 holdout 和 ComboSciPlex 上并不总是弱于 full，所以不要过度声称 spectral propagation 总是提升精度。更稳妥的表述是：graph-structured conditioning 是主要性能来源，spectral propagation 提供解释性机制，其定量收益依赖 setting。

## 2. 最小补实验清单

### P0：刷新当前汇总表

目的：让后续论文表格基于最新产物，而不是旧的 `20260617/20260618` summary。

命令：

```bash
cd /mnt/infini-data/test/quan_space/codespace/aidd_0604/code_0602_opo

/mnt/infini-data/test/quan_space/envs/aidd/bin/python scripts/summarize_experiments.py \
  --root outputs \
  --output outputs/experiment_summary_refreshed_20260622.csv
```

验收标准：

```text
outputs/experiment_summary_refreshed_20260622.csv 存在
complete / failed / planned 状态与当前 run 目录一致
adaptive gate 已完成的 run 不再被误标为 running/planned
```

### P0：补 Norman additive baseline 的 4 个 fold

当前失败 run：

```text
outputs/20260603_norman_baselines_0602/01_additive_f0_s42_g0
outputs/20260603_norman_baselines_0602/03_additive_f1_s42_g1
outputs/20260603_norman_baselines_0602/05_additive_f2_s42_g2
outputs/20260603_norman_baselines_0602/09_additive_f4_s42_g0
```

失败原因：

```text
evaluate_baseline.py 默认 --missing-single error。
部分 test double perturbation 的某个 single perturbation delta 不在 train 中，
例如 ZC3HAV1、OSR2、TGFBR2 等，导致 KeyError。
```

处理策略：

使用 no-leakage fallback：

```text
--missing-single zero
```

含义：如果 train 中没有某个 single perturbation delta，则该 single delta 按 0 处理。这个策略不会泄漏 test 信息，并且 ComboSciPlex baseline 已经采用同类策略。

建议输出到新 study，避免覆盖旧失败记录：

```text
outputs/20260622_norman_additive_baseline_zero_0602/
```

建议命令：

```bash
cd /mnt/infini-data/test/quan_space/codespace/aidd_0604/code_0602_opo

PY=/mnt/infini-data/test/quan_space/envs/aidd/bin/python
OUT=outputs/20260622_norman_additive_baseline_zero_0602

CUDA_VISIBLE_DEVICES=0 $PY scripts/evaluate_baseline.py \
  --config configs/norman.yaml \
  --output-dir $OUT/additive_f0_s42_missing_zero \
  --baseline additive \
  --fold 0 \
  --seed 42 \
  --missing-single zero

CUDA_VISIBLE_DEVICES=1 $PY scripts/evaluate_baseline.py \
  --config configs/norman.yaml \
  --output-dir $OUT/additive_f1_s42_missing_zero \
  --baseline additive \
  --fold 1 \
  --seed 42 \
  --missing-single zero

CUDA_VISIBLE_DEVICES=2 $PY scripts/evaluate_baseline.py \
  --config configs/norman.yaml \
  --output-dir $OUT/additive_f2_s42_missing_zero \
  --baseline additive \
  --fold 2 \
  --seed 42 \
  --missing-single zero

CUDA_VISIBLE_DEVICES=3 $PY scripts/evaluate_baseline.py \
  --config configs/norman.yaml \
  --output-dir $OUT/additive_f4_s42_missing_zero \
  --baseline additive \
  --fold 4 \
  --seed 42 \
  --missing-single zero
```

验收标准：

```text
每个 run 都有：
agg_results.csv
results.csv
scdfm_evaluation_summary.json
train.log 无 Traceback / KeyError
```

### P0：生成 Norman holdout Single / Double 子集表

目的：scDFM reported table 按 holdout Single / Double 分开报，我们必须拆分才能对齐。

命令：

```bash
cd /mnt/infini-data/test/quan_space/codespace/aidd_0604/code_0602_opo

/mnt/infini-data/test/quan_space/envs/aidd/bin/python scripts/summarize_holdout_subsets.py \
  --root outputs/20260603_holdout_full_0602 \
  --output outputs/20260603_holdout_full_0602/holdout_single_double_summary.csv
```

验收标准：

```text
outputs/20260603_holdout_full_0602/holdout_single_double_summary.csv 存在
至少包含 Single 和 Double 两类结果
可与 paper_baselines/scdfm_reported_metrics.csv 的 norman_holdout 对齐
```

### P0：补齐 scDFM 主表指标

目标指标：

```text
L2
Pearson Delta-hat
Pearson Delta-hat20
DS / discrimination_score_l2
```

当前问题：

`experiment_summary_current.csv` 里部分 run 的 `pearson_delta_hat` 和 `pearson_delta_hat20` 是 NaN。需要确认是 cell_eval 没输出，还是 summary 脚本没读取。

检查命令：

```bash
cd /mnt/infini-data/test/quan_space/codespace/aidd_0604/code_0602_opo

/mnt/infini-data/test/quan_space/envs/aidd/bin/python - <<'PY'
import pandas as pd
from pathlib import Path

summary = pd.read_csv("outputs/experiment_summary_refreshed_20260622.csv")
cols = [
    "run_name",
    "pearson_delta",
    "mse",
    "mae",
    "l2_mean",
    "pearson_delta_hat",
    "pearson_delta_hat20",
    "discrimination_score_l2",
]
print(summary[[c for c in cols if c in summary.columns]].head(30).to_string(index=False))
print(summary[[c for c in ["l2_mean", "pearson_delta_hat", "pearson_delta_hat20"] if c in summary.columns]].isna().mean())
PY
```

验收标准：

```text
Norman additive / Norman holdout / ComboSciPlex 主结果均有 scDFM 对齐指标
如果某指标无法从 cell_eval 直接得到，需要补一个 post-hoc 统计脚本，并在论文中说明计算方式
```

### P1：整理 graph mechanism 消融表

需要进入主文或 appendix 的核心消融：

```text
full
graph_none
go_only
coexp_only
no_spectral_embedding
no_control_anchor
no_delta_corr
no_ot_coupling
```

已有产物：

```text
outputs/20260603_ablation_0602/
outputs/20260612_core_components_0602/
outputs/20260603_holdout_ablation_0602/
outputs/20260603_combosciplex_0602/
```

建议生成 paper-facing 表：

```text
outputs/paper_ablation_graph_mechanism_20260622.csv
```

验收标准：

```text
表中至少包含：
setting
variant
Pearson Delta
MSE
MAE
DE Spearman
DS
```

解释重点：

```text
dual graph / graph conditioning 是性能主来源；
single graph 有效但不足；
no graph 明显崩；
spectral propagation 的收益依赖 setting，不做过度 claim。
```

### P1：解释性 case study

目的：把“图结构驱动”从一个普通 architecture trick 变成 paper story。

建议产物：

```text
outputs/figures/
  propagation_heatmap_case_*.png
  target_to_response_path_case_*.png
  top_de_gene_boxplot_case_*.png
  fusion_weights_case_*.csv
```

建议选择 case：

1. Norman 中 scDFM paper 或 benchmark 常用的 perturbation case。
2. SpecFlow 明显优于 graph_none 的 case。
3. DE genes 数量足够、GO/coexp path 可解释的 case。

验收标准：

```text
至少 2-3 个 case；
每个 case 有 prediction vs true 的 DE gene 证据；
至少一个 case 有 GO/coexp propagation 或 fusion weight 解释。
```

### P2：重跑 mmd_on 消融

当前失败：

```text
outputs/20260603_ablation_0602/07_mmd_on_f1_s42_g3
```

失败原因：

```text
CUDA OutOfMemoryError
```

建议策略：

1. 降低 batch size。
2. 设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
3. 如果仍 OOM，则把 MMD 从主贡献中移除，只在 limitation 中说明。

建议新输出：

```text
outputs/20260622_mmd_retry_0602/mmd_on_f1_s42_bs_reduced/
```

验收标准：

```text
完成 200000 steps 并输出 cell_eval 指标；
如果失败，不影响主线。
```

## 3. 推荐执行顺序

### 第一批：不需要长时间 GPU 训练

```text
1. 刷新 summary
2. 生成 holdout Single / Double 子集表
3. 检查 scDFM 指标缺失情况
4. 重跑 Norman additive baseline zero fallback
```

说明：baseline evaluation 会用 GPU/CPU 生成 h5ad 和跑 cell_eval，但不是 200k-step 训练，成本远低于模型训练。

### 第二批：整理表格和解释性证据

```text
5. 生成 graph mechanism 消融表
6. 选择解释性 case
7. 导出 propagation / fusion / DE gene 可视化数据
```

### 第三批：可选训练

```text
8. mmd_on retry
9. ODE solver / ode_steps sensitivity
10. 如果 reviewer story 需要，再补 adaptive gate 的最终汇总
```

## 4. 开跑前检查清单

```bash
cd /mnt/infini-data/test/quan_space/codespace/aidd_0604/code_0602_opo

# 环境
/mnt/infini-data/test/quan_space/envs/aidd/bin/python - <<'PY'
import sys, pandas, yaml
from cell_eval import MetricsEvaluator
print(sys.executable)
print("pandas", pandas.__version__)
print("cell_eval ok")
PY

# 数据
ls -lh data/norman.h5ad data/split_results.pkl data/splits/norman_holdout.pkl

# 磁盘
df -h /mnt/infini-data

# 当前 GPU
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

## 5. 今日建议行动

今天优先完成：

```text
A. 刷新 summary
B. 生成 holdout single/double
C. 开 4 个 Norman additive baseline zero fallback
```

完成后再判断：

```text
是否需要跑 mmd_on retry；
是否需要写额外脚本补 pearson_delta_hat / pearson_delta_hat20；
解释性 case 选哪几个。
```
