# SpecFlow

**谱引导的控制锚定流匹配，用于单细胞扰动响应预测**
(Spectral-Guided Control-Anchored Flow Matching for Single-Cell Perturbation Prediction)

SpecFlow 从控制（未扰动）细胞状态出发，沿基因调控网络的谱嵌入把细胞"推"到扰动后状态。评估流程与 **scDFM 的 `cell_eval` 协议完全对齐**，报告的指标可直接与 scDFM benchmark 对比。

---

## 快速开始

> 公开仓库**不包含数据集、训练输出、checkpoint 或 cell_eval 产物**。请按下方
> “数据准备” 自行放置 `data/` 目录；`.gitignore` 默认忽略这些大文件。

> **两条命令即可。第 1 条必须先执行，否则会报 `ModuleNotFoundError: No module named 'specflow'`。**

```bash
# 1. 安装包（开发模式）+ 数据依赖 + scDFM cell_eval 评估依赖
pip install -e ".[data]"
pip install cell-eval
python - <<'PY'
from cell_eval import MetricsEvaluator
print("cell_eval import ok")
PY

# 2. 训练 + 自动用 cell_eval 评估（与 scDFM 对齐）
python scripts/train.py --config configs/norman.yaml --output-dir outputs/run1
```

第 2 条命令会：训练模型 → 加载最优 checkpoint → 在**同一个 split fold** 的测试集上用 `cell_eval` 评估 → 写出 `results.csv` / `agg_results.csv`。

---

## 安装

SpecFlow 是 `src/` 布局的包，**运行任何脚本前必须先安装**。

```bash
# 基础（仅模型 + 训练）
pip install -e .

# 加数据预处理（h5ad 读取、HVG 筛选）
pip install -e ".[data]"

# 加测试 / 开发
pip install -e ".[data,dev]"

# scDFM 对齐评估所需（单独安装；安装后必须能 import cell_eval）
pip install cell-eval
python - <<'PY'
from cell_eval import MetricsEvaluator
print("cell_eval import ok")
PY
```

| 依赖组 | 包 | 用途 |
|--------|-----|------|
| 基础 | torch, numpy, scipy, PyYAML, tqdm | 模型与训练 |
| `data` | anndata, scanpy | h5ad 加载、预处理 |
| `dev` | pytest | 测试 |
| `cell-eval`（独立） | cell_eval, polars | **scDFM 对齐评估（必需）** |

未安装可用的 `cell_eval` 模块时，`--evaluation-protocol cell_eval` 会报错；可临时改用 `--evaluation-protocol internal` 走内置指标，或加 `--write-anndata-only` 只写 `pred.h5ad`/`real.h5ad`。如果 `pip install cell-eval` 后仍然 `ModuleNotFoundError`，说明当前 Python/依赖解析到了不可用版本，请安装 scDFM 使用的 `cell_eval` 包版本或切换到 Python 3.10/3.11 后重装。

**系统要求**：Python ≥ 3.9，PyTorch ≥ 2.0（训练建议 CUDA，显存 ≥ 16GB）。服务器复现实验建议使用 Python 3.10/3.11；`anndata/scanpy` 写 h5ad 依赖 Pandas 2.x，因此依赖中固定了 `pandas<3`。

---

## 数据准备

本仓库只提供代码、配置和实验启动脚本，不随仓库分发数据。复现实验前需要准备：

```text
data/
  norman.h5ad                      # 单细胞扰动数据集（obs 含 condition 列）
  split_results.pkl                # scDFM 的交叉验证 split（5 个 fold）
  gene_ontology/
    go_annotations.gaf             # GO 注释文件（GAF 格式，双图模式需要）
```

- **norman.h5ad**：来自 scPerturb / Norman et al. 2019（CRISPRa）。原始约 19264 基因、已 log 变换；`obs[condition]` 形如 `GENE+ctrl`（单扰动）或 `GENEA+GENEB`（组合扰动）。`preprocess: true` 时管线做 HVG（5000）+ 强制保留扰动靶基因 ≈ 5029 基因，并缓存到 `data/.specflow_cache/`（HVG 只跑一次）。
- **split_results.pkl**：scDFM 官方 split 文件，5 个 fold 的列表，每个含 `train`/`test`（仅组合扰动；单扰动自动并入 train）。这是**对齐的关键**——训练和评估都从同一个文件、同一个 `split_fold` 取 split。
- **go_annotations.gaf**：从 [Gene Ontology](http://current.geneontology.org/annotations/) 下载。

Norman holdout 使用一个可复现生成的 folded pickle。准备好 `data/norman.h5ad`
后运行一次：

```bash
python scripts/build_holdout_split.py \
  --h5ad data/norman.h5ad \
  --output data/splits/norman_holdout.pkl
```

`configs/norman_holdout.yaml` 和 holdout experiment specs 都读取
`data/splits/norman_holdout.pkl`，并继承 0602 主模型设置。

---

## 训练与评估流程（cell_eval 对齐）

### 核心保证：split 一致性

训练时 `load_benchmark_h5ad` 用 `config.data.split_fold` 从 `split_path` 取 split；评估时 `evaluate_scdfm` 用**同一个** `split_fold` 取 split。因此测试条件绝不会泄漏进训练集——这是 cell_eval 数字可信的前提。

### 方式 A：一条命令（训练 + cell_eval，推荐）

```bash
python scripts/train.py --config configs/norman.yaml --output-dir outputs/run1
```

`--evaluation-protocol` 默认就是 `cell_eval`，训练结束自动评估。

### 方式 B：分两步（训练后单独评估 / 复跑评估）

```bash
# 训练（跳过自动评估）
python scripts/train.py --config configs/norman.yaml --output-dir outputs/run1 --skip-evaluation

# 用 cell_eval 评估指定 checkpoint
python scripts/evaluate_scdfm.py \
  --config configs/norman.yaml \
  --output-dir outputs/run1 \
  --checkpoint outputs/run1/best.pt \
  --fold 1
```

### 输出文件

| 文件 | 内容 |
|------|------|
| `best.pt` | 最优 checkpoint（含 EMA 权重） |
| `training_history.json` | 每个 epoch/step 的 loss、lr、`val_pearson_delta` |
| `training_summary.json` | 训练总结 |
| `data_summary.json` | 基因数、条件数、图边数、split |
| `graphs/{go,coexp}.npz` | 构建的图 |
| **`pred.h5ad` / `real.h5ad`** | 预测 / 真实表达（cell_eval 输入） |
| **`results.csv`** | **逐扰动的 cell_eval 指标** |
| **`agg_results.csv`** | **聚合的 cell_eval 指标（统计量）** |
| `scdfm_evaluation_summary.json` | 评估元信息（fold、测试条件、路径） |

### 训练预算对齐 scDFM

scDFM 用步数（step）而非 epoch。`configs/norman.yaml` 已设 `max_steps: 200000`，训练按优化器更新计步、每步更新 cosine LR、每 `eval_every_steps` 验证并存盘，周期性 checkpoint 落在 `outputs/.../checkpoints/step_*.pt`。

---

## 复现实验批次

多实验批次由 `scripts/launch_experiments.py` 生成不可变的
`run_config.yaml`、每卡队列脚本和 `launch_tmux.sh`。先 dry-run 审查命令，再启动。

**Norman additive fold0-4 主实验**

```bash
python scripts/launch_experiments.py \
  --spec configs/experiments/additive_folds_4gpu.yaml

bash outputs/20260603_additive_folds_0602/launch_tmux.sh
```

**Norman additive 消融**

```bash
python scripts/launch_experiments.py \
  --spec configs/experiments/ablation_4gpu.yaml

bash outputs/20260603_ablation_0602/launch_tmux.sh
```

**Norman holdout full**

```bash
python scripts/build_holdout_split.py \
  --h5ad data/norman.h5ad \
  --output data/splits/norman_holdout.pkl

python scripts/launch_experiments.py \
  --spec configs/experiments/holdout_full_4gpu.yaml

bash outputs/20260603_holdout_full_0602/launch_tmux.sh
```

**汇总结果**

```bash
python scripts/summarize_experiments.py \
  --root outputs \
  --output outputs/experiment_summary.csv
```

**合并 scDFM paper-reported baseline**

```bash
python scripts/build_paper_table.py \
  --local-summary outputs/experiment_summary.csv \
  --paper-baselines paper_baselines/scdfm_reported_metrics.csv \
  --setting norman_additive \
  --output outputs/paper_table_norman_additive.csv
```

更多实验排队、baseline、extra metrics 和 stale-config 处理见
[`EXPERIMENTS.md`](EXPERIMENTS.md)。

---

## 评估指标（cell_eval）

`results.csv` / `agg_results.csv` 的列与 scDFM 一致，主要包括：

| 指标 | 层次 | 含义 |
|------|------|------|
| `pearson_delta` | 相关性 | 预测 vs 真实**扰动效应**（与 control 的差）的 Pearson R — **最核心** |
| `mse` / `mae` | 点级 | 均值表达重建误差 |
| `overlap_at_N` / `precision_at_N` | DE 检索 | top-N 差异表达基因的重合 / 精确率 |
| `de_spearman_sig` | 生物学 | 显著 DE 基因上的 Spearman ρ |
| `de_direction_match` | 生物学 | DE 方向一致率 |
| `pr_auc` / `roc_auc` | 检索 | 识别 DE 基因的 PR/ROC 曲线下面积 |
| `discrimination_score_l1/l2/cosine` | 检索 | 能否区分不同扰动 |
| `clustering_agreement` | 全局 | 扰动间距离结构一致性 |

---

## 训练监控（pearson_delta）

flow matching loss 与生物学质量几乎不相关，因此训练中直接监控 `pearson_delta`：

- 每个验证点用 **EMA 权重**对 val 条件采样，计算 `val_pearson_delta`，并**按它（而非 loss）选择最优 checkpoint**。
- 进度条显示 `pΔ`，全部记录进 `training_history.json`。

scDFM 的 split 默认没有 val 集。若要启用监控，在 config 里设：

```yaml
data:
  val_from_train_fraction: 0.1   # 从 train 切 10% 作为监控用 val（默认 0 = 严格对齐、不切）
```

设为 0 时与 scDFM split 逐条对齐、不做监控（按 loss 选 checkpoint）。

---

## 配置说明

```yaml
data:
  h5ad_path: data/norman.h5ad
  split_path: data/split_results.pkl          # scDFM split 文件
  split_fold: 1                               # 训练 + 评估共用此 fold
  preprocess: true                            # 原始 h5ad 做 HVG；已预处理则 false
  preprocess_cache: true                      # 缓存 HVG 子集，只跑一次
  val_from_train_fraction: 0.1                # >0 启用 pearson_delta 监控

spectral:
  static: true            # S3: 静态位置编码（不做扰动图衰减），扰动走 e_p

model:
  dual_graph: true        # GO + 共表达双图（false 仅共表达）
  d_model: 128
  hidden_dim: 256
  spectral_dim: 64
  pert_dim: 32            # 扰动 embedding 维度（FiLM 注入用）
  spectral_propagation: true   # 创新1: 谱扰动传播算子（见下）
  propagation_channels: 8      # 传播滤波器通道数

flow:
  sigma: 0.2              # 残差噪声尺度（降低让信号主导）
  ot_coupling: true       # 创新3: OT 耦合 control->perturbed（见下）
  mmd_weight: 0.0

training:
  batch_size: 48          # OOM 时降到 32 / 16
  max_steps: 200000       # 步数模式（与 scDFM 对齐）；留空则用 max_epochs
  eval_every_steps: 5000
  learning_rate: 0.0001
  warmup_steps: 2000
  use_ema: true
  use_amp: true           # 混合精度，省约一半显存
  scheduler: cosine
  monitor_pearson_delta: true
```

预置配置：

| 配置 | 数据集 | 设定 | 说明 |
|------|--------|------|------|
| `configs/norman.yaml` | Norman | additive | **cell_eval 对齐的主基准** |
| `configs/norman_holdout.yaml` | Norman | holdout | 完全未见扰动的泛化测试 |
| `configs/combosciplex.yaml` | ComboSciplex | 药物组合 | 需要 drug→target 映射 |
| `configs/default.yaml` | — | additive | 快速冒烟 / 本地调试 |

---

## 后台运行（tmux）

GPU 训练耗时长，用 tmux 防断线：

```bash
# 交互式
tmux new -s specflow
python scripts/train.py --config configs/norman.yaml --output-dir outputs/run1 2>&1 | tee outputs/run1/train.log
# Ctrl+B 然后 D 脱离（训练继续）

# 或一条命令后台启动
tmux new -d -s specflow "python scripts/train.py --config configs/norman.yaml --output-dir outputs/run1 2>&1 | tee outputs/run1/train.log"
```

| 命令 | 作用 |
|------|------|
| `tmux attach -t specflow` | 重新连接查看 |
| `Ctrl+B` then `D` | 脱离（训练继续） |
| `tmux ls` | 列出会话 |
| `tmux kill-session -t specflow` | 终止（会停训练） |

`tee` 同时写屏幕和 `train.log`，缓冲区滚掉也能查完整日志。

---

## 模型架构

```
              ┌─► pert_encoder ───────► e_p ─────────────┐ (FiLM + 拼入 token)
pert_mask s ──┤                                          │
              └─► SpectralPropagation ─► h (B,G,C) ───┐   │  [创新1]
                  h = Φ·diag(g_θ(λ))·Φᵀ·s            │   │
ctrl_expr ──┐     (固定图谱基 + 可学习滤波器)          │   │
            ├─► 双图静态谱 (GO+coexp) ─► spectral      │   │
            │   (SignNet+多尺度+跨图融合, S3 固定)      │   │
            ▼                                          ▼   ▼
       GeneTokenEncoder [ctrl ‖ spectral ‖ mask ‖ e_p] ─► gene tokens
                                       │
                              AttentivePooling ─► cell condition
                                       │
   x_t, t ─► VelocityField [局部: x_t‖ctrl‖spectral‖mask‖h ; 全局: cond+FiLM(e_p)] ─► velocity
                                       │
            Euler ODE 积分: x_0=ctrl+σε → x_1   (训练: 残差流匹配, OT 耦合配对 [创新3])
```

**关键设计**：
- **控制锚定残差流匹配**：流起点 `x_0 = ctrl + σε`，速度目标 `x_1 - x_0` 即扰动残差（低维、稀疏），比从噪声生成完整状态简单得多。
- **扰动 embedding 强注入（FiLM, S1）**：`e_p = MLP(pert_mask)` 直接拼进 gene token，并对速度场每层做 feature-wise 调制（零初始化、恒等启动）。
- **静态谱位置编码（S3）**：双图随机游走拉普拉斯谱分解 + SignNet + 跨图融合，作为**固定的基因位置编码**（只算一次，不随扰动变）。避免"动态谱信号微弱 + 对 CRISPRa 方向错误"。`spectral.static: false` 可切回动态谱（消融）。

### 创新点（可切换，便于 A/B 消融）

**创新 1 — 谱扰动传播算子（`model.spectral_propagation`）**
把扰动指示向量 `s∈{0,1}^G` 在**固定图**上扩散到每个基因：

$$h = \Phi\,\mathrm{diag}(g_\theta(\lambda))\,\Phi^\top s \in \mathbb{R}^{G\times C}$$

`Φ,λ` 是基图（coexp）的特征向量/特征值（预计算一次），`g_θ` 是特征值上的小 MLP（学习扩散尺度），`h_i` = 基因 i "在扰动下游多深"。`h` 作为逐基因特征喂给速度场。
- vs scDFM 二值 mask（扰动不传播）、vs GEARS GNN（固定跳数、过平滑）：谱滤波**全局、多尺度、无过平滑、无层数限制**，且 `O(G·k)` 便宜。
- 正确实现了"扰动沿网络传播"的初衷（动态谱失败的那个目标）。
- 关 / 切回 FiLM-only：`model.spectral_propagation: false`。

**创新 3 — OT 耦合流（`flow.ot_coupling`）**
control 和 perturbed 是两个**未配对**群体。训练时在**每个条件组内**用最优传输（匈牙利算法，`scipy.linear_sum_assignment`）按表达相似度把 control 细胞配到最可能的 perturbed 细胞，替代随机配对，使流轨迹更直、更有生物意义。
- 仅依赖 scipy（无需 POT）。
- 关：`flow.ot_coupling: false`。

> 消融建议：以 `norman.yaml`（两者都开）为基准，分别关 `spectral_propagation` / `ot_coupling` / 把 `static` 改 false，对比 holdout 上的 pearson_delta，验证每个组件的贡献。

---

## 项目结构

```text
src/specflow/
  config.py              配置数据类
  experiment.py          端到端实验编排（训练 / 评估 / cell_eval）
  data/
    preprocessing.py     HVG 筛选、归一化、扰动靶基因保留、log 检测
    benchmark.py         h5ad 加载、split 解析、PreparedPerturbationData
    dataset.py           Dataset + 按条件分组的 DataLoader
  graph/
    coexp_graph.py       控制细胞 KNN 共表达图
    go_graph.py          GO 语义相似度图
    perturbation_aware.py 扰动感知边衰减
    spectral_embedding.py 随机游走拉普拉斯谱分解
    spectral_cache.py    扰动条件谱嵌入磁盘缓存
  model/
    specflow.py          顶层模型（含 pert_encoder）
    spectral_fusion.py   SignNet + 多尺度 + 跨图融合
    sign_net.py          符号不变编码（带 gradient checkpointing）
    spectral_propagation.py  创新1: 谱扰动传播算子
    gene_encoder.py      逐基因 token 编码（拼入 e_p）
    cell_aggregator.py   注意力池化
    velocity_field.py    速度场（FiLM 条件化 + 传播特征）
    time_embedding.py    时间嵌入
  flow/
    flow_matching.py     控制锚定流匹配损失（含创新3: OT 耦合）
    mmd_loss.py          多带宽 RBF MMD
    ode_solver.py        Euler ODE 采样
  training/
    trainer.py           训练循环（AMP / EMA / 调度器 / tqdm）
    ema.py               指数移动平均
  evaluation/
    metrics.py           内置指标
    evaluator.py         条件级评估 + pearson_delta 监控
    results.py           cell_eval 风格结果表
    scdfm_protocol.py    scDFM split 解析 + pred/real h5ad + run_cell_eval

scripts/
  train.py               训练入口（默认训练后跑 cell_eval）
  evaluate_scdfm.py      cell_eval 对齐评估
  evaluate.py            内置指标评估
  build_graphs.py        单独构图
  ablation.py            消融实验
```

---

## 常见问题

**`ModuleNotFoundError: No module named 'specflow'`**
先安装包：`pip install -e .`。

**`ModuleNotFoundError: No module named 'cell_eval'`**
安装评估依赖并验证：`pip install cell-eval`，然后运行 `python -c "from cell_eval import MetricsEvaluator"`。临时绕过可用 `--evaluation-protocol internal`。

**CUDA Out of Memory**
config 里把 `batch_size` 降到 16 / 8；确认 `use_amp: true`。SignNet 默认已开 gradient checkpointing。

**"data already log-transformed" 警告**
若 h5ad 已预处理，设 `preprocess: false`。开启 `preprocess: true` 时管线会自动检测 log 数据并跳过重复归一化。

**cell_eval 指标全是常数 / pearson_delta ≈ 0**
说明模型对所有扰动输出几乎相同（扰动信号没学到）。检查 `training_history.json` 里 `val_pearson_delta` 是否随训练上升；若一直接近 0，需排查扰动条件化是否生效。

---

## 测试

```bash
pytest                      # 全部测试
pytest tests/test_model_flow.py -q
```
