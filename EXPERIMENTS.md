# SpecFlow 0602 Experiment Runs

This folder is configured to use the 0602 baseline by default.

## Run Naming

Use one study folder per batch:

```text
outputs/YYYYMMDD_study_name/
```

Each run folder is generated as:

```text
NN_experiment_name_f{fold}_s{seed}_g{gpu}
```

Example:

```text
outputs/20260603_ablation_0602/00_full_f1_s42_g0
outputs/20260603_ablation_0602/01_no_spectral_propagation_f1_s42_g1
outputs/20260603_ablation_0602/02_no_ot_coupling_f1_s42_g2
outputs/20260603_ablation_0602/03_no_delta_corr_f1_s42_g3
```

Every run directory contains its own generated `run_config.yaml`, so later
analysis does not depend on remembering command-line overrides.

## One-off Experiments Without Writing a Spec

For a quick single run that just tweaks a few fields, override any config key
directly on `train.py` with repeatable `--set KEY=VALUE` (dotted keys, YAML
value parsing):

```bash
python scripts/train.py --config configs/norman.yaml \
  --output-dir outputs/probe_sigma03 \
  --set flow.sigma=0.3 \
  --set model.graph_mode=go \
  --set training.max_steps=50000
```

`--fold N` and `--split-path PATH` are applied to the config too, so **training
and the cell_eval evaluation always use the same split** — no need to keep them
in sync by hand.

## Spec `defaults` Block

To avoid repeating common fields (`seed`, `gpu`, `fold`, ...) on every
experiment, put them under a spec-level `defaults` block. Per-experiment values
override the defaults:

```yaml
defaults:
  seed: 42
  fold: 1
experiments:
  - name: full
    gpu: 0
  - name: go_only
    gpu: 1
    overrides:
      model.graph_mode: go
```

## Dry Run First

Always inspect commands before launching:

```bash
python scripts/launch_experiments.py \
  --spec configs/experiments/ablation_4gpu.yaml
```

For hyperparameters:

```bash
python scripts/launch_experiments.py \
  --spec configs/experiments/hparams_4gpu.yaml
```

The launcher writes generated configs and `launch_manifest.yaml` during dry run.
It also writes:

```text
run_gpu0.sh
run_gpu1.sh
run_gpu2.sh
run_gpu3.sh
launch_tmux.sh
```

The per-GPU scripts run all experiments assigned to that GPU sequentially, so a
card can keep working through its queue without manual intervention.

## Launch Four GPU Queues

After dry run, start all GPU queues:

```bash
bash outputs/20260603_ablation_0602/launch_tmux.sh
```

Or start a single GPU queue manually:

```bash
bash outputs/20260603_ablation_0602/run_gpu0.sh
```

The launcher can also run experiments sequentially in the current process:

```bash
python scripts/launch_experiments.py \
  --spec configs/experiments/ablation_4gpu.yaml \
  --launch
```

For true four-GPU usage, prefer `launch_tmux.sh`.

## Manual Command Template

Template:

```bash
tmux new -d -s sf_g0_full "cd /mnt/infini-data/test/quan_space/codespace/aidd/code_0602_opo && CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config outputs/20260603_ablation_0602/00_full_f1_s42_g0/run_config.yaml --output-dir outputs/20260603_ablation_0602/00_full_f1_s42_g0 2>&1 | tee outputs/20260603_ablation_0602/00_full_f1_s42_g0/train.log"
```

Replace the session name, GPU id, and run directory for each run.

## Summarize Results

After a batch finishes:

```bash
python scripts/summarize_experiments.py \
  --root outputs/20260603_ablation_0602 \
  --output outputs/20260603_ablation_0602/summary.csv
```

For all local outputs:

```bash
python scripts/summarize_experiments.py \
  --root outputs \
  --output outputs/experiment_summary.csv
```

The summary includes:

```text
pearson_delta, mse, mae, de_spearman_lfc_sig, de_direction_match,
pr_auc, roc_auc, discrimination_score_l2, best_val_pearson_delta,
steps_completed, fold, n_eval_genes
```

## Recommended First Batch

Run ablations first:

```text
full
no_spectral_propagation
no_ot_coupling
no_delta_corr
go_only
coexp_only
graph_none
mmd_on
```

Then run the small hyperparameter sweep:

```text
delta_001
delta_003
delta_005
sigma_015
sigma_025
prop_ch4
prop_ch16
infer_controls256
```

## scDFM-Aligned Experiment Specs

Main additive multi-fold runs:

```bash
python scripts/launch_experiments.py \
  --spec configs/experiments/additive_folds_4gpu.yaml

bash outputs/20260603_additive_folds_0602/launch_tmux.sh
```

Local Control/Additive baselines. These do not train a model; they write
`pred.h5ad` / `real.h5ad` and run `cell_eval`:

```bash
python scripts/launch_experiments.py \
  --spec configs/experiments/norman_baselines_4gpu.yaml

bash outputs/20260603_norman_baselines_0602/launch_tmux.sh
```

Holdout full-model runs.

**Prerequisite (once):** the holdout configs read `data/splits/norman_holdout.pkl`,
which is not shipped. Generate it with the scDFM "unseen" protocol first:

```bash
python scripts/build_holdout_split.py \
  --h5ad data/norman.h5ad \
  --output data/splits/norman_holdout.pkl
```

If `outputs/20260603_holdout_*` was generated before the split path changed to
`norman_holdout.pkl`, remove those stale planned directories and dry-run again:

```bash
rm -rf outputs/20260603_holdout_full_0602
rm -rf outputs/20260603_holdout_ablation_0602
```

The launcher marks existing run folders whose `run_config.yaml` differs from
the current spec as `exists_stale_config`. It will not silently overwrite
configs for directories that already contain logs or results; remove the stale
run directory or pass `--overwrite` deliberately.

Then launch:

```bash
python scripts/launch_experiments.py \
  --spec configs/experiments/holdout_full_4gpu.yaml
```

Core holdout ablations:

```bash
python scripts/launch_experiments.py \
  --spec configs/experiments/holdout_ablation_4gpu.yaml
```

After holdout full finishes, split its per-condition cell_eval results into
the scDFM paper's Single / Double subsets:

```bash
python scripts/summarize_holdout_subsets.py \
  --root outputs/20260603_holdout_full_0602
```

ComboSciPlex extension:

```bash
python scripts/launch_experiments.py \
  --spec configs/experiments/combosciplex_4gpu.yaml
```

## Baseline Evaluation Command

Single baseline run:

```bash
python scripts/evaluate_baseline.py \
  --config configs/norman.yaml \
  --output-dir outputs/baselines/additive_f1 \
  --baseline additive \
  --fold 1
```

For strict no-leakage additive evaluation, the default behavior is to fail when
a test target has no train single-perturbation delta. For diagnostic runs only,
you can use:

```bash
--missing-single zero
```

## Extra Metrics

Compute L2 and Delta-hat-style metrics after a run has `pred.h5ad` and
`real.h5ad`:

```bash
python scripts/compute_scdfm_extra_metrics.py \
  --pred-h5ad outputs/run1/pred.h5ad \
  --real-h5ad outputs/run1/real.h5ad \
  --output-dir outputs/run1 \
  --reference control
```

`l2_mean` is directly comparable to the scDFM table definition. Delta-hat
metrics depend on the reference choice, so the script records
`extra_metric_reference` in the summary.

Then refresh the experiment summary:

```bash
python scripts/summarize_experiments.py \
  --root outputs \
  --output outputs/experiment_summary.csv
```

## Paper Table Merge

The file below stores scDFM-paper-reported baselines for GEARS, CPA, scGPT,
Geneformer, STATE, CellFlow, and scDFM:

```text
paper_baselines/scdfm_reported_metrics.csv
```

Merge those rows with local results:

```bash
python scripts/build_paper_table.py \
  --local-summary outputs/experiment_summary.csv \
  --paper-baselines paper_baselines/scdfm_reported_metrics.csv \
  --setting norman_additive \
  --output outputs/paper_table_norman_additive.csv
```
