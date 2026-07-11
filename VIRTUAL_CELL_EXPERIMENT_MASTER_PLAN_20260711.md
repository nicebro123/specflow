# GeneGeoFlow virtual-cell experiment master plan

Updated: 2026-07-11

Remote code root:
`/mnt/infini-data/test/quan_space/codespace/aidd_0604/code_0602_opo`

Python environment:
`/mnt/infini-data/test/quan_space/envs/aidd/bin/python`

## Execution rules

1. The paper default is GeneGeoFlow without explicit spectral propagation.
2. Every run uses an immutable generated YAML and a new output directory.
3. Dry-run, configuration-diff audit, and a short smoke test precede each new
   implementation-dependent batch.
4. Strict-holdout test folds are never used to choose hyperparameters.
5. Seeds are fixed to `42`, `123`, and `3407`.
6. Existing completed results are reused only when their split, preprocessing,
   model definition, and evaluation protocol match exactly.

## Phase A: finish and correct the active queue

The active study is
`outputs/20260711_core_ablation_multifold_8gpu_0602`.

- Four no-propagation GeneGeoFlow runs are active for additive folds 0, 2, 3,
  and 4.
- Four GeneGeoFlow-Prop no-delta runs are active for the same folds and are
  retained as propagation-variant evidence.
- Four queued no-control-anchor runs were revised to use no propagation.
- Four queued no-OT runs were revised to use no propagation.

After this queue, add seven matched default-model runs so that no-control,
no-OT, and no-delta each have all five additive folds:

- no-control-anchor: fold 1;
- no-OT: fold 1;
- no-delta: folds 0, 1, 2, 3, and 4.

New GPU training runs after the active queue: **7**.

## Phase B: isolate biological gene geometry

Run on all five Norman strict-holdout folds using seed 42.

Already complete and reused:

- perturbation-conditioned dual-graph geometry;
- graph-free;
- explicit propagation;
- adaptive propagation gate.

New five-fold groups:

- GO only;
- coexpression only;
- low-frequency only;
- high-frequency only;
- fixed mean dual-graph fusion;
- random coordinates;
- shuffled spectral coordinates;
- capacity-matched learned gene embeddings.

Run the same eight new variants once on the fixed ComboSciPlex split.

New GPU training runs: `8 x 5 + 8 = 48`.

Implementation dependency:

- GO/coexpression, low/high frequency, and fixed fusion already have model
  switches but need experiment specs and tests.
- random, shuffled, and learned coordinates require new model/data switches,
  leakage-safe deterministic generation, and unit tests.

## Phase C: random-seed uncertainty

Methods:

- GeneGeoFlow;
- graph-free;
- GeneGeoFlow-Prop.

Existing seed-42 runs are reused. Add seeds 123 and 3407 on:

- all five Norman strict-holdout folds;
- the fixed ComboSciPlex split.

New GPU training runs: `3 x (5 + 1) x 2 = 36`.

Report both split variation and seed variation; do not label five data folds as
random-seed uncertainty.

## Phase D: matched external baselines

Use the exact GeneGeoFlow preprocessing, strict-holdout folds, ComboSciPlex
split, evaluation genes, and cell-eval protocol.

Norman strict holdout, five folds:

- scDFM;
- CellFlow;
- GEARS;
- CPA.

ComboSciPlex, fixed split:

- scDFM;
- CellFlow;
- GEARS when the target-map adapter is scientifically valid;
- chemCPA.

Planned GPU training runs: up to `4 x 5 + 4 = 24`.

Implementation dependency: the external repositories are not currently present
next to the GeneGeoFlow checkout. Each baseline needs a pinned revision, dataset
adapter, generated-config record, and protocol audit before full launch.

## Phase E: sample-level population metrics

Implement post-hoc metrics over saved `pred.h5ad` and `real.h5ad`:

- sliced Wasserstein distance;
- energy distance;
- residual-space MMD;
- prediction diversity;
- variance-coverage ratio.

The repository contains an energy-distance function, but current paper-facing
artifacts do not contain these metrics. Sliced Wasserstein and residual MMD need
new audited implementations.

First apply the metrics to GeneGeoFlow, graph-free, and GeneGeoFlow-Prop across
five strict-holdout folds and ComboSciPlex: **18 post-hoc jobs**. Apply the same
metrics to external baselines when their predictions become available.

The previously identified 23 scDFM extra-metric jobs are a separate CPU
post-processing batch and remain part of the execution plan.

## Phase F: matched default-model component ablations

Phase A completes five-fold additive evidence for:

- control anchor;
- OT coupling;
- delta-correlation loss.

Repeat all three components on all five strict-holdout folds and once on
ComboSciPlex. These confirmation runs are pre-registered before examining the
new additive test results.

New GPU training runs: `3 x (5 + 1) = 18`.

## Phase G: default GeneGeoFlow hyperparameters

Tune only on a dedicated training/validation split, not strict-holdout test
folds. Keep explicit propagation disabled.

Sweep one factor at a time around the current default:

- delta-correlation weight: `0`, `0.01`, and `0.05` (3 runs);
- control noise sigma: `0.1` and `0.3` (2 runs);
- GO/coexpression spectral dimensions: `16` and `64` (2 runs);
- low/high-frequency split ratio: `0.25` and `0.75` (2 runs);
- graph-source fusion: fixed mean and concatenation (2 runs).

After choosing one locked configuration, evaluate it once on all strict-holdout
folds and ComboSciPlex. Do not use test results for iterative selection.

New GPU training runs: `11` validation runs plus `6` locked evaluations, **17**
in total.

## Phase H: efficiency and mechanism evidence

Efficiency profiler outputs:

- trainable and total parameters;
- graph and eigensystem preprocessing time;
- step and epoch training time;
- per-cell inference latency;
- 30-step Euler integration cost;
- peak GPU memory.

Compare GeneGeoFlow, graph-free, GeneGeoFlow-Prop, scDFM, and CellFlow when the
external baselines are ready.

Mechanism export outputs:

- low/high-frequency gate distributions per perturbation;
- GO/coexpression fusion weights;
- representative perturbation-specific gene and pathway views;
- stability of gate summaries across folds and seeds.

These jobs reuse trained checkpoints but require audited weight-export and
profiling scripts.

## Scheduling order

1. Finish the corrected active queue and the seven matched additive runs.
2. While GPUs are busy, implement and smoke-test Phase B and Phase E.
3. Run the Phase B geometry matrix.
4. Run the Phase C seed matrix.
5. In parallel with internal runs, prepare and audit external baseline adapters.
6. Run Phase D external baselines.
7. Run the validation-only Phase G sweep and locked final evaluation.
8. Generate Phase E, H, tables, and mechanism figures from frozen outputs.

No GPU should receive an unaudited implementation-dependent experiment merely
to keep it occupied. Each released card receives the next dry-run-verified job
from the current phase.
