# SpecFlow Paper Draft and Audit

This document is grounded in the local repository at `C:\Users\Administrator\Desktop\specflow`.
It intentionally does not invent results. Any item not evidenced by code, README, configuration files, experiment specs, or bundled baseline tables is marked as TODO / Need confirmation.

## A. Repository-Grounded Method Report

### A.1 What Problem the Project Solves

SpecFlow targets **single-cell perturbation response prediction**. Given a pool of unperturbed control cells and a perturbation condition represented as one or more targeted genes, the model predicts the post-perturbation gene-expression distribution.

Evidence:

- The README names the method "Spectral-Guided Control-Anchored Flow Matching for Single-Cell Perturbation Prediction" and states that SpecFlow starts from control cells and moves them toward perturbed states along spectral gene-structure embeddings.
- `src/specflow/data/dataset.py` defines a pool-based perturbation dataset with `ctrl_expr`, `pert_expr`, `pert_mask`, and `condition`.
- `src/specflow/flow/flow_matching.py` defines a control-anchored conditional flow-matching objective.
- `configs/norman.yaml` and `configs/norman_holdout.yaml` configure Norman additive and holdout perturbation protocols.
- `configs/combosciplex.yaml` and `configs/datasets/combosciplex_target_genes.yaml` indicate an optional ComboSciPlex drug-combination extension.

The main scientifically meaningful prediction is not merely reconstructing expression values; it is predicting **perturbation effects**, i.e., expression changes relative to control. The repository emphasizes `pearson_delta` as the core metric.

### A.2 Inputs, Outputs, Training, and Inference

#### Inputs

The model and data pipeline use:

- `ctrl_expr`: a control-cell expression vector with shape `(B, G)`.
- `pert_expr`: a target perturbed-cell expression vector with shape `(B, G)` during training.
- `pert_mask`: a binary gene-aligned perturbation mask with shape `(B, G)`.
- `condition`: the perturbation condition name, such as `GENE+ctrl` or `GENEA+GENEB`.
- `spectral_embedding`: either a tensor of gene spectral coordinates or a mapping with `go` and `coexp` eigenvectors.

The data are unpaired. `PerturbationDataset.__getitem__` independently samples one control cell and one perturbed cell from condition-level pools.

#### Outputs

The model predicts a per-gene velocity vector:

```text
velocity = f_theta(x_t, t, ctrl_expr, pert_mask, spectral_embedding)
```

During inference, the Euler sampler integrates the learned velocity field from a noisy control state to obtain predicted perturbed expression. During evaluation, predictions are written to `pred.h5ad` and compared against `real.h5ad` through `cell_eval` when available.

#### Training Flow

The control-anchored flow objective in `src/specflow/flow/flow_matching.py` uses:

```text
x_0 = ctrl_expr + sigma * epsilon
x_t = (1 - t) * x_0 + t * pert_expr
target_velocity = pert_expr - x_0
loss = MSE(f_theta(x_t, t, ctrl_expr, pert_mask, spectral), target_velocity)
```

If `flow.ot_coupling` is enabled, `_ot_align_targets` solves a Hungarian assignment within each perturbation condition to reorder perturbed targets according to squared expression distance from controls.

The trainer in `src/specflow/training/trainer.py` also supports:

- `delta_corr_weight`: a Pearson-style loss between mean predicted velocity and mean target perturbation delta, grouped by condition.
- `mmd_weight`: an optional MMD loss over sampled predictions.
- EMA weights.
- AMP training on CUDA.
- cosine schedule and warmup.

The default Norman configs set:

- `flow.sigma: 0.2`
- `flow.ot_coupling: true`
- `flow.mmd_weight: 0.0`
- `flow.delta_corr_weight: 0.03`
- `training.max_steps: 200000`

#### Inference Flow

Inference uses `EulerSampler` in `src/specflow/flow/ode_solver.py` (not reproduced here, but referenced by trainer/evaluator). The README states the inference start is:

```text
x_0 = ctrl + sigma * epsilon
```

and the state is integrated to `x_1` with a fixed number of Euler steps. `configs/norman.yaml` sets `inference.ode_steps: 30`, `n_control_cells: 128`, and `n_samples: 10`.

### A.3 Core Modules and Code Mapping

#### Data and Splits

- `src/specflow/data/dataset.py`: unpaired control/perturbation pool dataset; condition-grouped batch sampler.
- `src/specflow/data/benchmark.py`: h5ad loading, target-map loading, split loading, HVG preprocessing cache, conversion to `PreparedPerturbationData`.
- `src/specflow/data/preprocessing.py`: preprocessing, perturbation mask construction, split creation.
- `scripts/build_holdout_split.py`: generates Norman holdout split.

#### Graph Construction and Spectral Features

- `src/specflow/graph/go_graph.py`: GO-derived gene graph using shared GO terms and top-k Jaccard similarity.
- `src/specflow/graph/coexp_graph.py`: control-derived absolute-Pearson coexpression graph.
- `src/specflow/graph/spectral_embedding.py`: random-walk Laplacian spectral embedding via symmetric normalized Laplacian eigensolver.
- `src/specflow/graph/perturbation_aware.py`: perturbation-dependent edge attenuation. This exists in code, but the default `spectral.static: true` configs indicate the main path uses static spectral coordinates.
- `src/specflow/graph/spectral_cache.py`: spectral embedding caching.

#### Model

- `src/specflow/model/specflow.py`: top-level velocity model.
- `src/specflow/model/spectral_fusion.py`: dual-graph spectral fusion with SignNet, macro/micro scale fusion, and GO/coexpression fusion.
- `src/specflow/model/sign_net.py`: sign-invariant encoding of eigenvectors.
- `src/specflow/model/gene_encoder.py`: per-gene token encoder using control expression, spectral features, perturbation mask, and perturbation embedding.
- `src/specflow/model/cell_aggregator.py`: attentive pooling over gene tokens.
- `src/specflow/model/velocity_field.py`: per-gene velocity predictor with time embedding, global conditioning, FiLM layers, optional propagation features.
- `src/specflow/model/spectral_propagation.py`: spectral perturbation propagation feature generator.
- `src/specflow/model/contextual_propagation.py`: graph-aware perturbation encoder and contextual local propagation extension.

#### Flow and Training

- `src/specflow/flow/flow_matching.py`: control-anchored flow matching and OT target alignment.
- `src/specflow/flow/ode_solver.py`: Euler sampler.
- `src/specflow/flow/mmd_loss.py`: optional MMD loss.
- `src/specflow/training/trainer.py`: optimizer, EMA, AMP, scheduler, delta-correlation auxiliary loss.
- `src/specflow/experiment.py`: end-to-end experiment assembly, training, checkpointing, and evaluation integration.

#### Evaluation

- `src/specflow/evaluation/scdfm_protocol.py`: scDFM split loading, evaluation gene selection, writing `pred.h5ad`/`real.h5ad`, running `cell_eval`.
- `src/specflow/evaluation/evaluator.py`: internal sampling-based metrics and validation `pearson_delta`.
- `src/specflow/evaluation/results.py`: cell-eval-style result rows.
- `scripts/evaluate_scdfm.py`: scDFM-compatible evaluation entry point.
- `scripts/evaluate_baseline.py`: control/additive baseline evaluation.
- `scripts/build_paper_table.py`: merges local summaries with `paper_baselines/scdfm_reported_metrics.csv`.

### A.4 Likely Method Contributions Supported by Code

The code supports the following contribution candidates:

1. **Control-anchored flow matching for perturbation response prediction.**  
   The flow starts near measured controls rather than from pure noise, and the velocity target is the control-to-perturbed residual.

2. **Gene-structure conditioning through GO and coexpression spectra.**  
   The repository implements GO graph construction, control coexpression graph construction, random-walk Laplacian spectral embeddings, SignNet encoders, macro/micro scale fusion, and cross-graph fusion.

3. **Perturbation-conditioned training for unpaired single-cell measurements.**  
   The code explicitly models unpaired control/perturbed pools and implements condition-wise OT coupling to reduce random-pairing noise.

4. **Metric-aligned training and evaluation.**  
   `delta_corr_weight` directly targets perturbation-effect direction, and the evaluation path aligns with scDFM `cell_eval`.

5. **Implemented but not yet established as the main protocol: graph-aware perturbation pooling and contextual local propagation.**  
   `GraphAwarePerturbationEncoder` and `ContextualLocalPropagation` exist and are documented, but the default Norman configs do not enable `perturbation_encoder: graph_pool` or `propagation_variant: contextual_local`. This should be treated as an implemented extension or candidate final method until confirmed by experiments.

### A.5 Explicit Evidence vs. Inference

#### Explicitly Present in Code or Configs

- Unpaired control/perturbation dataset.
- Control-anchored flow-matching objective.
- Optional condition-wise OT coupling.
- Optional delta-correlation auxiliary loss.
- Optional MMD loss.
- GO and coexpression graph builders.
- Spectral embedding and dual-graph fusion modules.
- Spectral propagation and contextual local propagation modules.
- Norman additive and holdout configs.
- ComboSciPlex config and target-map files.
- scDFM-compatible evaluation file writing and `cell_eval` invocation.
- Experiment launch specs for main runs, baselines, ablations, hyperparameters, and extensions.
- Paper-reported baseline metrics in `paper_baselines/scdfm_reported_metrics.csv`.

#### Inferred but Plausible

- The paper's primary biological claim should be about structured perturbation-effect prediction rather than generic generation.
- GO and coexpression graphs are intended to provide complementary functional and data-driven structure.
- OT coupling is intended to reduce noisy training velocities caused by random pairing in unpaired data.
- `delta_corr_weight` is intended to improve `pearson_delta` because it aligns mean perturbation residual direction.

#### Not Supported Without Additional Evidence

- Any claim that SpecFlow outperforms scDFM, CellFlow, GEARS, CPA, Geneformer, scGPT, or STATE.
- Any numerical improvement, p-value, mean/std, or rank.
- Any claim that contextual local propagation improves performance.
- Any claim that OT coupling improves performance.
- Any claim that dual graphs improve performance.
- Any runtime or memory efficiency claim beyond implemented AMP/EMA/checkpointing support.
- Any biological discovery claim about specific genes or pathways.

### A.6 Experiments Supported vs. Missing

#### Supported by Scripts and Configs

- Norman additive full model across folds via `configs/experiments/additive_folds_4gpu.yaml`.
- Control/additive baselines via `scripts/evaluate_baseline.py` and `configs/experiments/norman_baselines_4gpu.yaml`.
- Norman holdout full model via `configs/norman_holdout.yaml` and `configs/experiments/holdout_full_4gpu.yaml`.
- Holdout single/double summary via `scripts/summarize_holdout_subsets.py`.
- Core ablations via `configs/experiments/ablation_4gpu.yaml`, `core_components_4gpu.yaml`, and `holdout_ablation_4gpu.yaml`.
- ComboSciPlex extension via `configs/combosciplex.yaml` and `configs/experiments/combosciplex_4gpu.yaml`.
- Hyperparameter sweeps via `configs/experiments/hparams_4gpu.yaml`.
- Contextual local propagation experiments via `configs/experiments/contextual_local_paired_4gpu.yaml`.
- scDFM extra metrics via `scripts/compute_scdfm_extra_metrics.py`.

#### Missing in Current Local State

- No `data/` directory.
- No `outputs/` directory.
- No trained checkpoints.
- No `results.csv`, `agg_results.csv`, `training_history.json`, or logs.
- No local SpecFlow experiment summary table.
- No verified visualizations.
- No completed ablation evidence.
- No evidence that local baselines were run under the same protocol.

### A.7 Potential Code/Narrative Mismatches

1. **README architecture vs. default main configs.**  
   README's architecture section describes `graph_pool` and `contextual_local` propagation. However, `configs/norman.yaml` and `configs/norman_holdout.yaml` do not set `perturbation_encoder: graph_pool` or `propagation_variant: contextual_local`; they use the defaults from `configs/default.yaml`, i.e., `legacy` perturbation encoder and `spectral` propagation. This must be resolved before submission.

2. **"Innovation 1" ambiguity.**  
   Default configs comment "spectral perturbation propagation operator", while README later frames contextual local propagation as Innovation 1. The paper should either use the 0602 spectral-propagation model as the main method or update main configs/results to the contextual local version.

3. **Evaluation-alignment claim depends on unavailable artifacts.**  
   The code is designed for scDFM-compatible evaluation, but current local state lacks `data/`, `outputs/`, and `cell_eval` artifacts. A paper may state that the implementation supports this protocol, but not that experiments have already been completed.

4. **Biological mechanism visualizations are planned, not present.**  
   README and `EXPERIMENT_ORDER.md` discuss graph propagation heatmaps and case studies, but no output figures are present.

5. **Model expressivity limitation.**  
   `codex_check.md` notes that the velocity field is largely per-gene local MLP plus global conditioning, with limited explicit cross-gene interaction inside the velocity field. This should be acknowledged or addressed experimentally.

## B. Paper Narrative Design

### B.1 Candidate Titles

1. **SpecFlow: Spectral-Guided Control-Anchored Flow Matching for Single-Cell Perturbation Response Prediction**
2. **Control-Anchored Flow Matching over Gene-Structure Spectra for Single-Cell Perturbation Modeling**
3. **From Control Cells to Perturbed States: Graph-Spectral Flow Matching for Unpaired Single-Cell Perturbations**

Recommended title: **SpecFlow: Spectral-Guided Control-Anchored Flow Matching for Single-Cell Perturbation Response Prediction**.

### B.2 One-Sentence Core Contribution

SpecFlow models single-cell perturbation response as a control-anchored flow over gene-expression space, conditioning the velocity field on GO/coexpression spectral structure and perturbation masks while using OT coupling to reduce noise from unpaired control and perturbed cell populations.

### B.3 Three Main Contributions

1. **A control-anchored conditional flow-matching formulation** for single-cell perturbation prediction that learns perturbation residual dynamics from measured control states rather than generating expression profiles from pure noise.

2. **A graph-spectral conditioning architecture** that incorporates GO-derived functional relationships and control-derived coexpression structure through spectral embeddings, sign-invariant encoding, multi-scale fusion, and perturbation-aware conditioning.

3. **A reproducible scDFM-aligned evaluation framework** with external split loading, cell-eval-compatible prediction files, control/additive baselines, holdout protocols, and planned component ablations.

Note: contribution 3 is currently an implementation/protocol contribution. It becomes an empirical contribution only after local SpecFlow results are available.

### B.4 Overall Storyline

1. Single-cell perturbation datasets observe population-level control and perturbed cells, not paired before/after trajectories.
2. Predicting perturbation response requires modeling both cell state and the structured relationships among genes targeted or affected by perturbation.
3. Existing perturbation predictors often rely on direct condition embeddings or black-box response models; this can underuse gene structure and can be sensitive to unpaired sampling noise.
4. SpecFlow treats perturbation prediction as a conditional flow from measured control cells to perturbed states.
5. Gene structure is injected through GO/coexpression spectral coordinates and perturbation masks.
6. Training addresses unpaired observations through condition-wise OT coupling and metric-aligned residual correlation.
7. The empirical claim must be established through scDFM-aligned experiments, additive and holdout splits, baselines, and ablations.

### B.5 Method Section Organization

Recommended method subsections:

1. **Problem Setup and Notation**  
   Define control distribution, perturbation condition, perturbation mask, target perturbed distribution, unpaired observations.

2. **Control-Anchored Flow Matching**  
   Present `x_0 = x_c + sigma epsilon`, interpolation, target velocity, and MSE loss.

3. **Gene-Structure Spectral Conditioning**  
   Describe GO graph, coexpression graph, random-walk spectral embeddings, SignNet, multi-scale fusion, and cross-graph fusion.

4. **Perturbation Conditioning and Velocity Field**  
   Describe perturbation embedding, gene tokens, attentive pooling, FiLM-conditioned velocity predictor, and propagation features.

5. **Training with Unpaired Populations**  
   Describe random pool sampling, condition-wise OT coupling, delta-correlation auxiliary loss, optional MMD, EMA.

6. **Inference and Evaluation Interface**  
   Describe Euler integration and `cell_eval` artifact generation.

Add a short note: current default main protocol uses legacy perturbation encoder + spectral propagation; contextual local propagation is an implemented extension requiring separate confirmation.

### B.6 Experiment Section Organization

1. **Datasets and Splits**  
   Norman as primary; ComboSciPlex as optional extension. Use scDFM splits where available. TODO: add exact cell/gene/condition counts after data are available.

2. **Baselines**  
   Control and Additive local baselines are implemented. scDFM paper-reported baselines are bundled. TODO: clarify whether comparison uses paper-reported, rerun, or same-code reproduction.

3. **Metrics**  
   `pearson_delta`, MSE, MAE, DE Spearman, DS, and extra scDFM metrics L2, Delta-hat, Delta-hat20.

4. **Main Results**  
   Planned tables for Norman additive fold 0-4 and holdout single/double. TODO: fill SpecFlow numbers.

5. **Ablations**  
   No graph, GO-only, coexpression-only, no spectral propagation, no OT coupling, no delta-correlation loss, MMD-on, no control anchor, no spectral embedding.

6. **Mechanistic Analysis**  
   Planned propagation heatmaps, delta scatter, UMAP, top DE genes, routing summaries for contextual local variant.

### B.7 Claims That Can and Cannot Be Written

#### Claims Currently Safe to Write

- SpecFlow implements a control-anchored flow-matching model for single-cell perturbation prediction.
- The repository supports scDFM-compatible evaluation outputs through `pred.h5ad` and `real.h5ad`.
- The model conditions on GO and coexpression spectral features.
- The training code supports condition-wise OT coupling and a delta-correlation auxiliary objective.
- The experiment plan includes additive, holdout, baseline, and ablation protocols.

#### Claims That Must Be Marked TODO

- SpecFlow outperforms any baseline.
- SpecFlow improves unseen perturbation generalization.
- OT coupling improves performance.
- Spectral propagation improves performance.
- Dual-graph fusion improves performance.
- Contextual local propagation is superior to spectral propagation.
- SpecFlow is computationally efficient compared with baselines.
- Any claim involving exact numerical improvements.

## C. Full English Paper Draft

```latex
\title{SpecFlow: Spectral-Guided Control-Anchored Flow Matching for Single-Cell Perturbation Response Prediction}

\begin{abstract}
Predicting how single cells respond to genetic or chemical perturbations is a central problem in computational biology. Existing perturbation datasets typically provide unpaired populations of control and perturbed cells, making it difficult to learn biologically meaningful state transitions. We present SpecFlow, a spectral-guided control-anchored flow-matching framework for single-cell perturbation response prediction. SpecFlow starts from measured control-cell expression states and learns a conditional velocity field toward perturbed states. The velocity model is conditioned on perturbation masks and gene-structure information derived from Gene Ontology and control-cell coexpression graphs through spectral embeddings and dual-graph fusion. To reduce noise from unpaired observations, the training objective optionally aligns control and perturbed samples within each condition using an optimal-transport assignment and includes a perturbation-delta correlation term aligned with the primary biological metric. The repository implements scDFM-compatible data splits and cell_eval-style evaluation outputs for Norman and ComboSciPlex protocols. TODO: Insert validated main results after running SpecFlow on the aligned additive and holdout splits. Without completed local experiments, this draft reports the method and planned evaluation protocol but does not claim numerical superiority over existing methods.
\end{abstract}

\section{Introduction}

Single-cell perturbation assays measure how cellular expression states change after genetic or chemical interventions. Accurate computational models of these responses can help prioritize perturbations, interpret regulatory mechanisms, and generalize from observed interventions to unseen or combinatorial perturbations. A key difficulty is that most single-cell perturbation datasets are not paired at the cell level: one observes a population of control cells and a separate population of perturbed cells, but not the before-and-after trajectory of the same cell. This makes perturbation prediction simultaneously a conditional generation problem, a distribution-matching problem, and a structured biological modeling problem.

A natural formulation is to model the response as a transition from an observed control state to a perturbed state. However, directly learning such transitions from randomly paired control and perturbed cells can introduce noisy or biologically implausible velocity targets. Moreover, perturbation effects are mediated by structured relationships among genes. A model that treats perturbation labels as arbitrary condition identifiers may fail to exploit known functional relationships from Gene Ontology or data-driven coexpression structure.

We introduce SpecFlow, a spectral-guided control-anchored flow-matching framework for single-cell perturbation response prediction. SpecFlow learns a conditional velocity field in expression space. Instead of starting from pure noise, it starts near measured control-cell expression and learns the residual flow toward perturbed expression. The model conditions this flow on gene-aligned perturbation masks and spectral gene features computed from GO-derived and coexpression-derived graphs. In the current implementation, these features are fused by sign-invariant spectral encoders, multi-scale graph encoders, and adaptive cross-graph fusion. The training code further supports condition-wise optimal-transport coupling to reduce random pairing noise in unpaired perturbation populations.

The repository is designed around strict evaluation alignment with scDFM-style benchmarks. It includes loaders for external split files, Norman additive and holdout configurations, ComboSciPlex configuration files, control/additive baseline scripts, and utilities for writing prediction and reference AnnData objects consumed by cell_eval. At the time of this draft, the local repository does not include the datasets, checkpoints, training logs, or SpecFlow result files. Therefore, this paper draft describes the method and experimental protocol while marking all empirical findings as TODO.

Our intended contributions are:

\begin{enumerate}
    \item We formulate single-cell perturbation response prediction as control-anchored conditional flow matching, learning perturbation residual dynamics from measured control states.
    \item We implement a graph-spectral conditioning architecture that combines GO-derived functional graphs and control-derived coexpression graphs through spectral embeddings and dual-graph fusion.
    \item We provide a scDFM-aligned experimental framework, including split handling, cell_eval-compatible outputs, local control/additive baselines, holdout protocols, and component ablation configurations.
\end{enumerate}

TODO: Once experiments are run, revise the third contribution to report validated empirical findings rather than implementation readiness.

\section{Related Work}

\paragraph{Single-cell perturbation prediction.}
Single-cell perturbation modeling aims to predict cellular expression responses under genetic or chemical interventions. Relevant baselines in the bundled paper-baseline table include GEARS, CPA, scGPT, Geneformer, STATE, CellFlow, and scDFM. The current repository does not include implementations or rerun logs for these methods, but it includes paper-reported baseline metrics for Norman additive, Norman holdout, and ComboSciPlex settings. TODO: Add precise citations and describe each baseline only after confirming the target bibliography and comparison protocol.

\paragraph{Flow matching and continuous generative modeling.}
Flow matching learns a velocity field that transports samples from a source distribution to a target distribution. SpecFlow uses a conditional form of flow matching where the source is anchored at measured control expression rather than pure noise. This differs from generic unconditional generation because the initial state carries cell-specific expression information. TODO: Add formal comparison to conditional flow matching and rectified flow literature with citations.

\paragraph{Graph-structured biological modeling.}
Gene regulatory and functional relationships provide useful inductive biases for perturbation prediction. SpecFlow constructs a GO-derived graph from shared annotations and a coexpression graph from control expression. It then computes random-walk Laplacian spectral coordinates and feeds them through sign-invariant encoders. TODO: Add citations for graph neural perturbation models, GO-based gene representations, and spectral positional encodings.

\paragraph{Evaluation protocols for perturbation benchmarks.}
The repository aligns evaluation with scDFM's cell_eval protocol by writing matched `pred.h5ad` and `real.h5ad` files and preserving split consistency between training and evaluation. TODO: Cite scDFM and cell_eval protocol details.

\section{Problem Formulation}

Let $G$ denote the number of modeled genes. A control cell is represented by an expression vector $x_c \in \mathbb{R}^G$. A perturbation condition $p$ is represented by a gene-aligned binary mask $s_p \in \{0,1\}^G$, where $s_{p,g}=1$ indicates that gene $g$ is directly targeted. For each perturbation condition, the dataset provides an unpaired set of perturbed expression profiles $\mathcal{X}_p = \{x_p^{(i)}\}$ and a shared control pool $\mathcal{X}_c = \{x_c^{(j)}\}$.

The goal is to learn a conditional generator that maps sampled control cells and perturbation masks to a predicted perturbed-cell distribution:

\[
\hat{x}_p \sim q_\theta(\cdot \mid x_c, s_p).
\]

Evaluation focuses on both expression-level reconstruction and perturbation-effect recovery. The repository emphasizes Pearson correlation between predicted and true mean perturbation deltas:

\[
\Delta_p = \mathbb{E}[x_p \mid p] - \mathbb{E}[x_c],
\quad
\hat{\Delta}_p = \mathbb{E}[\hat{x}_p \mid p] - \mathbb{E}[x_c].
\]

The primary metric is `pearson_delta`, the Pearson correlation between $\Delta_p$ and $\hat{\Delta}_p$.

\section{Method}

\subsection{Overview}

SpecFlow consists of three main components:

\begin{enumerate}
    \item gene-structure encoders that build GO and coexpression spectral features;
    \item a perturbation- and control-conditioned velocity model;
    \item a control-anchored flow-matching objective with optional OT coupling and delta-correlation regularization.
\end{enumerate}

The top-level implementation is `src/specflow/model/specflow.py`. The flow-matching loss is implemented in `src/specflow/flow/flow_matching.py`, and training logic is implemented in `src/specflow/training/trainer.py`.

\subsection{Gene-Structure Spectral Conditioning}

SpecFlow constructs two gene graphs. The GO graph connects genes with shared Gene Ontology annotations using a top-$k$ Jaccard similarity graph. The coexpression graph is computed from control-cell expression using absolute Pearson correlation with thresholding and top-$k$ neighbors. For each graph, SpecFlow computes non-trivial eigenvectors of a random-walk Laplacian by solving the symmetric normalized Laplacian and converting the resulting eigenvectors to the random-walk basis.

Let $U^{GO} \in \mathbb{R}^{G \times K_{GO}}$ and $U^{coexp} \in \mathbb{R}^{G \times K_{coexp}}$ denote the spectral coordinates. Because eigenvector signs are arbitrary, SpecFlow uses SignNet encoders before downstream fusion. The spectral-fusion module separates low-frequency and high-frequency components into macro and micro partitions, applies learned projections, and fuses them with perturbation-conditioned attention. It then fuses GO and coexpression features through adaptive cross-graph fusion.

This produces a per-gene spectral feature matrix:

\[
Z_p = \mathrm{Fuse}(U^{GO}, U^{coexp}, s_p) \in \mathbb{R}^{G \times d_z}.
\]

In the default Norman configs, `spectral.static: true`, so spectral graph positions are fixed rather than recomputed per perturbation. TODO: Confirm whether the final paper uses the static 0602 protocol or the contextual-local extension.

\subsection{Perturbation and Cell Conditioning}

The model constructs a perturbation embedding $e_p$. In the default path, this is produced by an MLP over the binary perturbation mask. The implemented `graph_pool` extension instead pools sign-invariant GO/coexpression coordinates over targeted genes, but this is not enabled in the default Norman configs.

For each gene $g$, SpecFlow builds a token from control expression, spectral feature, perturbation mask entry, and perturbation embedding:

\[
h_g = \mathrm{GeneTokenEncoder}(x_{c,g}, Z_{p,g}, s_{p,g}, e_p).
\]

An attentive pooling module aggregates gene tokens into a cell-level condition vector $c_p$. The velocity field then receives local per-gene inputs and global conditioning:

\[
v_\theta(x_t,t,x_c,s_p,Z_p,e_p,c_p) \in \mathbb{R}^{G}.
\]

The velocity field uses a time embedding, a local projection over $(x_{t,g}, x_{c,g}, Z_{p,g}, s_{p,g})$, a global projection of $(c_p, \phi(t), e_p)$, residual MLP blocks, and FiLM-style conditioning.

\subsection{Perturbation Propagation Features}

The default configured protocol enables `model.spectral_propagation: true` with `propagation_variant: spectral` inherited from `configs/default.yaml`. This produces additional propagation features from the perturbation mask.

The repository also implements `ContextualLocalPropagation`, where each gene routes between null, GO, and coexpression one-hop propagation candidates. This extension excludes directly perturbed genes from propagated influence and initializes the null route with high probability. However, because the default main configs do not enable it, this draft treats it as an implemented extension rather than a validated main component.

\subsection{Control-Anchored Flow Matching}

For each training pair sampled from unpaired pools, SpecFlow defines:

\[
x_0 = a + \sigma \epsilon,
\quad
a =
\begin{cases}
x_c, & \text{if control anchoring is enabled},\\
0, & \text{otherwise},
\end{cases}
\]

where $\epsilon$ is Gaussian noise and $\sigma$ is a configured noise scale. Given target perturbed expression $x_1$, the model samples $t \sim \mathcal{U}(0,1)$ and interpolates:

\[
x_t = (1-t)x_0 + t x_1.
\]

The target velocity is:

\[
u_t = x_1 - x_0.
\]

The primary loss is:

\[
\mathcal{L}_{FM} =
\mathbb{E}\left[
\left\|
v_\theta(x_t,t,x_c,s_p,Z_p) - (x_1-x_0)
\right\|_2^2
\right].
\]

This objective is directly implemented in `ControlAnchoredFlowMatching.compute_loss`.

\subsection{OT Coupling for Unpaired Populations}

Since control and perturbed cells are not paired, random pairings can create noisy velocity targets. If `flow.ot_coupling` is enabled, SpecFlow groups samples by perturbation condition within a batch and solves a balanced assignment between control and perturbed cells using squared expression distance. The perturbed cells are then reordered before computing flow targets.

This is an implementation of minibatch condition-wise OT-style coupling using the Hungarian algorithm. TODO: Empirically validate whether it improves metrics relative to random pairing.

\subsection{Delta-Correlation Auxiliary Loss}

The trainer optionally adds a condition-grouped correlation loss between the mean predicted velocity and the mean target perturbation delta:

\[
\mathcal{L}_{\Delta} =
1 - \mathrm{corr}
\left(
\mathbb{E}_{i \in p}[v_\theta^{(i)}],
\mathbb{E}_{i \in p}[x_1^{(i)} - a^{(i)}]
\right).
\]

The default Norman config sets `flow.delta_corr_weight: 0.03`. This aligns training with the evaluation emphasis on perturbation deltas. TODO: Validate with a no-delta-correlation ablation.

\subsection{Inference}

At test time, SpecFlow samples control cells, initializes near control expression, and integrates the learned velocity field with a fixed-step Euler solver for `inference.ode_steps` steps. The generated cells are compared with observed perturbed cells condition by condition. For scDFM-compatible evaluation, predictions and references are written to AnnData files and passed to `cell_eval`.

\section{Experiments}

\subsection{Datasets}

\paragraph{Norman.}
The primary configuration targets the Norman CRISPRa perturbation dataset. The README states that the raw h5ad contains approximately 19,264 genes and that preprocessing selects 5,000 HVGs while retaining perturbation target genes. TODO: Confirm exact cell count, condition count, gene count after preprocessing, and fold construction from local data.

\paragraph{ComboSciPlex.}
The repository includes a ComboSciPlex configuration and a drug-to-target mapping file. TODO: Confirm whether this dataset is used in the main paper or only as an appendix extension.

\subsection{Splits and Protocol}

The intended main protocol uses scDFM-aligned splits. For Norman additive experiments, `configs/norman.yaml` reads `data/split_results.pkl` and `split_fold`. For holdout experiments, `configs/norman_holdout.yaml` reads `data/splits/norman_holdout.pkl`, generated by `scripts/build_holdout_split.py`.

The repository explicitly requires training, testing, baselines, ablations, gene set, and cell_eval protocol to match. TODO: Verify these constraints in actual run artifacts.

\subsection{Baselines}

The repository supports local non-parametric baselines:

\begin{itemize}
    \item Control: predict perturbation expression directly from sampled control cells.
    \item Additive: predict a combination perturbation as control plus the sum of single-perturbation deltas estimated from training conditions.
\end{itemize}

The file `paper_baselines/scdfm_reported_metrics.csv` also includes paper-reported results for Control, Additive, scGPT, Geneformer, GEARS, CPA, STATE, CellFlow, scDFM, and ComboSciPlex baselines where available. These are not local reruns unless explicitly verified. TODO: Decide whether the paper will compare against paper-reported baselines, rerun baselines, or both.

\subsection{Metrics}

The main metrics planned by the repository are:

\begin{itemize}
    \item Pearson Delta: correlation between predicted and true perturbation effects.
    \item MSE and MAE: expression reconstruction error.
    \item DE Spearman: agreement on differential-expression genes.
    \item DS: distributional similarity.
    \item L2, Pearson Delta-hat, and Pearson Delta-hat20: extra scDFM-aligned metrics.
\end{itemize}

\subsection{Main Results}

TODO: Run SpecFlow on Norman additive folds 0--4 and report mean and standard deviation.

\begin{table}[t]
\centering
\caption{Norman additive results under scDFM-aligned splits. TODO: Fill SpecFlow values from `agg_results.csv` after running fold 0--4. Baseline values must be labeled as paper-reported or rerun.}
\begin{tabular}{lccccc}
\toprule
Method & Pearson Delta & MSE & MAE & DE Spearman & DS \\
\midrule
Control & TODO & TODO & TODO & TODO & TODO \\
Additive & TODO & TODO & TODO & TODO & TODO \\
scDFM & TODO & TODO & TODO & TODO & TODO \\
SpecFlow & TODO & TODO & TODO & TODO & TODO \\
\bottomrule
\end{tabular}
\end{table}

TODO: Run SpecFlow on Norman holdout and split results into Single and Double subsets.

\begin{table}[t]
\centering
\caption{Norman holdout results. TODO: Fill values after running `configs/norman_holdout.yaml` and `scripts/summarize_holdout_subsets.py`.}
\begin{tabular}{llccccc}
\toprule
Subset & Method & Pearson Delta & MSE & MAE & DE Spearman & DS \\
\midrule
Single & SpecFlow & TODO & TODO & TODO & TODO & TODO \\
Double & SpecFlow & TODO & TODO & TODO & TODO & TODO \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Implementation Details}

The default Norman config uses:

\begin{itemize}
    \item `batch_size: 512`
    \item `max_steps: 200000`
    \item `learning_rate: 0.0003`
    \item `warmup_steps: 2000`
    \item `sigma: 0.2`
    \item `ot_coupling: true`
    \item `delta_corr_weight: 0.03`
    \item `mmd_weight: 0.0`
    \item `inference.ode_steps: 30`
    \item `inference.n_control_cells: 128`
\end{itemize}

TODO: Add hardware, wall-clock time, random seed details, and dependency versions from actual run logs.

\section{Analysis and Ablation Study}

The repository includes configs for the following ablations, but current local state contains no completed result files.

\paragraph{Graph structure ablations.}
Planned variants include GO-only, coexpression-only, graph-none, and no spectral embedding. These test whether performance comes from graph structure rather than generic conditioning. TODO: Run `configs/experiments/core_components_4gpu.yaml`.

\paragraph{Propagation ablations.}
Planned variants include disabling spectral propagation and varying propagation scale/channels. TODO: Run additive and holdout propagation ablations.

\paragraph{OT coupling ablation.}
The `no_ot_coupling` variant disables condition-wise Hungarian matching and reverts to random pairings. TODO: Quantify its effect on Pearson Delta and distributional metrics.

\paragraph{Delta-correlation ablation.}
The `no_delta_corr` variant sets `flow.delta_corr_weight=0.0`. TODO: Verify whether this auxiliary objective improves perturbation-effect correlation without hurting MSE/MAE.

\paragraph{Control anchor ablation.}
The `no_control_anchor` variant trains/samples without measured control expression as the anchor. TODO: Use this to validate the central control-anchoring claim.

\paragraph{Contextual local propagation extension.}
The repository implements `graph_pool` perturbation encoding and `contextual_local` propagation. Paired experiments compare `propagation_scale=0.0` and `1.0`. TODO: Decide whether this extension becomes the final main method after results are available.

\paragraph{Visualization.}
Planned figures include predicted-vs-true delta scatter plots, UMAPs, top-DE gene boxplots, graph propagation heatmaps, and target-to-response gene paths. TODO: Implement and save figures from trained checkpoints.

\section{Limitations}

First, the current local repository does not include datasets, checkpoints, logs, or SpecFlow result files. Therefore, no empirical performance claims can be made from the available materials alone.

Second, the main-method definition is not fully locked. README documentation emphasizes the `graph_pool` and `contextual_local` path, while the default Norman configs use the legacy perturbation encoder and spectral propagation. This mismatch must be resolved before submission.

Third, SpecFlow relies on external gene annotations and coexpression estimates. Missing or noisy GO annotations and dataset-specific coexpression artifacts may affect performance.

Fourth, the velocity field is largely a per-gene local MLP with global conditioning and optional propagation features. It does not implement a full cross-gene attention velocity model in the default path. This may limit the expressivity of higher-order gene interactions.

Fifth, OT coupling is performed within minibatches and conditions using expression distance. This is a pragmatic approximation, not a guarantee of true biological cell-state pairing.

Sixth, fixed-step Euler inference may be less accurate than higher-order ODE solvers for complex velocity fields. The repository currently uses Euler sampling in the main path.

\section{Conclusion}

We described SpecFlow, a spectral-guided control-anchored flow-matching framework for single-cell perturbation response prediction. The method learns perturbation residual dynamics from measured control states, conditions the velocity field on GO and coexpression spectral structure, and includes training mechanisms for unpaired perturbation populations. The repository provides a substantial implementation and an scDFM-aligned evaluation plan, but submission-ready empirical claims require completing the planned additive, holdout, baseline, ablation, and visualization experiments. Once those results are available, the manuscript should be revised to report verified findings rather than protocol-level claims.
```

## D. Reviewer-Style Self-Check

### D.1 Claims That May Be Too Strong

- "SpecFlow improves perturbation prediction" is too strong until SpecFlow metrics are available.
- "Graph structure improves generalization" is too strong without graph-none, GO-only, coexpression-only, and no-spectral-propagation ablations.
- "OT coupling makes trajectories more biological" should be softened to "is intended to reduce random-pairing noise" unless supported by ablation and possibly qualitative trajectory analysis.
- "Contextual local propagation is a main innovation" is too strong unless final main configs and results use it.
- "scDFM-compatible evaluation is fully aligned" should be qualified as implementation support until actual run artifacts show identical split, gene set, and cell_eval settings.

### D.2 Experiments With Insufficient Support

- Main Norman additive results across fold 0--4 are missing.
- Control/additive baselines under the exact same local split/gene/evaluation protocol are missing.
- Norman holdout single/double results are missing.
- Core ablations are missing.
- ComboSciPlex extension results are missing.
- Hyperparameter stability is missing.
- Runtime/resource comparison is missing.
- Mechanistic visualizations are missing.

### D.3 Parts That Currently Do Not Yet Read Like a Top-Tier Submission

- The experiment section is still a protocol plan, not a result section.
- Related work lacks citations and precise comparisons.
- The main contribution is partially engineering/protocol-oriented; it needs empirical evidence to become a strong ML paper.
- The method version is ambiguous because README and default configs emphasize different propagation variants.
- No statistical reporting exists yet.
- No figure artifacts exist yet.

### D.4 Source-Evidence Gaps

- No local evidence that `cell_eval` is installed or runnable.
- No local evidence that external split files exist.
- No evidence that preprocessing produces the expected final gene count.
- No evidence that scDFM paper-reported metrics are directly comparable to local SpecFlow runs.
- No evidence that model checkpoints use EMA or that best checkpoints were selected by `pearson_delta` in actual runs.
- No evidence that `contextual_local` is stable at full scale.

### D.5 Current Code/Test Audit

- `python -m pytest -q` currently reports 57 passed and 2 failed in this Windows environment.
- Failure 1: `tests/test_dataset_protocol_configs.py::test_holdout_configs_use_generated_folded_pickle` reads UTF-8 YAML with the locale default encoding, causing a GBK `UnicodeDecodeError` on Chinese comments.
- Failure 2: `tests/test_launch_experiments.py::test_shell_command_prefers_current_checkout_source` expects the generated shell command to include `PYTHONPATH=<tmp>/src`, but the observed command string does not include that exact fragment.
- These failures do not invalidate the paper draft itself, but they are relevant for repository readiness and should be fixed or explained before presenting the codebase as fully reproducible.

### D.6 Needed Clarifications From the Project Owner

- Target conference style: ICLR, NeurIPS, ACL, EMNLP, or other.
- Final main method version: 0602 legacy+spectral propagation, or graph_pool+contextual_local.
- Whether comparisons should use scDFM paper-reported numbers or local reruns.
- Whether ComboSciPlex is intended for the main paper or appendix.
- Whether any completed outputs exist outside `C:\Users\Administrator\Desktop\specflow`.
- Preferred paper length and template.

### D.7 How to Move Toward Top-Conference Quality

1. Lock the method version and configs.
2. Run the minimal strong experiment set: Norman additive fold 0--4, Control/Additive fold 0--4, Norman holdout full, holdout single/double summary, core ablations.
3. Report mean and standard deviation, not a single best run.
4. Use the same split, gene set, and cell_eval protocol for all compared local methods.
5. Add mechanistic figures only after quantitative results are stable.
6. Separate implementation conveniences from research contributions.
7. Add a rigorous related-work section with precise citations.

## E. TODO List for Making the Paper Submission-Ready

### E.1 Method and Scope Decisions

- TODO: Choose final main method variant:
  - Option A: 0602 default model (`legacy` perturbation encoder + `spectral` propagation).
  - Option B: newer `graph_pool` + `contextual_local` variant.
- TODO: Make README, configs, method text, and experiment specs consistent with the chosen variant.
- TODO: Decide whether `ContextualLocalPropagation` is main, appendix, or future work.

### E.2 Data and Environment

- TODO: Add `data/norman.h5ad`.
- TODO: Add `data/split_results.pkl`.
- TODO: Add `data/gene_ontology/go_annotations.gaf`.
- TODO: Generate `data/splits/norman_holdout.pkl`.
- TODO: Confirm `cell_eval` installation.
- TODO: Fix or document the two current pytest failures, then rerun `pytest`.

### E.3 Main Experiments

- TODO: Run Norman additive SpecFlow fold 0--4 using `configs/experiments/additive_folds_4gpu.yaml`.
- TODO: Run Control baseline fold 0--4.
- TODO: Run Additive baseline fold 0--4.
- TODO: Summarize results as mean ± std.
- TODO: Merge with scDFM paper-reported or rerun baseline table, clearly labeling source.

### E.4 Holdout Experiments

- TODO: Run Norman holdout full model.
- TODO: Split holdout results into Single and Double subsets.
- TODO: Compare with scDFM paper-reported or rerun holdout baselines.

### E.5 Ablations

- TODO: Run `no_spectral_propagation`.
- TODO: Run `no_ot_coupling`.
- TODO: Run `no_delta_corr`.
- TODO: Run `graph_none`.
- TODO: Run `go_only`.
- TODO: Run `coexp_only`.
- TODO: Run `no_control_anchor`.
- TODO: Run `no_spectral_embedding`.
- TODO: If contextual local is used, run paired `propagation_scale=0` vs `1` comparisons.

### E.6 Metrics and Tables

- TODO: Extract `pearson_delta`, MSE, MAE, DE Spearman, DS from `agg_results.csv`.
- TODO: Compute or extract L2, Pearson Delta-hat, Pearson Delta-hat20.
- TODO: Build Table 1: Norman additive.
- TODO: Build Table 2: Norman holdout single/double.
- TODO: Build Table 3: ablations.
- TODO: Build Appendix tables for all folds, hyperparameters, and runtime.

### E.7 Figures

- TODO: Draw method overview.
- TODO: Generate predicted delta vs true delta scatter.
- TODO: Generate UMAP for control, true perturbed, SpecFlow prediction, and baseline.
- TODO: Generate top-DE gene boxplots.
- TODO: Generate graph propagation heatmap.
- TODO: If contextual local is used, export and visualize routing probabilities.

### E.8 Writing

- TODO: Add formal citations.
- TODO: Replace all TODO result placeholders with verified values or remove claims.
- TODO: Add dataset statistics.
- TODO: Add training hardware and runtime.
- TODO: Add standard deviation and possibly confidence intervals.
- TODO: Tighten contribution language after results are known.

### E.9 Verification Checklist Before Submission

- TODO: Every table value traces to a run directory with `run_config.yaml`, `train.log`, `training_summary.json`, `results.csv`, `agg_results.csv`, and `scdfm_evaluation_summary.json`.
- TODO: Every baseline uses the same split/gene/evaluation protocol or is clearly labeled as paper-reported.
- TODO: Every ablation changes only the intended component.
- TODO: Main method configs match the method section.
- TODO: No result claim remains without a corresponding artifact.
- TODO: All TODO markers are resolved or explicitly left as future work.
