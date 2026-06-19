"""End-to-end experiment orchestration for h5ad benchmark runs."""

import json
import random
from contextlib import nullcontext
from dataclasses import asdict
from hashlib import sha1
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import sparse
import torch

from specflow.config import SpecFlowConfig
from specflow.data.benchmark import PreparedPerturbationData, load_benchmark_h5ad
from specflow.data.dataset import make_dataloader
from specflow.evaluation.evaluator import SpecFlowEvaluator
from specflow.evaluation.results import write_results_csv
from specflow.evaluation.scdfm_protocol import (
    resolve_scdfm_split,
    run_cell_eval,
    select_scdfm_eval_genes,
    write_scdfm_anndata,
)
from specflow.flow.ode_solver import EulerSampler
from specflow.graph.coexp_graph import CoexpressionGraphBuilder
from specflow.graph.go_graph import GOGraphBuilder
from specflow.graph.perturbation_aware import PerturbationAwareGraphModifier
from specflow.graph.spectral_cache import SpectralCache
from specflow.graph.spectral_embedding import SpectralEmbedding, SpectralResult
from specflow.model.specflow import SpecFlow
from specflow.training.trainer import SpecFlowTrainer


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _carve_val_from_train(
    prepared: "PreparedPerturbationData", fraction: float, seed: int
) -> None:
    """Move a deterministic fraction of train conditions into an empty val split.

    Used to enable pearson_delta monitoring when an external split (e.g. scDFM
    folds) defines no validation set. No-op when a val split already exists.
    """
    if fraction <= 0 or prepared.splits.get("val"):
        return
    train = list(prepared.splits.get("train", []))
    if len(train) < 2:
        return
    order = sorted(train)
    np.random.default_rng(seed).shuffle(order)
    n_val = max(1, int(round(len(order) * fraction)))
    n_val = min(n_val, len(order) - 1)
    held = set(order[:n_val])
    prepared.splits["val"] = sorted(held)
    prepared.splits["train"] = [c for c in train if c not in held]


def _sample_control_indices(
    control_expression: np.ndarray,
    n_cells: int,
    rng: np.random.Generator,
) -> np.ndarray:
    sample_count = min(n_cells, control_expression.shape[0])
    return rng.permutation(control_expression.shape[0])[:sample_count]


class _SingleGraphSpectralCache:
    """Perturbation-conditioned spectral cache for a single graph type."""

    def __init__(
        self,
        graph: sparse.spmatrix,
        n_components: int,
        modifier: PerturbationAwareGraphModifier,
        graph_type: str = "coexp",
        use_approximation: bool = False,
        static: bool = False,
    ) -> None:
        self.graph = sparse.csr_matrix(graph, dtype=np.float64)
        self.n_genes = self.graph.shape[0]
        self.graph_type = graph_type
        self.embedder = SpectralEmbedding(n_components=n_components)
        self.modifier = modifier
        self.use_approximation = use_approximation
        self.static = static
        self._base_result = self.embedder.fit(self.graph)
        self._memory: Dict[str, SpectralResult] = {}

    def _to_mask(self, pert_mask) -> np.ndarray:
        if torch.is_tensor(pert_mask):
            pert_mask = pert_mask.detach().cpu().numpy()
        return (np.asarray(pert_mask).reshape(-1) > 0).astype(np.uint8)

    def get(self, pert_mask) -> SpectralResult:
        if self.static:
            return self._base_result
        mask = self._to_mask(pert_mask)
        key = sha1(np.packbits(mask).tobytes()).hexdigest()[:16]
        if key in self._memory:
            return self._memory[key]
        if not mask.any():
            result = self._base_result
        else:
            modified = self.modifier.modify(self.graph, mask, self.graph_type)
            if self.use_approximation:
                result = self.embedder.fit_with_perturbation_update(
                    self.graph, modified, self._base_result
                )
            else:
                result = self.embedder.fit(modified)
        self._memory[key] = result
        return result

    def base_spectrum(self, graph_type: str = "coexp"):
        """Return (eigenvectors, eigenvalues) of the unperturbed base graph."""
        return self._base_result.eigenvectors, self._base_result.eigenvalues

    def precompute_all(self, perturbation_masks) -> None:
        if self.static:
            return  # base spectrum already computed in __init__
        for mask in perturbation_masks:
            self.get(mask)

    def batch_embeddings(self, pert_masks) -> torch.Tensor:
        is_tensor = torch.is_tensor(pert_masks)
        device = pert_masks.device if is_tensor else None
        dtype = (
            pert_masks.dtype
            if is_tensor and pert_masks.is_floating_point()
            else torch.float32
        )
        masks = pert_masks.detach().cpu().numpy() if is_tensor else np.asarray(pert_masks)
        if masks.ndim == 1:
            masks = masks[None, :]
        if self.static:
            base = torch.as_tensor(
                self._base_result.eigenvectors, dtype=dtype, device=device
            )
            return base.unsqueeze(0).expand(masks.shape[0], -1, -1)
        arrays = [self.get(m).eigenvectors for m in masks]
        return torch.as_tensor(np.stack(arrays, axis=0), dtype=dtype, device=device)


class ExperimentRunner:
    """Build graphs, train SpecFlow, and evaluate condition-level partitions."""

    def __init__(
        self,
        config: SpecFlowConfig,
        prepared_data: PreparedPerturbationData,
        output_dir: Optional[str] = None,
    ) -> None:
        self.config = config
        self.data = prepared_data
        self.output_dir = Path(output_dir or config.output.output_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.graphs = None
        self.spectral_cache = None
        self.model = None
        self.trainer = None
        self._validate_configuration()

    @classmethod
    def from_config(
        cls, config: SpecFlowConfig, output_dir: Optional[str] = None
    ) -> "ExperimentRunner":
        data = config.data
        prepared = load_benchmark_h5ad(
            data.h5ad_path,
            preprocess=data.preprocess,
            n_top_genes=data.n_top_genes,
            preprocess_cache=data.preprocess_cache,
            preprocess_cache_dir=data.preprocess_cache_dir,
            target_map_path=data.target_map_path,
            condition_key=data.condition_key,
            gene_key=data.gene_key,
            control_labels=data.control_labels,
            separator=data.separator,
            split_path=data.split_path,
            split_fold=data.split_fold,
            setting=data.setting,
            seed=data.seed,
            test_fraction=data.test_fraction,
            val_fraction=data.val_fraction,
        )
        _carve_val_from_train(prepared, data.val_from_train_fraction, data.seed)
        return cls(config, prepared, output_dir=output_dir)

    def _validate_configuration(self) -> None:
        if self.config.model.dual_graph:
            for name, components in (
                ("go_components", self.config.spectral.go_components),
                ("coexp_components", self.config.spectral.coexp_components),
            ):
                if components < 2 or components >= self.data.n_genes:
                    raise ValueError(
                        f"spectral.{name} must be in [2, n_genes - 1] for dual-graph fusion"
                    )
        else:
            spectral_dim = self.config.model.spectral_dim
            if spectral_dim < 2 or spectral_dim >= self.data.n_genes:
                raise ValueError(
                    f"model.spectral_dim must be in [2, n_genes - 1], got {spectral_dim}"
                )

    def seed_all(self) -> None:
        seed = self.config.data.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def build_graphs(self) -> Dict[str, sparse.csr_matrix]:
        coexp = CoexpressionGraphBuilder(
            k_neighbors=self.config.coexpression.k_neighbors,
            threshold=self.config.coexpression.threshold,
        ).build(self.data.control_expression)
        self.graphs = {"coexp": coexp}
        if self.config.model.dual_graph:
            go = GOGraphBuilder(
                self.data.gene_names,
                annotation_file=self.config.go.annotation_file,
                k_neighbors=self.config.go.k_neighbors,
                namespace=self.config.go.namespace,
            ).build()
            if go.nnz == 0:
                raise ValueError(
                    "GO graph has no edges; verify gene symbols, GO annotations, and namespace"
                )
            self.graphs["go"] = go
        return self.graphs

    def build_spectral_cache(self):
        if self.graphs is None:
            self.build_graphs()
        modifier = PerturbationAwareGraphModifier(
            alpha_go=self.config.perturbation_graph.alpha_go,
            alpha_coexp=self.config.perturbation_graph.alpha_coexp,
        )
        static = self.config.spectral.static
        if self.config.model.dual_graph:
            self.spectral_cache = SpectralCache(
                self.graphs,
                {
                    "go": self.config.spectral.go_components,
                    "coexp": self.config.spectral.coexp_components,
                },
                modifier=modifier,
                cache_dir=str(self.output_dir / self.config.spectral.cache_dir),
                use_approximation=self.config.spectral.use_perturbation_approx,
                static=static,
            )
        else:
            self.spectral_cache = _SingleGraphSpectralCache(
                self.graphs["coexp"],
                n_components=self.config.model.spectral_dim,
                modifier=modifier,
                graph_type="coexp",
                use_approximation=self.config.spectral.use_perturbation_approx,
                static=static,
            )
        return self.spectral_cache

    def precompute_spectra(self) -> Dict[str, object]:
        cache = self.spectral_cache or self.build_spectral_cache()
        cache.precompute_all(self.data.perturbation_map.values())
        graph_dir = self.output_dir / "graphs"
        graph_dir.mkdir(parents=True, exist_ok=True)
        for name, graph in self.graphs.items():
            sparse.save_npz(graph_dir / f"{name}.npz", graph)
        summary = {
            "n_genes": self.data.n_genes,
            "n_controls": int(self.data.control_expression.shape[0]),
            "n_conditions": len(self.data.perturbed_by_condition),
            "graph_edges": {
                name: int(graph.nnz // 2) for name, graph in self.graphs.items()
            },
            "splits": self.data.splits,
            "gene_names": self.data.gene_names,
        }
        _write_json(self.output_dir / "data_summary.json", summary)
        return summary

    def build_model(self) -> SpecFlow:
        mcfg = self.config.model
        spectral = self.config.spectral
        kwargs = dict(
            n_genes=self.data.n_genes,
            spectral_dim=mcfg.spectral_dim,
            d_model=mcfg.d_model,
            hidden_dim=mcfg.hidden_dim,
            n_velocity_layers=mcfg.n_velocity_layers,
            dual_graph=mcfg.dual_graph,
            pert_dim=mcfg.pert_dim,
            graph_dim=mcfg.graph_dim,
            use_spectral_embedding=mcfg.use_spectral_embedding,
            spectral_propagation=mcfg.spectral_propagation,
            propagation_channels=mcfg.propagation_channels,
            propagation_scale=mcfg.propagation_scale,
            propagation_gate=mcfg.propagation_gate,
            propagation_gate_init=mcfg.propagation_gate_init,
            perturbation_encoder=mcfg.perturbation_encoder,
            propagation_variant=mcfg.propagation_variant,
            local_propagation_hops=mcfg.local_propagation_hops,
            local_propagation_null_init=mcfg.local_propagation_null_init,
        )
        if mcfg.dual_graph:
            kwargs.update(
                go_components=spectral.go_components,
                coexp_components=spectral.coexp_components,
                macro_ratio=spectral.macro_ratio,
                graph_mode=mcfg.graph_mode,
                fusion_mode=mcfg.fusion_mode,
                scale_mode=mcfg.scale_mode,
            )
        self.model = SpecFlow(**kwargs)
        return self.model

    def _setup_propagation(self) -> None:
        """Install the fixed base-graph spectrum into the propagation module."""
        model = self.model
        if model is None:
            return
        contextual = getattr(model, "contextual_propagation", None)
        if contextual is not None:
            if self.graphs is None:
                self.build_graphs()
            contextual.set_graphs(
                self.graphs["go"],
                self.graphs["coexp"],
            )
            model.to(self.device)
            return
        if getattr(model, "propagation", None) is None:
            return
        cache = self.spectral_cache or self.build_spectral_cache()
        eigvecs, eigvals = cache.base_spectrum("coexp")
        model.propagation.set_basis(
            torch.as_tensor(np.asarray(eigvecs), dtype=torch.float32),
            torch.as_tensor(np.asarray(eigvals), dtype=torch.float32),
        )
        model.to(self.device)

    def _build_trainer(
        self,
        max_epochs: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> SpecFlowTrainer:
        cache = self.spectral_cache or self.build_spectral_cache()
        model = self.model or self.build_model()
        self._setup_propagation()
        flow = self.config.flow
        training = self.config.training
        step_mode = max_steps is not None
        schedule_length = max_steps if step_mode else (max_epochs or training.max_epochs)
        warmup_length = (
            training.warmup_steps
            if step_mode and training.warmup_steps is not None
            else (0 if step_mode else training.warmup_epochs)
        )
        self.trainer = SpecFlowTrainer(
            model,
            cache.batch_embeddings,
            sigma=flow.sigma,
            learning_rate=training.learning_rate,
            weight_decay=training.weight_decay,
            grad_clip=training.grad_clip,
            mmd_weight=flow.mmd_weight,
            mmd_interval=flow.mmd_interval,
            mmd_steps=flow.mmd_steps,
            delta_corr_weight=flow.delta_corr_weight,
            device=self.device,
            ema_decay=training.ema_decay if training.use_ema else 0.0,
            scheduler=training.scheduler,
            max_epochs=schedule_length,
            warmup_epochs=warmup_length,
            eta_min=training.eta_min,
            use_amp=training.use_amp,
            show_progress=training.show_progress,
            scheduler_interval="step" if step_mode else "epoch",
            ot_coupling=flow.ot_coupling,
            control_anchor=flow.control_anchor,
        )
        return self.trainer

    def _loader(self, partition: str, shuffle: bool):
        dataset = self.data.dataset(
            partition, samples_per_condition=self.config.data.samples_per_condition
        )
        if dataset is None:
            return None
        return make_dataloader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=shuffle,
            group_by_condition=self.config.training.group_by_condition,
        )

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / self.config.output.checkpoint_name

    def save_checkpoint(self, history, score: float, path: Optional[Path] = None) -> None:
        checkpoint_path = Path(path) if path is not None else self.checkpoint_path
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.trainer.model.state_dict(),
            "config": asdict(self.config),
            "gene_names": self.data.gene_names,
            "splits": self.data.splits,
            "history": history,
            "selection_score": score,
        }
        if self.trainer.ema is not None:
            payload["ema_state_dict"] = self.trainer.ema.state_dict()
        torch.save(payload, checkpoint_path)

    def _prune_interval_checkpoints(self) -> None:
        """Keep only the most recent N interval checkpoints to bound disk use.

        ``best.pt`` lives outside ``checkpoints/`` and is never pruned here.
        A negative ``keep_interval_checkpoints`` keeps everything.
        """
        keep = self.config.training.keep_interval_checkpoints
        if keep < 0:
            return
        ckpt_dir = self.output_dir / "checkpoints"
        if not ckpt_dir.is_dir():
            return

        def _step_num(path: Path) -> int:
            try:
                return int(path.stem.split("_")[1])
            except (IndexError, ValueError):
                return -1

        files = sorted(ckpt_dir.glob("step_*.pt"), key=_step_num)
        stale = files if keep == 0 else files[:-keep]
        for path in stale:
            try:
                path.unlink()
            except OSError:
                pass

    def load_checkpoint(self, path: Optional[str] = None) -> None:
        model = self.model or self.build_model()
        checkpoint = torch.load(
            path or self.checkpoint_path, map_location=self.device, weights_only=False
        )
        if checkpoint["gene_names"] != self.data.gene_names:
            raise ValueError("checkpoint genes do not match current dataset genes")
        if "ema_state_dict" in checkpoint:
            from specflow.training.ema import EMA
            ema = EMA(model, decay=0.0)
            ema.load_state_dict(checkpoint["ema_state_dict"])
            ema.apply_shadow()
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)

    def train(
        self,
        n_epochs: Optional[int] = None,
        max_steps: Optional[int] = None,
        eval_every_steps: Optional[int] = None,
    ) -> Dict[str, object]:
        resolved_steps = max_steps if max_steps is not None else self.config.training.max_steps
        if resolved_steps is not None:
            return self._train_steps(
                max_steps=resolved_steps,
                eval_every_steps=eval_every_steps or self.config.training.eval_every_steps,
            )
        return self._train_epochs(n_epochs=n_epochs)

    def _biological_validation(self, conditions: List[str]) -> Optional[float]:
        """Run the sampler on validation conditions and return mean pearson_delta.

        Uses EMA shadow weights when available so the monitored metric matches
        the weights that will be persisted for inference.
        """
        conditions = [
            c for c in conditions if c in self.data.perturbed_by_condition
        ]
        if not conditions:
            return None
        cache = self.spectral_cache or self.build_spectral_cache()
        if self.model is None:
            self.build_model()
        sampler = EulerSampler(
            self.model,
            sigma=self.config.flow.sigma,
            n_steps=self.config.inference.ode_steps,
            control_anchor=self.config.flow.control_anchor,
        )
        evaluator = SpecFlowEvaluator(sampler, cache.batch_embeddings)
        ema = self.trainer.ema if self.trainer is not None else None
        context = ema.shadow_context() if ema is not None else nullcontext()
        with context:
            value = evaluator.mean_pearson_delta(
                self.data,
                conditions,
                n_control_cells=self.config.inference.n_control_cells,
                n_samples=1,
                seed=self.config.data.seed,
            )
        return value

    @staticmethod
    def _selection_score(row: Dict[str, object]) -> float:
        """Lower is better. Prefer maximizing pearson_delta when available."""
        pearson = row.get("val_pearson_delta")
        if pearson is not None and pearson == pearson:  # not NaN
            return -float(pearson)
        return float(row.get("validation_loss", row["train_loss"]))

    def _train_epochs(self, n_epochs: Optional[int] = None) -> Dict[str, object]:
        self.seed_all()
        self.precompute_spectra()
        max_epochs = n_epochs or self.config.training.max_epochs
        trainer = self._build_trainer(max_epochs=max_epochs)
        train_loader = self._loader("train", shuffle=True)
        if train_loader is None:
            raise ValueError("train split is empty")
        validation_loader = self._loader("val", shuffle=False)
        val_conditions = self.data.splits.get("val", [])
        monitor_bio = self.config.training.monitor_pearson_delta and bool(val_conditions)
        history = []
        best_score = float("inf")
        epochs_without_improvement = 0
        epoch_iterable = range(max_epochs)
        if self.config.training.show_progress:
            from tqdm.auto import tqdm
            epoch_iterable = tqdm(epoch_iterable, desc="epochs")
        for epoch in epoch_iterable:
            row = {
                "epoch": epoch + 1,
                "train_loss": trainer.train_epoch(train_loader, epoch + 1, max_epochs),
                "lr": trainer.current_lr(),
            }
            if validation_loader is not None:
                row["validation_loss"] = trainer.evaluate_loss(
                    validation_loader, epoch + 1, max_epochs
                )
            if monitor_bio:
                row["val_pearson_delta"] = self._biological_validation(val_conditions)
            score = self._selection_score(row)
            history.append(row)
            if self.config.training.show_progress and hasattr(epoch_iterable, "set_postfix"):
                postfix = {
                    "train": f"{row['train_loss']:.4f}",
                    "lr": f"{row['lr']:.2e}",
                }
                if "validation_loss" in row:
                    postfix["val"] = f"{row['validation_loss']:.4f}"
                if row.get("val_pearson_delta") is not None:
                    postfix["pΔ"] = f"{row['val_pearson_delta']:.3f}"
                epoch_iterable.set_postfix(**postfix)
            if score < best_score:
                best_score = score
                epochs_without_improvement = 0
                self.save_checkpoint(history, score)
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= self.config.training.patience:
                break
        _write_json(self.output_dir / "training_history.json", history)
        if not self.checkpoint_path.exists():
            raise RuntimeError(
                "training finished without saving a checkpoint: the selection "
                "score was never valid (loss likely diverged to NaN/Inf). "
                "Lower training.learning_rate or tighten training.grad_clip."
            )
        self.load_checkpoint()
        result = {
            "checkpoint": str(self.checkpoint_path),
            "best_loss": best_score,
            "epochs_completed": len(history),
            "history": history,
        }
        _write_json(self.output_dir / "training_summary.json", result)
        return result

    def _train_steps(
        self,
        max_steps: int,
        eval_every_steps: int,
    ) -> Dict[str, object]:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if eval_every_steps < 1:
            raise ValueError("eval_every_steps must be positive")
        self.seed_all()
        self.precompute_spectra()
        trainer = self._build_trainer(max_steps=max_steps)
        train_loader = self._loader("train", shuffle=True)
        if train_loader is None:
            raise ValueError("train split is empty")
        validation_loader = self._loader("val", shuffle=False)
        val_conditions = self.data.splits.get("val", [])
        monitor_bio = self.config.training.monitor_pearson_delta and bool(val_conditions)
        history = []
        best_score = float("inf")
        step = 0
        losses = []
        progress = None
        if self.config.training.show_progress:
            from tqdm.auto import tqdm
            progress = tqdm(total=max_steps, desc="steps")
        while step < max_steps:
            for raw_batch in train_loader:
                if step >= max_steps:
                    break
                loss = trainer.train_batch(raw_batch)
                losses.append(loss)
                step += 1
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix(loss=f"{loss:.4f}", lr=f"{trainer.current_lr():.2e}")
                should_evaluate = step % eval_every_steps == 0 or step == max_steps
                if not should_evaluate:
                    continue
                row = {
                    "step": step,
                    "train_loss": float(np.mean(losses)),
                    "lr": trainer.current_lr(),
                }
                losses = []
                if validation_loader is not None:
                    row["validation_loss"] = trainer.evaluate_loss(validation_loader)
                if monitor_bio:
                    row["val_pearson_delta"] = self._biological_validation(val_conditions)
                score = self._selection_score(row)
                history.append(row)
                if progress is not None and row.get("val_pearson_delta") is not None:
                    progress.set_postfix(
                        loss=f"{row['train_loss']:.4f}",
                        lr=f"{trainer.current_lr():.2e}",
                        pΔ=f"{row['val_pearson_delta']:.3f}",
                    )
                interval_path = self.output_dir / "checkpoints" / f"step_{step}.pt"
                self.save_checkpoint(history, score, path=interval_path)
                self._prune_interval_checkpoints()
                if score < best_score:
                    best_score = score
                    self.save_checkpoint(history, score)
        if progress is not None:
            progress.close()
        _write_json(self.output_dir / "training_history.json", history)
        if not self.checkpoint_path.exists():
            raise RuntimeError(
                "training finished without saving a checkpoint: the selection "
                "score was never valid (loss likely diverged to NaN/Inf). "
                "Lower training.learning_rate or tighten training.grad_clip."
            )
        self.load_checkpoint()
        result = {
            "checkpoint": str(self.checkpoint_path),
            "best_loss": best_score,
            "steps_completed": step,
            "eval_every_steps": eval_every_steps,
            "mode": "steps",
            "history": history,
        }
        _write_json(self.output_dir / "training_summary.json", result)
        return result

    def evaluate(self, partition: str = "test") -> Dict[str, object]:
        torch.manual_seed(self.config.data.seed)  # reproducible ODE sampling noise
        cache = self.spectral_cache or self.build_spectral_cache()
        if self.model is None:
            self.build_model()
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(str(self.checkpoint_path))
        self.load_checkpoint()
        self._setup_propagation()
        sampler = EulerSampler(
            self.model,
            sigma=self.config.flow.sigma,
            n_steps=self.config.inference.ode_steps,
            control_anchor=self.config.flow.control_anchor,
        )
        evaluator = SpecFlowEvaluator(sampler, cache.batch_embeddings)
        self.model.reset_routing_stats()
        results = evaluator.evaluate_conditions(
            self.data,
            self.data.splits.get(partition, []),
            n_control_cells=self.config.inference.n_control_cells,
            n_samples=self.config.inference.n_samples,
            de_top_k=self.config.inference.de_top_k,
            seed=self.config.data.seed,
        )
        routing_summary = self.model.routing_summary()
        if routing_summary:
            results["routing_summary"] = routing_summary
        csv_path = self.output_dir / (
            "results.csv" if partition == "test" else f"results_{partition}.csv"
        )
        write_results_csv(csv_path, results["per_condition"])
        results["results_csv"] = str(csv_path)
        _write_json(self.output_dir / f"evaluation_{partition}.json", results)
        return results

    def evaluate_scdfm(
        self,
        checkpoint: Optional[str] = None,
        split_path: Optional[str] = None,
        fold: Optional[int] = None,
        split_method: Optional[str] = None,
        infer_top_gene: int = 1000,
        n_control_cells: Optional[int] = None,
        control_pert: str = "control",
        num_threads: int = 32,
        seed: Optional[int] = None,
        write_anndata_only: bool = False,
    ) -> Dict[str, object]:
        """Evaluate the test split through the scDFM cell_eval protocol."""

        checkpoint_path = Path(checkpoint or self.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(str(checkpoint_path))

        resolved_fold = self.config.data.split_fold if fold is None else fold
        resolved_split_path = (
            split_path if split_path is not None else self.config.data.split_path
        )
        setting = self.config.data.setting
        resolved_split_method = split_method or (
            setting if setting in {"additive", "combinations"} else "additive"
        )
        split = resolve_scdfm_split(
            resolved_split_path,
            self.data.perturbed_by_condition.keys(),
            dataset=self.config.data.dataset,
            fold=resolved_fold,
            split_method=resolved_split_method,
        )
        test_conditions = [
            condition
            for condition in split["test"]
            if condition in self.data.perturbed_by_condition
        ]
        missing_conditions = sorted(set(split["test"]) - set(test_conditions))
        if missing_conditions:
            preview = ", ".join(missing_conditions[:10])
            raise ValueError(
                f"scDFM split contains conditions absent from data: {preview}"
            )
        if not test_conditions:
            raise ValueError("scDFM split has no test conditions")

        cache = self.spectral_cache or self.build_spectral_cache()
        if self.model is None:
            self.build_model()
        self.load_checkpoint(str(checkpoint_path))
        self._setup_propagation()

        eval_gene_names = select_scdfm_eval_genes(
            self.config.data.h5ad_path,
            self.data.gene_names,
            test_conditions,
            condition_key=self.config.data.condition_key,
            gene_key=self.config.data.gene_key,
            control_labels=self.config.data.control_labels,
            infer_top_gene=infer_top_gene,
        )
        gene_to_idx = {gene: idx for idx, gene in enumerate(self.data.gene_names)}
        eval_indices = np.array([gene_to_idx[gene] for gene in eval_gene_names])

        sampler = EulerSampler(
            self.model,
            sigma=self.config.flow.sigma,
            n_steps=self.config.inference.ode_steps,
            control_anchor=self.config.flow.control_anchor,
        )
        resolved_seed = self.config.data.seed if seed is None else seed
        torch.manual_seed(resolved_seed)  # reproducible ODE sampling noise
        rng = np.random.default_rng(resolved_seed)
        sample_cells = (
            self.config.inference.n_control_cells
            if n_control_cells is None
            else n_control_cells
        )
        predicted = {}
        observed = {}
        self.model.reset_routing_stats()
        for condition in test_conditions:
            control_indices = _sample_control_indices(
                self.data.control_expression,
                sample_cells,
                rng,
            )
            controls_np = self.data.control_expression[control_indices]
            controls = torch.from_numpy(controls_np).to(self.device)
            mask_np = np.broadcast_to(
                self.data.perturbation_map[condition],
                controls_np.shape,
            ).copy()
            mask = torch.from_numpy(mask_np).to(self.device)
            samples = sampler.sample(
                controls,
                mask,
                cache.batch_embeddings,
                n_samples=1,
            )[0]
            # Predictions live in log1p space (>= 0); Euler integration of the
            # unconstrained velocity field can dip slightly negative. Clamp so
            # the written pred.h5ad is valid log-normalized data for cell_eval.
            samples = samples.clamp_min(0.0)
            predicted[condition] = samples.detach().cpu().numpy()[:, eval_indices]
            observed[condition] = self.data.perturbed_by_condition[condition][
                :, eval_indices
            ]

        paths = write_scdfm_anndata(
            str(self.output_dir),
            predicted_by_condition=predicted,
            observed_by_condition=observed,
            control_expression=self.data.control_expression[:, eval_indices],
            eval_gene_names=eval_gene_names,
            control_pert=control_pert,
        )
        result = {
            "protocol": "scdfm",
            "checkpoint": str(checkpoint_path),
            "fold": resolved_fold,
            "split_method": resolved_split_method,
            "n_test_conditions": len(test_conditions),
            "n_eval_genes": len(eval_gene_names),
            "test_conditions": test_conditions,
            **paths,
        }
        routing_summary = self.model.routing_summary()
        if routing_summary:
            result["routing_summary"] = routing_summary
        if not write_anndata_only:
            result.update(
                run_cell_eval(
                    paths["pred_h5ad"],
                    paths["real_h5ad"],
                    str(self.output_dir),
                    control_pert=control_pert,
                    num_threads=num_threads,
                )
            )

        summary_path = self.output_dir / "scdfm_evaluation_summary.json"
        _write_json(summary_path, result)
        result["summary_json"] = str(summary_path)
        return result
