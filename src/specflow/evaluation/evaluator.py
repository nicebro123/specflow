"""Sample-based evaluator for trained SpecFlow models."""

from typing import Dict, Iterable, Optional

import numpy as np
import torch

from specflow.evaluation.metrics import (
    compute_de_spearman,
    compute_distributional_similarity,
    compute_energy_distance,
    compute_mae,
    compute_mmd,
    compute_mse,
    compute_mse_de,
    compute_pearson_de,
    compute_pearson_mean,
    compute_perturbation_discrimination_score,
)
from specflow.evaluation.results import (
    add_discrimination_metrics,
    compute_result_metrics,
)
from specflow.flow.ode_solver import EulerSampler


class SpecFlowEvaluator:
    def __init__(self, sampler: EulerSampler, spectral_embedding) -> None:
        self.sampler = sampler
        self.spectral_embedding = spectral_embedding

    def evaluate_condition(
        self,
        controls: torch.Tensor,
        pert_mask: torch.Tensor,
        observed: torch.Tensor,
        n_samples: int = 1,
        de_top_k: int = 20,
        return_samples: bool = False,
    ):
        predicted = self.sampler.sample(
            controls,
            pert_mask,
            self.spectral_embedding,
            n_samples=n_samples,
        )
        metrics = {
            "mse": compute_mse(predicted, observed),
            "mae": compute_mae(predicted, observed),
            "mmd": compute_mmd(predicted, observed),
            "energy_distance": compute_energy_distance(predicted, observed),
            "ds": compute_distributional_similarity(predicted, observed),
            "pearson_mean": compute_pearson_mean(predicted, observed, controls),
            "pearson_de": compute_pearson_de(
                predicted, observed, controls, top_k=de_top_k
            ),
            "de_spearman": compute_de_spearman(
                predicted, observed, controls, top_k=de_top_k
            ),
            "mse_de": compute_mse_de(
                predicted, observed, controls, top_k=de_top_k
            ),
        }
        metrics.update(
            compute_result_metrics(
                predicted,
                observed,
                controls,
                de_top_k=de_top_k,
            )
        )
        return (metrics, predicted) if return_samples else metrics

    def mean_pearson_delta(
        self,
        prepared_data,
        conditions: Iterable[str],
        n_control_cells: int = 64,
        n_samples: int = 1,
        seed: int = 42,
    ) -> float:
        """Lightweight perturbation-effect correlation for in-training monitoring.

        Samples predictions for each condition and returns the mean Pearson R
        between predicted and observed mean perturbation effects (pearson_delta).
        Only this single biological metric is computed, keeping it cheap enough
        to run every validation checkpoint.
        """
        conditions = [
            c for c in conditions if c in prepared_data.perturbed_by_condition
        ]
        if not conditions:
            return float("nan")
        model_device = next(self.sampler.model.parameters()).device
        rng = np.random.default_rng(seed)
        values = []
        for condition in conditions:
            target = prepared_data.perturbed_by_condition[condition]
            sample_count = min(n_control_cells, target.shape[0])
            control_indices = rng.choice(
                prepared_data.control_expression.shape[0],
                size=sample_count,
                replace=prepared_data.control_expression.shape[0] < sample_count,
            )
            controls = torch.from_numpy(
                prepared_data.control_expression[control_indices]
            ).to(model_device)
            observed = torch.from_numpy(target[:sample_count]).to(model_device)
            mask = torch.from_numpy(
                np.broadcast_to(
                    prepared_data.perturbation_map[condition], controls.shape
                ).copy()
            ).to(model_device)
            predicted = self.sampler.sample(
                controls, mask, self.spectral_embedding, n_samples=n_samples
            )
            values.append(compute_pearson_mean(predicted, observed, controls))
        return float(np.mean(values)) if values else float("nan")

    def evaluate_conditions(
        self,
        prepared_data,
        conditions: Iterable[str],
        n_control_cells: int = 64,
        n_samples: int = 1,
        de_top_k: int = 20,
        seed: int = 42,
        verbose: bool = True,
    ) -> Dict[str, object]:
        """Evaluate all requested conditions and aggregate scalar metrics."""
        conditions = list(conditions)
        if not conditions:
            return {"per_condition": {}, "aggregate": {}}
        model_device = next(self.sampler.model.parameters()).device
        rng = np.random.default_rng(seed)
        per_condition = {}
        generated = {}
        predicted_means = {}
        observed_means = {}
        total = len(conditions)
        for idx, condition in enumerate(conditions):
            if verbose:
                print(
                    f"\r  Evaluating [{idx + 1}/{total}] {condition}",
                    end="", flush=True,
                )
            target = prepared_data.perturbed_by_condition[condition]
            sample_count = min(n_control_cells, target.shape[0])
            control_indices = rng.choice(
                prepared_data.control_expression.shape[0],
                size=sample_count,
                replace=prepared_data.control_expression.shape[0] < sample_count,
            )
            controls = torch.from_numpy(
                prepared_data.control_expression[control_indices]
            ).to(model_device)
            observed = torch.from_numpy(target[:sample_count]).to(model_device)
            mask = torch.from_numpy(
                np.broadcast_to(
                    prepared_data.perturbation_map[condition], controls.shape
                ).copy()
            ).to(model_device)
            metrics, samples = self.evaluate_condition(
                controls,
                mask,
                observed,
                n_samples=n_samples,
                de_top_k=de_top_k,
                return_samples=True,
            )
            per_condition[condition] = metrics
            generated[condition] = samples.cpu()
            predicted_means[condition] = (
                samples.detach()
                .cpu()
                .reshape(-1, samples.shape[-1])
                .mean(dim=0)
                .numpy()
            )
            observed_means[condition] = (
                observed.detach()
                .cpu()
                .reshape(-1, observed.shape[-1])
                .mean(dim=0)
                .numpy()
            )
        if verbose:
            print()

        add_discrimination_metrics(per_condition, predicted_means, observed_means)
        metric_names = list(next(iter(per_condition.values())).keys())
        aggregate = {
            name: float(np.mean([m[name] for m in per_condition.values()]))
            for name in metric_names
        }
        aggregate["pds"] = compute_perturbation_discrimination_score(generated)

        if verbose:
            _print_summary(aggregate)

        return {"per_condition": per_condition, "aggregate": aggregate}


def _print_summary(aggregate: Dict[str, float]) -> None:
    print("  ── Aggregate Results ──")
    display_order = [
        ("mse", "MSE", ".6f"),
        ("mae", "MAE", ".6f"),
        ("mse_de", "MSE (DE)", ".6f"),
        ("pearson_mean", "Pearson R (all)", ".4f"),
        ("pearson_de", "Pearson R (DE)", ".4f"),
        ("de_spearman", "Spearman ρ (DE)", ".4f"),
        ("mmd", "MMD", ".6f"),
        ("energy_distance", "Energy Dist", ".6f"),
        ("ds", "DS", ".4f"),
        ("pds", "PDS", ".4f"),
    ]
    for key, label, fmt in display_order:
        if key in aggregate:
            print(f"    {label:>20s}: {aggregate[key]:{fmt}}")
