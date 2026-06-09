"""Runnable synthetic demonstration for the dual-graph SpecFlow model."""

import argparse
import json
from typing import Dict

import numpy as np
import torch

from specflow.data.dataset import PerturbationDataset, make_dataloader
from specflow.data.preprocessing import build_perturbation_map
from specflow.evaluation.evaluator import SpecFlowEvaluator
from specflow.flow.ode_solver import EulerSampler
from specflow.graph.coexp_graph import CoexpressionGraphBuilder
from specflow.graph.go_graph import GOGraphBuilder
from specflow.graph.perturbation_aware import PerturbationAwareGraphModifier
from specflow.graph.spectral_cache import SpectralCache
from specflow.model.specflow import SpecFlow
from specflow.training.trainer import SpecFlowTrainer


def _synthetic_data(
    seed: int = 42, n_cells: int = 96, n_genes: int = 24
):
    rng = np.random.default_rng(seed)
    genes = [f"G{idx}" for idx in range(n_genes)]
    latent = rng.normal(size=(n_cells, 4))
    loading = rng.normal(scale=0.4, size=(4, n_genes))
    controls = (latent @ loading + rng.normal(scale=0.15, size=(n_cells, n_genes))).astype(
        np.float32
    )
    conditions = ["G0", "G1", "G0+G1"]
    perturbation_map = build_perturbation_map(conditions, genes)
    perturbed: Dict[str, np.ndarray] = {}
    for name in conditions:
        mask = perturbation_map[name]
        smooth_effect = mask + 0.3 * np.roll(mask, 1) + 0.2 * np.roll(mask, 2)
        perturbed[name] = (
            controls + smooth_effect + rng.normal(scale=0.12, size=controls.shape)
        ).astype(np.float32)
    return controls, perturbed, perturbation_map


def _synthetic_go_annotations(n_genes: int):
    return {
        f"G{index}": {
            f"module_{index // 4}",
            f"cross_module_{index % 4}",
        }
        for index in range(n_genes)
    }


def run_synthetic() -> None:
    parser = argparse.ArgumentParser(
        description="Train dual-graph SpecFlow on synthetic perturbation data."
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    controls, perturbed, perturbation_map = _synthetic_data(seed=args.seed)
    dataset = PerturbationDataset(
        controls,
        perturbed,
        perturbation_map,
        samples_per_condition=48,
    )
    loader = make_dataloader(dataset, batch_size=32)

    coexp_graph = CoexpressionGraphBuilder(k_neighbors=5, threshold=0.25).build(
        controls
    )
    go_graph = GOGraphBuilder(
        [f"G{idx}" for idx in range(controls.shape[1])], k_neighbors=5
    ).build_from_annotations(_synthetic_go_annotations(controls.shape[1]))
    cache = SpectralCache(
        {"go": go_graph, "coexp": coexp_graph},
        {"go": 8, "coexp": 8},
        modifier=PerturbationAwareGraphModifier(alpha_go=0.3, alpha_coexp=0.1),
    )
    model = SpecFlow(
        n_genes=controls.shape[1],
        spectral_dim=16,
        d_model=48,
        hidden_dim=96,
        n_velocity_layers=2,
        dual_graph=True,
        go_components=8,
        coexp_components=8,
        pert_dim=24,
        graph_dim=24,
    )
    trainer = SpecFlowTrainer(
        model,
        cache.batch_embeddings,
        sigma=0.1,
        learning_rate=2e-3,
        mmd_weight=0.02,
        mmd_interval=2,
        mmd_steps=3,
    )
    history = trainer.fit(loader, n_epochs=args.epochs)

    condition = "G0+G1"
    control_tensor = torch.from_numpy(controls[:32]).to(trainer.device)
    mask_tensor = torch.from_numpy(
        np.broadcast_to(perturbation_map[condition], control_tensor.shape).copy()
    ).to(trainer.device)
    evaluator = SpecFlowEvaluator(
        EulerSampler(trainer.model, sigma=0.1, n_steps=20),
        cache.batch_embeddings,
    )
    metrics = evaluator.evaluate_condition(
        control_tensor,
        mask_tensor,
        torch.from_numpy(perturbed[condition][:32]).to(trainer.device),
        n_samples=1,
    )
    print(json.dumps({"history": history, "evaluation": metrics}, indent=2))


if __name__ == "__main__":
    run_synthetic()
