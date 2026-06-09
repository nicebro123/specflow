import numpy as np
import torch

from specflow.data.dataset import PerturbationDataset, make_dataloader
from specflow.data.preprocessing import build_perturbation_map
from specflow.evaluation.metrics import compute_mae, compute_mmd, compute_mse
from specflow.graph.coexp_graph import CoexpressionGraphBuilder
from specflow.graph.spectral_embedding import SpectralEmbedding
from specflow.model.specflow import SpecFlow
from specflow.training.trainer import SpecFlowTrainer


def test_one_epoch_training_and_metrics_execute_end_to_end():
    rng = np.random.default_rng(7)
    torch.manual_seed(7)
    controls = rng.normal(size=(20, 6)).astype(np.float32)
    masks = build_perturbation_map(["G0", "G1"], [f"G{i}" for i in range(6)])
    targets = {
        name: controls + mask + rng.normal(scale=0.05, size=controls.shape)
        for name, mask in masks.items()
    }
    dataset = PerturbationDataset(controls, targets, masks, samples_per_condition=8)
    loader = make_dataloader(dataset, batch_size=8)
    graph = CoexpressionGraphBuilder(k_neighbors=2, threshold=0.1).build(controls)
    spectral = torch.from_numpy(SpectralEmbedding(n_components=2).fit_transform(graph))
    model = SpecFlow(6, 2, d_model=12, hidden_dim=24, n_velocity_layers=1)
    trainer = SpecFlowTrainer(model, spectral, sigma=0.1, learning_rate=1e-3)

    history = trainer.fit(loader, n_epochs=1)
    assert len(history) == 1
    assert np.isfinite(history[0]["train_loss"])

    pred = torch.from_numpy(targets["G0"])
    true = torch.from_numpy(targets["G1"])
    assert compute_mse(pred, true) >= 0
    assert compute_mae(pred, true) >= 0
    assert compute_mmd(pred, true) >= 0
