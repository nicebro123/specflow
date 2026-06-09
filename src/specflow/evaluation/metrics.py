"""Point, distribution, and biological response metrics for perturbation prediction."""

from typing import Iterable, Mapping

import numpy as np
from scipy.stats import spearmanr, pearsonr
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_cell_matrix(samples: torch.Tensor) -> torch.Tensor:
    if samples.ndim < 2:
        raise ValueError("samples must contain cells and genes")
    return samples.reshape(-1, samples.shape[-1]).float()


def _mean_shift(
    samples: torch.Tensor, controls: torch.Tensor
) -> torch.Tensor:
    return _as_cell_matrix(samples).mean(dim=0) - _as_cell_matrix(controls).mean(dim=0)


def _top_de_indices(
    observed: torch.Tensor, controls: torch.Tensor, top_k: int
) -> np.ndarray:
    shift = torch.abs(
        _as_cell_matrix(observed).mean(dim=0) - _as_cell_matrix(controls).mean(dim=0)
    ).cpu().numpy()
    n_select = min(top_k, shift.size)
    return np.argsort(shift)[-n_select:]


# ---------------------------------------------------------------------------
# Point-level metrics (compare means)
# ---------------------------------------------------------------------------

def compute_mse(pred_samples: torch.Tensor, true_samples: torch.Tensor) -> float:
    pred_mean = _as_cell_matrix(pred_samples).mean(dim=0)
    true_mean = _as_cell_matrix(true_samples).mean(dim=0)
    return float(torch.mean((pred_mean - true_mean) ** 2).cpu())


def compute_mae(pred_samples: torch.Tensor, true_samples: torch.Tensor) -> float:
    pred_mean = _as_cell_matrix(pred_samples).mean(dim=0)
    true_mean = _as_cell_matrix(true_samples).mean(dim=0)
    return float(torch.mean(torch.abs(pred_mean - true_mean)).cpu())


# ---------------------------------------------------------------------------
# Correlation metrics
# ---------------------------------------------------------------------------

def compute_pearson_mean(
    pred_samples: torch.Tensor,
    true_samples: torch.Tensor,
    control_samples: torch.Tensor,
) -> float:
    """Pearson R between predicted and observed mean perturbation effects."""
    pred_shift = _mean_shift(pred_samples, control_samples).cpu().numpy()
    true_shift = _mean_shift(true_samples, control_samples).cpu().numpy()
    r, _ = pearsonr(pred_shift, true_shift)
    return float(r) if np.isfinite(r) else 0.0


def compute_pearson_de(
    pred_samples: torch.Tensor,
    true_samples: torch.Tensor,
    control_samples: torch.Tensor,
    top_k: int = 20,
) -> float:
    """Pearson R on top-k differentially expressed genes."""
    if top_k < 2:
        raise ValueError("top_k must be at least two")
    idx = _top_de_indices(true_samples, control_samples, top_k)
    pred_shift = _mean_shift(pred_samples, control_samples).cpu().numpy()
    true_shift = _mean_shift(true_samples, control_samples).cpu().numpy()
    r, _ = pearsonr(pred_shift[idx], true_shift[idx])
    return float(r) if np.isfinite(r) else 0.0


def compute_de_spearman(
    pred_samples: torch.Tensor,
    true_samples: torch.Tensor,
    control_samples: torch.Tensor,
    top_k: int = 20,
) -> float:
    """Spearman rho between predicted and observed perturbation effects on top DE genes."""
    if top_k < 2:
        raise ValueError("top_k must be at least two")
    idx = _top_de_indices(true_samples, control_samples, top_k)
    pred_shift = _mean_shift(pred_samples, control_samples).cpu().numpy()
    true_shift = _mean_shift(true_samples, control_samples).cpu().numpy()
    rho = spearmanr(pred_shift[idx], true_shift[idx]).statistic
    return float(rho) if np.isfinite(rho) else 0.0


def compute_mse_de(
    pred_samples: torch.Tensor,
    true_samples: torch.Tensor,
    control_samples: torch.Tensor,
    top_k: int = 20,
) -> float:
    """MSE restricted to top-k differentially expressed genes."""
    if top_k < 2:
        raise ValueError("top_k must be at least two")
    idx = _top_de_indices(true_samples, control_samples, top_k)
    pred_mean = _as_cell_matrix(pred_samples).mean(dim=0).cpu().numpy()
    true_mean = _as_cell_matrix(true_samples).mean(dim=0).cpu().numpy()
    return float(np.mean((pred_mean[idx] - true_mean[idx]) ** 2))


# ---------------------------------------------------------------------------
# Distribution-level metrics
# ---------------------------------------------------------------------------

def _rbf_kernel(
    left: torch.Tensor, right: torch.Tensor, bandwidths: Iterable[float]
) -> torch.Tensor:
    distance_sq = torch.cdist(left, right) ** 2
    kernel = torch.zeros_like(distance_sq)
    for bandwidth in bandwidths:
        if bandwidth <= 0:
            raise ValueError("MMD bandwidths must be positive")
        kernel = kernel + torch.exp(-distance_sq / (2.0 * bandwidth ** 2))
    return kernel


def compute_mmd(
    pred_samples: torch.Tensor,
    true_samples: torch.Tensor,
    bandwidths: Iterable[float] = (0.1, 0.5, 1.0, 5.0),
) -> float:
    """Biased multi-bandwidth RBF MMD estimate."""
    pred = _as_cell_matrix(pred_samples)
    true = _as_cell_matrix(true_samples)
    mmd = (
        _rbf_kernel(pred, pred, bandwidths).mean()
        + _rbf_kernel(true, true, bandwidths).mean()
        - 2.0 * _rbf_kernel(pred, true, bandwidths).mean()
    )
    return float(torch.clamp(mmd, min=0.0).cpu())


def compute_energy_distance(
    pred_samples: torch.Tensor,
    true_samples: torch.Tensor,
) -> float:
    """Energy distance: E(P,Q) = 2E||X-Y|| - E||X-X'|| - E||Y-Y'||."""
    pred = _as_cell_matrix(pred_samples)
    true = _as_cell_matrix(true_samples)
    cross = torch.cdist(pred, true).mean()
    intra_pred = torch.cdist(pred, pred).mean()
    intra_true = torch.cdist(true, true).mean()
    ed = 2.0 * cross - intra_pred - intra_true
    return float(torch.clamp(ed, min=0.0).cpu())


def compute_distributional_similarity(
    pred_samples: torch.Tensor,
    true_samples: torch.Tensor,
) -> float:
    """DS score: bounded distributional similarity based on energy distance.

    DS = 1 / (1 + energy_distance), ranges in (0, 1] with 1 = identical.
    """
    ed = compute_energy_distance(pred_samples, true_samples)
    return 1.0 / (1.0 + ed)


# ---------------------------------------------------------------------------
# Retrieval-level metrics
# ---------------------------------------------------------------------------

def compute_perturbation_discrimination_score(
    generated_by_condition: Mapping[str, torch.Tensor],
) -> float:
    """Leave-one-out nearest-neighbor condition discrimination accuracy (PDS)."""
    matrices = []
    labels = []
    for condition, samples in generated_by_condition.items():
        matrix = _as_cell_matrix(samples)
        matrices.append(matrix)
        labels.extend([condition] * matrix.shape[0])
    if not matrices:
        raise ValueError("generated_by_condition must be non-empty")
    combined = torch.cat(matrices, dim=0)
    if combined.shape[0] < 2:
        return 0.0
    distances = torch.cdist(combined, combined)
    distances.fill_diagonal_(float("inf"))
    nearest = distances.argmin(dim=1).cpu().tolist()
    correct = sum(labels[i] == labels[n] for i, n in enumerate(nearest))
    return correct / len(labels)
