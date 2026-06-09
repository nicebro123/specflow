"""CSV result table helpers for perturbation-level evaluation."""

import csv
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch


RESULT_COLUMNS = [
    "perturbation",
    "overlap_at_N",
    "overlap_at_50",
    "overlap_at_100",
    "overlap_at_200",
    "overlap_at_500",
    "precision_at_N",
    "precision_at_50",
    "precision_at_100",
    "precision_at_200",
    "precision_at_500",
    "de_spearman_sig",
    "de_direction_match",
    "de_spearman_lfc_sig",
    "de_sig_genes_recall",
    "de_nsig_counts_real",
    "de_nsig_counts_pred",
    "pr_auc",
    "roc_auc",
    "pearson_delta",
    "mse",
    "mae",
    "mse_delta",
    "mae_delta",
    "discrimination_score_l1",
    "discrimination_score_l2",
    "discrimination_score_cosine",
    "pearson_edistance",
    "clustering_agreement",
]


def _cells(samples: torch.Tensor) -> np.ndarray:
    return samples.detach().cpu().reshape(-1, samples.shape[-1]).float().numpy()


def _safe_corr(left: np.ndarray, right: np.ndarray, method: str = "pearson") -> float:
    if left.size < 2 or right.size < 2:
        return 0.0
    if np.allclose(left, left[0]) or np.allclose(right, right[0]):
        return 0.0
    if method == "spearman":
        value = spearmanr(left, right).statistic
    else:
        value, _ = pearsonr(left, right)
    return float(value) if np.isfinite(value) else 0.0


def _top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(max(int(k), 1), scores.size)
    return np.argsort(scores)[-k:]


def _overlap_metrics(abs_true: np.ndarray, abs_pred: np.ndarray, k: int) -> tuple:
    true_idx = set(_top_indices(abs_true, k).tolist())
    pred_idx = set(_top_indices(abs_pred, k).tolist())
    if not true_idx or not pred_idx:
        return 0.0, 0.0
    intersection = len(true_idx & pred_idx)
    return intersection / len(true_idx), intersection / len(pred_idx)


def _sig_masks(
    true_delta: np.ndarray,
    pred_delta: np.ndarray,
    controls: np.ndarray,
    observed: np.ndarray,
    min_genes: int,
) -> tuple:
    pooled_se = np.sqrt(
        controls.var(axis=0, ddof=0) / max(controls.shape[0], 1)
        + observed.var(axis=0, ddof=0) / max(observed.shape[0], 1)
    )
    pooled_se = np.maximum(pooled_se, 1e-8)
    true_z = np.abs(true_delta) / pooled_se
    pred_z = np.abs(pred_delta) / pooled_se
    true_mask = true_z >= 2.0
    pred_mask = pred_z >= 2.0

    if true_mask.sum() < min_genes:
        true_mask = np.zeros_like(true_mask, dtype=bool)
        true_mask[_top_indices(np.abs(true_delta), min_genes)] = True
    if pred_mask.sum() < min_genes:
        pred_mask = np.zeros_like(pred_mask, dtype=bool)
        pred_mask[_top_indices(np.abs(pred_delta), min_genes)] = True
    return true_mask, pred_mask


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(scores)[::-1]
    ranked = labels[order].astype(bool)
    hits = np.cumsum(ranked)
    precision = hits / (np.arange(ranked.size) + 1)
    return float((precision * ranked).sum() / positives)


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return 0.5
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, scores.size + 1)
    pos_rank_sum = ranks[labels.astype(bool)].sum()
    auc = (pos_rank_sum - positives * (positives + 1) / 2) / (
        positives * negatives
    )
    return float(auc) if np.isfinite(auc) else 0.5


def compute_result_metrics(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    controls: torch.Tensor,
    de_top_k: int = 20,
) -> Dict[str, float]:
    """Compute perturbation-level metrics for the final results CSV."""

    pred_cells = _cells(predicted)
    true_cells = _cells(observed)
    ctrl_cells = _cells(controls)
    pred_mean = pred_cells.mean(axis=0)
    true_mean = true_cells.mean(axis=0)
    ctrl_mean = ctrl_cells.mean(axis=0)
    pred_delta = pred_mean - ctrl_mean
    true_delta = true_mean - ctrl_mean
    abs_pred = np.abs(pred_delta)
    abs_true = np.abs(true_delta)
    n_genes = abs_true.size
    min_sig = min(max(int(de_top_k), 1), n_genes)
    true_sig, pred_sig = _sig_masks(
        true_delta, pred_delta, ctrl_cells, true_cells, min_genes=min_sig
    )

    row: Dict[str, float] = {}
    n_real = int(true_sig.sum())
    n_pred = int(pred_sig.sum())
    n_eval = max(n_real, 1)
    real_set = set(np.flatnonzero(true_sig).tolist())
    pred_top_n = set(_top_indices(abs_pred, n_eval).tolist())
    intersection_n = len(real_set & pred_top_n)
    union_n = len(real_set | pred_top_n)
    row["overlap_at_N"] = intersection_n / union_n if union_n else 0.0
    row["precision_at_N"] = intersection_n / len(pred_top_n) if pred_top_n else 0.0
    for k in (50, 100, 200, 500):
        overlap, precision = _overlap_metrics(abs_true, abs_pred, k)
        row[f"overlap_at_{k}"] = overlap
        row[f"precision_at_{k}"] = precision

    sig_idx = np.flatnonzero(true_sig)
    row["de_spearman_sig"] = _safe_corr(
        pred_delta[sig_idx], true_delta[sig_idx], method="spearman"
    )
    row["de_spearman_lfc_sig"] = row["de_spearman_sig"]
    row["de_direction_match"] = float(
        np.mean(np.sign(pred_delta[sig_idx]) == np.sign(true_delta[sig_idx]))
    )
    pred_sig_idx = set(np.flatnonzero(pred_sig).tolist())
    row["de_sig_genes_recall"] = (
        len(set(sig_idx.tolist()) & pred_sig_idx) / len(sig_idx)
        if len(sig_idx)
        else 0.0
    )
    row["de_nsig_counts_real"] = float(n_real)
    row["de_nsig_counts_pred"] = float(n_pred)
    labels = true_sig.astype(int)
    row["pr_auc"] = _average_precision(labels, abs_pred)
    row["roc_auc"] = _roc_auc(labels, abs_pred)
    row["pearson_delta"] = _safe_corr(pred_delta, true_delta)
    row["mse_delta"] = float(np.mean((pred_delta - true_delta) ** 2))
    row["mae_delta"] = float(np.mean(np.abs(pred_delta - true_delta)))
    return row


def add_discrimination_metrics(
    per_condition: Dict[str, Dict[str, float]],
    predicted_means: Mapping[str, np.ndarray],
    observed_means: Mapping[str, np.ndarray],
) -> None:
    """Add condition retrieval and global distance agreement metrics in-place."""

    conditions = list(per_condition)
    if not conditions:
        return
    observed_matrix = np.stack([observed_means[name] for name in conditions])
    predicted_matrix = np.stack([predicted_means[name] for name in conditions])
    true_pairwise = _pairwise_distances(observed_matrix, metric="l2")
    pred_pairwise = _pairwise_distances(predicted_matrix, metric="l2")
    clustering_agreement = _upper_triangle_corr(pred_pairwise, true_pairwise)

    for condition in conditions:
        pred_mean = predicted_means[condition]
        for metric, column in (
            ("l1", "discrimination_score_l1"),
            ("l2", "discrimination_score_l2"),
            ("cosine", "discrimination_score_cosine"),
        ):
            distances = np.array(
                [
                    _distance(pred_mean, observed_means[other], metric=metric)
                    for other in conditions
                ]
            )
            self_distance = distances[conditions.index(condition)]
            per_condition[condition][column] = float(np.mean(self_distance <= distances))

        pearson_distances = np.array(
            [
                _distance(pred_mean, observed_means[other], metric="correlation")
                for other in conditions
            ]
        )
        per_condition[condition]["pearson_edistance"] = float(
            1.0 - pearson_distances.mean()
        )
        per_condition[condition]["clustering_agreement"] = clustering_agreement


def _distance(left: np.ndarray, right: np.ndarray, metric: str) -> float:
    if metric == "l1":
        return float(np.mean(np.abs(left - right)))
    if metric == "l2":
        return float(np.linalg.norm(left - right))
    if metric == "cosine":
        denom = np.linalg.norm(left) * np.linalg.norm(right)
        return 1.0 - float(np.dot(left, right) / denom) if denom > 0 else 1.0
    if metric == "correlation":
        return 1.0 - _safe_corr(left, right)
    raise ValueError(f"unknown distance metric: {metric}")


def _pairwise_distances(matrix: np.ndarray, metric: str) -> np.ndarray:
    n_rows = matrix.shape[0]
    distances = np.zeros((n_rows, n_rows), dtype=float)
    for i in range(n_rows):
        for j in range(i + 1, n_rows):
            value = _distance(matrix[i], matrix[j], metric=metric)
            distances[i, j] = value
            distances[j, i] = value
    return distances


def _upper_triangle_corr(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape[0] < 2:
        return 1.0
    idx = np.triu_indices(left.shape[0], k=1)
    return _safe_corr(left[idx], right[idx])


def write_results_csv(
    path: Path,
    per_condition: Mapping[str, Mapping[str, float]],
) -> None:
    """Write a benchmark-style per-perturbation results CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for condition, metrics in per_condition.items():
            row = {"perturbation": condition}
            for column in RESULT_COLUMNS[1:]:
                row[column] = metrics.get(column, "")
            writer.writerow(row)
