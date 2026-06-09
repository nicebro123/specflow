"""Compute extra scDFM-style table metrics from pred.h5ad and real.h5ad.

L2 mean is directly comparable to the scDFM table definition. The Delta-hat
metrics require a reference expression. This script records the reference mode
in the output so paper tables do not silently mix definitions.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or right.size < 2:
        return float("nan")
    if np.allclose(left, left[0]) or np.allclose(right, right[0]):
        return float("nan")
    value = np.corrcoef(left, right)[0, 1]
    return float(value) if np.isfinite(value) else float("nan")


def _mean_corr(left: np.ndarray, right: np.ndarray) -> float:
    n = min(left.shape[0], right.shape[0])
    if n == 0:
        return float("nan")
    values = [_safe_corr(left[idx], right[idx]) for idx in range(n)]
    return float(np.nanmean(values)) if not np.all(np.isnan(values)) else float("nan")


def _labels(adata, pert_col: str) -> np.ndarray:
    if pert_col not in adata.obs:
        raise KeyError(f"AnnData.obs does not contain {pert_col!r}")
    return np.asarray(adata.obs[pert_col].astype(str))


def _matrix(adata, mask: np.ndarray) -> np.ndarray:
    matrix = adata[mask].X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _condition_rows(
    pred,
    real,
    pert_col: str,
    control_pert: str,
    reference: str,
    top_k: int,
) -> List[Dict[str, object]]:
    pred_labels = _labels(pred, pert_col)
    real_labels = _labels(real, pert_col)
    control_lower = control_pert.lower()
    pred_control = _matrix(pred, np.char.lower(pred_labels.astype(str)) == control_lower)
    real_control = _matrix(real, np.char.lower(real_labels.astype(str)) == control_lower)
    if pred_control.size == 0 or real_control.size == 0:
        raise ValueError("pred.h5ad and real.h5ad must both contain control cells")
    control_mean = real_control.mean(axis=0)

    conditions = sorted(
        set(real_labels.tolist()) | set(pred_labels.tolist()),
        key=str,
    )
    rows = []
    for condition in conditions:
        if condition.lower() == control_lower:
            continue
        pred_cells = _matrix(pred, pred_labels == condition)
        real_cells = _matrix(real, real_labels == condition)
        if pred_cells.size == 0 or real_cells.size == 0:
            continue

        pred_mean = pred_cells.mean(axis=0)
        real_mean = real_cells.mean(axis=0)
        if reference == "control":
            ref = control_mean
        elif reference == "real-mean":
            ref = real_mean
        else:
            raise ValueError("reference must be 'control' or 'real-mean'")

        pred_residual = pred_cells[: real_cells.shape[0]] - ref
        real_residual = real_cells[: pred_cells.shape[0]] - ref
        n_pair = min(pred_residual.shape[0], real_residual.shape[0])
        pred_residual = pred_residual[:n_pair]
        real_residual = real_residual[:n_pair]
        variances = real_residual.var(axis=0)
        top_idx = np.argsort(variances)[-min(top_k, variances.size) :]

        rows.append(
            {
                "perturbation": condition,
                "l2_mean": float(np.linalg.norm(pred_mean - real_mean)),
                "pearson_delta_hat": _mean_corr(pred_residual, real_residual),
                "pearson_delta_hat20": _mean_corr(
                    pred_residual[:, top_idx],
                    real_residual[:, top_idx],
                ),
                "n_pred_cells": int(pred_cells.shape[0]),
                "n_real_cells": int(real_cells.shape[0]),
                "n_paired_cells": int(n_pair),
            }
        )
    return rows


def _summary(rows: Iterable[Dict[str, object]], reference: str) -> Dict[str, object]:
    rows = list(rows)
    output: Dict[str, object] = {
        "reference": reference,
        "n_conditions": len(rows),
    }
    for metric in ("l2_mean", "pearson_delta_hat", "pearson_delta_hat20"):
        values = np.asarray([row[metric] for row in rows], dtype=float)
        output[metric] = float(np.nanmean(values)) if values.size else float("nan")
        output[f"{metric}_std"] = (
            float(np.nanstd(values, ddof=1)) if values.size > 1 else float("nan")
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-h5ad", required=True)
    parser.add_argument("--real-h5ad", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pert-col", default="perturbation")
    parser.add_argument("--control-pert", default="control")
    parser.add_argument("--reference", choices=("control", "real-mean"), default="control")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("extra metric computation requires anndata") from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred = ad.read_h5ad(args.pred_h5ad)
    real = ad.read_h5ad(args.real_h5ad)
    rows = _condition_rows(
        pred,
        real,
        pert_col=args.pert_col,
        control_pert=args.control_pert,
        reference=args.reference,
        top_k=args.top_k,
    )
    csv_path = output_dir / "extra_metrics.csv"
    fieldnames = [
        "perturbation",
        "l2_mean",
        "pearson_delta_hat",
        "pearson_delta_hat20",
        "n_pred_cells",
        "n_real_cells",
        "n_paired_cells",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = _summary(rows, args.reference)
    summary["extra_metrics_csv"] = str(csv_path)
    summary_path = output_dir / "extra_metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
