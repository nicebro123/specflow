"""scDFM-compatible evaluation protocol utilities."""

import pickle
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

import numpy as np
from scipy import sparse
import yaml


def scdfm_norman_split(
    conditions: Iterable[str],
    fold: int = 0,
    split_method: str = "additive",
    test_fraction: float = 0.3,
    separator: str = "+",
) -> Mapping[str, List[str]]:
    """Reproduce scDFM's Norman additive/combinations fold construction."""

    if fold < 0 or fold >= 5:
        raise ValueError("scDFM Norman split fold must be in [0, 4]")
    if split_method not in {"additive", "combinations"}:
        raise ValueError("generated scDFM splits support additive or combinations")
    perturbations = np.array(sorted(set(str(condition) for condition in conditions)))
    double_perturbations = np.array(
        [condition for condition in perturbations if "ctrl" not in condition]
    )
    rng = np.random.default_rng(42 + fold)
    shuffled = double_perturbations.copy()
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * test_fraction)
    test = shuffled[:split_idx].tolist()

    if split_method == "combinations":
        remove_genes = set()
        for condition in test[:15]:
            remove_genes.update(token for token in condition.split(separator) if token)
        test = test[:15] + [f"{gene}+control" for gene in sorted(remove_genes)]

    test_set = set(test)
    train = [condition for condition in perturbations.tolist() if condition not in test_set]
    return {"train": train, "val": [], "test": test}


def load_scdfm_split(
    path: str,
    fold: int = 0,
) -> Mapping[str, List[str]]:
    """Load a scDFM split file from pickle, JSON, or YAML."""

    split_path = Path(path)
    if not split_path.exists():
        raise FileNotFoundError(str(split_path))
    if split_path.suffix.lower() in {".pkl", ".pickle"}:
        with split_path.open("rb") as handle:
            raw = pickle.load(handle)
    else:
        with split_path.open("r", encoding="utf-8") as handle:
            if split_path.suffix.lower() == ".json":
                import json

                raw = json.load(handle)
            else:
                raw = yaml.safe_load(handle)

    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        if fold < 0 or fold >= len(raw):
            raise ValueError(f"fold {fold} is outside split file with {len(raw)} folds")
        raw = raw[fold]
    if not isinstance(raw, Mapping):
        raise ValueError("scDFM split file must contain a split mapping or list of folds")
    return {
        "train": [str(condition) for condition in raw.get("train", [])],
        "val": [str(condition) for condition in raw.get("val", [])],
        "test": [str(condition) for condition in raw.get("test", [])],
    }


def resolve_scdfm_split(
    split_path: Optional[str],
    conditions: Iterable[str],
    dataset: str,
    fold: int = 0,
    split_method: str = "additive",
    data_root: str = "data",
) -> Mapping[str, List[str]]:
    """Resolve the strict scDFM test split, preferring an external split file."""

    if split_path:
        return load_scdfm_split(split_path, fold=fold)
    default_path = Path(data_root) / dataset / "split_results.pkl"
    if default_path.exists():
        return load_scdfm_split(str(default_path), fold=fold)
    if dataset not in {"norman", "norman_umi_go_filtered"}:
        raise ValueError(
            "Provide --split-path for scDFM-compatible evaluation on this dataset"
        )
    return scdfm_norman_split(
        conditions,
        fold=fold,
        split_method=split_method,
    )


def select_scdfm_eval_genes(
    h5ad_path: str,
    model_gene_names: Sequence[str],
    test_conditions: Iterable[str],
    condition_key: str = "condition",
    gene_key: Optional[str] = None,
    control_labels: Iterable[str] = ("ctrl", "control"),
    infer_top_gene: int = 1000,
) -> List[str]:
    """Select scDFM-style test HVGs after aligning to the model gene space."""

    if infer_top_gene < 1:
        raise ValueError("infer_top_gene must be positive")
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError("scDFM-compatible evaluation requires scanpy") from exc

    adata = sc.read_h5ad(h5ad_path)
    if condition_key not in adata.obs:
        raise KeyError(f"AnnData.obs does not contain {condition_key!r}")
    if gene_key is not None:
        if gene_key not in adata.var:
            raise KeyError(f"AnnData.var does not contain {gene_key!r}")
        adata.var_names = [str(gene) for gene in adata.var[gene_key].tolist()]
    if len(set(adata.var_names)) != adata.n_vars:
        raise ValueError("AnnData gene names must be unique for scDFM evaluation")

    missing = [gene for gene in model_gene_names if gene not in adata.var_names]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            "checkpoint genes are absent from the raw h5ad gene list: "
            f"{preview}"
        )
    adata = adata[:, list(model_gene_names)].copy()

    labels = np.asarray(adata.obs[condition_key].astype(str))
    controls = {label.lower() for label in control_labels}
    tests = set(str(condition) for condition in test_conditions)
    eval_mask = np.array(
        [label in tests or label.lower() in controls for label in labels],
        dtype=bool,
    )
    eval_adata = adata[eval_mask].copy()
    if eval_adata.n_obs == 0:
        raise ValueError("scDFM evaluation split has no cells")

    n_select = min(infer_top_gene, eval_adata.n_vars)
    sc.pp.highly_variable_genes(eval_adata, n_top_genes=n_select, subset=False)
    hvg = eval_adata.var
    eval_genes = [
        str(gene)
        for gene in eval_adata.var_names[
            np.asarray(hvg["highly_variable"], dtype=bool)
        ].tolist()
    ]
    if len(eval_genes) > n_select:
        if "highly_variable_rank" in hvg:
            ranked = hvg.loc[eval_genes, "highly_variable_rank"]
            if ranked.notna().any():
                eval_genes = ranked.sort_values().index.astype(str).tolist()
        elif "dispersions_norm" in hvg:
            eval_genes = (
                hvg.loc[eval_genes, "dispersions_norm"]
                .sort_values(ascending=False)
                .index.astype(str)
                .tolist()
            )
        eval_genes = eval_genes[:n_select]
    if not eval_genes:
        raise ValueError("scDFM evaluation selected no genes")
    return eval_genes


def to_dense_float32(matrix) -> np.ndarray:
    """Convert a dense or sparse matrix to float32 numpy."""

    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def write_scdfm_anndata(
    output_dir: str,
    predicted_by_condition: Mapping[str, np.ndarray],
    observed_by_condition: Mapping[str, np.ndarray],
    control_expression: np.ndarray,
    eval_gene_names: Sequence[str],
    control_pert: str = "control",
) -> Mapping[str, str]:
    """Write pred.h5ad and real.h5ad in the format consumed by cell_eval."""

    try:
        import anndata as ad
        import pandas as pd
    except ImportError as exc:
        raise ImportError("scDFM-compatible evaluation requires anndata and pandas") from exc

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pred_arrays = [np.asarray(control_expression, dtype=np.float32)]
    real_arrays = [np.asarray(control_expression, dtype=np.float32)]
    pred_labels = [control_pert] * pred_arrays[0].shape[0]
    real_labels = [control_pert] * real_arrays[0].shape[0]
    for condition in predicted_by_condition:
        pred = np.asarray(predicted_by_condition[condition], dtype=np.float32)
        real = np.asarray(observed_by_condition[condition], dtype=np.float32)
        pred_arrays.append(pred)
        real_arrays.append(real)
        pred_labels.extend([condition] * pred.shape[0])
        real_labels.extend([condition] * real.shape[0])

    var = pd.DataFrame(index=[str(gene) for gene in eval_gene_names])
    pred_obs = pd.DataFrame(
        {"perturbation": pred_labels},
        index=[f"pred_{idx}" for idx in range(len(pred_labels))],
    )
    real_obs = pd.DataFrame(
        {"perturbation": real_labels},
        index=[f"real_{idx}" for idx in range(len(real_labels))],
    )
    pred_adata = ad.AnnData(
        X=np.concatenate(pred_arrays, axis=0),
        obs=pred_obs,
        var=var.copy(),
    )
    real_adata = ad.AnnData(
        X=np.concatenate(real_arrays, axis=0),
        obs=real_obs,
        var=var.copy(),
    )
    pred_path = output / "pred.h5ad"
    real_path = output / "real.h5ad"
    pred_adata.write_h5ad(pred_path)
    real_adata.write_h5ad(real_path)
    return {"pred_h5ad": str(pred_path), "real_h5ad": str(real_path)}


def run_cell_eval(
    pred_h5ad: str,
    real_h5ad: str,
    output_dir: str,
    control_pert: str = "control",
    pert_col: str = "perturbation",
    num_threads: int = 32,
) -> Mapping[str, str]:
    """Run the same cell_eval MetricsEvaluator used by scDFM."""

    try:
        import anndata as ad
        from cell_eval import MetricsEvaluator
    except ImportError as exc:
        raise ImportError(
            "scDFM-compatible metrics require the cell_eval package used by scDFM"
        ) from exc

    output = Path(output_dir)
    pred = ad.read_h5ad(pred_h5ad)
    real = ad.read_h5ad(real_h5ad)
    evaluator = MetricsEvaluator(
        adata_pred=pred,
        adata_real=real,
        control_pert=control_pert,
        pert_col=pert_col,
        num_threads=num_threads,
    )
    results, agg_results = evaluator.compute()
    results_path = output / "results.csv"
    aggregate_path = output / "agg_results.csv"
    results.write_csv(results_path)
    agg_results.write_csv(aggregate_path)
    return {
        "results_csv": str(results_path),
        "agg_results_csv": str(aggregate_path),
    }
