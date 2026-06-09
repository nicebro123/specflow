"""Preprocessing helpers for single-cell perturbation datasets."""

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


def _is_log_transformed(adata, max_value_threshold: float = 30.0) -> bool:
    """Heuristic: if max expression < threshold, data is likely already log1p-transformed."""
    from scipy import sparse
    X = adata.X
    if sparse.issparse(X):
        return float(X.max()) < max_value_threshold
    return float(np.max(X)) < max_value_threshold


def load_and_preprocess(
    h5ad_path: str,
    n_top_genes: int = 5000,
    target_sum: float = 1e4,
    keep_genes: Optional[Iterable[str]] = None,
):
    """Load an h5ad file and run the standard Scanpy preprocessing pipeline.

    Perturbation target genes listed in *keep_genes* are always retained
    even if they fall outside the HVG selection.
    """
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError(
            "load_and_preprocess requires the optional dependency set: "
            "python -m pip install -e '.[data]'"
        ) from exc

    path = Path(h5ad_path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    adata = sc.read_h5ad(path)
    sc.pp.filter_cells(adata, min_genes=1)
    sc.pp.filter_genes(adata, min_cells=1)

    already_log = _is_log_transformed(adata)
    if not already_log:
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)

    # HVG ranking on log-normalized data uses the 'seurat' flavor (dispersion).
    n_select = min(n_top_genes, adata.n_vars)
    sc.pp.highly_variable_genes(
        adata, n_top_genes=n_select, flavor="seurat", subset=False
    )

    # Always retain perturbation target genes, even outside the HVG budget, so
    # every condition's perturbed genes remain modeled.
    if keep_genes:
        forced = sorted(set(keep_genes) & set(adata.var_names))
        if forced:
            adata.var.loc[forced, "highly_variable"] = True

    selected = adata.var["highly_variable"].to_numpy()
    if not selected.any():
        raise ValueError("highly_variable_genes selected no genes")
    return adata[:, selected].copy()


def build_perturbation_map(
    conditions: Iterable[str],
    gene_names: Iterable[str],
    separator: str = "+",
    control_labels: Iterable[str] = ("ctrl", "control", "non-targeting"),
    target_map: Optional[Mapping[str, Sequence[str]]] = None,
) -> Dict[str, np.ndarray]:
    """Map named perturbations to gene-aligned multi-hot masks.

    A condition such as ``A+B`` maps to non-zero entries for genes ``A`` and
    ``B``. Tokens such as ``ctrl`` in ``A+ctrl`` are ignored. ``target_map``
    allows drug or alias conditions to resolve to modeled target genes.
    """
    genes = list(gene_names)
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
    controls = {label.lower() for label in control_labels}
    mapping: Dict[str, np.ndarray] = {}

    for condition in conditions:
        mask = np.zeros(len(genes), dtype=np.float32)
        if condition.lower() not in controls:
            if target_map and condition in target_map:
                targets = list(target_map[condition])
            else:
                targets = []
                for item in condition.split(separator):
                    item = item.strip()
                    if not item or item.lower() in controls:
                        continue
                    if target_map and item in target_map:
                        targets.extend(target_map[item])
                    else:
                        targets.append(item)
            missing = [target for target in targets if target not in gene_to_idx]
            if missing:
                raise ValueError(
                    "Perturbed genes absent from modeled gene list for "
                    f"{condition!r}: {missing}"
                )
            for target in targets:
                mask[gene_to_idx[target]] = 1.0
        mapping[condition] = mask
    return mapping


def create_splits(
    conditions: Iterable[str],
    setting: str = "additive",
    seed: int = 42,
    test_fraction: float = 0.2,
    val_fraction: float = 0.1,
    separator: str = "+",
    control_labels: Iterable[str] = ("ctrl", "control", "non-targeting"),
) -> Mapping[str, List[str]]:
    """Create reproducible condition-level train/validation/test splits.

    In ``additive`` mode single perturbations stay in training and combination
    conditions are divided between splits. ``holdout`` divides every
    non-control condition, providing a basic unseen-perturbation split.
    Public benchmark split files should replace this helper in paper runs.
    """
    unique = sorted(set(conditions))
    if setting not in {"additive", "holdout"}:
        raise ValueError("setting must be 'additive' or 'holdout'")
    rng = np.random.default_rng(seed)
    controls = {label.lower() for label in control_labels}

    def n_targets(condition: str) -> int:
        return sum(
            item.strip().lower() not in controls
            for item in condition.split(separator)
            if item.strip()
        )

    singles = [condition for condition in unique if n_targets(condition) <= 1]
    combinations = [condition for condition in unique if n_targets(condition) > 1]
    candidates = combinations if setting == "additive" else unique
    candidates = list(candidates)
    rng.shuffle(candidates)

    n_test = int(round(len(candidates) * test_fraction))
    n_val = int(round(len(candidates) * val_fraction))
    if candidates and test_fraction > 0:
        n_test = max(1, n_test)
    test = candidates[:n_test]
    val = candidates[n_test : n_test + n_val]
    held_out = set(test + val)
    train = [condition for condition in unique if condition not in held_out]
    if setting == "additive":
        train = sorted(set(train + singles))
    return {"train": train, "val": val, "test": test}
