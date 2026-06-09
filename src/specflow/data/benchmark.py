"""Adapters for h5ad-based perturbation benchmark datasets."""

import json
import pickle
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from scipy import sparse
import yaml

from specflow.data.dataset import PerturbationDataset
from specflow.data.preprocessing import (
    build_perturbation_map,
    create_splits,
    load_and_preprocess,
)


def _to_dense_float32(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def load_target_map(path: Optional[str]) -> Optional[Mapping[str, Sequence[str]]]:
    """Load condition or drug target definitions from JSON/YAML."""
    if path is None:
        return None
    target_path = Path(path)
    if not target_path.exists():
        raise FileNotFoundError(str(target_path))
    with target_path.open("r", encoding="utf-8") as handle:
        if target_path.suffix.lower() == ".json":
            raw = json.load(handle)
        else:
            raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("target map must be a mapping of condition/token to genes")
    output = {}
    for condition, genes in raw.items():
        if isinstance(genes, str):
            genes = [genes]
        output[str(condition)] = [str(gene) for gene in genes]
    return output


def load_condition_splits(
    path: str,
    available_conditions: Iterable[str],
    control_labels: Iterable[str] = ("ctrl", "control", "non-targeting"),
    fold: int = 0,
) -> Dict[str, List[str]]:
    """Read externally supplied benchmark splits and validate their coverage."""
    split_path = Path(path)
    if not split_path.exists():
        raise FileNotFoundError(str(split_path))
    is_folded_pickle = split_path.suffix.lower() in {".pkl", ".pickle"}
    if is_folded_pickle:
        with split_path.open("rb") as handle:
            raw = pickle.load(handle)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            if fold < 0 or fold >= len(raw):
                raise ValueError(f"fold {fold} is outside split file with {len(raw)} folds")
            raw = raw[fold]
    else:
        with split_path.open("r", encoding="utf-8") as handle:
            if split_path.suffix.lower() == ".json":
                raw = json.load(handle)
            else:
                raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("split file must contain train, val, and test lists")
    controls = {label.lower() for label in control_labels}
    conditions = {
        condition for condition in available_conditions if condition.lower() not in controls
    }
    splits = {}
    for name in ("train", "val", "test"):
        values = raw.get(name, [])
        if not isinstance(values, list):
            raise ValueError(f"split {name!r} must be a list")
        splits[name] = [
            str(condition)
            for condition in values
            if str(condition).lower() not in controls
        ]
    flattened = [condition for values in splits.values() for condition in values]
    if len(flattened) != len(set(flattened)):
        raise ValueError("split file assigns a condition to multiple partitions")
    unknown = set(flattened) - conditions
    missing = conditions - set(flattened)
    if unknown:
        raise ValueError(f"split file contains unknown conditions: {sorted(unknown)}")
    if missing:
        if not is_folded_pickle:
            raise ValueError(f"split file omits available conditions: {sorted(missing)}")
        splits["train"].extend(sorted(missing))
    return splits


@dataclass
class PreparedPerturbationData:
    """Model-ready expression pools and condition-level partition metadata."""

    gene_names: List[str]
    control_expression: np.ndarray
    perturbed_by_condition: Dict[str, np.ndarray]
    perturbation_map: Dict[str, np.ndarray]
    splits: Dict[str, List[str]]

    @property
    def n_genes(self) -> int:
        return len(self.gene_names)

    def dataset(
        self, partition: str, samples_per_condition: Optional[int] = None
    ) -> Optional[PerturbationDataset]:
        conditions = self.splits.get(partition, [])
        if not conditions:
            return None
        return PerturbationDataset(
            self.control_expression,
            self.perturbed_by_condition,
            self.perturbation_map,
            conditions=conditions,
            samples_per_condition=samples_per_condition,
        )


def prepare_anndata(
    adata,
    condition_key: str = "condition",
    gene_key: Optional[str] = None,
    control_labels: Iterable[str] = ("ctrl", "control", "non-targeting"),
    separator: str = "+",
    target_map: Optional[Mapping[str, Sequence[str]]] = None,
    split_path: Optional[str] = None,
    split_fold: int = 0,
    setting: str = "additive",
    seed: int = 42,
    test_fraction: float = 0.2,
    val_fraction: float = 0.1,
) -> PreparedPerturbationData:
    """Convert an AnnData object into unpaired control/perturbation pools."""
    if condition_key not in adata.obs:
        raise KeyError(f"AnnData.obs does not contain condition key {condition_key!r}")
    if gene_key is not None:
        if gene_key not in adata.var:
            raise KeyError(f"AnnData.var does not contain gene key {gene_key!r}")
        gene_names = [str(gene) for gene in adata.var[gene_key].tolist()]
    else:
        gene_names = [str(gene) for gene in adata.var_names.tolist()]
    if len(gene_names) != len(set(gene_names)):
        raise ValueError("modeled gene names must be unique")

    labels = np.asarray(adata.obs[condition_key].astype(str))
    expression = _to_dense_float32(adata.X)
    controls = {label.lower() for label in control_labels}
    control_indices = np.array([label.lower() in controls for label in labels])
    if not control_indices.any():
        raise ValueError("no control cells found under configured control_labels")
    control_expression = expression[control_indices]

    conditions = sorted(
        {label for label in labels if label.lower() not in controls}
    )
    if not conditions:
        raise ValueError("no perturbed conditions found in AnnData")
    perturbed = {
        condition: expression[labels == condition]
        for condition in conditions
    }
    perturbation_map = build_perturbation_map(
        conditions,
        gene_names,
        separator=separator,
        control_labels=control_labels,
        target_map=target_map,
    )
    if split_path:
        splits = load_condition_splits(
            split_path,
            conditions,
            control_labels,
            fold=split_fold,
        )
    else:
        splits = dict(
            create_splits(
                conditions,
                setting=setting,
                seed=seed,
                test_fraction=test_fraction,
                val_fraction=val_fraction,
                separator=separator,
                control_labels=control_labels,
            )
        )
    return PreparedPerturbationData(
        gene_names=gene_names,
        control_expression=control_expression,
        perturbed_by_condition=perturbed,
        perturbation_map=perturbation_map,
        splits=splits,
    )


def _extract_perturbation_targets(
    h5ad_path: str,
    condition_key: str = "condition",
    separator: str = "+",
    control_labels: Iterable[str] = ("ctrl", "control", "non-targeting"),
    target_map_path: Optional[str] = None,
) -> List[str]:
    """Scan condition labels to collect perturbation target gene names."""
    try:
        import anndata as ad
    except ImportError:
        return []
    adata = ad.read_h5ad(h5ad_path, backed="r")
    if condition_key not in adata.obs:
        return []
    controls = {label.lower() for label in control_labels}
    target_map = load_target_map(target_map_path)
    targets = set()
    for label in adata.obs[condition_key].astype(str).unique():
        if label.lower() in controls:
            continue
        if target_map and label in target_map:
            targets.update(target_map[label])
        else:
            for token in label.split(separator):
                token = token.strip()
                if token and token.lower() not in controls:
                    if target_map and token in target_map:
                        targets.update(target_map[token])
                    else:
                        targets.add(token)
    return sorted(targets)


def _preprocess_cache_path(
    h5ad_path: str,
    n_top_genes: int,
    keep_genes: Sequence[str],
    cache_dir: Optional[str],
) -> Path:
    """Deterministic cache path for a preprocessed (HVG-subset) dataset."""
    src = Path(h5ad_path)
    mtime = int(src.stat().st_mtime)
    key_source = (
        f"{src.resolve()}|{mtime}|{n_top_genes}|{','.join(sorted(keep_genes or []))}"
    )
    digest = sha1(key_source.encode("utf-8")).hexdigest()[:16]
    base = Path(cache_dir) if cache_dir else src.parent / ".specflow_cache"
    return base / f"{src.stem}_hvg{n_top_genes}_{digest}.h5ad"


def load_benchmark_h5ad(
    h5ad_path: str,
    preprocess: bool = False,
    n_top_genes: int = 5000,
    target_map_path: Optional[str] = None,
    preprocess_cache: bool = True,
    preprocess_cache_dir: Optional[str] = None,
    **kwargs,
) -> PreparedPerturbationData:
    """Read a benchmark h5ad file, optionally preprocessing raw counts.

    When ``preprocess`` is set, the expensive HVG selection is run once and the
    resulting gene-subset dataset is cached to disk (keyed by source mtime,
    ``n_top_genes``, and the retained perturbation genes). Subsequent runs reuse
    the cache, skipping the full-matrix HVG pass.
    """
    if preprocess:
        try:
            import anndata as ad
        except ImportError as exc:
            raise ImportError(
                "h5ad loading requires: python -m pip install -e '.[data]'"
            ) from exc
        keep_genes = _extract_perturbation_targets(
            h5ad_path,
            condition_key=kwargs.get("condition_key", "condition"),
            separator=kwargs.get("separator", "+"),
            control_labels=kwargs.get("control_labels", ("ctrl", "control", "non-targeting")),
            target_map_path=target_map_path,
        )
        cache_path = (
            _preprocess_cache_path(
                h5ad_path, n_top_genes, keep_genes, preprocess_cache_dir
            )
            if preprocess_cache
            else None
        )
        if cache_path is not None and cache_path.exists():
            adata = ad.read_h5ad(cache_path)
        else:
            adata = load_and_preprocess(
                h5ad_path, n_top_genes=n_top_genes, keep_genes=keep_genes
            )
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                adata.write_h5ad(cache_path)
    else:
        try:
            import anndata as ad
        except ImportError as exc:
            raise ImportError(
                "h5ad loading requires: python -m pip install -e '.[data]'"
            ) from exc
        adata = ad.read_h5ad(h5ad_path)
    return prepare_anndata(
        adata,
        target_map=load_target_map(target_map_path),
        **kwargs,
    )
