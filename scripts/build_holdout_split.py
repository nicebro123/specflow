"""Generate a Norman *holdout* (scDFM "unseen") split file for SpecFlow.

scDFM defines its holdout/generalization setting as ``split_method='unseen'`` in
``src/data_process/data.py``: per fold it holds out a handful of perturbation
*genes*, and the test set is every condition whose target genes include at least
one held-out gene (both the held-out single perturbations and any double
perturbation that involves them). Everything else is training data.

The official scDFM repository does **not** ship this split file -- it is created
at runtime into ``split_results_unseen.pkl`` -- so we reproduce the same protocol
here. Two intentional, documented differences make the result usable by SpecFlow:

* the per-fold gene shuffle is *seeded* (``seed + fold``) so our split is
  reproducible (scDFM's unseen branch used an unseeded ``random.shuffle``);
* the test set only ever contains condition labels that actually exist in the
  AnnData, using their exact strings, so SpecFlow's evaluation never trips over a
  phantom ``GENE+control`` label that is absent from the data.

The output is a list of ``{"train", "test", "holdout_genes"}`` dicts (one per
fold), pickled to the path SpecFlow's holdout configs point at. Because it is a
*folded pickle*, SpecFlow's ``load_condition_splits`` auto-assigns every
condition not listed in ``test`` to ``train`` -- so the test set is what matters.

Usage (run once on the machine that has data/norman.h5ad):

  python scripts/build_holdout_split.py \
    --h5ad data/norman.h5ad \
    --output data/splits/norman_holdout.pkl
"""

import argparse
import pickle
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


def _read_conditions(h5ad_path: str, condition_key: str) -> List[str]:
    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError("building the holdout split requires anndata") from exc
    adata = ad.read_h5ad(h5ad_path, backed="r")
    if condition_key not in adata.obs:
        raise KeyError(f"AnnData.obs has no {condition_key!r} column")
    return [str(c) for c in adata.obs[condition_key].astype(str).unique()]


def _targets(condition: str, separator: str, controls: set) -> Tuple[str, ...]:
    out = []
    for token in condition.split(separator):
        token = token.strip()
        if token and token.lower() not in controls:
            out.append(token)
    return tuple(out)


def build_folds(
    conditions: Sequence[str],
    control_labels: Sequence[str],
    separator: str,
    n_holdout_genes: int,
    n_folds: int,
    seed: int,
) -> List[dict]:
    controls = {str(c).strip().lower() for c in control_labels}
    # Non-control conditions and their target genes (exact data labels).
    perturbed = [c for c in conditions if _targets(c, separator, controls)]
    gene_pool = sorted({g for c in perturbed for g in _targets(c, separator, controls)})
    if n_holdout_genes >= len(gene_pool):
        raise ValueError(
            f"n_holdout_genes={n_holdout_genes} must be < gene pool size {len(gene_pool)}"
        )

    folds = []
    for fold in range(n_folds):
        rng = np.random.default_rng(seed + fold)
        holdout = set(rng.choice(gene_pool, size=n_holdout_genes, replace=False).tolist())
        test = [
            c for c in perturbed
            if set(_targets(c, separator, controls)) & holdout
        ]
        train = [c for c in perturbed if c not in set(test)]
        if not test:
            raise RuntimeError(f"fold {fold}: empty test set")
        if not train:
            raise RuntimeError(f"fold {fold}: empty train set")
        folds.append(
            {
                "train": sorted(train),
                "test": sorted(test),
                "holdout_genes": sorted(holdout),
            }
        )
    return folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", default="data/norman.h5ad")
    parser.add_argument("--condition-key", default="condition")
    parser.add_argument("--output", default="data/splits/norman_holdout.pkl")
    parser.add_argument("--separator", default="+")
    parser.add_argument(
        "--control-labels", nargs="+", default=["ctrl", "control"]
    )
    parser.add_argument("--n-holdout-genes", type=int, default=12)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    conditions = _read_conditions(args.h5ad, args.condition_key)
    folds = build_folds(
        conditions,
        control_labels=args.control_labels,
        separator=args.separator,
        n_holdout_genes=args.n_holdout_genes,
        n_folds=args.folds,
        seed=args.seed,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(folds, handle)

    print(f"Wrote {len(folds)} folds to {out_path}")
    for i, fold in enumerate(folds):
        print(
            f"  fold {i}: {len(fold['test'])} test / {len(fold['train'])} train "
            f"conditions; holdout genes: {', '.join(fold['holdout_genes'])}"
        )


if __name__ == "__main__":
    main()
