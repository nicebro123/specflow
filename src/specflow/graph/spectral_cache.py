"""Cached perturbation-conditioned spectral embeddings for multiple graphs."""

from hashlib import sha1
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
from scipy import sparse
import torch

from specflow.graph.perturbation_aware import PerturbationAwareGraphModifier
from specflow.graph.spectral_embedding import SpectralEmbedding, SpectralResult


class SpectralCache:
    """Compute and cache GO/coexpression spectra keyed by perturbation mask."""

    def __init__(
        self,
        base_graphs: Mapping[str, sparse.spmatrix],
        n_components: Mapping[str, int],
        modifier: Optional[PerturbationAwareGraphModifier] = None,
        cache_dir: Optional[str] = None,
        use_approximation: bool = False,
        static: bool = False,
    ) -> None:
        self.static = static
        required = {"go", "coexp"}
        if set(base_graphs) != required or set(n_components) != required:
            raise ValueError("base_graphs and n_components must contain 'go' and 'coexp'")
        self.graphs = {
            name: sparse.csr_matrix(graph, dtype=np.float64)
            for name, graph in base_graphs.items()
        }
        shape = self.graphs["go"].shape
        if shape != self.graphs["coexp"].shape or shape[0] != shape[1]:
            raise ValueError("GO and coexpression graphs must have the same square shape")
        self.n_genes = shape[0]
        self.embedders = {
            name: SpectralEmbedding(n_components=n_components[name])
            for name in required
        }
        self.modifier = modifier or PerturbationAwareGraphModifier()
        self.use_approximation = use_approximation
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._graph_digests = {
            name: self._graph_digest(graph) for name, graph in self.graphs.items()
        }
        self._memory: Dict[str, SpectralResult] = {}
        self._base_results = {
            name: embedder.fit(self.graphs[name])
            for name, embedder in self.embedders.items()
        }

    def base_spectrum(self, graph_type: str = "coexp"):
        """Return (eigenvectors, eigenvalues) of the unperturbed base graph."""
        result = self._base_results[graph_type]
        return result.eigenvectors, result.eigenvalues

    @staticmethod
    def _graph_digest(graph: sparse.csr_matrix) -> str:
        digest = sha1()
        digest.update(str(graph.shape).encode("ascii"))
        digest.update(graph.data.tobytes())
        digest.update(graph.indices.tobytes())
        digest.update(graph.indptr.tobytes())
        return digest.hexdigest()[:12]

    def _mask_array(self, pert_mask) -> np.ndarray:
        if torch.is_tensor(pert_mask):
            pert_mask = pert_mask.detach().cpu().numpy()
        mask = (np.asarray(pert_mask).reshape(-1) > 0).astype(np.uint8)
        if mask.shape != (self.n_genes,):
            raise ValueError(f"pert_mask must have shape ({self.n_genes},)")
        return mask

    def _key(self, graph_type: str, mask: np.ndarray) -> str:
        mask_digest = sha1(np.packbits(mask).tobytes()).hexdigest()[:12]
        method = "approx" if self.use_approximation else "exact"
        components = self.embedders[graph_type].n_components
        alpha = self.modifier.alpha[graph_type]
        return (
            f"{graph_type}-{self._graph_digests[graph_type]}-k{components}-"
            f"a{alpha:.8g}-{mask_digest}-{method}"
        )

    def _disk_path(self, key: str) -> Optional[Path]:
        return self.cache_dir / f"{key}.npz" if self.cache_dir else None

    def get(self, pert_mask, graph_type: str) -> SpectralResult:
        if self.static:
            return self._base_results[graph_type]
        mask = self._mask_array(pert_mask)
        key = self._key(graph_type, mask)
        if key in self._memory:
            return self._memory[key]
        disk_path = self._disk_path(key)
        if disk_path is not None and disk_path.exists():
            loaded = np.load(disk_path)
            result = SpectralResult(loaded["eigenvalues"], loaded["eigenvectors"])
            self._memory[key] = result
            return result

        if not mask.any():
            result = self._base_results[graph_type]
        else:
            modified = self.modifier.modify(self.graphs[graph_type], mask, graph_type)
            if self.use_approximation:
                result = self.embedders[graph_type].fit_with_perturbation_update(
                    self.graphs[graph_type], modified, self._base_results[graph_type]
                )
            else:
                result = self.embedders[graph_type].fit(modified)
        self._memory[key] = result
        if disk_path is not None:
            np.savez_compressed(
                disk_path,
                eigenvalues=result.eigenvalues,
                eigenvectors=result.eigenvectors,
            )
        return result

    def precompute_all(self, perturbation_masks: Iterable[np.ndarray]) -> None:
        if self.static:
            return  # base spectra already computed in __init__
        for pert_mask in perturbation_masks:
            for graph_type in ("go", "coexp"):
                self.get(pert_mask, graph_type)

    def batch_embeddings(self, pert_masks):
        """Return a model-ready dual-graph embedding dictionary for a batch."""
        is_tensor = torch.is_tensor(pert_masks)
        device = pert_masks.device if is_tensor else None
        dtype = pert_masks.dtype if is_tensor and pert_masks.is_floating_point() else torch.float32
        masks = pert_masks.detach().cpu().numpy() if is_tensor else np.asarray(pert_masks)
        if masks.ndim == 1:
            masks = masks[None, :]
        if masks.ndim != 2 or masks.shape[1] != self.n_genes:
            raise ValueError("pert_masks must have shape (G,) or (B, G)")
        batch_size = masks.shape[0]
        embeddings = {}
        for graph_type in ("go", "coexp"):
            if self.static:
                base = torch.as_tensor(
                    self._base_results[graph_type].eigenvectors,
                    dtype=dtype,
                    device=device,
                )
                embeddings[graph_type] = base.unsqueeze(0).expand(batch_size, -1, -1)
            else:
                array = np.stack(
                    [self.get(mask, graph_type).eigenvectors for mask in masks],
                    axis=0,
                )
                embeddings[graph_type] = torch.as_tensor(
                    array, dtype=dtype, device=device
                )
        return embeddings
