"""Gene Ontology-derived functional similarity graph."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Set

import numpy as np
from scipy import sparse


_ASPECTS = {
    "biological_process": "P",
    "molecular_function": "F",
    "cellular_component": "C",
    "all": None,
}


@dataclass
class GOGraphBuilder:
    """Build a sparse gene-gene graph from shared Gene Ontology terms."""

    gene_names: Iterable[str]
    annotation_file: Optional[str] = None
    k_neighbors: int = 20
    namespace: str = "biological_process"

    def __post_init__(self) -> None:
        self.gene_names = list(self.gene_names)
        if not self.gene_names:
            raise ValueError("gene_names must be non-empty")
        if len(self.gene_names) != len(set(self.gene_names)):
            raise ValueError("gene_names must be unique")
        if self.k_neighbors < 0:
            raise ValueError("k_neighbors must be non-negative")
        if self.namespace not in _ASPECTS:
            raise ValueError(f"unsupported GO namespace: {self.namespace!r}")

    def parse_annotations(self) -> Dict[str, Set[str]]:
        """Parse GAF 2.x annotations for the modeled genes."""
        if self.annotation_file is None:
            raise ValueError("annotation_file must be provided to parse GAF annotations")
        path = Path(self.annotation_file)
        if not path.exists():
            raise FileNotFoundError(str(path))

        terms = {gene: set() for gene in self.gene_names}
        requested_aspect = _ASPECTS[self.namespace]
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line or line.startswith("!"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 9:
                    continue
                symbol, qualifier, go_term, aspect = (
                    fields[2],
                    fields[3],
                    fields[4],
                    fields[8],
                )
                if symbol not in terms or "NOT" in qualifier.split("|"):
                    continue
                if requested_aspect is None or requested_aspect == aspect:
                    terms[symbol].add(go_term)
        return terms

    def _term_matrix(self, annotations: Mapping[str, Iterable[str]]) -> sparse.csr_matrix:
        term_names = sorted(
            {
                term
                for gene in self.gene_names
                for term in annotations.get(gene, ())
            }
        )
        if not term_names:
            return sparse.csr_matrix((len(self.gene_names), 0), dtype=np.float64)
        term_to_index = {term: index for index, term in enumerate(term_names)}
        rows = []
        columns = []
        for row, gene in enumerate(self.gene_names):
            for term in set(annotations.get(gene, ())):
                rows.append(row)
                columns.append(term_to_index[term])
        data = np.ones(len(rows), dtype=np.float64)
        return sparse.csr_matrix(
            (data, (rows, columns)),
            shape=(len(self.gene_names), len(term_names)),
        )

    def build_from_annotations(
        self, annotations: Mapping[str, Iterable[str]]
    ) -> sparse.csr_matrix:
        """Create a symmetric top-k Jaccard graph from a gene-to-terms map."""
        binary = self._term_matrix(annotations)
        n_genes = len(self.gene_names)
        if binary.shape[1] == 0:
            return sparse.csr_matrix((n_genes, n_genes), dtype=np.float64)

        intersections = (binary @ binary.T).tocoo()
        term_count = np.asarray(binary.sum(axis=1)).ravel()
        unions = term_count[intersections.row] + term_count[intersections.col] - intersections.data
        values = np.divide(
            intersections.data,
            unions,
            out=np.zeros_like(intersections.data, dtype=np.float64),
            where=unions > 0,
        )
        jaccard = sparse.csr_matrix(
            (values, (intersections.row, intersections.col)),
            shape=(n_genes, n_genes),
        )
        jaccard.setdiag(0.0)
        jaccard.eliminate_zeros()

        if self.k_neighbors == 0:
            return sparse.csr_matrix((n_genes, n_genes), dtype=np.float64)

        rows = []
        columns = []
        data = []
        for row in range(n_genes):
            start, end = jaccard.indptr[row], jaccard.indptr[row + 1]
            row_values = jaccard.data[start:end]
            row_columns = jaccard.indices[start:end]
            if row_values.size > self.k_neighbors:
                selected = np.argpartition(row_values, -self.k_neighbors)[-self.k_neighbors :]
                row_values = row_values[selected]
                row_columns = row_columns[selected]
            rows.extend([row] * row_values.size)
            columns.extend(row_columns.tolist())
            data.extend(row_values.tolist())
        directed = sparse.csr_matrix(
            (data, (rows, columns)), shape=(n_genes, n_genes), dtype=np.float64
        )
        graph = (directed + directed.T) * 0.5
        graph.eliminate_zeros()
        return graph.tocsr()

    def build(self) -> sparse.csr_matrix:
        return self.build_from_annotations(self.parse_annotations())
