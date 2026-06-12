"""Configuration objects for SpecFlow training and dual-graph modeling."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

import yaml


def set_by_path(payload: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``payload["a"]["b"] = value`` from a ``"a.b"`` dotted key.

    Intermediate mappings are created on demand. Raises ``ValueError`` if the
    key is malformed or traverses a non-mapping node.
    """
    if not dotted_key or dotted_key.startswith(".") or dotted_key.endswith("."):
        raise ValueError(f"invalid override key: {dotted_key!r}")
    current: MutableMapping[str, Any] = payload
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        child = current[part]
        if not isinstance(child, MutableMapping):
            raise ValueError(f"cannot set {dotted_key!r}: {part!r} is not a mapping")
        current = child
    current[parts[-1]] = value


def apply_overrides(
    config: Mapping[str, Any], overrides: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Return a deep copy of ``config`` with dotted-key ``overrides`` applied."""
    from copy import deepcopy

    merged = deepcopy(dict(config))
    for key, value in (overrides or {}).items():
        set_by_path(merged, str(key), value)
    return merged


def parse_override_value(raw: str) -> Any:
    """Parse a ``KEY=VALUE`` CLI value using YAML scalar rules.

    So ``0.3`` becomes a float, ``true``/``false`` booleans, ``null`` None,
    ``8`` an int, and anything else a string.
    """
    return yaml.safe_load(raw)


@dataclass
class DataConfig:
    dataset: str = "norman"
    h5ad_path: str = "data/norman.h5ad"
    condition_key: str = "condition"
    gene_key: Optional[str] = None
    control_labels: List[str] = field(
        default_factory=lambda: ["ctrl", "control", "non-targeting"]
    )
    separator: str = "+"
    target_map_path: Optional[str] = None
    split_path: Optional[str] = None
    split_fold: int = 0
    preprocess: bool = False
    n_top_genes: int = 5000
    preprocess_cache: bool = True
    preprocess_cache_dir: Optional[str] = None
    setting: str = "additive"
    seed: int = 42
    test_fraction: float = 0.2
    val_fraction: float = 0.1
    samples_per_condition: Optional[int] = None
    # When an external split has no validation set (e.g. scDFM folds), carve
    # this fraction of train conditions into a monitoring val set. 0 disables
    # this and keeps the split byte-for-byte aligned with the source.
    val_from_train_fraction: float = 0.0


@dataclass
class CoexpressionConfig:
    k_neighbors: int = 20
    threshold: float = 0.3


@dataclass
class GOConfig:
    annotation_file: str = "data/gene_ontology/go_annotations.gaf"
    namespace: str = "biological_process"
    k_neighbors: int = 20


@dataclass
class PerturbationGraphConfig:
    alpha_go: float = 0.1
    alpha_coexp: float = 0.05
    edge_dropout: float = 0.0
    edge_dropout_seed: int = 42


@dataclass
class SpectralConfig:
    n_components: int = 32
    go_components: int = 32
    coexp_components: int = 32
    macro_ratio: float = 0.5
    use_perturbation_approx: bool = False
    cache_dir: str = "cache/spectral"
    # When True (S3), spectral embeddings are a static positional encoding from
    # the base graph; the perturbation-aware edge attenuation is NOT applied and
    # the eigendecomposition is computed once. Perturbation specificity is
    # carried entirely by the injected perturbation embedding (e_p). When False,
    # the legacy per-perturbation dynamic spectrum is used (ablation only).
    static: bool = True


@dataclass
class ModelConfig:
    d_model: int = 128
    hidden_dim: int = 256
    n_velocity_layers: int = 3
    spectral_dim: int = 64
    graph_dim: int = 32
    pert_dim: int = 32
    dual_graph: bool = True
    graph_mode: str = "dual"
    fusion_mode: str = "adaptive"
    scale_mode: str = "multi"
    use_spectral_embedding: bool = True
    # Innovation 1: propagate the perturbation indicator over the fixed graph via
    # learnable spectral filters (per-gene "perturbation influence"), fed to the
    # velocity field. Realizes signal propagation without per-perturbation
    # eigendecomposition.
    spectral_propagation: bool = False
    propagation_channels: int = 8


@dataclass
class FlowConfig:
    sigma: float = 0.5
    mmd_weight: float = 0.1
    mmd_interval: int = 10
    mmd_steps: int = 8
    # Match the mean perturbation residual direction within each condition.
    # This directly targets pearson_delta-style evaluation.
    delta_corr_weight: float = 0.0
    # Innovation 3: couple control and perturbed cells within each condition via
    # optimal transport (Hungarian) before flow matching, instead of random
    # pairing — straighter, more biologically meaningful trajectories.
    ot_coupling: bool = False
    # Core control-anchor ablation switch. When disabled, flow matching and
    # sampling start from an unconditioned zero anchor instead of measured
    # control expression, and the model does not receive control expression.
    control_anchor: bool = True


@dataclass
class TrainingConfig:
    batch_size: int = 32
    max_epochs: int = 100
    max_steps: Optional[int] = None
    eval_every_steps: int = 5000
    # In step mode, a full checkpoint is written to ``checkpoints/step_N.pt`` at
    # every evaluation. Keep only the most recent N to bound disk use across
    # long sweeps. 0 keeps none (best.pt is always kept); a negative value keeps
    # all interval checkpoints.
    keep_interval_checkpoints: int = 3
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    group_by_condition: bool = True
    patience: int = 20
    ema_decay: float = 0.999
    use_ema: bool = True
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    warmup_steps: Optional[int] = None
    eta_min: float = 1e-6
    use_amp: bool = True
    show_progress: bool = True
    monitor_pearson_delta: bool = True


@dataclass
class InferenceConfig:
    n_samples: int = 10
    n_control_cells: int = 64
    ode_steps: int = 50
    de_top_k: int = 20


@dataclass
class OutputConfig:
    output_dir: str = "outputs/default"
    checkpoint_name: str = "best.pt"


@dataclass
class SpecFlowConfig:
    data: DataConfig = field(default_factory=DataConfig)
    coexpression: CoexpressionConfig = field(default_factory=CoexpressionConfig)
    go: GOConfig = field(default_factory=GOConfig)
    perturbation_graph: PerturbationGraphConfig = field(
        default_factory=PerturbationGraphConfig
    )
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "SpecFlowConfig":
        graph = config.get("graph", {})
        coexpression = graph.get("coexp", config.get("coexpression", {}))
        return cls(
            data=DataConfig(**config.get("data", {})),
            coexpression=CoexpressionConfig(**coexpression),
            go=GOConfig(**graph.get("go", {})),
            perturbation_graph=PerturbationGraphConfig(
                **graph.get("perturbation", {})
            ),
            spectral=SpectralConfig(**config.get("spectral", {})),
            model=ModelConfig(**config.get("model", {})),
            flow=FlowConfig(**config.get("flow", {})),
            training=TrainingConfig(**config.get("training", {})),
            inference=InferenceConfig(**config.get("inference", {})),
            output=OutputConfig(**config.get("output", {})),
        )

    @classmethod
    def from_yaml(
        cls, path: str, overrides: Optional[Mapping[str, Any]] = None
    ) -> "SpecFlowConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if overrides:
            raw = apply_overrides(raw, overrides)
        return cls.from_dict(raw)
