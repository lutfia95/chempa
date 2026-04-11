from __future__ import annotations

import dataclasses
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class InputConfig:
    path: str
    smiles_column: str = "InputSMILES"
    id_columns: List[str] = field(default_factory=lambda: ["Label", "CID"])
    text_columns: List[str] = field(default_factory=lambda: ["IUPACName"])


@dataclass(frozen=True)
class OutputConfig:
    run_name: str = "primary_amins"
    root_dir: str = "workflow/results"


@dataclass(frozen=True)
class StandardizationConfig:
    parent_strategy: str = "fragment"
    deduplicate: bool = True
    drop_invalid: bool = True


@dataclass(frozen=True)
class FingerprintConfig:
    radius: int = 2
    n_bits: int = 2048


@dataclass(frozen=True)
class EmbeddingConfig:
    backend: str = "molformer"
    model_name: str = "ibm/MoLFormer-XL-both-10pct"
    batch_size: int = 16
    max_length: int = 256
    pooling: str = "mean"
    normalize: bool = True
    device: str = "auto"
    local_files_only: bool = False


@dataclass(frozen=True)
class ReductionConfig:
    pca_components: int = 50
    umap_cluster_components: int = 15
    umap_plot_components: int = 2
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "cosine"
    random_state: int = 42


@dataclass(frozen=True)
class ClusteringConfig:
    algorithm: str = "hdbscan"
    min_cluster_size: Optional[int] = None
    min_cluster_fraction: float = 0.01
    min_cluster_floor: int = 10
    min_samples: Optional[int] = None
    cluster_selection_method: str = "eom"


@dataclass(frozen=True)
class SummaryConfig:
    top_scaffolds_per_cluster: int = 5
    top_neighbors: int = 5
    descriptor_columns: List[str] = field(
        default_factory=lambda: ["MW", "logP", "HBD", "HBA", "TPSA", "RB", "Rings"]
    )


@dataclass(frozen=True)
class RuntimeConfig:
    random_state: int = 42
    numba_cache_dir: str = "/tmp/numba_cache"
    matplotlib_config_dir: str = "/tmp/mplconfig"


@dataclass(frozen=True)
class PipelineConfig:
    input: InputConfig
    output: OutputConfig
    standardization: StandardizationConfig = field(default_factory=StandardizationConfig)
    fingerprints: FingerprintConfig = field(default_factory=FingerprintConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reduction: ReductionConfig = field(default_factory=ReductionConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    summary: SummaryConfig = field(default_factory=SummaryConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @property
    def output_dir(self) -> Path:
        return Path(self.output.root_dir) / self.output.run_name

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def _construct(section_type: Any, raw: Optional[Dict[str, Any]]) -> Any:
    return section_type(**(raw or {}))


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    return PipelineConfig(
        input=_construct(InputConfig, raw.get("input")),
        output=_construct(OutputConfig, raw.get("output")),
        standardization=_construct(StandardizationConfig, raw.get("standardization")),
        fingerprints=_construct(FingerprintConfig, raw.get("fingerprints")),
        embedding=_construct(EmbeddingConfig, raw.get("embedding")),
        reduction=_construct(ReductionConfig, raw.get("reduction")),
        clustering=_construct(ClusteringConfig, raw.get("clustering")),
        summary=_construct(SummaryConfig, raw.get("summary")),
        runtime=_construct(RuntimeConfig, raw.get("runtime")),
    )

