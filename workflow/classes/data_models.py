from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class PipelinePaths:
    output_dir: Path
    standardized_tsv: Path
    valid_tsv: Path
    embeddings_npy: Path
    fingerprints_npz: Path
    pca_npy: Path
    umap_cluster_npy: Path
    umap_plot_tsv: Path
    clusters_tsv: Path
    cluster_metrics_json: Path
    cluster_representatives_tsv: Path
    cluster_scaffolds_tsv: Path
    cluster_descriptors_tsv: Path
    nearest_neighbors_embedding_tsv: Path
    nearest_neighbors_fingerprint_tsv: Path
    library_overview_json: Path
    executive_summary_txt: Path
    super_regions_tsv: Path
    cluster_interpretation_tsv: Path
    outlier_summary_tsv: Path
    similarity_agreement_summary_json: Path
    functional_group_summary_tsv: Path
    scaffold_family_summary_tsv: Path
    property_landscape_json: Path
    chemical_landscape_report_txt: Path
    plot_clusters_png: Path
    plot_mw_png: Path
    plot_logp_png: Path
    plot_tpsa_png: Path
    plot_rings_png: Path
    plot_super_regions_png: Path
    plot_outliers_png: Path
    plot_cluster_size_histogram_png: Path
    plot_functional_groups_png: Path
    plot_scaffold_families_png: Path
    plot_property_profile_png: Path
    metadata_json: Path

    def to_dict(self) -> Dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


@dataclass(frozen=True)
class PipelineRunResult:
    paths: PipelinePaths
    valid_rows: int
    invalid_rows: int
    duplicate_rows_removed: int
    cluster_count: int
    noise_count: int

    def to_json(self) -> str:
        payload = asdict(self)
        payload["paths"] = self.paths.to_dict()
        return json.dumps(payload, indent=2, sort_keys=True)


@dataclass(frozen=True)
class NeighborRow:
    source_row_id: int
    neighbor_rank: int
    neighbor_row_id: int
    score: float
    similarity_type: str
    source_smiles: str
    neighbor_smiles: str


@dataclass(frozen=True)
class ClusterRepresentative:
    cluster_id: int
    row_id: int
    label: str
    standardized_smiles: str
    distance_to_centroid: float


@dataclass(frozen=True)
class ClusterMetricSummary:
    cluster_count: int
    noise_count: int
    non_noise_count: int
    silhouette_score: float | None
    cluster_sizes: Dict[int, int]


@dataclass(frozen=True)
class StandardizationSummary:
    total_rows: int
    invalid_rows: int
    duplicate_rows_removed: int
    valid_rows: int
    parent_strategy: str
