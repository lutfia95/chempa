from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from workflow.classes.data_models import PipelinePaths
from workflow.config import PipelineConfig


class DatasetLoader:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def load(self) -> pd.DataFrame:
        input_path = Path(self.config.input.path)
        df = pd.read_csv(input_path, sep="\t", dtype=str, keep_default_na=False)
        smiles_column = self.config.input.smiles_column
        if smiles_column not in df.columns:
            raise ValueError(f"Missing SMILES column '{smiles_column}' in {input_path}")

        df = df.copy()
        df.insert(0, "row_id", range(1, len(df) + 1))
        return df


class ResultPathsFactory:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def build(self) -> PipelinePaths:
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        return PipelinePaths(
            output_dir=output_dir,
            standardized_tsv=output_dir / "standardized.tsv",
            valid_tsv=output_dir / "valid_molecules.tsv",
            embeddings_npy=output_dir / "embeddings.npy",
            fingerprints_npz=output_dir / "morgan_fingerprints.npz",
            pca_npy=output_dir / "pca_50.npy",
            umap_cluster_npy=output_dir / "umap_cluster.npy",
            umap_plot_tsv=output_dir / "umap_2d.tsv",
            clusters_tsv=output_dir / "clusters.tsv",
            cluster_metrics_json=output_dir / "cluster_metrics.json",
            cluster_representatives_tsv=output_dir / "cluster_representatives.tsv",
            cluster_scaffolds_tsv=output_dir / "cluster_scaffolds.tsv",
            cluster_descriptors_tsv=output_dir / "cluster_descriptors.tsv",
            nearest_neighbors_embedding_tsv=output_dir / "nearest_neighbors_embedding.tsv",
            nearest_neighbors_fingerprint_tsv=output_dir / "nearest_neighbors_fingerprint.tsv",
            plot_clusters_png=output_dir / "umap_clusters.png",
            plot_mw_png=output_dir / "umap_mw.png",
            plot_logp_png=output_dir / "umap_logp.png",
            metadata_json=output_dir / "run_metadata.json",
        )


class ResultWriter:
    @staticmethod
    def write_tsv(df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, sep="\t", index=False)

    @staticmethod
    def write_json(payload: Dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

