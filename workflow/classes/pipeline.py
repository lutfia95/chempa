from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy import sparse

from workflow.classes.clustering import HDBSCANClusterer
from workflow.classes.data_models import PipelineRunResult
from workflow.classes.embeddings import EmbeddingFactory
from workflow.classes.fingerprints import MorganFingerprintComputer
from workflow.classes.io import DatasetLoader, ResultPathsFactory, ResultWriter
from workflow.classes.reduction import DimensionalityReducer
from workflow.classes.standardization import RDKitStandardizer
from workflow.classes.summaries import ClusterSummarizer
from workflow.config import PipelineConfig


class MoleculeClusteringPipeline:
    def __init__(self, config: PipelineConfig, config_path: Path | None = None) -> None:
        self.config = config
        self.config_path = config_path
        self.loader = DatasetLoader(config)
        self.paths = ResultPathsFactory(config).build()
        self.writer = ResultWriter()
        self.standardizer = RDKitStandardizer(config)
        self.fingerprints = MorganFingerprintComputer(config)
        self.embeddings = EmbeddingFactory(config)
        self.reducer = DimensionalityReducer(config)
        self.clusterer = HDBSCANClusterer(config)
        self.summarizer = ClusterSummarizer(config)

    def run(self) -> PipelineRunResult:
        raw_df = self.loader.load()
        standardized_df, standardization_summary = self.standardizer.run(raw_df)
        self.writer.write_tsv(standardized_df, self.paths.standardized_tsv)

        fingerprint_dense, fingerprint_sparse = self.fingerprints.compute(standardized_df)
        sparse.save_npz(self.paths.fingerprints_npz, fingerprint_sparse)

        embeddings = self.embeddings.build_embeddings(standardized_df, fingerprint_dense)
        np.save(self.paths.embeddings_npy, embeddings)

        pca_space, cluster_space, plot_space = self.reducer.reduce(embeddings)
        np.save(self.paths.pca_npy, pca_space)
        np.save(self.paths.umap_cluster_npy, cluster_space)

        labels, metrics = self.clusterer.cluster(cluster_space)

        valid_df = standardized_df.copy()
        valid_df["cluster_id"] = labels
        valid_df["umap_x"] = plot_space[:, 0]
        valid_df["umap_y"] = plot_space[:, 1]
        self.writer.write_tsv(valid_df, self.paths.valid_tsv)
        self.writer.write_tsv(
            valid_df[["row_id", "standardized_smiles", "cluster_id", "umap_x", "umap_y"]],
            self.paths.umap_plot_tsv,
        )
        self.writer.write_tsv(valid_df, self.paths.clusters_tsv)

        representatives = self.summarizer.representatives(valid_df, cluster_space, labels)
        scaffolds = self.summarizer.scaffold_table(valid_df, labels)
        descriptor_summary = self.summarizer.descriptor_summary(valid_df, labels)

        embedding_neighbors = self.embeddings.cosine_top_k(embeddings, self.config.summary.top_neighbors)
        embedding_neighbors = self.summarizer.decorate_neighbor_rows(
            valid_df,
            embedding_neighbors,
            score_column="similarity",
            similarity_type="cosine_embedding",
        )

        fingerprint_neighbors = self.fingerprints.tanimoto_top_k(
            fingerprint_dense.astype(np.int16),
            self.config.summary.top_neighbors,
        )
        fingerprint_neighbors = self.summarizer.decorate_neighbor_rows(
            valid_df,
            fingerprint_neighbors,
            score_column="similarity",
            similarity_type="tanimoto_fingerprint",
        )

        overview = self.summarizer.build_library_overview(
            valid_df,
            labels,
            cluster_space,
            embedding_neighbors,
            fingerprint_neighbors,
        )
        chemical_landscape = self.summarizer.build_chemical_landscape(valid_df, labels)

        self.writer.write_tsv(representatives, self.paths.cluster_representatives_tsv)
        self.writer.write_tsv(scaffolds, self.paths.cluster_scaffolds_tsv)
        self.writer.write_tsv(descriptor_summary, self.paths.cluster_descriptors_tsv)
        self.writer.write_tsv(embedding_neighbors, self.paths.nearest_neighbors_embedding_tsv)
        self.writer.write_tsv(fingerprint_neighbors, self.paths.nearest_neighbors_fingerprint_tsv)
        self.writer.write_json(overview["library_overview"], self.paths.library_overview_json)
        self.writer.write_text(overview["executive_summary"], self.paths.executive_summary_txt)
        self.writer.write_tsv(overview["super_regions"], self.paths.super_regions_tsv)
        self.writer.write_tsv(overview["cluster_interpretation"], self.paths.cluster_interpretation_tsv)
        self.writer.write_tsv(overview["outlier_summary"], self.paths.outlier_summary_tsv)
        self.writer.write_json(overview["similarity_agreement"], self.paths.similarity_agreement_summary_json)
        self.writer.write_tsv(chemical_landscape["functional_group_summary"], self.paths.functional_group_summary_tsv)
        self.writer.write_tsv(chemical_landscape["scaffold_family_summary"], self.paths.scaffold_family_summary_tsv)
        self.writer.write_json(chemical_landscape["property_landscape"], self.paths.property_landscape_json)
        self.writer.write_text(chemical_landscape["chemical_landscape_report"], self.paths.chemical_landscape_report_txt)
        self.writer.write_json(self.summarizer.metrics_payload(metrics), self.paths.cluster_metrics_json)
        self.summarizer.make_plots(valid_df, labels, plot_space, self.paths)

        metadata = {
            "config": self.config.to_dict(),
            "config_path": str(self.config_path) if self.config_path else None,
            "standardization_summary": asdict(standardization_summary),
            "cluster_metrics": self.summarizer.metrics_payload(metrics),
            "library_overview": overview["library_overview"],
            "similarity_agreement": overview["similarity_agreement"],
            "property_landscape": chemical_landscape["property_landscape"],
            "paths": self.paths.to_dict(),
        }
        self.writer.write_json(metadata, self.paths.metadata_json)

        return PipelineRunResult(
            paths=self.paths,
            valid_rows=int(len(valid_df)),
            invalid_rows=int(standardization_summary.invalid_rows),
            duplicate_rows_removed=int(standardization_summary.duplicate_rows_removed),
            cluster_count=int(metrics.cluster_count),
            noise_count=int(metrics.noise_count),
        )
