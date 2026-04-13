from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import silhouette_score

from workflow.classes.data_models import ClusterMetricSummary
from workflow.config import PipelineConfig


LOGGER = logging.getLogger(__name__)


class HDBSCANClusterer:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def cluster(self, cluster_space: np.ndarray) -> tuple[np.ndarray, ClusterMetricSummary]:
        model = self._build_model(cluster_space.shape[0])
        LOGGER.info("Fitting HDBSCAN on array with shape %s", cluster_space.shape)
        labels = model.fit_predict(cluster_space)

        unique_clusters = sorted(int(value) for value in np.unique(labels) if int(value) != -1)
        non_noise_mask = labels != -1
        non_noise_count = int(non_noise_mask.sum())
        noise_count = int((labels == -1).sum())

        silhouette = None
        if len(unique_clusters) >= 2 and non_noise_count >= 2:
            silhouette = float(silhouette_score(cluster_space[non_noise_mask], labels[non_noise_mask]))

        cluster_sizes = {
            cluster_id: int((labels == cluster_id).sum())
            for cluster_id in unique_clusters
        }
        metrics = ClusterMetricSummary(
            cluster_count=len(unique_clusters),
            noise_count=noise_count,
            non_noise_count=non_noise_count,
            silhouette_score=silhouette,
            cluster_sizes=cluster_sizes,
        )
        LOGGER.info(
            "HDBSCAN metrics: clusters=%d non_noise=%d noise=%d silhouette=%s",
            metrics.cluster_count,
            metrics.non_noise_count,
            metrics.noise_count,
            "n/a" if metrics.silhouette_score is None else f"{metrics.silhouette_score:.4f}",
        )
        return labels.astype(int), metrics

    def _build_model(self, n_rows: int) -> Any:
        min_cluster_size = self.config.clustering.min_cluster_size
        if min_cluster_size is None:
            fraction = round(self.config.clustering.min_cluster_fraction * n_rows)
            min_cluster_size = max(self.config.clustering.min_cluster_floor, fraction)

        min_samples = self.config.clustering.min_samples
        LOGGER.info(
            "Building HDBSCAN model with min_cluster_size=%d min_samples=%s method=%s",
            min_cluster_size,
            min_samples,
            self.config.clustering.cluster_selection_method,
        )
        try:
            from sklearn.cluster import HDBSCAN

            return HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_method=self.config.clustering.cluster_selection_method,
            )
        except Exception:
            try:
                import hdbscan
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "HDBSCAN is required for clustering. Install scikit-learn with HDBSCAN support "
                    "or the standalone hdbscan package."
                ) from exc

            return hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_method=self.config.clustering.cluster_selection_method,
            )
