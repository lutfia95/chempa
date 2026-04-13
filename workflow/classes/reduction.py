from __future__ import annotations

import logging
import os
from typing import Any, Tuple

import numpy as np
from sklearn.decomposition import PCA

from workflow.config import PipelineConfig


LOGGER = logging.getLogger(__name__)


class DimensionalityReducer:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def reduce(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(embeddings) == 0:
            raise ValueError("No embeddings available for reduction.")

        pca_dims = min(
            self.config.reduction.pca_components,
            embeddings.shape[0],
            embeddings.shape[1],
        )
        pca = PCA(n_components=pca_dims, random_state=self.config.reduction.random_state)
        LOGGER.info("Fitting PCA with %d components", pca_dims)
        pca_space = pca.fit_transform(embeddings)

        cluster_dims = min(self.config.reduction.umap_cluster_components, max(2, pca_space.shape[0] - 1), pca_space.shape[1])
        plot_dims = min(self.config.reduction.umap_plot_components, max(2, pca_space.shape[0] - 1), pca_space.shape[1])

        # The clustering UMAP keeps more dimensions than the plotting UMAP to preserve local density structure.
        cluster_space = self._run_umap(pca_space, cluster_dims)
        plot_space = self._run_umap(pca_space, plot_dims)
        return pca_space.astype(np.float32), cluster_space.astype(np.float32), plot_space.astype(np.float32)

    def _run_umap(self, data: np.ndarray, n_components: int) -> np.ndarray:
        umap = self._import_umap()
        LOGGER.info(
            "Running UMAP with n_components=%d, n_neighbors=%d, min_dist=%.3f, metric=%s",
            n_components,
            self.config.reduction.umap_n_neighbors,
            self.config.reduction.umap_min_dist,
            self.config.reduction.umap_metric,
        )
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=self.config.reduction.umap_n_neighbors,
            min_dist=self.config.reduction.umap_min_dist,
            metric=self.config.reduction.umap_metric,
            random_state=self.config.reduction.random_state,
        )
        return reducer.fit_transform(data)

    def _import_umap(self) -> Any:
        os.environ.setdefault("NUMBA_CACHE_DIR", self.config.runtime.numba_cache_dir)
        try:
            import umap
        except Exception as exc:
            raise RuntimeError(
                "UMAP is required for dimensionality reduction. "
                "Install a working umap-learn environment or fix the numba cache configuration."
            ) from exc

        return umap
