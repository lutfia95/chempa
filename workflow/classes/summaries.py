from __future__ import annotations

import os
from collections import Counter
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from workflow.classes.data_models import ClusterRepresentative, ClusterMetricSummary
from workflow.config import PipelineConfig


class ClusterSummarizer:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def representatives(self, df: pd.DataFrame, cluster_space: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        id_columns = self._present_columns(df, self.config.input.id_columns)

        for cluster_id in sorted(int(value) for value in np.unique(labels) if int(value) != -1):
            mask = labels == cluster_id
            subset = df.loc[mask].reset_index(drop=True)
            subset_space = cluster_space[mask]
            centroid = subset_space.mean(axis=0, keepdims=True)
            distances = np.linalg.norm(subset_space - centroid, axis=1)
            winner = int(np.argmin(distances))
            row = subset.iloc[winner]
            record = ClusterRepresentative(
                cluster_id=cluster_id,
                row_id=int(row["row_id"]),
                label=str(row[id_columns[0]]) if id_columns else "",
                standardized_smiles=str(row["standardized_smiles"]),
                distance_to_centroid=float(distances[winner]),
            )
            output = asdict(record)
            for column in id_columns[1:]:
                output[column] = str(row[column])
            rows.append(output)

        return pd.DataFrame(rows)

    def scaffold_table(self, df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        top_n = self.config.summary.top_scaffolds_per_cluster
        for cluster_id in sorted(int(value) for value in np.unique(labels) if int(value) != -1):
            subset = df.loc[labels == cluster_id]
            counts = Counter(str(value) for value in subset["murcko_scaffold"].tolist() if str(value))
            for scaffold, count in counts.most_common(top_n):
                rows.append(
                    {
                        "cluster_id": cluster_id,
                        "murcko_scaffold": scaffold,
                        "count": int(count),
                    }
                )
        return pd.DataFrame(rows)

    def descriptor_summary(self, df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        descriptor_columns = self._present_columns(df, self.config.summary.descriptor_columns)
        rows: List[Dict[str, Any]] = []
        for cluster_id in sorted(int(value) for value in np.unique(labels) if int(value) != -1):
            subset = df.loc[labels == cluster_id].copy()
            subset[descriptor_columns] = subset[descriptor_columns].apply(pd.to_numeric, errors="coerce")
            row: Dict[str, Any] = {
                "cluster_id": cluster_id,
                "cluster_size": int(len(subset)),
            }
            for column in descriptor_columns:
                row[f"{column}_mean"] = float(subset[column].mean())
                row[f"{column}_min"] = float(subset[column].min())
                row[f"{column}_max"] = float(subset[column].max())
            rows.append(row)
        return pd.DataFrame(rows)

    def decorate_neighbor_rows(
        self,
        df: pd.DataFrame,
        raw_neighbors: pd.DataFrame,
        score_column: str,
        similarity_type: str,
    ) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        if raw_neighbors.empty:
            return pd.DataFrame(rows)

        for item in raw_neighbors.to_dict(orient="records"):
            source_row = df.iloc[int(item["source_index"])]
            neighbor_row = df.iloc[int(item["neighbor_index"])]
            rows.append(
                {
                    "source_row_id": int(source_row["row_id"]),
                    "neighbor_rank": int(item["neighbor_rank"]),
                    "neighbor_row_id": int(neighbor_row["row_id"]),
                    "similarity_type": similarity_type,
                    "score": float(item[score_column]),
                    "source_smiles": str(source_row["standardized_smiles"]),
                    "neighbor_smiles": str(neighbor_row["standardized_smiles"]),
                }
            )
        return pd.DataFrame(rows)

    def metrics_payload(self, metrics: ClusterMetricSummary) -> Dict[str, Any]:
        return asdict(metrics)

    def make_plots(self, df: pd.DataFrame, labels: np.ndarray, plot_space: np.ndarray, output_paths: Any) -> None:
        plt = self._import_matplotlib()
        plot_df = df.copy()
        plot_df["cluster_id"] = labels
        plot_df["umap_x"] = plot_space[:, 0]
        plot_df["umap_y"] = plot_space[:, 1]
        plot_df["MW"] = pd.to_numeric(plot_df["MW"], errors="coerce")
        plot_df["logP"] = pd.to_numeric(plot_df["logP"], errors="coerce")

        self._scatter(plt, plot_df, "cluster_id", output_paths.plot_clusters_png, categorical=True)
        self._scatter(plt, plot_df, "MW", output_paths.plot_mw_png, categorical=False)
        self._scatter(plt, plot_df, "logP", output_paths.plot_logp_png, categorical=False)

    def _scatter(self, plt: Any, df: pd.DataFrame, color_column: str, path: Any, categorical: bool) -> None:
        fig, ax = plt.subplots(figsize=(8, 6))
        if categorical:
            scatter = ax.scatter(
                df["umap_x"],
                df["umap_y"],
                c=df[color_column],
                cmap="tab20",
                s=18,
                alpha=0.85,
            )
        else:
            scatter = ax.scatter(
                df["umap_x"],
                df["umap_y"],
                c=df[color_column],
                cmap="viridis",
                s=18,
                alpha=0.85,
            )
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_title(f"UMAP colored by {color_column}")
        fig.colorbar(scatter, ax=ax)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)

    def _present_columns(self, df: pd.DataFrame, columns: List[str]) -> List[str]:
        return [column for column in columns if column in df.columns]

    def _import_matplotlib(self) -> Any:
        os.environ.setdefault("MPLCONFIGDIR", self.config.runtime.matplotlib_config_dir)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            raise RuntimeError("matplotlib is required for plotting the UMAP outputs.") from exc

        return plt

