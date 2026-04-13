from __future__ import annotations

import os
from collections import Counter
from dataclasses import asdict
from typing import Any, Dict, List

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

    def build_library_overview(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        cluster_space: np.ndarray,
        embedding_neighbors: pd.DataFrame,
        fingerprint_neighbors: pd.DataFrame,
    ) -> Dict[str, Any]:
        cluster_ids = sorted(int(value) for value in np.unique(labels) if int(value) != -1)
        cluster_summary = self._cluster_summary_rows(df, labels)
        agreement_payload, agreement_map = self._similarity_agreement_payload(
            embedding_neighbors,
            fingerprint_neighbors,
        )
        super_regions = self._super_region_table(df, labels, cluster_space, cluster_summary)
        interpretations = self._cluster_interpretation_table(cluster_summary)
        outliers = self._outlier_table(df, labels, cluster_space, agreement_map)

        non_noise_sizes = np.array(
            [int((labels == cluster_id).sum()) for cluster_id in cluster_ids],
            dtype=float,
        )
        n_rows = int(len(df))
        noise_fraction = float((labels == -1).sum() / n_rows) if n_rows else 0.0
        largest_cluster_fraction = float(non_noise_sizes.max() / n_rows) if len(non_noise_sizes) else 0.0
        effective_cluster_count = self._effective_cluster_count(non_noise_sizes)
        redundancy_estimate = self._redundancy_estimate(embedding_neighbors, fingerprint_neighbors)
        diversity_level = self._diversity_level(effective_cluster_count, redundancy_estimate, noise_fraction)
        map_structure = self._map_structure(largest_cluster_fraction, noise_fraction, effective_cluster_count)
        dominant_cluster_ids = [int(row["cluster_id"]) for row in cluster_summary[:3]]
        outlier_cluster_count = int(outliers["cluster_id"].nunique()) if not outliers.empty else 0

        overview = {
            "library_overview": {
                "n_molecules": n_rows,
                "n_clusters": len(cluster_ids),
                "noise_fraction": round(noise_fraction, 4),
                "effective_cluster_count": round(effective_cluster_count, 2),
                "largest_cluster_fraction": round(largest_cluster_fraction, 4),
                "redundancy_estimate": round(redundancy_estimate, 4),
                "diversity_level": diversity_level,
                "map_structure": map_structure,
                "n_super_regions": int(super_regions["super_region_id"].nunique()) if not super_regions.empty else 0,
                "dominant_cluster_ids": dominant_cluster_ids,
                "outlier_cluster_count": outlier_cluster_count,
            },
            "super_regions": super_regions,
            "cluster_interpretation": interpretations,
            "outlier_summary": outliers,
            "similarity_agreement": agreement_payload,
            "executive_summary": self._executive_summary_text(
                n_rows=n_rows,
                cluster_count=len(cluster_ids),
                diversity_level=diversity_level,
                map_structure=map_structure,
                noise_fraction=noise_fraction,
                largest_cluster_fraction=largest_cluster_fraction,
                redundancy_estimate=redundancy_estimate,
                dominant_cluster_ids=dominant_cluster_ids,
                super_regions=super_regions,
                agreement_payload=agreement_payload,
            ),
        }
        return overview

    def build_chemical_landscape(self, df: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
        functional_groups = self._functional_group_summary(df)
        scaffold_families = self._scaffold_family_summary(df)
        property_landscape = self._property_landscape(df, labels)
        report_text = self._chemical_landscape_report_text(
            df=df,
            labels=labels,
            functional_groups=functional_groups,
            scaffold_families=scaffold_families,
            property_landscape=property_landscape,
        )
        return {
            "functional_group_summary": functional_groups,
            "scaffold_family_summary": scaffold_families,
            "property_landscape": property_landscape,
            "chemical_landscape_report": report_text,
        }

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

    def _functional_group_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        patterns = self._functional_group_patterns()
        counts = {name: 0 for name, _ in patterns}
        use_rdkit = all(pattern is not None for _, pattern in patterns)
        chem = self._try_import_rdkit_chem() if use_rdkit else None
        for smiles in df.get("standardized_smiles", pd.Series(dtype=str)).astype(str).tolist():
            if chem is not None:
                mol = chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                for name, pattern in patterns:
                    if mol.HasSubstructMatch(pattern):
                        counts[name] += 1
            else:
                for name in counts:
                    if self._functional_group_fallback_match(name, smiles):
                        counts[name] += 1

        total = max(len(df), 1)
        rows = [
            {
                "functional_group": name,
                "molecule_count": int(count),
                "fraction_of_library": round(float(count / total), 4),
            }
            for name, count in counts.items()
        ]
        return pd.DataFrame(rows).sort_values(["molecule_count", "functional_group"], ascending=[False, True])

    def _scaffold_family_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        total = max(len(df), 1)
        scaffold_series = df.get("murcko_scaffold", pd.Series(dtype=str)).astype(str)
        scaffold_counter = Counter(scaffold for scaffold in scaffold_series.tolist() if scaffold)
        for scaffold, count in scaffold_counter.most_common(15):
            rows.append(
                {
                    "scaffold_family": scaffold,
                    "molecule_count": int(count),
                    "fraction_of_library": round(float(count / total), 4),
                    "family_type": self._scaffold_family_type(scaffold),
                }
            )

        no_scaffold_count = int(sum(1 for scaffold in scaffold_series.tolist() if not scaffold))
        rows.append(
            {
                "scaffold_family": "[no_murcko_scaffold]",
                "molecule_count": no_scaffold_count,
                "fraction_of_library": round(float(no_scaffold_count / total), 4),
                "family_type": "acyclic_or_fragment_like",
            }
        )
        return pd.DataFrame(rows)

    def _property_landscape(self, df: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
        total = max(len(df), 1)
        mw = pd.to_numeric(df.get("MW", pd.Series(dtype=float)), errors="coerce")
        logp = pd.to_numeric(df.get("logP", pd.Series(dtype=float)), errors="coerce")
        tpsa = pd.to_numeric(df.get("TPSA", pd.Series(dtype=float)), errors="coerce")
        rb = pd.to_numeric(df.get("RB", pd.Series(dtype=float)), errors="coerce")
        rings = pd.to_numeric(df.get("Rings", pd.Series(dtype=float)), errors="coerce")

        return {
            "mw_profile": self._distribution_buckets(mw, [(250, "small"), (450, "mid_size"), (float("inf"), "large")]),
            "logp_profile": self._distribution_buckets(logp, [(1, "low"), (3, "balanced"), (float("inf"), "lipophilic")]),
            "tpsa_profile": self._distribution_buckets(tpsa, [(60, "low_polarity"), (90, "moderate_polarity"), (float("inf"), "high_polarity")]),
            "flexibility_profile": self._distribution_buckets(rb, [(2, "rigid"), (5, "mixed"), (float("inf"), "flexible")]),
            "ring_profile": self._distribution_buckets(rings, [(1, "acyclic"), (3, "single_or_bicyclic"), (float("inf"), "polycyclic")]),
            "noise_fraction": round(float((labels == -1).sum() / total), 4),
        }

    def _chemical_landscape_report_text(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        functional_groups: pd.DataFrame,
        scaffold_families: pd.DataFrame,
        property_landscape: Dict[str, Any],
    ) -> str:
        total = len(df)
        cluster_count = len([value for value in np.unique(labels) if int(value) != -1])
        top_groups = functional_groups.head(5)["functional_group"].tolist() if not functional_groups.empty else []
        top_scaffolds = scaffold_families.head(5)["scaffold_family"].tolist() if not scaffold_families.empty else []
        lines = [
            f"Library size: {total} molecules.",
            f"Non-noise cluster count: {cluster_count}.",
            "This report combines structural chemistry, property balance, and cluster-level organization.",
            "Most common functional groups: " + (", ".join(top_groups) if top_groups else "n/a") + ".",
            "Most common scaffold families: " + (", ".join(top_scaffolds) if top_scaffolds else "n/a") + ".",
            "Molecular weight profile: " + self._bucket_summary_text(property_landscape["mw_profile"]) + ".",
            "Lipophilicity profile: " + self._bucket_summary_text(property_landscape["logp_profile"]) + ".",
            "Polarity profile: " + self._bucket_summary_text(property_landscape["tpsa_profile"]) + ".",
            "Flexibility profile: " + self._bucket_summary_text(property_landscape["flexibility_profile"]) + ".",
            "Ring-system profile: " + self._bucket_summary_text(property_landscape["ring_profile"]) + ".",
            f"Noise fraction from clustering: {property_landscape['noise_fraction']:.1%}.",
        ]
        return "\n".join(lines) + "\n"

    def _cluster_summary_rows(self, df: pd.DataFrame, labels: np.ndarray) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for cluster_id in sorted(int(value) for value in np.unique(labels) if int(value) != -1):
            subset = df.loc[labels == cluster_id].copy()
            size = int(len(subset))
            scaffolds = Counter(str(value) for value in subset.get("murcko_scaffold", pd.Series(dtype=str)).tolist() if str(value))
            dominant_scaffold, dominant_count = ("", 0)
            if scaffolds:
                dominant_scaffold, dominant_count = scaffolds.most_common(1)[0]

            mw_mean = self._safe_mean(subset, "MW")
            logp_mean = self._safe_mean(subset, "logP")
            tpsa_mean = self._safe_mean(subset, "TPSA")
            rb_mean = self._safe_mean(subset, "RB")
            rings_mean = self._safe_mean(subset, "Rings")
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "cluster_size": size,
                    "cluster_fraction": float(size / max(len(df), 1)),
                    "dominant_scaffold": dominant_scaffold,
                    "dominant_scaffold_fraction": float(dominant_count / max(size, 1)),
                    "MW_mean": mw_mean,
                    "logP_mean": logp_mean,
                    "TPSA_mean": tpsa_mean,
                    "RB_mean": rb_mean,
                    "Rings_mean": rings_mean,
                }
            )
        rows.sort(key=lambda item: item["cluster_size"], reverse=True)
        return rows

    def _super_region_table(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        cluster_space: np.ndarray,
        cluster_summary: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        cluster_ids = [int(row["cluster_id"]) for row in cluster_summary]
        if not cluster_ids:
            return pd.DataFrame(
                columns=[
                    "super_region_id",
                    "member_clusters",
                    "molecule_count",
                    "fraction_of_library",
                    "dominant_scaffolds",
                    "mw_range",
                    "logp_range",
                    "region_density",
                    "representative_smiles",
                ]
            )

        centroids = []
        for cluster_id in cluster_ids:
            subset_space = cluster_space[labels == cluster_id]
            centroids.append(subset_space.mean(axis=0))
        centroid_matrix = np.vstack(centroids)
        assignments = self._super_region_assignments(centroid_matrix, cluster_ids)

        rows: List[Dict[str, Any]] = []
        total_rows = max(len(df), 1)
        for super_region_id in sorted(set(assignments.values())):
            member_clusters = [cluster_id for cluster_id in cluster_ids if assignments[cluster_id] == super_region_id]
            subset = df.loc[np.isin(labels, member_clusters)].copy()
            scaffolds = Counter(str(value) for value in subset.get("murcko_scaffold", pd.Series(dtype=str)).tolist() if str(value))
            top_scaffolds = " | ".join(scaffold for scaffold, _ in scaffolds.most_common(3))
            cluster_centers = centroid_matrix[[cluster_ids.index(cluster_id) for cluster_id in member_clusters]]
            region_center = cluster_centers.mean(axis=0, keepdims=True)
            spread = np.linalg.norm(cluster_centers - region_center, axis=1).mean() if len(cluster_centers) else 0.0
            rows.append(
                {
                    "super_region_id": int(super_region_id),
                    "member_clusters": ",".join(str(cluster_id) for cluster_id in member_clusters),
                    "molecule_count": int(len(subset)),
                    "fraction_of_library": round(float(len(subset) / total_rows), 4),
                    "dominant_scaffolds": top_scaffolds,
                    "mw_range": self._range_text(subset, "MW"),
                    "logp_range": self._range_text(subset, "logP"),
                    "region_density": self._density_label(spread),
                    "representative_smiles": str(subset.iloc[0]["standardized_smiles"]) if len(subset) else "",
                }
            )
        return pd.DataFrame(rows).sort_values(["molecule_count", "super_region_id"], ascending=[False, True])

    def _cluster_interpretation_table(self, cluster_summary: List[Dict[str, Any]]) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for row in cluster_summary:
            descriptor_profile = self._descriptor_profile(row)
            rows.append(
                {
                    "cluster_id": int(row["cluster_id"]),
                    "cluster_size": int(row["cluster_size"]),
                    "cluster_fraction": round(float(row["cluster_fraction"]), 4),
                    "dominant_scaffold": row["dominant_scaffold"],
                    "descriptor_profile": descriptor_profile,
                    "structural_pattern": self._structural_pattern(row, descriptor_profile),
                    "confidence": self._confidence_label(float(row["dominant_scaffold_fraction"])),
                }
            )
        return pd.DataFrame(rows)

    def _outlier_table(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        cluster_space: np.ndarray,
        agreement_map: Dict[int, float],
    ) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for index, row in df.reset_index(drop=True).iterrows():
            cluster_id = int(labels[index])
            score = float(agreement_map.get(int(row["row_id"]), 0.0))
            if cluster_id == -1:
                rows.append(
                    {
                        "row_id": int(row["row_id"]),
                        "cluster_id": -1,
                        "outlier_type": "noise_point",
                        "reason": "HDBSCAN labeled this molecule as noise.",
                        "embedding_isolation_score": 1.0,
                        "fingerprint_agreement": round(score, 4),
                    }
                )
                continue

            subset_space = cluster_space[labels == cluster_id]
            centroid = subset_space.mean(axis=0, keepdims=True)
            distances = np.linalg.norm(subset_space - centroid, axis=1)
            threshold = float(np.quantile(distances, 0.95)) if len(distances) >= 5 else float(np.max(distances))
            row_distance = float(np.linalg.norm(cluster_space[index] - centroid[0]))
            if row_distance >= threshold or score <= 0.1:
                reason = "Far from cluster center." if row_distance >= threshold else "Embedding and fingerprint neighborhoods disagree."
                rows.append(
                    {
                        "row_id": int(row["row_id"]),
                        "cluster_id": cluster_id,
                        "outlier_type": "peripheral_member" if row_distance >= threshold else "neighbor_disagreement",
                        "reason": reason,
                        "embedding_isolation_score": round(row_distance, 4),
                        "fingerprint_agreement": round(score, 4),
                    }
                )
        if not rows:
            return pd.DataFrame(columns=["row_id", "cluster_id", "outlier_type", "reason", "embedding_isolation_score", "fingerprint_agreement"])
        return pd.DataFrame(rows).sort_values(
            ["outlier_type", "embedding_isolation_score"],
            ascending=[True, False],
        )

    def _similarity_agreement_payload(
        self,
        embedding_neighbors: pd.DataFrame,
        fingerprint_neighbors: pd.DataFrame,
    ) -> tuple[Dict[str, Any], Dict[int, float]]:
        emb_map = self._neighbor_sets(embedding_neighbors)
        fp_map = self._neighbor_sets(fingerprint_neighbors)
        source_ids = sorted(set(emb_map) | set(fp_map))
        overlap_rows = []
        agreement_map: Dict[int, float] = {}
        for source_id in source_ids:
            emb_set = emb_map.get(source_id, set())
            fp_set = fp_map.get(source_id, set())
            union = emb_set | fp_set
            score = float(len(emb_set & fp_set) / len(union)) if union else 0.0
            agreement_map[source_id] = score
            overlap_rows.append(score)

        mean_agreement = float(np.mean(overlap_rows)) if overlap_rows else 0.0
        high_sources = sum(score >= 0.6 for score in overlap_rows)
        low_sources = sum(score <= 0.2 for score in overlap_rows)
        payload = {
            "mean_neighbor_agreement": round(mean_agreement, 4),
            "high_agreement_source_count": int(high_sources),
            "low_agreement_source_count": int(low_sources),
            "possible_scaffold_hop_fraction": round(float(low_sources / max(len(overlap_rows), 1)), 4),
        }
        return payload, agreement_map

    def _neighbor_sets(self, neighbors: pd.DataFrame) -> Dict[int, set[int]]:
        mapping: Dict[int, set[int]] = {}
        if neighbors.empty:
            return mapping
        for row in neighbors.to_dict(orient="records"):
            source = int(row["source_row_id"])
            mapping.setdefault(source, set()).add(int(row["neighbor_row_id"]))
        return mapping

    def _effective_cluster_count(self, non_noise_sizes: np.ndarray) -> float:
        if len(non_noise_sizes) == 0 or float(non_noise_sizes.sum()) == 0.0:
            return 0.0
        probs = non_noise_sizes / non_noise_sizes.sum()
        entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)))
        return float(np.exp(entropy))

    def _redundancy_estimate(
        self,
        embedding_neighbors: pd.DataFrame,
        fingerprint_neighbors: pd.DataFrame,
    ) -> float:
        per_source: Dict[int, bool] = {}
        for row in embedding_neighbors.to_dict(orient="records"):
            source = int(row["source_row_id"])
            per_source[source] = per_source.get(source, False) or float(row["score"]) >= 0.9
        for row in fingerprint_neighbors.to_dict(orient="records"):
            source = int(row["source_row_id"])
            per_source[source] = per_source.get(source, False) or float(row["score"]) >= 0.85
        if not per_source:
            return 0.0
        return float(sum(per_source.values()) / len(per_source))

    def _diversity_level(self, effective_cluster_count: float, redundancy_estimate: float, noise_fraction: float) -> str:
        if effective_cluster_count >= 8 and redundancy_estimate <= 0.35:
            return "high"
        if effective_cluster_count <= 3 or redundancy_estimate >= 0.65:
            return "low"
        if noise_fraction >= 0.3 and effective_cluster_count >= 5:
            return "high"
        return "moderate"

    def _map_structure(self, largest_cluster_fraction: float, noise_fraction: float, effective_cluster_count: float) -> str:
        if largest_cluster_fraction >= 0.4 and effective_cluster_count <= 4:
            return "compact"
        if noise_fraction >= 0.25:
            return "diffuse"
        if effective_cluster_count >= 8:
            return "broad"
        return "mixed"

    def _super_region_assignments(self, centroid_matrix: np.ndarray, cluster_ids: List[int]) -> Dict[int, int]:
        if len(cluster_ids) <= 1:
            return {cluster_id: 1 for cluster_id in cluster_ids}

        n_regions = min(max(2, int(round(np.sqrt(len(cluster_ids))))), 10, len(cluster_ids))
        try:
            from sklearn.cluster import KMeans

            model = KMeans(n_clusters=n_regions, random_state=self.config.runtime.random_state, n_init=10)
            labels = model.fit_predict(centroid_matrix)
            return {cluster_id: int(label) + 1 for cluster_id, label in zip(cluster_ids, labels)}
        except Exception:
            return {cluster_id: index + 1 for index, cluster_id in enumerate(cluster_ids)}

    def _descriptor_profile(self, row: Dict[str, Any]) -> str:
        size = "small" if row["MW_mean"] < 250 else "large" if row["MW_mean"] >= 450 else "mid-size"
        polarity = "polar" if row["TPSA_mean"] >= 90 else "lipophilic" if row["logP_mean"] >= 3 else "balanced"
        flexibility = "flexible" if row["RB_mean"] >= 5 else "rigid" if row["RB_mean"] <= 2 else "mixed-flexibility"
        return f"{size}, {polarity}, {flexibility}"

    def _structural_pattern(self, row: Dict[str, Any], descriptor_profile: str) -> str:
        scaffold = str(row["dominant_scaffold"]).lower()
        if row["MW_mean"] < 250 and row["TPSA_mean"] >= 60:
            return "small polar fragments"
        if row["logP_mean"] >= 3 and row["Rings_mean"] >= 2:
            return "bulky lipophilic ring systems"
        if row["RB_mean"] >= 5:
            return "flexible linker-like molecules"
        if "n" in scaffold or row["TPSA_mean"] >= 70:
            return "heteroatom-rich chemotypes"
        if row["Rings_mean"] >= 3 and row["RB_mean"] <= 2:
            return "rigid polycyclic cores"
        return f"balanced drug-like chemotypes ({descriptor_profile})"

    def _confidence_label(self, dominant_scaffold_fraction: float) -> str:
        if dominant_scaffold_fraction >= 0.6:
            return "high"
        if dominant_scaffold_fraction >= 0.35:
            return "medium"
        return "low"

    def _executive_summary_text(
        self,
        n_rows: int,
        cluster_count: int,
        diversity_level: str,
        map_structure: str,
        noise_fraction: float,
        largest_cluster_fraction: float,
        redundancy_estimate: float,
        dominant_cluster_ids: List[int],
        super_regions: pd.DataFrame,
        agreement_payload: Dict[str, Any],
    ) -> str:
        lines = [
            f"Library size: {n_rows} molecules.",
            f"Overall diversity: {diversity_level}.",
            f"Map structure: {map_structure}.",
            f"Non-noise cluster count: {cluster_count}.",
            f"Noise fraction: {noise_fraction:.1%}.",
            f"Largest cluster fraction: {largest_cluster_fraction:.1%}.",
            f"Redundancy estimate: {redundancy_estimate:.1%}.",
        ]
        if dominant_cluster_ids:
            lines.append("Dominant clusters by size: " + ", ".join(str(value) for value in dominant_cluster_ids) + ".")
        if not super_regions.empty:
            top_region = super_regions.iloc[0]
            lines.append(
                "Largest super-region: "
                f"{int(top_region['super_region_id'])} "
                f"({int(top_region['molecule_count'])} molecules; scaffolds: {top_region['dominant_scaffolds'] or 'n/a'})."
            )
        lines.append(
            "Embedding/fingerprint agreement: "
            f"{agreement_payload['mean_neighbor_agreement']:.2f} mean overlap."
        )
        lines.append(
            f"Possible scaffold-hop fraction: {agreement_payload['possible_scaffold_hop_fraction']:.1%}."
        )
        return "\n".join(lines) + "\n"

    def _safe_mean(self, df: pd.DataFrame, column: str) -> float:
        if column not in df.columns:
            return float("nan")
        return float(pd.to_numeric(df[column], errors="coerce").mean())

    def _range_text(self, df: pd.DataFrame, column: str) -> str:
        if column not in df.columns or df.empty:
            return ""
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            return ""
        return f"{series.min():.1f}-{series.max():.1f}"

    def _density_label(self, spread: float) -> str:
        if spread <= 0.5:
            return "high"
        if spread <= 1.5:
            return "medium"
        return "low"

    def _functional_group_patterns(self) -> List[tuple[str, Any]]:
        specs = [
            ("amine", "[NX3;H2,H1;!$(NC=O)]"),
            ("amide", "[NX3][CX3](=[OX1])[#6]"),
            ("carboxylic_acid", "[CX3](=O)[OX2H1]"),
            ("ester", "[CX3](=O)[OX2][#6]"),
            ("alcohol", "[OX2H][#6;!$(C=O)]"),
            ("ether", "[OD2]([#6])[#6]"),
            ("ketone_or_aldehyde", "[CX3]=[OX1]"),
            ("halogen", "[F,Cl,Br,I]"),
            ("nitrile", "[CX2]#N"),
            ("sulfonamide_or_sulfone", "[SX4](=O)(=O)"),
            ("heteroaromatic_ring", "[a;r][n,o,s]"),
        ]
        chem = self._try_import_rdkit_chem()
        if chem is None:
            return [(name, None) for name, _ in specs]
        return [(name, chem.MolFromSmarts(smarts)) for name, smarts in specs]

    def _import_rdkit_chem(self) -> Any:
        try:
            from rdkit import Chem
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "RDKit is required for chemical landscape summaries."
            ) from exc
        return Chem

    def _try_import_rdkit_chem(self) -> Any | None:
        try:
            from rdkit import Chem
        except ModuleNotFoundError:
            return None
        return Chem

    def _scaffold_family_type(self, scaffold: str) -> str:
        if not scaffold:
            return "acyclic_or_fragment_like"
        lower = scaffold.lower()
        hetero = sum(lower.count(char) for char in ["n", "o", "s"])
        ring_markers = lower.count("1") + lower.count("2") + lower.count("3")
        if hetero >= 2:
            return "heteroaromatic_or_heterocyclic"
        if ring_markers >= 4:
            return "polycyclic_core"
        if "c1" in lower:
            return "aryl_dominant"
        return "mixed_scaffold"

    def _distribution_buckets(self, series: pd.Series, thresholds: List[tuple[float, str]]) -> Dict[str, float]:
        clean = series.dropna()
        total = max(len(clean), 1)
        counts = {label: 0 for _, label in thresholds}
        for value in clean.tolist():
            for cutoff, label in thresholds:
                if value < cutoff:
                    counts[label] += 1
                    break
        return {label: round(float(count / total), 4) for label, count in counts.items()}

    def _bucket_summary_text(self, buckets: Dict[str, float]) -> str:
        ordered = sorted(buckets.items(), key=lambda item: item[1], reverse=True)
        top = ordered[:2]
        return ", ".join(f"{label} {value:.1%}" for label, value in top)

    def _functional_group_fallback_match(self, name: str, smiles: str) -> bool:
        upper = smiles.upper()
        if name == "amine":
            return "N" in upper and "NC(=O)" not in upper and "N=C" not in upper
        if name == "amide":
            return "NC(=O)" in upper or "N(C)C(=O)" in upper or "C(=O)N" in upper
        if name == "carboxylic_acid":
            return "C(=O)O" in upper and "[O-]" not in upper
        if name == "ester":
            return "C(=O)O" in upper and "C(=O)O" != upper
        if name == "alcohol":
            return "O" in upper and "C(=O)O" not in upper
        if name == "ether":
            return "O" in upper and "C(=O)" not in upper
        if name == "ketone_or_aldehyde":
            return "C(=O)" in upper
        if name == "halogen":
            return any(token in upper for token in ["F", "CL", "BR", "I"])
        if name == "nitrile":
            return "#N" in upper
        if name == "sulfonamide_or_sulfone":
            return "S(=O)(=O)" in upper
        if name == "heteroaromatic_ring":
            return any(token in smiles for token in ["n", "o", "s"])
        return False

    def _import_matplotlib(self) -> Any:
        os.environ.setdefault("MPLCONFIGDIR", self.config.runtime.matplotlib_config_dir)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            raise RuntimeError("matplotlib is required for plotting the UMAP outputs.") from exc

        return plt
