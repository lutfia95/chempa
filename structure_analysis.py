from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors


try:
    import umap

    _HAVE_UMAP = True
except Exception:
    _HAVE_UMAP = False

from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest


PathLike = Union[str, Path]


@dataclass(frozen=True)
class PipelineOutputs:
    """Paths to the main output files."""
    merged_tsv: Path
    embedding_tsv: Path
    clusters_tsv: Path
    representatives_tsv: Path
    outliers_tsv: Path
    filtered_tsv: Path
    plots_dir: Path


class ChemSpacePipeline:
    def __init__(
        self,
        input_unique_tsv: PathLike,
        outdir: PathLike,
        smiles_col: str = "SMILES",
        label_cols: Optional[Sequence[str]] = None,
        fp_radius: int = 2,
        fp_nbits: int = 2048,
        n_clusters: int = 2000,
        rep_per_cluster: int = 1,
        outliers: int = 500,
        sample_embed: int = 50000,
        random_state: int = 0,
    ) -> None:
        self.input_unique_tsv = Path(input_unique_tsv)
        self.outdir = Path(outdir)
        self.smiles_col = smiles_col
        self.label_cols = list(label_cols) if label_cols else ["Label", "File", "Index"]
        self.fp_radius = fp_radius
        self.fp_nbits = fp_nbits
        self.n_clusters = n_clusters
        self.rep_per_cluster = rep_per_cluster
        self.outliers = outliers
        self.sample_embed = sample_embed
        self.random_state = random_state

        self.outdir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.outdir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.df: Optional[pd.DataFrame] = None
        self.mols: Optional[List[Optional[Chem.Mol]]] = None
        self.fps: Optional[List[Optional[DataStructs.cDataStructs.ExplicitBitVect]]] = None
        self.fp_bits: Optional[np.ndarray] = None  # shape: (n, nbits)
        self.embedding: Optional[np.ndarray] = None  # shape: (n, 2)
        self.cluster_ids: Optional[np.ndarray] = None


    def run(self) -> PipelineOutputs:
        """
        Run full pipeline and write outputs to outdir.
        Returns paths to key files.
        """
        df = self._read()
        mols = self._parse_smiles(df)
        df = self._ensure_descriptors(df, mols)
        fps, fp_bits = self._compute_fingerprints(mols)

        emb = self._compute_embedding(fp_bits, sample_n=self.sample_embed)

        cluster_ids = self._cluster_embedding(emb, n_clusters=self.n_clusters)

        reps_df = self._pick_representatives(df, emb, cluster_ids, per_cluster=self.rep_per_cluster)

        outliers_df = self._detect_outliers(df, top_n=self.outliers)

        filtered_df = self._apply_rule_filters(df)

        merged_tsv = self.outdir / "merged_with_features.tsv"
        embedding_tsv = self.outdir / "embedding_2d.tsv"
        clusters_tsv = self.outdir / "clusters.tsv"
        representatives_tsv = self.outdir / "representatives.tsv"
        outliers_tsv = self.outdir / "outliers.tsv"
        filtered_tsv = self.outdir / "filtered_pass_rules.tsv"

        df_out = df.copy()
        df_out["emb_x"] = emb[:, 0]
        df_out["emb_y"] = emb[:, 1]
        df_out["cluster_id"] = cluster_ids.astype(int)

        df_out.to_csv(merged_tsv, sep="\t", index=False)

        df_out[[self.smiles_col, *self._safe_cols(df_out, self.label_cols), "emb_x", "emb_y"]].to_csv(
            embedding_tsv, sep="\t", index=False
        )

        df_out[[self.smiles_col, *self._safe_cols(df_out, self.label_cols), "cluster_id"]].to_csv(
            clusters_tsv, sep="\t", index=False
        )

        reps_df.to_csv(representatives_tsv, sep="\t", index=False)
        outliers_df.to_csv(outliers_tsv, sep="\t", index=False)
        filtered_df.to_csv(filtered_tsv, sep="\t", index=False)

        self._plot_histograms(df_out)
        self._plot_embedding(df_out, color_by="MW")
        self._plot_embedding(df_out, color_by="logP")
        self._plot_embedding(df_out, color_by="TPSA")
        self._plot_embedding_clusters(df_out)

        return PipelineOutputs(
            merged_tsv=merged_tsv,
            embedding_tsv=embedding_tsv,
            clusters_tsv=clusters_tsv,
            representatives_tsv=representatives_tsv,
            outliers_tsv=outliers_tsv,
            filtered_tsv=filtered_tsv,
            plots_dir=self.plots_dir,
        )


    def _read(self) -> pd.DataFrame:
        df = pd.read_csv(self.input_unique_tsv, sep="\t", dtype=str, keep_default_na=False)
        if self.smiles_col not in df.columns:
            raise ValueError(f"Missing SMILES column '{self.smiles_col}' in {self.input_unique_tsv}")
        self.df = df
        return df


    def _parse_smiles(self, df: pd.DataFrame) -> List[Optional[Chem.Mol]]:
        mols: List[Optional[Chem.Mol]] = []
        bad = 0
        for smi in df[self.smiles_col].astype(str).tolist():
            smi = (smi or "").strip()
            m = Chem.MolFromSmiles(smi) if smi else None
            if m is None:
                bad += 1
            mols.append(m)
        df["is_valid_smiles"] = ["1" if m is not None else "0" for m in mols]
        self.mols = mols
        if bad > 0:
            # Keep going, but those rows will mostly drop out of downstream computations.
            pass
        return mols


    def _ensure_descriptors(self, df: pd.DataFrame, mols: List[Optional[Chem.Mol]]) -> pd.DataFrame:
        want = ["MW", "logP", "HBD", "HBA", "TPSA", "RB", "Rings", "HeavyAtoms", "NumAtoms"]
        for c in want:
            if c not in df.columns:
                df[c] = np.nan

        for c in want:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        for i, m in enumerate(mols):
            if m is None:
                continue

            if math.isnan(df.at[i, "NumAtoms"]):
                df.at[i, "NumAtoms"] = float(m.GetNumAtoms())
            if math.isnan(df.at[i, "HeavyAtoms"]):
                df.at[i, "HeavyAtoms"] = float(m.GetNumHeavyAtoms())
            if math.isnan(df.at[i, "MW"]):
                df.at[i, "MW"] = float(Descriptors.MolWt(m))
            if math.isnan(df.at[i, "logP"]):
                df.at[i, "logP"] = float(Crippen.MolLogP(m))
            if math.isnan(df.at[i, "HBD"]):
                df.at[i, "HBD"] = float(Lipinski.NumHDonors(m))
            if math.isnan(df.at[i, "HBA"]):
                df.at[i, "HBA"] = float(Lipinski.NumHAcceptors(m))
            if math.isnan(df.at[i, "TPSA"]):
                df.at[i, "TPSA"] = float(rdMolDescriptors.CalcTPSA(m))
            if math.isnan(df.at[i, "RB"]):
                df.at[i, "RB"] = float(Lipinski.NumRotatableBonds(m))
            if math.isnan(df.at[i, "Rings"]):
                df.at[i, "Rings"] = float(rdMolDescriptors.CalcNumRings(m))

        return df

    def _compute_fingerprints(
        self, mols: List[Optional[Chem.Mol]]
    ) -> Tuple[List[Optional[DataStructs.cDataStructs.ExplicitBitVect]], np.ndarray]:

        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
        from rdkit import DataStructs

        morgan = GetMorganGenerator(radius=self.fp_radius, fpSize=self.fp_nbits)

        fps = []
        bits = np.zeros((len(mols), self.fp_nbits), dtype=np.uint8)

        for i, m in enumerate(mols):
            if m is None:
                fps.append(None)
                continue
            fp = morgan.GetFingerprint(m)  # ExplicitBitVect
            fps.append(fp)
            arr = np.zeros((self.fp_nbits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            bits[i, :] = arr.astype(np.uint8)

        self.fps = fps
        self.fp_bits = bits
        return fps, bits

    def _compute_embedding(self, fp_bits: np.ndarray, sample_n: int) -> np.ndarray:
        n = fp_bits.shape[0]
        rng = np.random.default_rng(self.random_state)

        idx_fit = np.arange(n)
        if sample_n is not None and sample_n > 0 and n > sample_n:
            idx_fit = rng.choice(n, size=sample_n, replace=False)

        if _HAVE_UMAP:
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                metric="jaccard",  # good default for binary fingerprints
                random_state=self.random_state,
            )
            reducer.fit(fp_bits[idx_fit])
            emb = reducer.transform(fp_bits).astype(np.float32)
        else:
            pca = PCA(n_components=2, random_state=self.random_state)
            emb = pca.fit_transform(fp_bits.astype(np.float32)).astype(np.float32)

        self.embedding = emb
        return emb


    def _cluster_embedding(self, emb: np.ndarray, n_clusters: int) -> np.ndarray:

        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            batch_size=4096,
            n_init="auto",
        )
        cluster_ids = km.fit_predict(emb)
        self.cluster_ids = cluster_ids
        return cluster_ids

    def _pick_representatives(
        self, df: pd.DataFrame, emb: np.ndarray, cluster_ids: np.ndarray, per_cluster: int
    ) -> pd.DataFrame:

        cols = [self.smiles_col, *self._safe_cols(df, self.label_cols)]
        out_rows: List[Dict[str, object]] = []

        cluster_to_idx: Dict[int, np.ndarray] = {}
        for cid in np.unique(cluster_ids):
            cluster_to_idx[int(cid)] = np.where(cluster_ids == cid)[0]

        for cid, idxs in cluster_to_idx.items():
            pts = emb[idxs]
            centroid = pts.mean(axis=0)
            d = np.sqrt(((pts - centroid) ** 2).sum(axis=1))
            order = np.argsort(d)[: max(1, per_cluster)]
            for j in order:
                i = int(idxs[j])
                row = {c: df.at[i, c] if c in df.columns else "" for c in cols}
                row["cluster_id"] = cid
                row["distance_to_centroid"] = float(d[j])
                out_rows.append(row)

        return pd.DataFrame(out_rows)


    def _detect_outliers(self, df: pd.DataFrame, top_n: int) -> pd.DataFrame:

        desc_cols = ["MW", "logP", "HBD", "HBA", "TPSA", "RB", "Rings", "HeavyAtoms", "NumAtoms"]
        X = df[desc_cols].copy()
        X = X.apply(pd.to_numeric, errors="coerce").fillna(X.median(numeric_only=True))

        iso = IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=self.random_state,
            n_jobs=-1,
        )
        iso.fit(X)
        score = iso.decision_function(X)
        df2 = df.copy()
        df2["outlier_score"] = score

        df2 = df2.sort_values("outlier_score", ascending=True)
        cols = [self.smiles_col, *self._safe_cols(df2, self.label_cols), "outlier_score", *desc_cols]
        return df2.loc[:, cols].head(top_n)


    def _apply_rule_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        for c in ["MW", "logP", "HBD", "HBA", "TPSA", "RB"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")

        passed = (
            (d["MW"] <= 500)
            & (d["logP"] <= 5)
            & (d["HBD"] <= 5)
            & (d["HBA"] <= 10)
            & (d["TPSA"] <= 140)
            & (d["RB"] <= 10)
        )

        out = d.loc[passed].copy()
        out["passed_rules"] = "1"
        return out


    def _plot_histograms(self, df: pd.DataFrame) -> None:

        cols = ["MW", "logP", "HBD", "HBA", "TPSA", "RB", "Rings", "HeavyAtoms", "NumAtoms"]
        for c in cols:
            if c not in df.columns:
                continue
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if s.empty:
                continue

            plt.figure()
            plt.hist(s.values, bins=50)
            plt.title(f"Histogram: {c}")
            plt.xlabel(c)
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(self.plots_dir / f"hist_{c}.png", dpi=200, bbox_inches="tight")
            plt.close()

    def _plot_embedding(self, df: pd.DataFrame, color_by: str) -> None:
        if "emb_x" not in df.columns or "emb_y" not in df.columns:
            return
        if color_by not in df.columns:
            return

        x = df["emb_x"].astype(float).values
        y = df["emb_y"].astype(float).values
        c = pd.to_numeric(df[color_by], errors="coerce").values

        plt.figure()
        sc = plt.scatter(x, y, c=c, s=2)
        plt.title(f"Embedding (2D) colored by {color_by}")
        plt.xlabel("emb_x")
        plt.ylabel("emb_y")
        plt.colorbar(sc, label=color_by)
        plt.tight_layout()
        plt.savefig(self.plots_dir / f"embed_color_{color_by}.png", dpi=200, bbox_inches="tight")
        plt.close()

    def _plot_embedding_clusters(self, df: pd.DataFrame) -> None:

        if "emb_x" not in df.columns or "emb_y" not in df.columns or "cluster_id" not in df.columns:
            return

        x = df["emb_x"].astype(float).values
        y = df["emb_y"].astype(float).values
        cid = df["cluster_id"].astype(int).values

        sizes = pd.Series(cid).value_counts().to_dict()
        cs = np.array([sizes[int(k)] for k in cid], dtype=float)

        # bin sizes to avoid a million colors
        # bins: [1-2], [3-5], [6-10], [11-20], [21-50], [51+]
        bins = np.digitize(cs, bins=[2, 5, 10, 20, 50])
        plt.figure()
        sc = plt.scatter(x, y, c=bins, s=2)
        plt.title("Embedding colored by cluster-size bin")
        plt.xlabel("emb_x")
        plt.ylabel("emb_y")
        plt.colorbar(sc, label="cluster size bin")
        plt.tight_layout()
        plt.savefig(self.plots_dir / "embed_cluster_size_bins.png", dpi=200, bbox_inches="tight")
        plt.close()


    @staticmethod
    def _safe_cols(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
        return [c for c in cols if c in df.columns]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to SMILES-unique TSV")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--smiles-col", default="SMILES", help="SMILES column name")
    ap.add_argument("--n-clusters", type=int, default=2000, help="KMeans clusters in embedding space")
    ap.add_argument("--rep-per-cluster", type=int, default=1, help="Representatives per cluster")
    ap.add_argument("--outliers", type=int, default=500, help="Number of outliers to write")
    ap.add_argument("--sample-embed", type=int, default=50000, help="Fit embedding on this many points (0=all)")
    ap.add_argument("--fp-radius", type=int, default=2, help="Morgan radius (2=ECFP4)")
    ap.add_argument("--fp-nbits", type=int, default=2048, help="Fingerprint bits")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    args = ap.parse_args()

    sample_n = args.sample_embed if args.sample_embed and args.sample_embed > 0 else None

    p = ChemSpacePipeline(
        input_unique_tsv=args.input,
        outdir=args.outdir,
        smiles_col=args.smiles_col,
        fp_radius=args.fp_radius,
        fp_nbits=args.fp_nbits,
        n_clusters=args.n_clusters,
        rep_per_cluster=args.rep_per_cluster,
        outliers=args.outliers,
        sample_embed=sample_n if sample_n is not None else 10**18,
        random_state=args.seed,
    )
    outs = p.run()

    print("Wrote:")
    print("  merged:", outs.merged_tsv)
    print("  embedding:", outs.embedding_tsv)
    print("  clusters:", outs.clusters_tsv)
    print("  representatives:", outs.representatives_tsv)
    print("  outliers:", outs.outliers_tsv)
    print("  filtered:", outs.filtered_tsv)
    print("  plots:", outs.plots_dir)
    if not _HAVE_UMAP:
        print("UMAP not installed")


if __name__ == "__main__":
    main()