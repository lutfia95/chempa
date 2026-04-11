from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from workflow.config import PipelineConfig


class MorganFingerprintComputer:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def compute(self, df: pd.DataFrame) -> Tuple[np.ndarray, sparse.csr_matrix]:
        Chem, DataStructs, GetMorganGenerator = self._imports()
        smiles = df["standardized_smiles"].astype(str).tolist()

        generator = GetMorganGenerator(
            radius=self.config.fingerprints.radius,
            fpSize=self.config.fingerprints.n_bits,
        )
        dense = np.zeros((len(smiles), self.config.fingerprints.n_bits), dtype=np.uint8)

        for index, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise RuntimeError(f"Unexpected invalid standardized SMILES during fingerprinting: {smi}")
            fp = generator.GetFingerprint(mol)
            DataStructs.ConvertToNumpyArray(fp, dense[index])

        return dense, sparse.csr_matrix(dense)

    def tanimoto_top_k(self, dense_bits: np.ndarray, top_k: int) -> pd.DataFrame:
        rows = []
        if len(dense_bits) <= 1:
            return pd.DataFrame(rows)

        intersections = dense_bits @ dense_bits.T
        bit_counts = dense_bits.sum(axis=1, keepdims=True)
        unions = bit_counts + bit_counts.T - intersections
        similarities = np.divide(intersections, unions, out=np.zeros_like(intersections, dtype=float), where=unions > 0)

        for source_index in range(similarities.shape[0]):
            scores = similarities[source_index].copy()
            scores[source_index] = -1.0
            neighbor_indices = np.argsort(scores)[::-1][:top_k]
            for rank, neighbor_index in enumerate(neighbor_indices, start=1):
                rows.append(
                    {
                        "source_index": int(source_index),
                        "neighbor_rank": rank,
                        "neighbor_index": int(neighbor_index),
                        "similarity": float(scores[neighbor_index]),
                    }
                )
        return pd.DataFrame(rows)

    def _imports(self) -> Tuple[Any, Any, Any]:
        try:
            from rdkit import Chem, DataStructs
            from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "RDKit is required for Morgan fingerprint generation. "
                "Install RDKit in the runtime environment before running the workflow."
            ) from exc

        return Chem, DataStructs, GetMorganGenerator

