from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from workflow.config import PipelineConfig


class EmbeddingFactory:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def build_embeddings(self, df: pd.DataFrame, fingerprint_bits: np.ndarray) -> np.ndarray:
        backend = self.config.embedding.backend.strip().lower()
        if backend == "molformer":
            embeddings = self._molformer_embeddings(df["standardized_smiles"].astype(str).tolist())
        elif backend == "morgan_bits":
            embeddings = fingerprint_bits.astype(np.float32)
        else:
            raise ValueError("Unsupported embedding backend. Expected 'molformer' or 'morgan_bits'.")

        if self.config.embedding.normalize:
            embeddings = normalize(embeddings)
        return embeddings.astype(np.float32)

    def cosine_top_k(self, embeddings: np.ndarray, top_k: int) -> pd.DataFrame:
        rows = []
        if len(embeddings) <= 1:
            return pd.DataFrame(rows)

        similarity = embeddings @ embeddings.T
        for source_index in range(similarity.shape[0]):
            scores = similarity[source_index].copy()
            scores[source_index] = -np.inf
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

    def _molformer_embeddings(self, smiles: list[str]) -> np.ndarray:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ModuleNotFoundError as exc:
            message = str(exc)
            if "transformers.onnx" in message:
                raise RuntimeError(
                    "The installed transformers package is incompatible with the MoLFormer custom code. "
                    "Use transformers==4.48.3 for this workflow."
                ) from exc
            raise RuntimeError(
                "The MoLFormer backend requires torch and transformers. "
                "Install those packages or switch embedding.backend to 'morgan_bits'."
            ) from exc

        device = self._resolve_device(torch)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.embedding.model_name,
                local_files_only=self.config.embedding.local_files_only,
                trust_remote_code=True,
            )
            model = AutoModel.from_pretrained(
                self.config.embedding.model_name,
                local_files_only=self.config.embedding.local_files_only,
                trust_remote_code=True,
            )
        except ModuleNotFoundError as exc:
            message = str(exc)
            if "transformers.onnx" in message:
                raise RuntimeError(
                    "The installed transformers package is incompatible with the MoLFormer custom code. "
                    "Use transformers==4.48.3 for this workflow."
                ) from exc
            raise
        model.to(device)
        model.eval()

        batches = []
        batch_size = max(1, self.config.embedding.batch_size)
        with torch.no_grad():
            for start in range(0, len(smiles), batch_size):
                batch_smiles = smiles[start : start + batch_size]
                encoded = tokenizer(
                    batch_smiles,
                    padding=True,
                    truncation=True,
                    max_length=self.config.embedding.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                outputs = model(**encoded)
                token_embeddings = outputs.last_hidden_state
                attention_mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = self._pool(token_embeddings, attention_mask)
                batches.append(pooled.detach().cpu().numpy())

        if not batches:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack(batches)

    def _pool(self, token_embeddings: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
        pooling = self.config.embedding.pooling.strip().lower()
        masked = token_embeddings * attention_mask
        if pooling == "mean":
            denom = attention_mask.sum(dim=1).clamp(min=1)
            return masked.sum(dim=1) / denom
        if pooling == "cls":
            return token_embeddings[:, 0, :]
        raise ValueError("Unsupported pooling. Expected 'mean' or 'cls'.")

    def _resolve_device(self, torch: "object") -> str:
        requested = self.config.embedding.device.strip().lower()
        if requested != "auto":
            return requested
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
