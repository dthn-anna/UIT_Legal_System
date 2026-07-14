from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ArtifactBundle:
    root: Path
    config: dict
    metadata: pd.DataFrame
    embeddings: np.ndarray | None

    @classmethod
    def load(cls, root: Path, load_embeddings: bool) -> "ArtifactBundle":
        required = [
            root / "retriever_config.json",
            root / "passage_metadata.parquet",
            root / "bm25_index",
            root / "dense_model",
        ]
        if load_embeddings:
            required.append(root / "corpus_embeddings.npy")

        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing retrieval artifacts:\n- " + "\n- ".join(missing)
            )

        with (root / "retriever_config.json").open("r", encoding="utf-8") as file:
            config = json.load(file)

        metadata = pd.read_parquet(root / "passage_metadata.parquet")
        metadata = metadata.sort_values("embedding_index").reset_index(drop=True)

        embeddings = None
        if load_embeddings:
            embeddings = np.load(root / "corpus_embeddings.npy", mmap_mode="r")
            if embeddings.shape[0] != len(metadata):
                raise ValueError(
                    "corpus_embeddings.npy rows do not match passage metadata."
                )

        return cls(
            root=root,
            config=config,
            metadata=metadata,
            embeddings=embeddings,
        )
