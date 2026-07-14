from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd

from app.models.schemas import SourceResult
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import BM25Retriever
from app.retrieval.types import RankedItem


class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        sparse: BM25Retriever,
        metadata: pd.DataFrame,
        rrf_k: int,
        dense_top_k: int,
        bm25_top_k: int,
        final_top_k: int,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.rrf_k = rrf_k
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.final_top_k = final_top_k

        self.metadata = {
            str(row["passage_id"]): row
            for row in metadata.to_dict(orient="records")
        }

    async def search(
        self,
        question: str,
        query_embedding: np.ndarray,
        top_k: int | None = None,
    ) -> list[SourceResult]:
        dense_results, sparse_results = await asyncio.gather(
            self.dense.search(query_embedding, self.dense_top_k),
            self.sparse.search(question, self.bm25_top_k),
        )

        fused = self._rrf(dense_results, sparse_results)
        output_k = top_k or self.final_top_k
        return self._hydrate(fused[:output_k], dense_results, sparse_results)

    def _rrf(
        self,
        dense_results: list[RankedItem],
        sparse_results: list[RankedItem],
    ) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}

        for item in dense_results:
            scores[item.passage_id] = (
                scores.get(item.passage_id, 0.0)
                + 1.0 / (self.rrf_k + item.rank)
            )

        for item in sparse_results:
            scores[item.passage_id] = (
                scores.get(item.passage_id, 0.0)
                + 1.0 / (self.rrf_k + item.rank)
            )

        return sorted(scores.items(), key=lambda item: item[1], reverse=True)

    def _hydrate(
        self,
        fused: list[tuple[str, float]],
        dense_results: list[RankedItem],
        sparse_results: list[RankedItem],
    ) -> list[SourceResult]:
        dense_map = {item.passage_id: item for item in dense_results}
        sparse_map = {item.passage_id: item for item in sparse_results}

        results = []
        for passage_id, score in fused:
            row = self.metadata.get(passage_id)
            if row is None:
                continue

            dense_item = dense_map.get(passage_id)
            sparse_item = sparse_map.get(passage_id)

            results.append(
                SourceResult(
                    passage_id=passage_id,
                    document=str(row.get("document", "")),
                    article=str(row.get("article", "")),
                    content=str(row.get("content", "")),
                    score=float(score),
                    dense_rank=dense_item.rank if dense_item else None,
                    bm25_rank=sparse_item.rank if sparse_item else None,
                    dense_score=dense_item.score if dense_item else None,
                    bm25_score=sparse_item.score if sparse_item else None,
                )
            )

        return results
