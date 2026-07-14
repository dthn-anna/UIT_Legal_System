from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

import numpy as np
from supabase import Client, create_client

from app.retrieval.types import RankedItem


class DenseRetriever(ABC):
    @abstractmethod
    async def search(self, embedding: np.ndarray, top_k: int) -> list[RankedItem]:
        raise NotImplementedError


class LocalNumpyDenseRetriever(DenseRetriever):
    def __init__(
        self,
        embeddings: np.ndarray,
        passage_ids: list[str],
        min_score: float = -1.0,
    ) -> None:
        self.embeddings = embeddings
        self.passage_ids = passage_ids
        self.min_score = min_score

    def _search_sync(self, embedding: np.ndarray, top_k: int) -> list[RankedItem]:
        scores = self.embeddings @ embedding
        top_k = min(top_k, len(scores))

        if top_k <= 0:
            return []

        candidate_indices = np.argpartition(scores, -top_k)[-top_k:]
        sorted_indices = candidate_indices[
            np.argsort(scores[candidate_indices])[::-1]
        ]

        results = []
        for rank, index in enumerate(sorted_indices, start=1):
            score = float(scores[index])
            if score < self.min_score:
                continue
            results.append(
                RankedItem(
                    passage_id=self.passage_ids[int(index)],
                    rank=rank,
                    score=score,
                )
            )
        return results

    async def search(self, embedding: np.ndarray, top_k: int) -> list[RankedItem]:
        return await asyncio.to_thread(self._search_sync, embedding, top_k)


class SupabaseDenseRetriever(DenseRetriever):
    def __init__(
        self,
        url: str,
        service_role_key: str,
        rpc_name: str,
    ) -> None:
        if not url or not service_role_key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required."
            )
        self.client: Client = create_client(url, service_role_key)
        self.rpc_name = rpc_name

    def _search_sync(self, embedding: np.ndarray, top_k: int) -> list[RankedItem]:
        response = self.client.rpc(
            self.rpc_name,
            {
                "query_embedding": embedding.astype(float).tolist(),
                "match_count": top_k,
            },
        ).execute()

        rows = response.data or []
        return [
            RankedItem(
                passage_id=str(row["passage_id"]),
                rank=rank,
                score=float(row["similarity"]),
            )
            for rank, row in enumerate(rows, start=1)
        ]

    async def search(self, embedding: np.ndarray, top_k: int) -> list[RankedItem]:
        return await asyncio.to_thread(self._search_sync, embedding, top_k)
