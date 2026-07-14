from __future__ import annotations

import asyncio
from pathlib import Path

import bm25s

from app.retrieval.types import RankedItem


class BM25Retriever:
    def __init__(self, index_path: Path, passage_ids: list[str]) -> None:
        self.retriever = bm25s.BM25.load(
            str(index_path),
            load_corpus=False,
        )
        self.passage_ids = passage_ids

    def _search_sync(self, question: str, top_k: int) -> list[RankedItem]:
        query_tokens = bm25s.tokenize(
            [question],
            stopwords=None,
            stemmer=None,
        )
        indices, scores = self.retriever.retrieve(
            query_tokens,
            k=min(top_k, len(self.passage_ids)),
        )

        return [
            RankedItem(
                passage_id=self.passage_ids[int(index)],
                rank=rank,
                score=float(scores[0][rank - 1]),
            )
            for rank, index in enumerate(indices[0], start=1)
        ]

    async def search(self, question: str, top_k: int) -> list[RankedItem]:
        return await asyncio.to_thread(self._search_sync, question, top_k)
