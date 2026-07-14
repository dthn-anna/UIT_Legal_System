import pandas as pd
import pytest

from app.retrieval.hybrid import HybridRetriever
from app.retrieval.types import RankedItem


class DummyDense:
    async def search(self, embedding, top_k):
        return []


class DummySparse:
    async def search(self, question, top_k):
        return []


def build_retriever():
    metadata = pd.DataFrame(
        [
            {
                "passage_id": "a",
                "document": "D1",
                "article": "A1",
                "content": "C1",
            },
            {
                "passage_id": "b",
                "document": "D2",
                "article": "A2",
                "content": "C2",
            },
        ]
    )
    return HybridRetriever(
        dense=DummyDense(),
        sparse=DummySparse(),
        metadata=metadata,
        rrf_k=60,
        dense_top_k=10,
        bm25_top_k=10,
        final_top_k=2,
    )


def test_rrf_rewards_items_in_both_lists():
    retriever = build_retriever()

    dense = [
        RankedItem("a", 1, 0.9),
        RankedItem("b", 2, 0.8),
    ]
    sparse = [
        RankedItem("b", 1, 8.0),
    ]

    fused = retriever._rrf(dense, sparse)

    assert fused[0][0] == "b"
    assert fused[0][1] > fused[1][1]


def test_hydrate_preserves_ranks():
    retriever = build_retriever()

    dense = [RankedItem("a", 1, 0.9)]
    sparse = [RankedItem("a", 2, 7.0)]

    result = retriever._hydrate([("a", 0.03)], dense, sparse)[0]

    assert result.passage_id == "a"
    assert result.dense_rank == 1
    assert result.bm25_rank == 2
    assert result.score == pytest.approx(0.03)
