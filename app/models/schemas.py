from typing import Literal

from pydantic import BaseModel, Field, field_validator


class RetrievalRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=8, ge=1, le=30)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Cần bổ sung câu hỏi!")
        return value


class AskRequest(RetrievalRequest):
    generate: bool = True


class SourceResult(BaseModel):
    passage_id: str
    document: str
    article: str
    content: str
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None
    dense_score: float | None = None
    bm25_score: float | None = None


class RetrievalResponse(BaseModel):
    question: str
    results: list[SourceResult]


class AskResponse(BaseModel):
    question: str
    answer: str | None
    grounded: bool
    generator_backend: str
    sources: list[SourceResult]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    dense_backend: str
    generator_backend: str
    model_loaded: bool
    bm25_loaded: bool
    ollama_reachable: bool
    corpus_size: int
