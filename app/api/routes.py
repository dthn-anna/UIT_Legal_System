from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_container
from app.models.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    RetrievalRequest,
    RetrievalResponse,
)
from app.services.container import ServiceContainer

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(
    container: ServiceContainer = Depends(get_container),
) -> HealthResponse:
    return await container.health()


@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(
    payload: RetrievalRequest,
    container: ServiceContainer = Depends(get_container),
) -> RetrievalResponse:
    try:
        results = await container.rag.retrieve(
            question=payload.question,
            top_k=payload.top_k,
        )
        return RetrievalResponse(question=payload.question, results=results)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/ask", response_model=AskResponse)
async def ask(
    payload: AskRequest,
    container: ServiceContainer = Depends(get_container),
) -> AskResponse:
    try:
        return await container.rag.answer(
            question=payload.question,
            top_k=payload.top_k,
            generate=payload.generate,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
