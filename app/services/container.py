from __future__ import annotations

import logging

from app.core.config import Settings
from app.generation.ollama import OllamaGenerator
from app.models.schemas import HealthResponse
from app.retrieval.artifacts import ArtifactBundle
from app.retrieval.dense import LocalNumpyDenseRetriever, SupabaseDenseRetriever
from app.retrieval.encoder import QueryEncoder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.sparse import BM25Retriever
from app.services.rag import RAGService

logger = logging.getLogger(__name__)


class ServiceContainer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.bundle: ArtifactBundle | None = None
        self.rag: RAGService | None = None
        self.generator: OllamaGenerator | None = None

    async def startup(self) -> None:
        load_local_embeddings = self.settings.dense_backend == "local_numpy"
        self.bundle = ArtifactBundle.load(
            self.settings.artifact_dir,
            load_embeddings=load_local_embeddings,
        )

        config = self.bundle.config
        passage_ids = self.bundle.metadata["passage_id"].astype(str).tolist()

        encoder = QueryEncoder(
            model_path=str(self.settings.artifact_dir / config["dense_model_path"]),
            query_format=config["query_format"],
            query_instruction=config.get("query_instruction", ""),
            max_query_length=int(config["max_query_length"]),
            device=self.settings.device,
        )

        if self.settings.dense_backend == "local_numpy":
            if self.bundle.embeddings is None:
                raise RuntimeError("Local embeddings were not loaded.")
            dense = LocalNumpyDenseRetriever(
                embeddings=self.bundle.embeddings,
                passage_ids=passage_ids,
                min_score=self.settings.min_dense_score,
            )
        else:
            dense = SupabaseDenseRetriever(
                url=self.settings.supabase_url,
                service_role_key=self.settings.supabase_service_role_key,
                rpc_name=self.settings.supabase_match_rpc,
            )

        sparse = BM25Retriever(
            index_path=self.settings.artifact_dir / config["bm25_index_path"],
            passage_ids=passage_ids,
        )

        retriever = HybridRetriever(
            dense=dense,
            sparse=sparse,
            metadata=self.bundle.metadata,
            rrf_k=self.settings.rrf_k,
            dense_top_k=self.settings.dense_top_k,
            bm25_top_k=self.settings.bm25_top_k,
            final_top_k=self.settings.final_top_k,
        )

        self.generator = OllamaGenerator(
            base_url=self.settings.ollama_base_url,
            model=self.settings.ollama_model,
            num_ctx=self.settings.ollama_num_ctx,
            temperature=self.settings.ollama_temperature,
            top_p=self.settings.ollama_top_p,
            timeout_seconds=self.settings.ollama_timeout_seconds,
        )

        self.rag = RAGService(
            encoder=encoder,
            retriever=retriever,
            generator=self.generator,
        )

        logger.info(
            "Startup complete: corpus=%s, dense=%s, generator=ollama",
            len(self.bundle.metadata),
            self.settings.dense_backend,
        )

    async def shutdown(self) -> None:
        if self.generator is not None:
            await self.generator.close()

    async def health(self) -> HealthResponse:
        if self.bundle is None or self.rag is None or self.generator is None:
            return HealthResponse(
                status="degraded",
                dense_backend=self.settings.dense_backend,
                generator_backend="ollama",
                model_loaded=False,
                bm25_loaded=False,
                ollama_reachable=False,
                corpus_size=0,
            )

        ollama_reachable = await self.generator.health()
        return HealthResponse(
            status="ok" if ollama_reachable else "degraded",
            dense_backend=self.settings.dense_backend,
            generator_backend="ollama",
            model_loaded=True,
            bm25_loaded=True,
            ollama_reachable=ollama_reachable,
            corpus_size=len(self.bundle.metadata),
        )
