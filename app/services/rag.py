from app.generation.ollama import OllamaGenerator
from app.models.schemas import AskResponse, SourceResult
from app.retrieval.encoder import QueryEncoder
from app.retrieval.hybrid import HybridRetriever


class RAGService:
    def __init__(
        self,
        encoder: QueryEncoder,
        retriever: HybridRetriever,
        generator: OllamaGenerator,
    ) -> None:
        self.encoder = encoder
        self.retriever = retriever
        self.generator = generator

    async def retrieve(self, question: str, top_k: int) -> list[SourceResult]:
        query_embedding = await self.encoder.encode(question)
        return await self.retriever.search(
            question=question,
            query_embedding=query_embedding,
            top_k=top_k,
        )

    async def answer(
        self,
        question: str,
        top_k: int,
        generate: bool,
    ) -> AskResponse:
        sources = await self.retrieve(question, top_k)

        if not generate:
            return AskResponse(
                question=question,
                answer=None,
                grounded=bool(sources),
                generator_backend=self.generator.backend_name,
                sources=sources,
            )

        if not sources:
            return AskResponse(
                question=question,
                answer="Không tìm thấy nguồn quy định phù hợp để trả lời.",
                grounded=False,
                generator_backend=self.generator.backend_name,
                sources=[],
            )

        answer = await self.generator.generate(question, sources)
        return AskResponse(
            question=question,
            answer=answer,
            grounded=True,
            generator_backend=self.generator.backend_name,
            sources=sources,
        )
