from __future__ import annotations

import asyncio

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class QueryEncoder:
    def __init__(
        self,
        model_path: str,
        query_format: str,
        query_instruction: str = "",
        max_query_length: int = 128,
        device: str = "auto",
    ) -> None:
        resolved_device = self._resolve_device(device)

        model_kwargs = {}

        if resolved_device.startswith("cuda"):
            model_kwargs["torch_dtype"] = torch.bfloat16

        self.model = SentenceTransformer(
            model_path,
            device=resolved_device,
            model_kwargs=model_kwargs,
        )

        self.model.max_seq_length = max_query_length
        self.query_format = query_format
        self.query_instruction = query_instruction

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device

        if torch.cuda.is_available():
            return "cuda"

        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return "mps"

        return "cpu"

    def format_query(self, question: str) -> str:
        return self.query_format.format(
            question=question.strip(),
            instruction=self.query_instruction,
        )

    def encode_sync(self, question: str) -> np.ndarray:
        formatted_query = self.format_query(question)

        embedding = self.model.encode(
            [formatted_query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]

        return np.asarray(
            embedding,
            dtype=np.float32,
        )

    async def encode(self, question: str) -> np.ndarray:
        return await asyncio.to_thread(
            self.encode_sync,
            question,
        )