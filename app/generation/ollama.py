from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.generation.prompt import SYSTEM_PROMPT, build_user_prompt
from app.models.schemas import SourceResult


class OllamaGenerator:
    backend_name = "ollama"

    def __init__(
        self,
        base_url: str,
        model: str,
        num_ctx: int,
        temperature: float,
        top_p: float,
        timeout_seconds: int,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.top_p = top_p
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(float(timeout_seconds)))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def generate(self, question: str, sources: list[SourceResult]) -> str:
        response = await self.client.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_user_prompt(question, sources),
                    },
                ],
                "options": {
                    "num_ctx": self.num_ctx,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                },
            },
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload["message"]["content"]).strip()

    async def health(self) -> bool:
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        await self.client.aclose()
