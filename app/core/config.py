from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "School Regulation RAG"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:5173",
        ]
    )

    artifact_dir: Path = Path("./artifacts/hybrid_retriever_e5")
    device: str = "auto"

    dense_backend: Literal["local_numpy", "supabase"] = "local_numpy"
    dense_top_k: int = 30
    bm25_top_k: int = 30
    final_top_k: int = 8
    rrf_k: int = 60
    min_dense_score: float = -1.0

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    ollama_num_ctx: int = 8192
    ollama_temperature: float = 0.1
    ollama_top_p: float = 0.9
    ollama_timeout_seconds: int = 120

    supabase_url: str = ""
    supabase_service_role_key: str = ""
    supabase_table: str = "regulation_chunks"
    supabase_match_rpc: str = "match_regulation_chunks"

    @field_validator("artifact_dir")
    @classmethod
    def resolve_artifact_dir(cls, value: Path) -> Path:
        return value.expanduser().resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
