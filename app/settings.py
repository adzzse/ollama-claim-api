import os
from functools import lru_cache

from pydantic import BaseModel

from app.env import load_runtime_env


DEFAULT_MODEL = "evidencopilot:latest"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"


class Settings(BaseModel):
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    ollama_model: str = DEFAULT_MODEL
    ollama_embedding_model: str = DEFAULT_EMBEDDING_MODEL

    @classmethod
    def from_env(cls) -> "Settings":
        load_runtime_env()
        return cls(
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
            ollama_model=os.getenv("OLLAMA_MODEL", DEFAULT_MODEL),
            ollama_embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        )


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    return Settings.from_env()
