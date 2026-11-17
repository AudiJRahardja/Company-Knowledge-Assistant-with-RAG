"""Settings, read from environment variables (and the .env file in the project root)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    docs_dir: Path
    storage_dir: Path
    profiles_dir: Path
    log_path: Path
    embedding_model: str
    embedding_device: str
    llm_provider: str
    gemini_api_key: str
    gemini_model: str
    gemini_fallback_models: tuple[str, ...]
    gemini_timeout_s: int
    ollama_url: str
    ollama_model: str


def get_settings() -> Settings:
    storage_dir = PROJECT_ROOT / "storage"
    return Settings(
        docs_dir=PROJECT_ROOT / "data" / "docs",
        storage_dir=storage_dir,
        profiles_dir=PROJECT_ROOT / "profiles",
        log_path=storage_dir / "logs" / "chat_log.jsonl",
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        embedding_device=os.getenv("EMBEDDING_DEVICE", "auto"),
        llm_provider=os.getenv("LLM_PROVIDER", "gemini").lower(),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        gemini_fallback_models=tuple(
            m.strip()
            for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.1-flash-lite,gemini-3.8-flash").split(",")
            if m.strip()
        ),
        gemini_timeout_s=int(os.getenv("GEMINI_TIMEOUT_S", "30")),
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
    )
