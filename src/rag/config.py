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
    retrieval_hybrid: bool
    retrieval_merge_twins: bool
    reranker_model: str  # empty = no reranker
    query_rewrite: bool
    split_questions: bool
    hide_superseded: bool
    # Phase 4: trust and security
    max_question_chars: int
    rate_limit_external: int   # requests per minute per visitor (IP address)
    rate_limit_internal: int   # requests per minute per signed-in user
    jwt_secret: str
    jwt_ttl_minutes: int
    users_path: Path
    security_dir: Path
    # Phase 5: tools
    mock_db_path: Path
    order_lookup_limit: int    # failed order verifications allowed per visitor per 10 minutes


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
        retrieval_hybrid=_flag("RETRIEVAL_HYBRID", True),
        retrieval_merge_twins=_flag("RETRIEVAL_MERGE_TWINS", True),
        reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3").strip(),
        query_rewrite=_flag("QUERY_REWRITE", True),
        split_questions=_flag("RETRIEVAL_SPLIT_QUESTIONS", True),
        hide_superseded=_flag("RETRIEVAL_HIDE_SUPERSEDED", True),
        max_question_chars=int(os.getenv("MAX_QUESTION_CHARS", "1000")),
        rate_limit_external=int(os.getenv("RATE_LIMIT_EXTERNAL", "10")),
        rate_limit_internal=int(os.getenv("RATE_LIMIT_INTERNAL", "30")),
        jwt_secret=os.getenv("JWT_SECRET", ""),
        jwt_ttl_minutes=int(os.getenv("JWT_TTL_MINUTES", "480")),
        users_path=PROJECT_ROOT / "data" / "mock" / "users.yaml",
        security_dir=storage_dir / "security",
        mock_db_path=storage_dir / "mock" / "brightnest.sqlite",
        order_lookup_limit=int(os.getenv("ORDER_LOOKUP_LIMIT", "5")),
    )


def _flag(name: str, default: bool) -> bool:
    """Read an on/off setting: 1/0, true/false, on/off, yes/no."""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in ("1", "true", "on", "yes")
