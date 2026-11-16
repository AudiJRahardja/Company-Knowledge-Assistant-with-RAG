"""One small interface for the language model, with two backends: Gemini (API) and Ollama (local).

The rest of the code only calls `llm.generate(system, messages)`, so switching providers is a one-line
change in .env (LLM_PROVIDER=gemini or ollama).

Quick check:  python -m rag.llm            (sends a test prompt to the configured provider)
              python -m rag.llm --models   (lists Gemini models your key can use)
"""

import sys
from dataclasses import dataclass
from typing import Literal, Protocol

import httpx

from rag.config import Settings, get_settings

TEMPERATURE = 0.2  # low = more factual and consistent, which is what we want for policy answers


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    content: str


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    name: str

    def generate(self, system: str, messages: list[Message]) -> str: ...


class GeminiClient:
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise LLMError("GEMINI_API_KEY is empty. Add it to your .env file.")
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.name = f"gemini:{model}"

    def generate(self, system: str, messages: list[Message]) -> str:
        from google.genai import types

        contents = [
            types.Content(role="model" if m.role == "assistant" else "user", parts=[types.Part(text=m.content)])
            for m in messages
        ]
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=system, temperature=TEMPERATURE),
            )
        except Exception as exc:  # the SDK raises several error types; we show one clear message
            raise LLMError(f"Gemini request failed: {exc}") from exc
        return (response.text or "").strip()

    def list_models(self) -> list[str]:
        return [m.name for m in self._client.models.list()]


class OllamaClient:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = f"ollama:{model}"

    def generate(self, system: str, messages: list[Message]) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {"temperature": TEMPERATURE},
        }
        try:
            r = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=180)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Ollama request failed ({exc}). Is Ollama running and is '{self.model}' pulled?"
            ) from exc
        return r.json()["message"]["content"].strip()


def get_llm(settings: Settings | None = None) -> LLMClient:
    s = settings or get_settings()
    if s.llm_provider == "gemini":
        return GeminiClient(s.gemini_api_key, s.gemini_model)
    if s.llm_provider == "ollama":
        return OllamaClient(s.ollama_url, s.ollama_model)
    raise LLMError(f"unknown LLM_PROVIDER {s.llm_provider!r}; use 'gemini' or 'ollama'")


if __name__ == "__main__":
    llm = get_llm()
    if "--models" in sys.argv:
        if not isinstance(llm, GeminiClient):
            sys.exit("--models only works with LLM_PROVIDER=gemini")
        print("\n".join(llm.list_models()))
    else:
        print(f"Provider: {llm.name}")
        print(llm.generate("Reply in one short sentence.", [Message("user", "Say hello in Indonesian.")]))
