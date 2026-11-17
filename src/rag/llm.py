"""One small interface for the language model, with two backends: Gemini (API) and Ollama (local).

The rest of the code only calls `llm.generate(system, messages)`, so switching providers is a one-line
change in .env (LLM_PROVIDER=gemini or ollama).

Quick check:  python -m rag.llm            (sends a test prompt to the configured provider)
              python -m rag.llm --models   (lists Gemini models your key can use)
"""

import sys
import time
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
    last_model: str

    def generate(self, system: str, messages: list[Message]) -> str: ...


RETRYABLE_STATUS = {429, 500, 502, 503}  # rate limited, or busy for a moment: worth one more try
SKIP_STATUS = {404, 504}  # model retired (404) or too slow (504): go straight to the next model
RETRY_DELAY_S = 2.0


class GeminiClient:
    """Gemini with a time limit, one quick retry, and backup models.

    Popular models are sometimes overloaded (HTTP 503 "high demand"). Without a time limit the request can
    hang for minutes. So each attempt gets `timeout_s` seconds; if the model is busy we retry once, then
    move on to the next model in `fallback_models`. Errors that retrying can't fix (for example an invalid
    API key) are raised immediately.
    """

    def __init__(self, api_key: str, model: str, fallback_models: tuple[str, ...] = (), timeout_s: int = 30,
                 attempts_per_model: int = 2):
        if not api_key:
            raise LLMError("GEMINI_API_KEY is empty. Add it to your .env file.")
        from google import genai
        from google.genai import types

        self._client = genai.Client(
            api_key=api_key,
            # attempts=1 turns off the SDK's own hidden retries so our retry logic below is the only one.
            http_options=types.HttpOptions(
                timeout=timeout_s * 1000, retry_options=types.HttpRetryOptions(attempts=1)
            ),
        )
        self.models = (model,) + tuple(m for m in fallback_models if m != model)
        self.attempts_per_model = attempts_per_model
        self.name = f"gemini:{model}"
        self.last_model = model  # which model produced the most recent answer

    def generate(self, system: str, messages: list[Message]) -> str:
        from google.genai import errors, types

        contents = [
            types.Content(role="model" if m.role == "assistant" else "user", parts=[types.Part(text=m.content)])
            for m in messages
        ]
        config = types.GenerateContentConfig(system_instruction=system, temperature=TEMPERATURE)
        failures = []
        for model in self.models:
            for attempt in range(1, self.attempts_per_model + 1):
                try:
                    response = self._client.models.generate_content(
                        model=model, contents=contents, config=config
                    )
                    self.last_model = model
                    return (response.text or "").strip()
                except errors.APIError as exc:
                    if exc.code in SKIP_STATUS:
                        failures.append(f"{model}: HTTP {exc.code}")
                        break
                    if exc.code not in RETRYABLE_STATUS:
                        raise LLMError(f"Gemini request failed ({model}): {exc}") from exc
                    failures.append(f"{model} try {attempt}: HTTP {exc.code}")
                except Exception as exc:  # our timeout or a dropped connection: this model is struggling
                    failures.append(f"{model}: {type(exc).__name__}")
                    break
                if attempt < self.attempts_per_model:
                    time.sleep(RETRY_DELAY_S)
        raise LLMError(
            "Gemini is busy or unreachable right now. Please try again in a minute. "
            f"(Tried: {'; '.join(failures)})"
        )

    def list_models(self) -> list[str]:
        return [m.name for m in self._client.models.list()]


class OllamaClient:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = f"ollama:{model}"
        self.last_model = model

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
        return GeminiClient(s.gemini_api_key, s.gemini_model, s.gemini_fallback_models, s.gemini_timeout_s)
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
