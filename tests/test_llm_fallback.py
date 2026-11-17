"""Retry and fallback behaviour of GeminiClient, tested with a fake API (no network needed)."""

import pytest
from google.genai import errors

import rag.llm as llm_module
from rag.llm import GeminiClient, LLMError, Message


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    """Plays back a script of outcomes, one per call: an exception to raise, or text to return."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        outcome = self.script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return FakeResponse(outcome)


def busy():
    return errors.ServerError(503, {"error": {"message": "high demand", "status": "UNAVAILABLE"}})


def make_client(monkeypatch, script):
    monkeypatch.setattr(llm_module, "RETRY_DELAY_S", 0)
    client = GeminiClient("fake-key", "main-model", fallback_models=("backup-model",))
    fake = FakeModels(script)
    client._client = type("FakeGenaiClient", (), {"models": fake})()
    return client, fake


def ask(client):
    return client.generate("system", [Message("user", "hi")])


def test_retries_then_succeeds_on_same_model(monkeypatch):
    client, fake = make_client(monkeypatch, [busy(), "hello"])
    assert ask(client) == "hello"
    assert fake.calls == ["main-model", "main-model"]
    assert client.last_model == "main-model"


def test_falls_back_to_backup_model_when_main_stays_busy(monkeypatch):
    client, fake = make_client(monkeypatch, [busy(), TimeoutError(), "from backup"])
    assert ask(client) == "from backup"
    assert fake.calls == ["main-model", "main-model", "backup-model"]
    assert client.last_model == "backup-model"


def test_timeout_skips_to_backup_without_retrying(monkeypatch):
    client, fake = make_client(monkeypatch, [TimeoutError(), "from backup"])
    assert ask(client) == "from backup"
    assert fake.calls == ["main-model", "backup-model"]


def test_retired_model_is_skipped(monkeypatch):
    gone = errors.ClientError(404, {"error": {"message": "no longer available", "status": "NOT_FOUND"}})
    client, fake = make_client(monkeypatch, [gone, "from backup"])
    assert ask(client) == "from backup"
    assert fake.calls == ["main-model", "backup-model"]  # no pointless retry of a missing model


def test_non_retryable_error_fails_immediately(monkeypatch):
    bad_key = errors.ClientError(400, {"error": {"message": "API key not valid", "status": "INVALID_ARGUMENT"}})
    client, fake = make_client(monkeypatch, [bad_key])
    with pytest.raises(LLMError, match="API key not valid"):
        ask(client)
    assert fake.calls == ["main-model"]


def test_clear_error_when_everything_is_busy(monkeypatch):
    client, fake = make_client(monkeypatch, [busy()] * 4)
    with pytest.raises(LLMError, match="busy or unreachable"):
        ask(client)
    assert len(fake.calls) == 4
