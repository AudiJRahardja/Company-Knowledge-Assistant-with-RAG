"""HTTP API for both assistants.

    uvicorn api.main:app --port 8000        then open http://localhost:8000/docs to try it

Endpoints
    GET  /health            is the server up
    POST /auth/login        username + password -> token (internal assistant only)
    POST /chat/internal     needs "Authorization: Bearer <token>"; the token's role decides what is readable
    POST /chat/external     public, no login

Protection
    - rate limits: per visitor IP on /chat/external, per user on /chat/internal, per IP on /auth/login
    - input limits: question length, number and size of history messages (HTTP 422 if exceeded)
    - everything in rag.guard and the pipeline (injection screening, output check, hand-offs)

The conversation history comes from the client, so a client could send fake "assistant" turns. That is why
the pipeline screens earlier user messages for injection and the output check runs on every answer; a
production version would keep the history on the server instead.
"""

import logging
import secrets
import threading
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from rag.auth import AuthError, User, UserDirectory, create_token, read_token
from rag.config import Settings, get_settings
from rag.guard import InputRejected, RateLimiter
from rag.llm import LLMError, Message
from rag.pipeline import Answer, Assistant, build_assistant
from rag.tools import ToolContext

log = logging.getLogger("api")

MAX_HISTORY = 6
MAX_HISTORY_CHARS = 2000


# ---------------------------------------------------------------- request / response shapes

class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=MAX_HISTORY_CHARS)


class ChatRequest(BaseModel):
    question: str
    history: list[HistoryMessage] = Field(default_factory=list, max_length=MAX_HISTORY)


class Source(BaseModel):
    number: int
    title: str
    section: str
    collection: str
    last_updated: str
    cited: bool


class ChatResponse(BaseModel):
    answer: str
    status: str            # answered | not_found | needs_info | blocked_input | blocked_output
    language: str
    handoff: str           # who to contact when the assistant could not answer ("" otherwise)
    sources: list[Source]
    search_query: str
    tools_used: list[str]
    latency_ms: int
    model: str


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    name: str
    role: str


def to_response(answer: Answer) -> ChatResponse:
    return ChatResponse(
        answer=answer.text, status=answer.status, language=answer.language, handoff=answer.handoff,
        sources=[Source(number=i, title=h.title, section=h.section, collection=h.collection,
                        last_updated=h.last_updated, cited=i in answer.cited)
                 for i, h in enumerate(answer.sources, start=1)],
        search_query=answer.search_query, tools_used=answer.tools_used, latency_ms=answer.latency_ms,
        model=answer.model,
    )


# ---------------------------------------------------------------- the assistants

class AssistantPool:
    """One assistant per (profile, role), created on first use. They share the embedding model, the
    reranker and the LLM client. One question is answered at a time (one GPU, local database files)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._assistants: dict[tuple[str, str | None], Assistant] = {}
        self._embedder = None
        self._llm = None
        self.lock = threading.Lock()

    def get(self, profile: str, role: str | None = None) -> Assistant:
        key = (profile, role)
        if key not in self._assistants:
            from rag.embeddings import Embedder
            from rag.llm import get_llm

            s = self.settings
            self._embedder = self._embedder or Embedder(s.embedding_model, s.embedding_device)
            self._llm = self._llm or get_llm(s)
            self._assistants[key] = build_assistant(profile, embedder=self._embedder, llm=self._llm,
                                                    settings=s, role=role)
        return self._assistants[key]

    def ask(self, profile: str, role: str | None, question: str, history: list[Message],
            context: ToolContext) -> Answer:
        with self.lock:
            return self.get(profile, role).ask(question, history, context)


# ---------------------------------------------------------------- app

def create_app(settings: Settings | None = None, pool=None, users: UserDirectory | None = None) -> FastAPI:
    """Build the app. Tests pass a fake pool and users; normally everything comes from settings."""
    s = settings or get_settings()
    users = users or UserDirectory.from_file(s.users_path)
    pool = pool or AssistantPool(s)
    secret = s.jwt_secret
    if not secret:
        secret = secrets.token_urlsafe(32)
        log.warning("JWT_SECRET is not set: using a random one, so sign-ins end when the server restarts.")

    limits = {
        "external": RateLimiter(s.rate_limit_external),
        "internal": RateLimiter(s.rate_limit_internal),
        "login": RateLimiter(5),
    }
    app = FastAPI(title="Brightnest Assistants API", version="0.4")
    bearer = HTTPBearer(auto_error=False)

    def rate_limit(name: str, key: str) -> None:
        if not limits[name].allow(key):
            wait = limits[name].retry_after(key)
            raise HTTPException(429, f"Too many requests. Please wait {wait} seconds.",
                                headers={"Retry-After": str(wait)})

    def current_user(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> User:
        if creds is None:
            raise HTTPException(401, "Sign in first (POST /auth/login).", headers={"WWW-Authenticate": "Bearer"})
        try:
            return read_token(creds.credentials, secret, users)
        except AuthError as exc:
            raise HTTPException(401, str(exc), headers={"WWW-Authenticate": "Bearer"}) from exc

    def answer(profile: str, role: str | None, body: ChatRequest, context: ToolContext) -> ChatResponse:
        history = [Message(m.role, m.content) for m in body.history]
        try:
            return to_response(pool.ask(profile, role, body.question, history, context))
        except InputRejected as exc:
            raise HTTPException(422, str(exc)) from exc
        except LLMError as exc:
            raise HTTPException(503, "The assistant is busy right now. Please try again in a minute.") from exc

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/auth/login", response_model=LoginResponse)
    def login(body: LoginRequest, request: Request) -> LoginResponse:
        rate_limit("login", client_ip(request))
        try:
            user = users.authenticate(body.username, body.password)
        except AuthError as exc:
            raise HTTPException(401, "Wrong username or password.") from exc
        token = create_token(user, secret, s.jwt_ttl_minutes)
        return LoginResponse(access_token=token, name=user.name, role=user.role)

    @app.post("/chat/external", response_model=ChatResponse)
    def chat_external(body: ChatRequest, request: Request) -> ChatResponse:
        ip = client_ip(request)
        rate_limit("external", ip)
        return answer("external", None, body, ToolContext(client_id=ip))

    @app.post("/chat/internal", response_model=ChatResponse)
    def chat_internal(body: ChatRequest, user: User = Depends(current_user)) -> ChatResponse:
        rate_limit("internal", user.username)
        # The tools' "you" is the user from the signed token, never anything in the message.
        return answer("internal", user.role, body, ToolContext(username=user.username, client_id=user.username))

    return app


def client_ip(request: Request) -> str:
    # Behind a reverse proxy this would read the X-Forwarded-For header set by that proxy (Phase 6).
    return request.client.host if request.client else "unknown"


app = create_app()
