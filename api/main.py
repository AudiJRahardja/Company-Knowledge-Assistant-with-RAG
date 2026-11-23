"""HTTP API for both assistants.

    uvicorn api.main:app --port 8000        then open http://localhost:8000 (demo shop with the chat widget),
                                            http://localhost:8000/web/internal.html (employee chat) or
                                            http://localhost:8000/docs (try the endpoints)

Endpoints
    GET  /health            is the server up, and which assistants it offers
    POST /auth/login        username + password -> token (internal assistant only)
    POST /chat/internal     needs "Authorization: Bearer <token>"; the token's role decides what is readable
    POST /chat/external     public, no login
    POST /feedback          thumbs up / down for one answer (by the answer_id in the chat response)
    GET  /web/...           the chat widget script (chat.js), the internal chat page and the demo shop page

Settings (Phase 6)
    ASSISTANTS=external     a server that only offers the customer assistant: no login, no internal chat.
                            Deployed next to an index that holds only public documents (see README).
    WIDGET_ORIGINS          other websites allowed to call the API from the browser (the embedded widget)
    TRUST_PROXY=1           behind a reverse proxy: take the visitor IP from X-Forwarded-For

Protection
    - rate limits: per visitor IP on /chat/external and /feedback, per user on /chat/internal, per IP on login
    - input limits: question length, number and size of history messages (HTTP 422 if exceeded)
    - everything in rag.guard and the pipeline (injection screening, output check, hand-offs)
    - answers are shown in the browser as text, never as HTML (web/chat.js), so an answer cannot inject a
      script into the page

The conversation history comes from the client, so a client could send fake "assistant" turns. That is why
the pipeline screens earlier user messages for injection and the output check runs on every answer; a
production version would keep the history on the server instead.
"""

import logging
import secrets
import threading
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag.auth import AuthError, User, UserDirectory, create_token, read_token
from rag.chatlog import ChatLog
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
    collection: str        # public | internal | restricted, or "tool" for a tool result
    last_updated: str
    cited: bool
    text: str              # the section itself, shown when the user clicks a citation


class ChatResponse(BaseModel):
    answer_id: str         # send this to POST /feedback
    answer: str
    status: str            # answered | not_found | needs_info | blocked_input | blocked_output
    language: str
    handoff: str           # who to contact when the assistant could not answer ("" otherwise)
    sources: list[Source]
    search_query: str
    tools_used: list[str]
    latency_ms: int
    model: str


class FeedbackRequest(BaseModel):
    answer_id: str = Field(min_length=32, max_length=32)
    rating: Literal[1, -1]  # thumbs up / down
    comment: str = Field(default="", max_length=500)


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
        answer_id=answer.id, answer=answer.text, status=answer.status, language=answer.language,
        handoff=answer.handoff,
        sources=[Source(number=i, title=h.title, section=h.section, collection=h.collection,
                        last_updated=h.last_updated, cited=i in answer.cited, text=h.text)
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

def create_app(settings: Settings | None = None, pool=None, users: UserDirectory | None = None,
               chat_log: ChatLog | None = None) -> FastAPI:
    """Build the app. Tests pass a fake pool, users and log; normally everything comes from settings."""
    s = settings or get_settings()
    pool = pool or AssistantPool(s)
    chat_log = chat_log or ChatLog(s.chat_db_path)
    internal = "internal" in s.assistants
    external = "external" in s.assistants
    users = users or (UserDirectory.from_file(s.users_path) if internal else None)
    secret = s.jwt_secret
    if internal and not secret:
        secret = secrets.token_urlsafe(32)
        log.warning("JWT_SECRET is not set: using a random one, so sign-ins end when the server restarts.")

    limits = {
        "external": RateLimiter(s.rate_limit_external),
        "internal": RateLimiter(s.rate_limit_internal),
        "login": RateLimiter(5),
        "feedback": RateLimiter(30),
    }
    app = FastAPI(title="Brightnest Assistants API", version="0.6")
    bearer = HTTPBearer(auto_error=False)
    if s.widget_origins:
        # Lets the listed websites call the API from the browser (the embedded customer widget). No cookies
        # are used anywhere (the internal chat sends its token in a header), so credentials stay off.
        app.add_middleware(CORSMiddleware, allow_origins=list(s.widget_origins), allow_methods=["POST"],
                           allow_headers=["Content-Type"], allow_credentials=False)

    def client_ip(request: Request) -> str:
        if s.trust_proxy:
            # Our proxy appends the address it saw, so the right-most entry is the one we can trust
            # (anything to the left of it was sent by the visitor and could be made up).
            forwarded = request.headers.get("x-forwarded-for", "")
            if forwarded.strip():
                return forwarded.split(",")[-1].strip()
        return request.client.host if request.client else "unknown"

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
        return {"status": "ok", "assistants": list(s.assistants)}

    @app.post("/feedback")
    def feedback(body: FeedbackRequest, request: Request) -> dict:
        rate_limit("feedback", client_ip(request))
        if not chat_log.add_feedback(body.answer_id, body.rating, body.comment):
            raise HTTPException(404, "Unknown answer.")
        return {"status": "ok"}

    if internal:
        @app.post("/auth/login", response_model=LoginResponse)
        def login(body: LoginRequest, request: Request) -> LoginResponse:
            rate_limit("login", client_ip(request))
            try:
                user = users.authenticate(body.username, body.password)
            except AuthError as exc:
                raise HTTPException(401, "Wrong username or password.") from exc
            token = create_token(user, secret, s.jwt_ttl_minutes)
            return LoginResponse(access_token=token, name=user.name, role=user.role)

        @app.post("/chat/internal", response_model=ChatResponse)
        def chat_internal(body: ChatRequest, user: User = Depends(current_user)) -> ChatResponse:
            rate_limit("internal", user.username)
            # The tools' "you" is the user from the signed token, never anything in the message.
            return answer("internal", user.role, body,
                          ToolContext(username=user.username, client_id=user.username))

    if external:
        @app.post("/chat/external", response_model=ChatResponse)
        def chat_external(body: ChatRequest, request: Request) -> ChatResponse:
            ip = client_ip(request)
            rate_limit("external", ip)
            return answer("external", None, body, ToolContext(client_id=ip))

    if s.web_dir.is_dir():
        # Static pages only. On an external-only server the internal page is still there but useless:
        # its endpoints don't exist.
        app.mount("/web", StaticFiles(directory=s.web_dir), name="web")

        @app.get("/", include_in_schema=False)
        def home() -> RedirectResponse:
            return RedirectResponse("/web/shop-demo.html" if external else "/web/internal.html")

    return app


app = create_app()
