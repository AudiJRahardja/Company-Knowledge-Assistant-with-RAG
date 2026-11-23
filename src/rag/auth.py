"""Mock login for the internal assistant: users with roles, hashed passwords, signed tokens (JWT).

- Users live in data/mock/users.yaml (demo accounts only). Passwords are stored as salted PBKDF2 hashes,
  never as plain text.
- After a successful login the user gets a JWT: a small signed token that says who they are and their
  role, valid for JWT_TTL_MINUTES. Every request to /chat/internal sends it back. The signature (HMAC with
  JWT_SECRET) means a user cannot edit their token to claim the "hr" role.
- The role in the token decides which documents the assistant may open (see profiles.py and store.py).

This is a demo login. A real company would use its single sign-on (Google Workspace / Azure AD) instead.
"""

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jwt
import yaml

from rag.profiles import ROLES

ALGORITHM = "HS256"
PBKDF2_ITERATIONS = 200_000


@dataclass(frozen=True)
class User:
    username: str
    name: str
    role: str


class AuthError(Exception):
    pass


def hash_password(password: str, salt: bytes | None = None, iterations: int = PBKDF2_ITERATIONS) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iterations, salt_hex, digest_hex = stored.split("$")
    except ValueError:
        return False
    if scheme != "pbkdf2_sha256":
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations))
    return hmac.compare_digest(digest.hex(), digest_hex)  # constant time: no timing hints


class UserDirectory:
    def __init__(self, records: dict[str, dict]):
        for username, rec in records.items():
            if rec.get("role") not in ROLES:
                raise ValueError(f"user {username!r} has unknown role {rec.get('role')!r}")
        self._records = records

    @classmethod
    def from_file(cls, path: Path) -> "UserDirectory":
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(data.get("users", {}))

    def authenticate(self, username: str, password: str) -> User:
        rec = self._records.get(username)
        # Check a dummy hash for unknown users too, so the response time doesn't reveal which usernames exist.
        stored = rec["password_hash"] if rec else hash_password("dummy", b"0" * 16)
        if not verify_password(password, stored) or rec is None:
            raise AuthError("wrong username or password")
        return User(username=username, name=rec["name"], role=rec["role"])

    def all(self) -> list[User]:
        return [User(username=u, name=r["name"], role=r["role"]) for u, r in self._records.items()]

    def get(self, username: str) -> User | None:
        rec = self._records.get(username)
        return User(username=username, name=rec["name"], role=rec["role"]) if rec else None


def create_token(user: User, secret: str, ttl_minutes: int) -> str:
    now = datetime.now(timezone.utc)
    claims = {"sub": user.username, "name": user.name, "role": user.role,
              "iat": now, "exp": now + timedelta(minutes=ttl_minutes)}
    return jwt.encode(claims, secret, algorithm=ALGORITHM)


def read_token(token: str, secret: str, users: UserDirectory) -> User:
    """Check the signature and expiry, and that the user still exists with the same role."""
    try:
        claims = jwt.decode(token, secret, algorithms=[ALGORITHM])  # only HS256: no "alg: none" tricks
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("the session has expired, please sign in again") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthError("invalid token") from exc
    user = users.get(claims.get("sub", ""))
    if user is None or user.role != claims.get("role"):
        raise AuthError("this account no longer exists or its role changed, please sign in again")
    return user
