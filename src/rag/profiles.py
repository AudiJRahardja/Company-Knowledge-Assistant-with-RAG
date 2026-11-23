"""Assistant profiles: the YAML files that turn the one shared engine into two different assistants."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from rag.store import COLLECTIONS
from rag.tools import SPECS

AUTH_MODES = ("anonymous", "login")
ROLES = ("employee", "manager", "hr")
LANGUAGES = ("en", "id")


@dataclass(frozen=True)
class Profile:
    name: str
    display_name: str
    description: str
    auth: str
    collections: tuple[str, ...]          # what every user of this assistant may search
    top_k: int
    system_prompt: str
    fallback_message: dict[str, str]      # "where to get help" text, per language (en, id)
    role_collections: dict[str, tuple[str, ...]] = field(default_factory=dict)  # extra collections per role
    tools: tuple[str, ...] = ()           # tool names from rag.tools.SPECS

    def collections_for(self, role: str | None) -> tuple[str, ...]:
        """All collections a user with this role may search."""
        if self.auth == "login" and role not in ROLES:
            raise ProfileError(f"{self.name}: unknown role {role!r}; expected one of {ROLES}")
        extra = self.role_collections.get(role, ()) if role else ()
        return self.collections + tuple(c for c in extra if c not in self.collections)

    def fallback(self, language: str) -> str:
        return self.fallback_message.get(language) or self.fallback_message["en"]


class ProfileError(ValueError):
    pass


def load_profile(name: str, profiles_dir: Path) -> Profile:
    path = profiles_dir / f"{name}.yaml"
    if not path.exists():
        raise ProfileError(f"no profile file at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    fallback = data["fallback_message"]
    if isinstance(fallback, str):
        fallback = {"en": fallback}
    profile = Profile(
        name=data["name"],
        display_name=data["display_name"],
        description=data.get("description", ""),
        auth=data["auth"],
        collections=tuple(data["collections"]),
        top_k=int(data.get("top_k", 5)),
        system_prompt=data["system_prompt"].strip(),
        fallback_message={lang: text.strip() for lang, text in fallback.items()},
        role_collections={role: tuple(cols) for role, cols in (data.get("role_collections") or {}).items()},
        tools=tuple(data.get("tools") or ()),
    )
    _validate(profile)
    return profile


def _validate(p: Profile) -> None:
    if p.auth not in AUTH_MODES:
        raise ProfileError(f"{p.name}: auth must be one of {AUTH_MODES}")
    unknown = set(p.collections) - set(COLLECTIONS)
    if unknown:
        raise ProfileError(f"{p.name}: unknown collections {sorted(unknown)}")
    # Config-level guard: an assistant for anonymous users can never be pointed at non-public data,
    # even by a typo in the YAML file.
    if p.auth == "anonymous" and p.collections != ("public",):
        raise ProfileError(f"{p.name}: anonymous assistants may only use the 'public' collection")
    if not 1 <= p.top_k <= 20:
        raise ProfileError(f"{p.name}: top_k must be between 1 and 20")
    unknown_tools = set(p.tools) - set(SPECS)
    if unknown_tools:
        raise ProfileError(f"{p.name}: unknown tools {sorted(unknown_tools)}")
    # Tools that read "your own" employee data need a signed-in user.
    if p.auth == "anonymous" and any(SPECS[t].needs_user for t in p.tools):
        raise ProfileError(f"{p.name}: anonymous assistants can't have tools that need a signed-in employee")
    if "en" not in p.fallback_message or set(p.fallback_message) - set(LANGUAGES):
        raise ProfileError(f"{p.name}: fallback_message needs an 'en' text (and optionally 'id')")
    # Restricted data is only ever reachable through a role, never for every user of an assistant.
    if "restricted" in p.collections:
        raise ProfileError(f"{p.name}: 'restricted' can only be granted per role (role_collections)")
    if p.role_collections and p.auth != "login":
        raise ProfileError(f"{p.name}: role_collections need auth: login")
    for role, cols in p.role_collections.items():
        if role not in ROLES:
            raise ProfileError(f"{p.name}: unknown role {role!r} in role_collections")
        if set(cols) - set(COLLECTIONS):
            raise ProfileError(f"{p.name}: unknown collections for role {role!r}")
