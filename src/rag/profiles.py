"""Assistant profiles: the YAML files that turn the one shared engine into two different assistants."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from rag.store import COLLECTIONS

AUTH_MODES = ("anonymous", "login")


@dataclass(frozen=True)
class Profile:
    name: str
    display_name: str
    description: str
    auth: str
    collections: tuple[str, ...]
    top_k: int
    system_prompt: str
    fallback_message: str


class ProfileError(ValueError):
    pass


def load_profile(name: str, profiles_dir: Path) -> Profile:
    path = profiles_dir / f"{name}.yaml"
    if not path.exists():
        raise ProfileError(f"no profile file at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    profile = Profile(
        name=data["name"],
        display_name=data["display_name"],
        description=data.get("description", ""),
        auth=data["auth"],
        collections=tuple(data["collections"]),
        top_k=int(data.get("top_k", 5)),
        system_prompt=data["system_prompt"].strip(),
        fallback_message=data["fallback_message"].strip(),
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
