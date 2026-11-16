import pytest

from rag.answer import build_user_message, extract_citations
from rag.config import get_settings
from rag.profiles import ProfileError, load_profile

PROFILES_DIR = get_settings().profiles_dir


def test_external_profile_is_public_only():
    profile = load_profile("external", PROFILES_DIR)
    assert profile.auth == "anonymous"
    assert profile.collections == ("public",)


def test_internal_profile_has_no_restricted_access_yet():
    profile = load_profile("internal", PROFILES_DIR)
    assert "restricted" not in profile.collections


def test_anonymous_profile_with_internal_collection_is_rejected(tmp_path):
    (tmp_path / "bad.yaml").write_text(
        "name: bad\ndisplay_name: Bad\nauth: anonymous\ncollections: [public, internal]\n"
        "system_prompt: hi\nfallback_message: bye\n",
        encoding="utf-8",
    )
    with pytest.raises(ProfileError, match="anonymous"):
        load_profile("bad", tmp_path)


@pytest.mark.parametrize("text, n, expected", [
    ("Returns take 30 days [1].", 3, [1]),
    ("See [2][3] and again [2].", 3, [2, 3]),
    ("Grouped [1, 3].", 3, [1, 3]),
    ("Out of range [7] is ignored [1].", 3, [1]),
    ("No citations here.", 3, []),
])
def test_extract_citations(text, n, expected):
    assert extract_citations(text, n) == expected


def test_user_message_wraps_sources():
    msg = build_user_message("Hi?", [])
    assert msg.startswith("<sources>") and "Question: Hi?" in msg
