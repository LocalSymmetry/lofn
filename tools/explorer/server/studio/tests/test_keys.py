"""Key-alias resolution: standard names AND legacy config.yaml aliases; env
before .env.studio; no key material ever surfaced by configured_map."""
from __future__ import annotations

from pathlib import Path

from studio.keys import KeyResolver, PROVIDER_KEY_NAMES


def _resolver(repo_root: Path, env=None, dotenv=None) -> KeyResolver:
    return KeyResolver(repo_root=repo_root, env=env or {}, dotenv=dotenv or {})


def test_standard_names_resolve(repo_root: Path):
    env = {
        "ANTHROPIC_API_KEY": "a1",
        "OPENAI_API_KEY": "o1",
        "OPENROUTER_API_KEY": "r1",
        "POE_API_KEY": "p1",
        "GEMINI_API_KEY": "g1",
    }
    r = _resolver(repo_root, env=env)
    assert r.resolve("anthropic") == "a1"
    assert r.resolve("openai") == "o1"
    assert r.resolve("openrouter") == "r1"
    assert r.resolve("poe") == "p1"
    assert r.resolve("gemini") == "g1"


def test_legacy_aliases_resolve(repo_root: Path):
    env = {
        "ANTHROPIC_API": "a2",
        "OPENAI_API": "o2",
        "OPEN_ROUTER_API_KEY": "r2",
        "POE_API": "p2",
        "GOOGLE_API_KEY": "g2",
    }
    r = _resolver(repo_root, env=env)
    assert r.resolve("anthropic") == "a2"
    assert r.resolve("openai") == "o2"
    assert r.resolve("openrouter") == "r2"
    assert r.resolve("poe") == "p2"
    assert r.resolve("gemini") == "g2"


def test_standard_name_wins_over_legacy_alias(repo_root: Path):
    env = {"ANTHROPIC_API_KEY": "standard", "ANTHROPIC_API": "legacy"}
    r = _resolver(repo_root, env=env)
    assert r.resolve("anthropic") == "standard"


def test_env_beats_dotenv(repo_root: Path):
    r = _resolver(repo_root, env={"ANTHROPIC_API_KEY": "from-env"},
                  dotenv={"ANTHROPIC_API_KEY": "from-file"})
    assert r.resolve("anthropic") == "from-env"


def test_dotenv_used_when_env_absent(repo_root: Path):
    r = _resolver(repo_root, env={}, dotenv={"POE_API": "from-file"})
    assert r.resolve("poe") == "from-file"


def test_empty_string_is_not_configured(repo_root: Path):
    r = _resolver(repo_root, env={"OPENAI_API_KEY": ""})
    assert r.resolve("openai") is None
    assert r.is_configured("openai") is False


def test_mock_is_keyless_and_configured(repo_root: Path):
    r = _resolver(repo_root, env={})
    assert r.resolve("mock") is None
    assert r.is_configured("mock") is True


def test_configured_map_has_no_key_material(repo_root: Path):
    env = {"ANTHROPIC_API_KEY": "supersecret"}
    r = _resolver(repo_root, env=env)
    m = r.configured_map()
    assert m["anthropic"] is True
    assert m["openai"] is False
    assert m["mock"] is True
    # The map is bools only — no key strings anywhere in it.
    for v in m.values():
        assert isinstance(v, bool)


def test_dotenv_file_parsing(tmp_path: Path):
    p = tmp_path / ".env.studio"
    p.write_text(
        "# a comment\n"
        "export ANTHROPIC_API_KEY=sk-ant-fromfile\n"
        "OPENAI_API='quoted-value'\n"
        "\n"
        'GEMINI_API_KEY = "spaced"\n',
        encoding="utf-8",
    )
    r = KeyResolver.from_root(tmp_path, env={})
    assert r.resolve("anthropic") == "sk-ant-fromfile"
    assert r.resolve("openai") == "quoted-value"
    assert r.resolve("gemini") == "spaced"


def test_every_provider_has_key_names():
    # Contract check: the five real providers each have a standard + legacy name.
    for provider in ("anthropic", "openai", "openrouter", "poe", "gemini"):
        names = PROVIDER_KEY_NAMES[provider]
        assert len(names) >= 2, provider
