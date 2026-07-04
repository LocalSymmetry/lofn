"""Redaction with canary keys (acceptance gate S2). No planted key survives."""
from __future__ import annotations

from studio.keys import redact, redact_with_values, REDACTED

# Canary keys shaped like each provider's real format.
CANARIES = {
    "anthropic": "sk-ant-api03-" + "A" * 40,
    "anthropic_oauth": "sk-ant-oat01-" + "B" * 40,
    "openai": "sk-proj-" + "C" * 40,
    "openrouter": "sk-or-v1-" + "d" * 40,
    "gemini": "AIza" + "E" * 35,
}


def test_all_canary_shapes_scrubbed_from_string():
    for name, key in CANARIES.items():
        text = f"call failed with key {key} in the traceback"
        out = redact(text)
        assert key not in out, f"{name} canary leaked"
        assert REDACTED in out


def test_env_style_assignment_scrubbed():
    text = "ANTHROPIC_API_KEY=sk-ant-secretvalue123456789 loaded"
    out = redact(text)
    assert "sk-ant-secretvalue123456789" not in out
    assert "ANTHROPIC_API_KEY" in out   # keep the name, drop the value
    assert REDACTED in out


def test_legacy_alias_assignment_scrubbed():
    text = "POE_API = 'poe-opaque-token-abcdef0123456789'"
    out = redact(text)
    assert "poe-opaque-token-abcdef0123456789" not in out
    assert "POE_API" in out


def test_dict_values_by_known_key_name_redacted():
    obj = {
        "ANTHROPIC_API_KEY": "sk-ant-whatever",
        "api_key": "anything-here-at-all",
        "authorization": "Bearer sk-xxxxxxxxxxxxxxxxxxxx",
        "note": "clean text stays",
        "nested": {"OPENAI_API_KEY": "sk-proj-nested-secret"},
    }
    out = redact(obj)
    assert out["ANTHROPIC_API_KEY"] == REDACTED
    assert out["api_key"] == REDACTED
    assert out["authorization"] == REDACTED
    assert out["note"] == "clean text stays"
    assert out["nested"]["OPENAI_API_KEY"] == REDACTED


def test_redact_with_literal_values_catches_opaque_tokens():
    # A Poe/OpenRouter token that doesn't match a known shape must still be
    # scrubbed when we know the literal value (runlog/SSE path).
    opaque = "totally-opaque-token-9f8e7d6c5b4a3210"
    text = f"provider returned: {opaque} (ok)"
    out = redact_with_values(text, [opaque])
    assert opaque not in out
    assert REDACTED in out


def test_lists_are_recursed():
    obj = ["clean", "sk-ant-api03-" + "Z" * 40, {"api_key": "leak"}]
    out = redact(obj)
    assert out[0] == "clean"
    assert "sk-ant-api03-" not in out[1]
    assert out[2]["api_key"] == REDACTED


def test_scalars_pass_through():
    assert redact(42) == 42
    assert redact(True) is True
    assert redact(None) is None
