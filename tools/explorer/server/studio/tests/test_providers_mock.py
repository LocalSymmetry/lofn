"""Mock dialect record/replay determinism ($0 — no live calls)."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from studio.keys import KeyResolver
from studio.providers import (
    ProviderClient,
    ProviderRequest,
    ProviderResult,
    Usage,
    write_mock_fixture,
)


def _drain(client: ProviderClient, req: ProviderRequest):
    async def go():
        deltas = []
        final = None
        async for chunk in client.complete(req):
            if chunk.final is not None:
                final = chunk.final
            elif chunk.text:
                deltas.append(chunk.text)
        return deltas, final

    return asyncio.run(go())


def _req(model="claude-opus-4-8", content="hello world"):
    return ProviderRequest(
        model=model,
        messages=[{"role": "user", "content": content}],
        system="you are a test",
        max_tokens=64,
        params={"temperature": 0.5},
    )


def test_fingerprint_is_stable_and_key_independent():
    a = _req()
    b = _req()
    assert a.fingerprint() == b.fingerprint()
    # Different content -> different fingerprint.
    assert _req(content="other").fingerprint() != a.fingerprint()


def test_param_ordering_does_not_change_fingerprint():
    r1 = ProviderRequest(model="m", messages=[{"role": "user", "content": "x"}],
                         params={"a": 1, "b": 2})
    r2 = ProviderRequest(model="m", messages=[{"role": "user", "content": "x"}],
                         params={"b": 2, "a": 1})
    assert r1.fingerprint() == r2.fingerprint()


def test_replay_is_deterministic(tmp_fixtures: Path, repo_root: Path):
    req = _req()
    result = ProviderResult(
        text="The quick brown fox jumps over the lazy dog.",
        usage=Usage(in_=11, out=9),
        stop_reason="end_turn",
        model="claude-opus-4-8",
    )
    write_mock_fixture(tmp_fixtures, req, result)

    keys = KeyResolver.from_root(repo_root)
    client = ProviderClient("mock", keys, fixtures_dir=tmp_fixtures)

    d1, f1 = _drain(client, req)
    d2, f2 = _drain(client, req)

    # Reassembled text is identical and equals the fixture.
    assert "".join(d1) == result.text
    assert "".join(d1) == "".join(d2)
    assert f1.text == f2.text == result.text
    assert f1.usage.as_dict() == {"in": 11, "out": 9}
    assert f1.stop_reason == "end_turn"


def test_missing_fixture_raises(tmp_fixtures: Path, repo_root: Path):
    from studio.providers import ProviderError

    keys = KeyResolver.from_root(repo_root)
    client = ProviderClient("mock", keys, fixtures_dir=tmp_fixtures)
    with pytest.raises(ProviderError):
        _drain(client, _req(content="never recorded"))


def test_record_mode_writes_fixture_then_replays(tmp_fixtures: Path, repo_root: Path):
    """Simulate 'record a live run': write via write_mock_fixture, then a mock
    client replays the exact same result — the record→replay round-trip."""
    req = _req(model="gpt-5-mini", content="record me")
    result = ProviderResult(text="recorded output", usage=Usage(in_=3, out=2),
                            stop_reason="stop", model="gpt-5-mini", cost=0.00012)
    path = write_mock_fixture(tmp_fixtures, req, result)
    assert path.exists()

    keys = KeyResolver.from_root(repo_root)
    client = ProviderClient("mock", keys, fixtures_dir=tmp_fixtures)
    _, final = _drain(client, req)
    assert final.text == "recorded output"
    assert final.cost == 0.00012
