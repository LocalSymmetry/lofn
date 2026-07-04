"""ProviderClient — the studio's execution surface across five providers + mock.

The engine (WS3) speaks ONE async interface to every provider:

    async for chunk in client.complete(ProviderRequest(...)):
        ...            # StreamChunk(text=...) as tokens arrive
    # the final chunk is a StreamChunk(final=ProviderResult(text, usage, stop_reason))

Dialects (plan §7, decisions §13):

  * ``anthropic``     — native Anthropic SDK. Adaptive thinking on 4.6+ per the
                        current claude-api conventions ({"type": "adaptive"}, NOT
                        the legacy budget_tokens). Streaming. ``count_tokens`` for
                        estimates. A capability table STRIPS unsupported params
                        (e.g. temperature on 4.7+/Fable) instead of 400ing.
  * ``openai-compat`` — native OpenAI SDK; base_url covers OpenAI + OpenRouter (+
                        optionally the Gemini OpenAI endpoint). OpenRouter adds
                        Bearer + HTTP-Referer / X-Title headers and PREFERS the
                        per-request cost it returns.
  * ``poe``           — fastapi_poe ``fp.get_bot_response`` (ported from the legacy
                        PoeLLM). bot_name = model minus the "Poe-" prefix. NO stop
                        sequences.
  * ``gemini``        — native google.genai ``generate_content`` (ported from the
                        legacy GeminiLLM). Verified as the primary Gemini path.
  * ``mock``          — record/replay from ``studio/fixtures/`` + a record mode
                        that captures a live run into fixtures. All engine/CI tests
                        run on mock — CI spends $0.

Keys never appear in logs, run.json, SSE, or any HTTP response — resolution lives
in ``keys.py`` and a redaction filter guards every text path. A key only travels
to its own provider's endpoint.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional

from .keys import KeyResolver

# ---------------------------------------------------------------------------
# The unified request / response contract.
# ---------------------------------------------------------------------------


@dataclass
class ProviderRequest:
    """One text-to-text completion request (no tool use, no structured output)."""

    model: str
    messages: List[Dict[str, str]]          # [{"role": "user"|"assistant", "content": str}]
    system: Optional[str] = None
    max_tokens: int = 4096
    stream: bool = True
    params: Dict[str, object] = field(default_factory=dict)   # temperature, thinking, ...

    def fingerprint(self) -> str:
        """Stable hash of the semantically-load-bearing request fields.

        Drives mock fixture lookup: same model+system+messages+max_tokens+params
        -> same fixture. Keys are never part of this (they're not on the request).
        """
        payload = {
            "model": self.model,
            "system": self.system or "",
            "messages": self.messages,
            "max_tokens": self.max_tokens,
            "params": _canonical(self.params),
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class Usage:
    """Token usage. ``in_`` / ``out`` chosen so the JSON is {"in":..,"out":..}."""

    in_: int = 0
    out: int = 0

    def as_dict(self) -> dict:
        return {"in": self.in_, "out": self.out}


@dataclass
class ProviderResult:
    """The terminal result of a completion."""

    text: str
    usage: Usage
    stop_reason: Optional[str] = None
    model: Optional[str] = None
    # OpenRouter (and any provider that reports it) returns an actual per-request
    # cost — PREFER it over the pricing.yaml estimate when present.
    cost: Optional[float] = None
    raw: Optional[dict] = None

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "usage": self.usage.as_dict(),
            "stop_reason": self.stop_reason,
            "model": self.model,
            "cost": self.cost,
        }


@dataclass
class StreamChunk:
    """One streamed unit. Either an incremental ``text`` delta, or the ``final``
    ProviderResult on the last chunk (text may repeat the full text there)."""

    text: str = ""
    final: Optional[ProviderResult] = None


def _canonical(obj):
    """JSON-canonicalize params so equivalent dicts fingerprint identically."""
    if isinstance(obj, dict):
        return {k: _canonical(obj[k]) for k in sorted(obj)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(x) for x in obj]
    return obj


# ---------------------------------------------------------------------------
# Provider → dialect routing. A model id carries a prefix that names its dialect
# (heritage: model_defaults.yaml uses Poe- / OR- prefixes). The studio also lets
# the caller name the provider explicitly on the request (via params["provider"]).
# ---------------------------------------------------------------------------

def dialect_for(provider: str) -> str:
    return {
        "anthropic": "anthropic",
        "openai": "openai-compat",
        "openrouter": "openai-compat",
        "gemini": "gemini",
        "poe": "poe",
        "mock": "mock",
    }.get(provider, provider)


# ---------------------------------------------------------------------------
# Anthropic capability table — which request params each model family accepts.
# The dialect STRIPS unsupported params (per claude-api: temperature/top_p/top_k
# and budget_tokens are removed on 4.7+/Fable and 400 if sent), rather than
# passing them through and eating a 400.
# ---------------------------------------------------------------------------

# Models on the modern surface: adaptive thinking only, NO sampling params.
_ANTHROPIC_NO_SAMPLING = (
    "claude-opus-4-8", "claude-opus-4-7",
    "claude-sonnet-5",
    "claude-fable-5", "claude-mythos-5", "claude-mythos-preview",
)
# Models where adaptive thinking is available (4.6+).
_ANTHROPIC_ADAPTIVE_THINKING = (
    "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6",
    "claude-sonnet-5", "claude-sonnet-4-6",
    "claude-fable-5", "claude-mythos-5", "claude-mythos-preview",
)
# Fable rejects an explicit thinking:{type:"disabled"} — omit thinking entirely.
_ANTHROPIC_THINKING_ALWAYS_ON = (
    "claude-fable-5", "claude-mythos-5", "claude-mythos-preview",
)


def _anthropic_kwargs(req: ProviderRequest) -> dict:
    """Build the create() kwargs, stripping params the model would 400 on."""
    model = req.model
    kwargs: dict = {
        "model": model,
        "max_tokens": req.max_tokens,
        "messages": req.messages,
    }
    if req.system:
        kwargs["system"] = req.system

    params = dict(req.params or {})
    # provider/routing hints are not API params
    for internal in ("provider", "tier"):
        params.pop(internal, None)

    supports_sampling = not any(model.startswith(m) for m in _ANTHROPIC_NO_SAMPLING)
    if supports_sampling:
        for p in ("temperature", "top_p", "top_k"):
            if p in params:
                kwargs[p] = params.pop(p)
    else:
        for p in ("temperature", "top_p", "top_k"):
            params.pop(p, None)   # strip — would 400

    # Thinking: adaptive on 4.6+; strip legacy budget_tokens.
    thinking = params.pop("thinking", None)
    params.pop("budget_tokens", None)
    supports_adaptive = any(model.startswith(m) for m in _ANTHROPIC_ADAPTIVE_THINKING)
    always_on = any(model.startswith(m) for m in _ANTHROPIC_THINKING_ALWAYS_ON)
    if supports_adaptive and not always_on:
        # Default to adaptive thinking on the modern models unless the caller
        # explicitly disabled it. Fable is always-on — omit the field entirely.
        if thinking is None:
            kwargs["thinking"] = {"type": "adaptive"}
        elif isinstance(thinking, dict):
            kwargs["thinking"] = thinking
        elif thinking is False:
            kwargs["thinking"] = {"type": "disabled"}
        elif thinking is True:
            kwargs["thinking"] = {"type": "adaptive"}

    # effort lives under output_config
    effort = params.pop("effort", None)
    if effort:
        kwargs["output_config"] = {"effort": effort}

    # anything left over is passed through (forward-compat)
    kwargs.update(params)
    return kwargs


# ---------------------------------------------------------------------------
# The client.
# ---------------------------------------------------------------------------


class ProviderError(RuntimeError):
    """A provider call failed (redaction is the caller's responsibility)."""


class ProviderClient:
    """Async, streaming, key-safe client across all supported providers."""

    def __init__(
        self,
        provider: str,
        keys: KeyResolver,
        *,
        fixtures_dir: Optional[Path] = None,
        record: bool = False,
        record_target: Optional[str] = None,
    ):
        """``provider`` is the provider id (anthropic/openai/openrouter/gemini/
        poe/mock). ``fixtures_dir`` is where the mock dialect reads/writes; for a
        real dialect in ``record`` mode, captured calls are written there too."""
        self.provider = provider
        self.dialect = dialect_for(provider)
        self.keys = keys
        self.fixtures_dir = fixtures_dir
        self.record = record
        # When recording a real provider, this is the underlying real dialect.
        self.record_target = record_target or provider

    # -- key access (never logged) -----------------------------------------
    def _key(self) -> Optional[str]:
        return self.keys.resolve(self.provider)

    def configured(self) -> bool:
        return self.keys.is_configured(self.provider)

    # -- public: complete ---------------------------------------------------
    async def complete(self, req: ProviderRequest) -> AsyncIterator[StreamChunk]:
        """Stream a completion. Yields text deltas then a final chunk.

        The mock dialect replays a recorded fixture. A real dialect in ``record``
        mode captures its final result to a fixture as it streams.
        """
        if self.dialect == "mock":
            async for chunk in self._complete_mock(req):
                yield chunk
            return

        if self.dialect == "anthropic":
            gen = self._complete_anthropic(req)
        elif self.dialect == "openai-compat":
            gen = self._complete_openai(req)
        elif self.dialect == "poe":
            gen = self._complete_poe(req)
        elif self.dialect == "gemini":
            gen = self._complete_gemini(req)
        else:
            raise ProviderError(f"unknown dialect: {self.dialect}")

        final: Optional[ProviderResult] = None
        async for chunk in gen:
            if chunk.final is not None:
                final = chunk.final
            yield chunk

        if self.record and final is not None and self.fixtures_dir is not None:
            _write_fixture(self.fixtures_dir, req, final)

    # -- estimate: count_tokens (Anthropic) or heuristic -------------------
    async def count_tokens(self, req: ProviderRequest) -> int:
        """Best available input-token estimate for ``req``.

        Anthropic has a real ``count_tokens`` endpoint. Everyone else gets the
        chars/4 heuristic, and the caller labels it as such.
        """
        if self.dialect == "anthropic":
            try:
                client = self._anthropic_client()
                kwargs = {"model": req.model, "messages": req.messages}
                if req.system:
                    kwargs["system"] = req.system
                res = await client.messages.count_tokens(**kwargs)
                return int(res.input_tokens)
            except Exception:
                pass  # fall through to heuristic
        return heuristic_token_count(req)

    # -- dialect: anthropic -------------------------------------------------
    def _anthropic_client(self):
        import anthropic

        return anthropic.AsyncAnthropic(api_key=self._key())

    async def _complete_anthropic(self, req: ProviderRequest) -> AsyncIterator[StreamChunk]:
        client = self._anthropic_client()
        kwargs = _anthropic_kwargs(req)
        text_parts: List[str] = []
        try:
            async with client.messages.stream(**kwargs) as stream:
                async for delta in stream.text_stream:
                    text_parts.append(delta)
                    yield StreamChunk(text=delta)
                msg = await stream.get_final_message()
        except Exception as e:  # pragma: no cover - network path
            raise ProviderError(f"anthropic error: {type(e).__name__}") from e

        usage = Usage(
            in_=getattr(msg.usage, "input_tokens", 0) or 0,
            out=getattr(msg.usage, "output_tokens", 0) or 0,
        )
        text = "".join(text_parts)
        yield StreamChunk(final=ProviderResult(
            text=text, usage=usage,
            stop_reason=getattr(msg, "stop_reason", None),
            model=getattr(msg, "model", req.model),
        ))

    # -- dialect: openai-compat (OpenAI + OpenRouter + optional Gemini) -----
    def _openai_client(self):
        from openai import AsyncOpenAI

        base_url = None
        default_headers: Dict[str, str] = {}
        if self.provider == "openrouter":
            base_url = "https://openrouter.ai/api/v1"
            # OpenRouter attribution headers (optional but recommended).
            default_headers = {
                "HTTP-Referer": "https://github.com/LocalSymmetry/lofn",
                "X-Title": "Lofn Creative Studio",
            }
        elif self.provider == "gemini":
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        # provider "openai" -> default base_url
        return AsyncOpenAI(
            api_key=self._key(),
            base_url=base_url,
            default_headers=default_headers or None,
        )

    async def _complete_openai(self, req: ProviderRequest) -> AsyncIterator[StreamChunk]:
        client = self._openai_client()
        messages = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.extend(req.messages)

        # OpenRouter: ask for usage (incl. its per-request cost) in the stream.
        extra_body: Dict[str, object] = {}
        if self.provider == "openrouter":
            extra_body["usage"] = {"include": True}

        params = dict(req.params or {})
        for internal in ("provider", "tier", "thinking", "budget_tokens", "effort"):
            params.pop(internal, None)

        create_kwargs = dict(
            model=req.model,
            messages=messages,
            max_tokens=req.max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            **params,
        )
        if extra_body:
            create_kwargs["extra_body"] = extra_body

        text_parts: List[str] = []
        usage = Usage()
        cost: Optional[float] = None
        stop_reason: Optional[str] = None
        model_out = req.model
        try:
            stream = await client.chat.completions.create(**create_kwargs)
            async for event in stream:
                if getattr(event, "choices", None):
                    ch = event.choices[0]
                    delta = getattr(ch.delta, "content", None)
                    if delta:
                        text_parts.append(delta)
                        yield StreamChunk(text=delta)
                    if getattr(ch, "finish_reason", None):
                        stop_reason = ch.finish_reason
                u = getattr(event, "usage", None)
                if u:
                    usage = Usage(
                        in_=getattr(u, "prompt_tokens", 0) or 0,
                        out=getattr(u, "completion_tokens", 0) or 0,
                    )
                    # OpenRouter reports actual cost under usage.cost.
                    cost = getattr(u, "cost", None)
                if getattr(event, "model", None):
                    model_out = event.model
        except Exception as e:  # pragma: no cover - network path
            raise ProviderError(f"openai-compat error: {type(e).__name__}") from e

        yield StreamChunk(final=ProviderResult(
            text="".join(text_parts), usage=usage,
            stop_reason=stop_reason, model=model_out, cost=cost,
        ))

    # -- dialect: poe (fastapi_poe) — ported from legacy PoeLLM -------------
    async def _complete_poe(self, req: ProviderRequest) -> AsyncIterator[StreamChunk]:
        import fastapi_poe as fp

        # bot_name = model minus the "Poe-" prefix (legacy convention).
        bot_name = req.model.split("-", 1)[1] if req.model.startswith("Poe-") else req.model

        # Poe takes a flat message list; fold the system prompt into a leading
        # system message. NO stop sequences (Poe rejects them).
        protocol_messages = []
        if req.system:
            protocol_messages.append(fp.ProtocolMessage(role="system", content=req.system))
        for m in req.messages:
            role = "bot" if m.get("role") == "assistant" else "user"
            protocol_messages.append(fp.ProtocolMessage(role=role, content=m["content"]))

        text_parts: List[str] = []
        try:
            async for partial in fp.get_bot_response(
                messages=protocol_messages,
                bot_name=bot_name,
                api_key=self._key(),
            ):
                if isinstance(partial, fp.PartialResponse) and partial.text:
                    text_parts.append(partial.text)
                    yield StreamChunk(text=partial.text)
        except Exception as e:  # pragma: no cover - network path
            raise ProviderError(f"poe error: {type(e).__name__}") from e

        text = "".join(text_parts)
        # Poe does not report token usage; leave it zeroed (caller marks unpriced
        # via chars/4 if it wants a cost). stop_reason unknown.
        yield StreamChunk(final=ProviderResult(
            text=text,
            usage=Usage(out=len(text) // 4),
            stop_reason="end_turn",
            model=req.model,
        ))

    # -- dialect: gemini (native google.genai) — ported from legacy GeminiLLM
    async def _complete_gemini(self, req: ProviderRequest) -> AsyncIterator[StreamChunk]:
        from google import genai
        from google.genai import types as gtypes

        client = genai.Client(api_key=self._key())

        # Fold system + turns into a single content string (Lofn steps are
        # single-shot prompt->text; no multi-turn tool loops in v2).
        prompt_parts: List[str] = []
        if req.system:
            prompt_parts.append(req.system)
        for m in req.messages:
            prompt_parts.append(m["content"])
        prompt = "\n\n".join(prompt_parts)

        config = gtypes.GenerateContentConfig(max_output_tokens=req.max_tokens)
        temperature = (req.params or {}).get("temperature")
        if temperature is not None:
            config.temperature = temperature

        text_parts: List[str] = []
        usage = Usage()
        try:
            stream = await client.aio.models.generate_content_stream(
                model=req.model, contents=prompt, config=config,
            )
            async for event in stream:
                t = getattr(event, "text", None)
                if t:
                    text_parts.append(t)
                    yield StreamChunk(text=t)
                meta = getattr(event, "usage_metadata", None)
                if meta:
                    usage = Usage(
                        in_=getattr(meta, "prompt_token_count", 0) or 0,
                        out=getattr(meta, "candidates_token_count", 0) or 0,
                    )
        except Exception as e:  # pragma: no cover - network path
            raise ProviderError(f"gemini error: {type(e).__name__}") from e

        yield StreamChunk(final=ProviderResult(
            text="".join(text_parts), usage=usage,
            stop_reason="end_turn", model=req.model,
        ))

    # -- dialect: mock (record/replay) -------------------------------------
    async def _complete_mock(self, req: ProviderRequest) -> AsyncIterator[StreamChunk]:
        if self.fixtures_dir is None:
            raise ProviderError("mock provider requires a fixtures_dir")
        fixture = _read_fixture(self.fixtures_dir, req)
        if fixture is None:
            # A retry re-call appends a repair note to the SAME resolved prompt, so
            # its fingerprint differs. Fall back to a fixture for the same model
            # whose recorded user content is a PREFIX of this request's — i.e. the
            # base call. This lets the engine's repair loop replay deterministically
            # against the base fixture (the artifact is re-served, still broken, so
            # retries exhaust and the pair quarantines).
            fixture = _read_fixture_prefix_fallback(self.fixtures_dir, req)
        if fixture is None:
            raise ProviderError(
                f"no mock fixture for model='{req.model}' "
                f"(fingerprint {req.fingerprint()[:12]}) in {self.fixtures_dir}"
            )
        result = _fixture_to_result(fixture)
        # Replay deterministically: stream in fixed-size chunks so the engine's
        # streaming path is exercised identically every run.
        text = result.text
        step = max(1, len(text) // 8) if text else 1
        for i in range(0, len(text), step):
            yield StreamChunk(text=text[i:i + step])
        yield StreamChunk(final=result)


# ---------------------------------------------------------------------------
# Heuristic token estimate (non-Anthropic providers) — chars/4, labeled so.
# ---------------------------------------------------------------------------

def heuristic_token_count(req: ProviderRequest) -> int:
    chars = len(req.system or "")
    for m in req.messages:
        chars += len(m.get("content", ""))
    return max(1, chars // 4)


# ---------------------------------------------------------------------------
# Fixture storage (mock dialect + record mode). One JSON file per fingerprint.
# ---------------------------------------------------------------------------

def _fixture_path(fixtures_dir: Path, req: ProviderRequest) -> Path:
    fp = req.fingerprint()
    return fixtures_dir / f"{req.model}__{fp[:16]}.json"


def _write_fixture(fixtures_dir: Path, req: ProviderRequest, result: ProviderResult) -> Path:
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    path = _fixture_path(fixtures_dir, req)
    payload = {
        "request": {
            "model": req.model,
            "system": req.system,
            "messages": req.messages,
            "max_tokens": req.max_tokens,
            "params": _canonical(req.params),
        },
        "fingerprint": req.fingerprint(),
        "result": result.as_dict(),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _read_fixture(fixtures_dir: Path, req: ProviderRequest) -> Optional[dict]:
    path = _fixture_path(fixtures_dir, req)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_fixture_prefix_fallback(fixtures_dir: Path, req: ProviderRequest) -> Optional[dict]:
    """Find a fixture (same model) whose recorded user content is a PREFIX of the
    request's user content. Serves the engine's retry re-calls (which append a
    repair note to the base prompt) without pre-authoring per-repair fixtures.

    Only the LONGEST matching prefix is used, so a base call and its retry both map
    to the base fixture deterministically. Returns None if none match."""
    if not fixtures_dir.exists():
        return None
    try:
        req_user = req.messages[-1].get("content", "") if req.messages else ""
    except (IndexError, AttributeError):
        return None
    if not req_user:
        return None
    best: Optional[dict] = None
    best_len = -1
    for path in fixtures_dir.glob(f"{req.model}__*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rmsgs = data.get("request", {}).get("messages") or []
        if not rmsgs:
            continue
        fx_user = rmsgs[-1].get("content", "")
        if fx_user and req_user.startswith(fx_user) and len(fx_user) > best_len:
            best = data
            best_len = len(fx_user)
    return best


def _fixture_to_result(fixture: dict) -> ProviderResult:
    r = fixture.get("result", {})
    usage = r.get("usage", {}) or {}
    return ProviderResult(
        text=r.get("text", ""),
        usage=Usage(in_=usage.get("in", 0), out=usage.get("out", 0)),
        stop_reason=r.get("stop_reason"),
        model=r.get("model"),
        cost=r.get("cost"),
    )


def write_mock_fixture(fixtures_dir: Path, req: ProviderRequest, result: ProviderResult) -> Path:
    """Public helper: hand-author a mock fixture (used by tests)."""
    return _write_fixture(fixtures_dir, req, result)
