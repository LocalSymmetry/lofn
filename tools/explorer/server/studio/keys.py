"""API-key resolution and redaction (plan §7, decision §13.6).

Two jobs, both security-critical (acceptance gate S2):

1. **Resolution** — find a provider's key from the environment first, then from
   ``.env.studio`` at the repo root (gitignored, never served by the file API).
   Accept BOTH the standard names (``ANTHROPIC_API_KEY`` …) AND the legacy
   ``config.yaml`` aliases the operator already uses (``ANTHROPIC_API`` …).

2. **Redaction** — a filter that scrubs key-shaped strings out of any text before
   it reaches run.json, the runlog, an SSE frame, an error message, or an HTTP
   response. Keys only ever leave the machine to their own provider's endpoint.

DUMB / LOCAL-FIRST: this module never phones home. ``.env.studio`` is read by the
server process only; it is NOT in the file API's write allowlist and no endpoint
returns key material — ``/api/studio/providers`` reports ``configured: bool`` only.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# ---------------------------------------------------------------------------
# Provider → accepted env var names. First present, non-empty wins; the
# standard *_API_KEY name is checked before the legacy config.yaml alias.
# ---------------------------------------------------------------------------

PROVIDER_KEY_NAMES: Dict[str, List[str]] = {
    # provider id : [standard, ...legacy aliases]
    "anthropic":  ["ANTHROPIC_API_KEY", "ANTHROPIC_API"],
    "openai":     ["OPENAI_API_KEY", "OPENAI_API"],
    "openrouter": ["OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY"],
    "poe":        ["POE_API_KEY", "POE_API"],
    "gemini":     ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
}

# The mock provider needs no key.
KEYLESS_PROVIDERS = {"mock"}


# ---------------------------------------------------------------------------
# .env.studio loading (repo root, gitignored). A tiny dotenv parser — we do not
# take a python-dotenv dependency; the format we need is KEY=VALUE lines.
# ---------------------------------------------------------------------------

def _parse_dotenv(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        # strip a single layer of matching quotes
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        if key:
            out[key] = val
    return out


def load_env_studio(path: Path) -> Dict[str, str]:
    """Read ``.env.studio`` into a dict; empty dict if it does not exist."""
    try:
        return _parse_dotenv(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except OSError:
        return {}


@dataclass
class KeyResolver:
    """Resolve provider keys from env, then a ``.env.studio`` overlay.

    Precedence: process environment first (an operator export wins), then the
    ``.env.studio`` file at the repo root. Never mutates ``os.environ``.
    """

    repo_root: Path
    env: Dict[str, str] = field(default_factory=lambda: dict(os.environ))
    dotenv: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_root(cls, repo_root: Path, env: Optional[Dict[str, str]] = None) -> "KeyResolver":
        dotenv = load_env_studio(repo_root / ".env.studio")
        return cls(
            repo_root=repo_root,
            env=dict(env if env is not None else os.environ),
            dotenv=dotenv,
        )

    def _lookup(self, name: str) -> Optional[str]:
        v = self.env.get(name)
        if v:
            return v
        v = self.dotenv.get(name)
        if v:
            return v
        return None

    def resolve(self, provider: str) -> Optional[str]:
        """Return the key for ``provider`` (or None). ``mock`` returns None."""
        if provider in KEYLESS_PROVIDERS:
            return None
        for name in PROVIDER_KEY_NAMES.get(provider, []):
            v = self._lookup(name)
            if v:
                return v
        return None

    def is_configured(self, provider: str) -> bool:
        """True if the provider has a usable key (mock is always configured)."""
        if provider in KEYLESS_PROVIDERS:
            return True
        return self.resolve(provider) is not None

    def configured_map(self) -> Dict[str, bool]:
        providers = list(PROVIDER_KEY_NAMES.keys()) + list(KEYLESS_PROVIDERS)
        return {p: self.is_configured(p) for p in providers}


# ---------------------------------------------------------------------------
# Redaction filter (acceptance gate S2). Scrub key-shaped strings from any text.
# ---------------------------------------------------------------------------

# Known provider key shapes plus a generic long-token fallback. These are
# deliberately broad — over-redacting a log line is safe; leaking a key is not.
_KEY_PATTERNS: List[re.Pattern] = [
    # Anthropic: sk-ant-... (api keys and oauth tokens sk-ant-oat01-...)
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    # OpenAI: sk-..., sk-proj-..., plus org/proj-scoped variants
    re.compile(r"sk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_\-]{16,}"),
    # OpenRouter: sk-or-v1-...
    re.compile(r"sk-or-[A-Za-z0-9_\-]{8,}"),
    # Google / Gemini: AIza...
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    # Poe tokens are opaque; covered by the assignment-based redactor below and
    # the generic bearer pattern here.
    re.compile(r"(?i)bearer\s+[A-Za-z0-9_\-\.=]{16,}"),
]

# KEY=VALUE / "KEY": "VALUE" style leaks for any of our known env var names.
_ALL_KEY_NAMES = sorted(
    {name for names in PROVIDER_KEY_NAMES.values() for name in names},
    key=len,
    reverse=True,
)
_ASSIGN_PATTERNS: List[re.Pattern] = [
    # KEY = value / KEY: value  (env / yaml style)
    re.compile(
        r"(?P<k>\b(?:%s)\b)\s*(?P<sep>[:=])\s*(?P<q>[\"']?)(?P<v>[^\s\"',}]+)(?P=q)"
        % "|".join(re.escape(n) for n in _ALL_KEY_NAMES)
    ),
]

REDACTED = "[REDACTED]"


def _redact_str(s: str) -> str:
    for pat in _ASSIGN_PATTERNS:
        s = pat.sub(lambda m: f"{m.group('k')}{m.group('sep')}{REDACTED}", s)
    for pat in _KEY_PATTERNS:
        s = pat.sub(REDACTED, s)
    return s


def redact(obj):
    """Recursively scrub key-shaped strings from a str / dict / list / scalar.

    Returns a new structure of the same shape with any key material replaced by
    ``[REDACTED]``. Dict VALUES whose KEY names a known secret are also redacted
    wholesale (so ``{"ANTHROPIC_API_KEY": "anything"}`` never survives).
    """
    if isinstance(obj, str):
        return _redact_str(obj)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, str) and k in _ALL_KEY_NAMES:
                out[k] = REDACTED
            elif isinstance(k, str) and k.lower() in {
                "api_key", "apikey", "authorization", "auth_token", "token", "secret"
            }:
                out[k] = REDACTED
            else:
                out[k] = redact(v)
        return out
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(redact(v) for v in obj)
    return obj


def redact_with_values(text: str, secrets: Iterable[Optional[str]]) -> str:
    """Redact known literal secret VALUES from ``text`` (belt-and-suspenders).

    Given the actual resolved key strings, scrub any exact occurrence even if it
    doesn't match a known shape (e.g. a Poe token). Then run the pattern-based
    ``_redact_str`` pass. Use this on runlog/SSE text where the resolver's live
    keys are known.
    """
    for s in secrets:
        if s and len(s) >= 8:
            text = text.replace(s, REDACTED)
    return _redact_str(text)
