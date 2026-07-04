"""Resolver — THE single prompt-resolution path (plan §6, WS2).

There is exactly ONE function that turns (flow, knobs, context, step, pair?) into
the bytes a provider sees: :func:`resolve`. Both the **Compile** preview and the
**engine** (WS3) call it — acceptance gate **S3** asserts the bytes Compile shows
are byte-identical to what the engine sends. Nothing else may assemble a prompt.

The resolution pipeline (plan §6, in order):

  1. Read the step's canon prompt file **read-only** (never mutated — S1).
  2. Apply overlay hunks — ordered EXACT-MATCH find→replace, apply-or-error,
     NEVER fuzzy (a hunk that doesn't match raises; a silent miss is a bug).
  3. Fill the CREATIVE CONTEXT ``{slot}`` tokens with the **FULL ICB verbatim**
     (the anti-personality-collapse rule: inject the whole block, never a
     name-reference / summary). This mirrors the streamlit-era
     ``system_prompt.replace("{slot}", value)`` slot-filling — see
     ``docs/reference/lofn-legacy-llm/llm_integration.py``.
  4. Append the **RUN PARAMETERS** block — a studio-owned, authoritative
     statement of the run's knob values that overrides any different number the
     step prose remembers (plan §3.3 mechanism 1).
  5. Attach a token estimate.

CANON IMMUTABILITY (S1): every canon file is opened read-only; the only bytes the
studio produces live in the returned :class:`ResolvedPrompt` (in memory) and, at
run time, under ``output/studio/**`` (engine-owned).

DUMB BACKEND: the resolver renders text only. It never runs a gate and never
computes a threshold — the RUN PARAMETERS block RESTATES knob values (which the
run-local ``gates.yaml`` also carries) but enforcement is the validators' job.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .flowspec import FlowSpec, find_repo_root, sha256_text
from .knobs import KnobSet


# ---------------------------------------------------------------------------
# Data types.
# ---------------------------------------------------------------------------

@dataclass
class ResolvedPrompt:
    """The bytes a provider will see, plus provenance meta.

    ``user`` is the assembled prompt (the step prose with slots filled + the RUN
    PARAMETERS block appended). ``system`` is optional — Lofn steps carry their
    directive inline, so v1 leaves it None; a flow MAY set a system tier prompt
    later without changing this contract. ``meta`` records everything the runlog
    needs for S3 byte-equality + provenance.
    """

    user: str
    system: Optional[str] = None
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def text_sha(self) -> str:
        """SHA of exactly the bytes that go on the wire (system + user).

        S3 records this in the runlog; Compile records the same; they must match.
        """
        return sha256_text((self.system or "") + "\x00" + self.user)

    def as_dict(self) -> dict:
        return {
            "system": self.system,
            "user": self.user,
            "meta": self.meta,
            "text_sha": self.text_sha,
        }


@dataclass
class OverlayHunk:
    """One exact-match find→replace edit."""

    find: str
    replace: str


class OverlayError(ValueError):
    """A hunk did not match exactly (apply-or-error; never fuzzy)."""


# ---------------------------------------------------------------------------
# Overlay parsing + application (plan §3.3 mechanism 2, §6 step 2).
# ---------------------------------------------------------------------------

# Overlay files are markdown with ordered hunks. Format (documented in the plan
# as "ordered exact-match find→replace hunks"):
#
#   <<<<<<< FIND
#   the exact text to find
#   =======
#   the exact replacement text
#   >>>>>>> REPLACE
#
# Blank lines and prose between hunks are ignored, so an overlay can be
# self-documenting. The delimiters mirror a git conflict marker so they're
# familiar and never collide with normal prose.
_HUNK_RE = re.compile(
    r"<{7}\s*FIND[ \t]*\n(?P<find>.*?)\n={7}[ \t]*\n(?P<replace>.*?)\n>{7}\s*REPLACE",
    re.DOTALL,
)


def parse_overlay(text: str) -> List[OverlayHunk]:
    """Parse an overlay patch into ordered hunks. Empty text -> no hunks."""
    hunks: List[OverlayHunk] = []
    for m in _HUNK_RE.finditer(text):
        hunks.append(OverlayHunk(find=m.group("find"), replace=m.group("replace")))
    return hunks


def apply_overlay(prompt: str, hunks: List[OverlayHunk], *, source: str = "<overlay>") -> str:
    """Apply hunks in order. EXACT match required — apply-or-error, never fuzzy.

    Each ``find`` must occur **exactly once** in the current text (after prior
    hunks applied). Zero matches or multiple matches raise ``OverlayError`` — the
    studio refuses to silently mis-edit a canon prompt. A hunk that finds and
    replaces the same string is a no-op but still must match exactly once.
    """
    out = prompt
    for i, h in enumerate(hunks):
        count = out.count(h.find)
        if count == 0:
            raise OverlayError(
                f"{source} hunk {i}: find-text not present (exact match required):\n"
                f"---\n{h.find[:400]}\n---"
            )
        if count > 1:
            raise OverlayError(
                f"{source} hunk {i}: find-text matches {count} times (must be unique):\n"
                f"---\n{h.find[:400]}\n---"
            )
        out = out.replace(h.find, h.replace, 1)
    return out


# ---------------------------------------------------------------------------
# CREATIVE CONTEXT slot filling (plan §6 step 3).
# ---------------------------------------------------------------------------

# A slot token in the canon prose looks like ``{input}`` / ``{personality}`` etc.
# We replace the WHOLE-WORD token, verbatim, with the context value — exactly the
# legacy ``.replace("{slot}", value)`` behavior, applied for every declared slot.
def _slot_token(slot: str) -> str:
    return "{" + slot + "}"


def fill_slots(prompt: str, context: Dict[str, str], slots: List[str]) -> str:
    """Replace each ``{slot}`` with its verbatim context value.

    ANTI-PERSONALITY-COLLAPSE (plan §6): inject the FULL value, never a
    name-reference. A slot declared by the flow but absent from ``context`` is
    filled with an empty string (legacy behavior: the streamlit template left an
    unsupplied optional slot — e.g. ``{image_context}`` — empty). Only the flow's
    declared ``slots`` are substituted; any other ``{...}`` in the prose (JSON
    schema braces, code fences) is left untouched.
    """
    out = prompt
    for slot in slots:
        token = _slot_token(slot)
        if token not in out:
            continue
        value = context.get(slot, "")
        # str() guards against a non-string context value sneaking in.
        out = out.replace(token, "" if value is None else str(value))
    return out


# ---------------------------------------------------------------------------
# RUN PARAMETERS block (plan §3.3 mechanism 1, §6 step 4).
# ---------------------------------------------------------------------------

_RUN_PARAMS_BEGIN = "===== BEGIN RUN PARAMETERS (studio — authoritative) ====="
_RUN_PARAMS_END = "===== END RUN PARAMETERS ====="


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


def render_run_parameters(knobset: KnobSet, *, pair: Optional[int] = None) -> str:
    """Render the authoritative RUN PARAMETERS block from resolved knob values.

    Explicit, universal, reversible (plan §3.3.1). It STATES the run's numbers and
    instructs the model that these override any different number remembered from
    the step prose. The block is deterministic given the resolved knobs (so S3
    byte-equality holds) — knob names iterate in a stable, curated order with an
    alphabetical tail for any extras, so adding a knob never reshuffles the rest.
    """
    r = knobset.resolved

    def val(name, default=None):
        return r.get(name, default)

    lines: List[str] = [_RUN_PARAMS_BEGIN, ""]
    lines.append(
        "These values are AUTHORITATIVE for this run. Where a number below "
        "differs from a number remembered from the step prose above, USE THE "
        "NUMBER BELOW."
    )
    lines.append("")

    # Curated, human-meaningful ordering (the portfolio shape first).
    n_pairs = val("n_pairs")
    if n_pairs is not None:
        acc = val("barbell_accessible")
        amb = val("barbell_ambitious")
        n_var = val("n_variations_per_pair")
        total = val("total_prompts")
        lines.append(
            f"- Select **exactly {_fmt(n_pairs)}** concept x medium pairs at "
            f"step 05"
            + (
                f" — {_fmt(acc)} ACCESSIBLE + {_fmt(amb)} AMBITIOUS across the "
                f"barbell."
                if acc is not None and amb is not None
                else "."
            )
        )
        if n_var is not None:
            lines.append(
                f"- Produce **exactly {_fmt(n_var)}** variations per pair"
                + (f" ({_fmt(total)} prompts total)." if total is not None else ".")
            )

    n_concepts = val("n_concepts")
    if n_concepts is not None:
        lines.append(
            f"- Diverge to at least **{_fmt(n_concepts)}** concepts at step 02 "
            f"before pruning."
        )

    panel_count = val("panel_count")
    voices = val("voices_per_panel")
    if panel_count is not None and voices is not None:
        lines.append(
            f"- Use **{_fmt(panel_count)} panels x {_fmt(voices)} voices = "
            f"{_fmt(int(panel_count) * int(voices))}** total voices."
        )
    n_flairs = val("n_flairs")
    if n_flairs is not None:
        lines.append(f"- Carry **{_fmt(n_flairs)}** Special Flairs throughout.")

    # A stable alphabetical tail for every OTHER knob, so the block is a complete,
    # deterministic record of the run's numbers without reshuffling on additions.
    _named = {
        "n_pairs", "barbell_accessible", "barbell_ambitious",
        "n_variations_per_pair", "total_prompts", "n_concepts",
        "panel_count", "voices_per_panel", "n_flairs",
    }
    extras = [k for k in sorted(r) if k not in _named]
    if extras:
        lines.append("")
        lines.append("Other run parameters (authoritative thresholds for this run):")
        for k in extras:
            lines.append(f"- `{k}` = {_fmt(r[k])}")

    if pair is not None:
        lines.append("")
        lines.append(f"This step is resolving for **pair {_fmt(pair)}** of the fan-out.")

    lines.append("")
    lines.append(_RUN_PARAMS_END)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Token estimation (cheap, provider-agnostic — the wire count comes at run time).
# ---------------------------------------------------------------------------

def estimate_tokens(system: Optional[str], user: str) -> int:
    """chars/4 heuristic (labeled as such, per plan §5 budget doctrine).

    The Compile view and the pre-run budget estimate use this. The engine, at run
    time, prefers the provider's real ``count_tokens`` (Anthropic) — that's a
    strictly-better number captured in usage, not a contradiction of this estimate.
    """
    chars = len(system or "") + len(user)
    return max(1, chars // 4)


# ---------------------------------------------------------------------------
# Step lookup helpers.
# ---------------------------------------------------------------------------

def find_step(flow: FlowSpec, step_id: str) -> dict:
    """Return the step dict with ``id == step_id`` (searches all stages)."""
    for stage in flow.data["stages"]:
        for step in stage.get("steps", []):
            if step.get("id") == step_id:
                return step
    raise KeyError(f"step {step_id!r} not in flow {flow.name}@{flow.version}")


def step_is_fanout(flow: FlowSpec, step_id: str) -> bool:
    """True if the step lives in a ``foreach: pair`` stage (needs a pair index)."""
    for stage in flow.data["stages"]:
        if any(s.get("id") == step_id for s in stage.get("steps", [])):
            return bool(stage.get("foreach"))
    return False


def _read_canon(repo_root: Path, rel_path: str) -> str:
    """Read a canon file READ-ONLY. Raises if it's missing (a flow that points at
    a non-existent prompt is a build error, not a silent empty prompt)."""
    p = (repo_root / rel_path).resolve()
    # Path jail: the canon file must live inside the repo (never escape via ..).
    if repo_root not in p.parents and p != repo_root:
        raise ValueError(f"canon path escapes repo root: {rel_path}")
    if not p.exists():
        raise FileNotFoundError(f"canon prompt file not found: {rel_path}")
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# THE single resolution path.
# ---------------------------------------------------------------------------

def resolve(
    flow: FlowSpec,
    knobs: KnobSet,
    context: Dict[str, str],
    step: str,
    pair: Optional[int] = None,
    *,
    repo_root: Optional[Path] = None,
) -> ResolvedPrompt:
    """Resolve ONE step to the exact bytes a provider will see (plan §6).

    Args:
      flow: the loaded FlowSpec (canon prompt paths + declared slots).
      knobs: the resolved KnobSet (the run's numbers).
      context: slot → verbatim value (the filled ICB pieces). Missing declared
        slots fill empty (legacy behavior).
      step: the step id, e.g. "00", "08", "11".
      pair: fan-out pair index (1-based), required for a ``foreach: pair`` step's
        RUN PARAMETERS note; ignored for coordinator steps.
      repo_root: override for testing; defaults to the discovered repo root.

    Returns a :class:`ResolvedPrompt`. Raises for a validator-kind step (no LLM
    prompt to resolve), an unmatched overlay hunk, or a missing canon file.
    """
    repo_root = repo_root or find_repo_root()
    stepdef = find_step(flow, step)

    if stepdef.get("kind") == "validator":
        raise ValueError(
            f"step {step!r} is kind:validator (run={stepdef.get('run')!r}); it has "
            f"no LLM prompt to resolve"
        )

    prompt_path = stepdef.get("prompt")
    if not prompt_path:
        raise ValueError(f"step {step!r} has no prompt path to resolve")

    # 1. Read canon prompt (read-only).
    raw = _read_canon(repo_root, prompt_path)
    canon_sha = sha256_text(raw)

    # 2. Overlay hunks (exact-match or error).
    overlay_rel = stepdef.get("overlay")
    overlay_applied = False
    overlay_sha: Optional[str] = None
    text = raw
    if overlay_rel:
        overlay_text = _read_canon(repo_root, overlay_rel)
        overlay_sha = sha256_text(overlay_text)
        hunks = parse_overlay(overlay_text)
        if hunks:
            text = apply_overlay(text, hunks, source=overlay_rel)
            overlay_applied = True

    # 3. Fill CREATIVE CONTEXT slots verbatim (full ICB, never a reference).
    slots = list(flow.data.get("context", {}).get("slots", []))
    text = fill_slots(text, context, slots)

    # 4. Append the authoritative RUN PARAMETERS block.
    run_params = render_run_parameters(
        knobs, pair=pair if step_is_fanout(flow, step) else None
    )
    user = text.rstrip("\n") + "\n\n" + run_params + "\n"

    # 5. Token estimate + provenance meta.
    tokens = estimate_tokens(None, user)
    tier = stepdef.get("tier")
    tier_def = (flow.data.get("model_tiers", {}) or {}).get(tier, {}) if tier else {}
    meta: Dict[str, object] = {
        "flow": flow.name,
        "flow_version": flow.version,
        "step": step,
        "pair": pair if step_is_fanout(flow, step) else None,
        "tier": tier,
        "model": tier_def.get("model"),
        "provider": tier_def.get("provider"),
        "prompt_path": prompt_path,
        "canon_sha": canon_sha,
        "overlay_path": overlay_rel,
        "overlay_sha": overlay_sha,
        "overlay_applied": overlay_applied,
        "slots_filled": [s for s in slots if _slot_token(s) in raw],
        "est_tokens_in": tokens,
        "token_estimate_method": "chars/4 heuristic",
    }

    resolved = ResolvedPrompt(user=user, system=None, meta=meta)
    meta["text_sha"] = resolved.text_sha
    return resolved
