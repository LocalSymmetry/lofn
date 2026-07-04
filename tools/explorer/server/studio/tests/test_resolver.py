"""WS2 resolver: overlay hunks, slot filling, RUN PARAMETERS, and the v1
CREATIVE-CONTEXT reproduction gate. All $0 — reads canon read-only, no provider.

The load-bearing test is ``test_reproduces_v1_creative_context``: the resolved
prompt, minus the appended RUN PARAMETERS block, is BYTE-IDENTICAL to the v1
pipeline's CREATIVE CONTEXT assembly (the step file with its {slot}s filled
verbatim — exactly the streamlit-era ``.replace("{slot}", value)`` behavior)."""
from __future__ import annotations

import pytest

from studio.knobs import KnobSet, load_preset
from studio.flowspec import find_repo_root, flows_dir, load_flowspec
from studio.resolver import (
    resolve, fill_slots, parse_overlay, apply_overlay, OverlayError,
    render_run_parameters, _RUN_PARAMS_BEGIN,
)

_REPO_ROOT = find_repo_root()


def _music_flow():
    return load_flowspec(flows_dir(_REPO_ROOT) / "lofn-music@1.flow.yaml")


def _classic():
    return load_preset(flows_dir(_REPO_ROOT), "classic-6x4")


_CTX = {
    "input": "A song about tide pools at dawn",
    "seed": "GOLDEN SEED BLOCK",
    "meta_prompt": "META PROMPT DIRECTIVE",
    "personality": "PERSONA DNA BLOCK",
    "concept_panel": "CONCEPT PANEL 6 VOICES",
    "medium_panel": "MEDIUM PANEL 6 VOICES",
    "marketing_panel": "MARKETING PANEL 6 VOICES",
    "flairs": "15 SPECIAL FLAIRS",
    "genres_list": "genre1\tgenre2",
    "frames_list": "frame1\tframe2",
    "image_context": "",
}


# ---------------------------------------------------------------------------
# Slot filling.
# ---------------------------------------------------------------------------

def test_fill_slots_verbatim():
    text = "Idea: {input}\nPersona: {personality}\nUntouched: {not_a_slot}"
    out = fill_slots(text, _CTX, ["input", "personality"])
    assert "A song about tide pools at dawn" in out
    assert "PERSONA DNA BLOCK" in out
    # A brace token that is NOT a declared slot is left alone (JSON schema braces).
    assert "{not_a_slot}" in out


def test_fill_slots_missing_declared_slot_is_empty():
    text = "img: [{image_context}]"
    out = fill_slots(text, {}, ["image_context"])
    assert out == "img: []"


# ---------------------------------------------------------------------------
# Overlay hunks — exact match, apply-or-error, never fuzzy.
# ---------------------------------------------------------------------------

_OVERLAY = """Some doc prose.

<<<<<<< FIND
exactly 6 pairs
=======
exactly 10 pairs
>>>>>>> REPLACE

More prose.

<<<<<<< FIND
4 variations
=======
2 variations
>>>>>>> REPLACE
"""


def test_parse_overlay():
    hunks = parse_overlay(_OVERLAY)
    assert len(hunks) == 2
    assert hunks[0].find == "exactly 6 pairs"
    assert hunks[0].replace == "exactly 10 pairs"


def test_apply_overlay_exact():
    src = "Produce exactly 6 pairs with 4 variations each."
    out = apply_overlay(src, parse_overlay(_OVERLAY))
    assert out == "Produce exactly 10 pairs with 2 variations each."


def test_apply_overlay_missing_hunk_errors():
    src = "Produce exactly 7 pairs."  # no 'exactly 6 pairs'
    with pytest.raises(OverlayError):
        apply_overlay(src, parse_overlay(_OVERLAY))


def test_apply_overlay_ambiguous_match_errors():
    hunks = parse_overlay(
        "<<<<<<< FIND\nfoo\n=======\nbar\n>>>>>>> REPLACE\n"
    )
    with pytest.raises(OverlayError):
        apply_overlay("foo foo", hunks)   # matches twice -> refuse


# ---------------------------------------------------------------------------
# RUN PARAMETERS block — deterministic + authoritative.
# ---------------------------------------------------------------------------

def test_run_parameters_states_knob_values():
    block = render_run_parameters(_classic())
    assert "exactly 6" in block
    assert "3 ACCESSIBLE + 3 AMBITIOUS" in block
    assert "AUTHORITATIVE" in block
    assert "USE THE NUMBER BELOW" in block


def test_run_parameters_deterministic():
    ks = _classic()
    assert render_run_parameters(ks) == render_run_parameters(ks)


def test_run_parameters_moves_with_knob():
    ks = _classic().with_value("n_pairs", 8).with_value("n_concepts", 16)
    block = render_run_parameters(ks)
    assert "exactly 8" in block
    assert "4 ACCESSIBLE + 4 AMBITIOUS" in block


# ---------------------------------------------------------------------------
# resolve() end-to-end.
# ---------------------------------------------------------------------------

def test_resolve_step00_fills_and_appends():
    rp = resolve(_music_flow(), _classic(), _CTX, "00", repo_root=_REPO_ROOT)
    assert "{input}" not in rp.user
    assert "A song about tide pools at dawn" in rp.user
    assert "PERSONA DNA BLOCK" in rp.user
    assert _RUN_PARAMS_BEGIN in rp.user
    assert rp.meta["step"] == "00"
    assert rp.meta["model"] == "claude-opus-4-8"
    assert rp.meta["est_tokens_in"] > 0
    assert rp.text_sha  # non-empty


def test_resolve_fanout_step_carries_pair():
    rp = resolve(_music_flow(), _classic(), _CTX, "08", pair=3, repo_root=_REPO_ROOT)
    assert rp.meta["pair"] == 3
    assert "pair 3" in rp.user


def test_resolve_coordinator_step_ignores_pair():
    rp = resolve(_music_flow(), _classic(), _CTX, "00", pair=3, repo_root=_REPO_ROOT)
    # Coordinator steps aren't per-pair; the pair note is suppressed.
    assert rp.meta["pair"] is None


def test_resolve_validator_step_raises():
    with pytest.raises(ValueError):
        resolve(_music_flow(), _classic(), _CTX, "distinctiveness", repo_root=_REPO_ROOT)


def test_resolve_deterministic_same_bytes():
    """S3 foundation: two resolutions of the same inputs are byte-identical."""
    a = resolve(_music_flow(), _classic(), _CTX, "00", repo_root=_REPO_ROOT)
    b = resolve(_music_flow(), _classic(), _CTX, "00", repo_root=_REPO_ROOT)
    assert a.user == b.user
    assert a.text_sha == b.text_sha


# ---------------------------------------------------------------------------
# THE reproduction gate (WS2 done-criterion).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("step_id", ["00", "01", "02", "05"])
def test_reproduces_v1_creative_context(step_id):
    """The resolved prompt, MINUS the RUN PARAMETERS block, equals the v1
    CREATIVE CONTEXT assembly: the canon step file with {slot}s filled verbatim."""
    flow = _music_flow()
    ks = _classic()
    rp = resolve(flow, ks, _CTX, step_id, repo_root=_REPO_ROOT)

    stepdef = next(
        s for stage in flow.data["stages"] for s in stage.get("steps", [])
        if s["id"] == step_id
    )
    raw = (_REPO_ROOT / stepdef["prompt"]).read_text(encoding="utf-8")
    slots = flow.data["context"]["slots"]
    v1_assembly = fill_slots(raw, _CTX, slots).rstrip("\n")

    idx = rp.user.index(_RUN_PARAMS_BEGIN)
    resolved_body = rp.user[:idx].rstrip("\n")

    assert resolved_body == v1_assembly


def test_canon_file_not_mutated():
    """S1: resolving reads canon read-only — the file's bytes are unchanged."""
    p = _REPO_ROOT / "skills/music/steps/00_Generate_Music_Aesthetics_And_Genres.md"
    before = p.read_bytes()
    resolve(_music_flow(), _classic(), _CTX, "00", repo_root=_REPO_ROOT)
    assert p.read_bytes() == before
