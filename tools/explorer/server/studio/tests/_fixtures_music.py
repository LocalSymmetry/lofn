"""Test helpers: author MOCK fixtures for a full music run ($0, deterministic).

The engine calls the resolver (which reads REAL canon prompt files) then the mock
provider, which looks a fixture up by the request fingerprint. So to drive a full
mock run we must, for the given flow+knobs+context, resolve every step exactly as
the engine will and write a fixture whose ``result.text`` is a canonical artifact
that PASSES that step's gates.

This module builds:
  * ``generic_artifact(step, marker)`` — a ≥800-char, non-repetitive canonical step
    artifact carrying the step's required marker (steps 00–07, 09).
  * ``music_song_artifact(...)`` — a full ## 1. MUSIC PROMPT (850–1000 chars) +
    ## 2. LYRICS (Theme/SONG FORM, full EMO headers, SFX cue, 60+ sung lines)
    artifact that passes the g.music_prompt / g.music_lyrics gates (steps 08/10/11).
  * ``author_run_fixtures(...)`` — walk the flow, resolve each step (incl. retries
    are not needed on the happy path), and write a fixture per request fingerprint.

None of this makes a network call.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from studio.flowspec import FlowSpec
from studio.knobs import KnobSet
from studio.providers import ProviderRequest, ProviderResult, Usage, write_mock_fixture
from studio.resolver import resolve, step_is_fanout
from studio.runstore import pair_label


# ---------------------------------------------------------------------------
# Artifact generators.
# ---------------------------------------------------------------------------

_MARKERS = {
    "00": "aesthetic emotion genre",
    "01": "essence facet style",
    "02": "concept",
    "03": "artist critique",
    "04": "medium",
    "05": "pair concept medium",
    "06": "facet",
    "07": "song guide",
    "09": "artist refined",
}


def _paragraph(seed: str, n: int) -> str:
    """A block of distinct, non-repetitive sentences (dodges the collapse check)."""
    words = ("dawn tide pools shimmer across the granite shelf while gulls wheel "
             "over the cold spray and the light bends through shallow water").split()
    lines = []
    for i in range(n):
        # rotate the word window so no two lines are identical
        w = words[i % len(words):] + words[:i % len(words)]
        lines.append(f"{seed} note {i}: " + " ".join(w[:12]) + ".")
    return "\n".join(lines)


def generic_artifact(step: str, *, is_music: bool = True) -> str:
    """A canonical, gate-passing artifact for a step with no heavy content gates."""
    marker = _MARKERS.get(step, "concept")
    body = _paragraph(f"step{step}", 22)
    header = f"# Step {step} artifact\n\n"
    # Ensure every required marker token appears verbatim.
    marker_line = "Markers present: " + ", ".join(marker.split()) + ".\n\n"
    extra = _paragraph(f"step{step}-detail", 14)
    text = header + marker_line + body + "\n\n" + extra + "\n"
    # pad to comfortably exceed 800 chars if short
    while len(text) < 1000:
        text += _paragraph(f"step{step}-pad{len(text)}", 6) + "\n"
    return text


def music_song_artifact(*, name_and_crime: bool = False) -> str:
    """A full music song artifact that passes g.music_prompt + g.music_lyrics.

    Set ``name_and_crime`` to embed a name+crime tuple (fires check_human_subjects
    -> HOLD_FOR_HUMAN) for the S7 test — the structure still passes validate_step.
    """
    # --- ## 1. MUSIC PROMPT : must be 850-1000 chars of non-heading body -----
    prompt_body = (
        "Dream-pop with shoegaze texture at a patient 82 BPM, an adult female "
        "vocalist singing close and unhurried over layered guitars washed in "
        "reverb and slow tremolo. The low end is a rounded analog bass that "
        "breathes with the kick while brushed drums keep a soft, tidal pulse. "
        "Wide stereo pads bloom under the chorus and a single bell motif answers "
        "each vocal phrase from the far right. Keep the mix airy and cool-toned, "
        "salt-spray bright in the highs and cavern-dark in the sub, so the song "
        "feels like standing on wet granite as the light comes up. Dynamics move "
        "from a whispered verse to a swelling, cathartic refrain and back to hush, "
        "never louder than it needs to be, always leaving room for the reverb tail "
        "to ring out over the retreating water and the distant, wheeling gulls."
    )
    # trim/pad to land inside 850-1000
    prompt_body = _fit_band(prompt_body, 870, 960)

    # --- ## 2. LYRICS --------------------------------------------------------
    def verse(theme_word: str, n: int) -> List[str]:
        # Each line is unique across the whole song (section index n woven in) so the
        # unique-line ratio stays well above the 0.45 collapse floor.
        openers = [
            "the {t} pulls the morning thin",
            "cold granite holds the {i}th light",
            "a gull unspools across the grey at {i}",
            "the shallow water learns my name again",
            "salt on the wind and {i} things more",
            "the tide remembers what I forgot in {i}",
            "a bell answers the {i}th distant shore",
            "I stand where spray meets stone at hour {i}",
        ]
        return [openers[k].format(t=theme_word, i=n * 8 + k) for k in range(8)]

    lyric_lines: List[str] = []
    # 8 sections * 8 lines = 64 sung lines (>= 60 floor)
    section_specs = [
        ("Intro", "TENDER", "Lead", "soft entry"),
        ("Verse 1", "LONGING", "Lead", "close mic"),
        ("Pre-Chorus", "RISING", "Lead", "build"),
        ("Chorus", "CATHARTIC", "Lead+Harmony", "full bloom"),
        ("Verse 2", "LONGING", "Lead", "close mic"),
        ("Bridge", "ACHE", "Lead", "strip back"),
        ("Chorus", "CATHARTIC", "Lead+Harmony", "full bloom"),
        ("Outro", "RELEASE", "Lead", "reverb tail"),
    ]
    body_parts = ["## 2. LYRICS", "", "[Theme: tide pools at dawn]", "[SONG FORM: Intro-Verse-Pre-Chorus-Chorus-Verse-Bridge-Chorus-Outro]", ""]
    for i, (sec, emo, voice, cue) in enumerate(section_specs):
        body_parts.append(f"[{sec} - EMO:{emo} - {voice} - {cue}]")
        for ln in verse("tide", i):
            body_parts.append(ln)
        body_parts.append("*soft wash of spray*")
        body_parts.append("")
    lyrics_block = "\n".join(body_parts)

    crime_block = ""
    if name_and_crime:
        # A name + a locating detail + a harm context -> identifying tuple HOLD.
        crime_block = (
            "\n[Verse 3 - EMO:GRIEF - Lead - documentary]\n"
            "Daniel Marsh was murdered in Vallejo that October night\n"
            "the courthouse steps still carry his name\n"
        )

    prompt_section = "## 1. MUSIC PROMPT\n\n" + prompt_body + "\n\n"
    text = (
        "# Song Package\n\n"
        + prompt_section
        + lyrics_block
        + crime_block
        + "\n"
    )
    return text


def _fit_band(s: str, lo: int, hi: int) -> str:
    """Grow/shrink ``s`` so its length lands within [lo, hi]."""
    filler = " The reverb tail rings on over the cold retreating water once more."
    while len(s) < lo:
        s = s + filler
    if len(s) > hi:
        s = s[:hi].rsplit(" ", 1)[0].rstrip(",. ") + "."
    return s


# ---------------------------------------------------------------------------
# Fixture authoring — resolve every step exactly as the engine will.
# ---------------------------------------------------------------------------

def _artifact_for_step(step_id: str, *, name_and_crime_on_08: bool = False,
                       break_step: Optional[str] = None) -> str:
    """Pick the artifact text for a step. ``break_step`` forces a gate failure by
    returning a too-short body (for the S6 forced-quarantine test)."""
    if break_step is not None and step_id == break_step:
        return "TOO SHORT — deliberately invalid artifact to force a hard gate fail."
    if step_id in ("08", "10", "11"):
        return music_song_artifact(name_and_crime=(name_and_crime_on_08 and step_id == "08"))
    return generic_artifact(step_id)


def author_run_fixtures(
    flow: FlowSpec,
    knobs: KnobSet,
    context: Dict[str, str],
    fixtures_dir: Path,
    repo_root: Path,
    *,
    n_pairs: int,
    name_and_crime_on_08: bool = False,
    break_step: Optional[str] = None,
) -> int:
    """Write one mock fixture per (step, pair) request the engine will issue.

    Mirrors the engine's request construction: model = tier model, system =
    resolved.system, messages = [{user: resolved.user}], max_tokens = tier
    max_tokens, params = tier params. Returns the number of fixtures written.
    """
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    tiers = flow.data.get("model_tiers", {}) or {}

    for stage in flow.data["stages"]:
        pairs: List[Optional[int]]
        if stage.get("foreach") == "pair":
            pairs = list(range(1, n_pairs + 1))
        else:
            pairs = [None]
        for step in stage.get("steps", []):
            if step.get("kind") == "validator":
                continue
            step_id = step["id"]
            tier = tiers.get(step.get("tier"), {})
            model = tier.get("model", "")
            for pair in pairs:
                if pair is not None and not step_is_fanout(flow, step_id):
                    continue
                resolved = resolve(flow, knobs, context, step_id, pair=pair, repo_root=repo_root)
                req = ProviderRequest(
                    model=model,
                    system=resolved.system,
                    messages=[{"role": "user", "content": resolved.user}],
                    max_tokens=int(tier.get("max_tokens", 4096) or 4096),
                    stream=True,
                    params=dict(tier.get("params", {}) or {}),
                )
                text = _artifact_for_step(
                    step_id,
                    name_and_crime_on_08=name_and_crime_on_08,
                    break_step=break_step,
                )
                result = ProviderResult(
                    text=text,
                    usage=Usage(in_=resolved.meta.get("est_tokens_in", 100), out=len(text) // 4),
                    stop_reason="end_turn",
                    model=model,
                )
                write_mock_fixture(fixtures_dir, req, result)
                written += 1
    return written
