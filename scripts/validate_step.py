#!/usr/bin/env python3
"""Minimal deterministic validator for Lofn step artifacts.

This is not a creative-quality judge; it catches collapsed/stub/template artifacts before
agents advance to the next step.

L4 deterministic backstop (upgrade-plan item 2.1 + Stage 5):
  - Numeric thresholds are the single-sourced values in ``vault/gates.yaml``.
    This script READS them from there and FAILS OPEN: a missing or unparseable
    gates.yaml logs a warning and falls back to built-in defaults that mirror it.
  - The countable subset (char-count bands, byte floors, taxonomy cardinality,
    banned-imperative-opener regex, prompt totals, sung-line band) is emitted as
    ``GATE_REPORT.json`` rows ``{pair, step, check, expected, actual, pass}`` that
    the subagent and ``lofn-qa`` paste as proof-of-fix evidence.
  - HARD FAILS are limited to UNAMBIGUOUS checks (banned imperative opener — with
    the offending token as evidence — and real-artist-name use). The repeated-line
    / n-gram collapse ratio is a FLAG ONLY, with the chorus/refrain block EXEMPT;
    a deliberate refrain must never auto-fail.
  - ``--meta-check`` scans the skill files for a restated numeric threshold that
    DISAGREES with gates.yaml and warns (Stage 5 prose/YAML disagreement check).
  - No "is it soulful" judgment lives in this script. Counts only.

Usage:
  validate_step.py <step> <file>            # validate one step artifact
  validate_step.py --gate-report <step> <file> [--out PATH]
                                            # also write GATE_REPORT.json rows
  validate_step.py --expected-prompts 4 08 <file>
                                            # current four-candidate music run;
                                            # legacy Step08 default stays six
  validate_step.py --gates PATH <step> <file>
                                            # validate against a specific gates.yaml
                                            # (Studio run-local, knob-adjusted thresholds)
  validate_step.py --meta-check [--root DIR]
                                            # prose/YAML threshold disagreement scan
  validate_step.py --help

Studio note (Creative Studio v2, backward-compatible): ``--gates PATH`` points the
run-local, knob-adjusted thresholds file at ``load_gates(path)`` (which already
accepted a path). WITHOUT the flag, behavior is byte-identical to before — the
default ``vault/gates.yaml`` path is used exactly as always. The flag only threads
an alternate gates file into the SAME code path.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parent))
from icb_span import load_icb, strip_icb  # the one ICB strip (2026-10-01) -- never re-derived


# --------------------------------------------------------------------------
# gates.yaml loading (FAIL-OPEN). Built-in defaults MIRROR vault/gates.yaml so a
# missing/broken YAML never hard-fails a valid run.
# --------------------------------------------------------------------------

DEFAULT_GATES = {
    "music_prompt_chars": [850, 1000],
    "suno_lyrics_field_max": 5000,
    "suno_lyrics_field_target": 3800,
    "suno_lyrics_field_target_band": [2600, 3600],
    # V6-19 (2026-11-25): length is budgeted in SUNG WORDS; lines are advisory.
    # Mirrors vault/gates.yaml. `length_gate: lines` restores the older behaviour.
    "length_gate": "words",
    "sung_words_target": [370, 430],
    "sung_words_band": [340, 460],
    "sung_words_hard_floor": 300,
    "sung_words_hard_ceiling": 520,
    "sung_chars_target": [1800, 2150],
    "sung_lines": [40, 100],
    "sung_lines_hard_floor": 45,
    "sung_lines_target": [56, 72],
    "mean_words_per_line_ceiling": 6.8,
    "lyrics_overhead_max_chars": 1100,
    "lyrics_overhead_flag_chars": 1300,
    "lyrics_overhead_flag_pct": 40.0,
    "disc_channel_in_lyrics_box": False,
    "step00_min_bytes": 2000,
    "taxonomy_cardinality": 50,
    "image_min_words": 80,
    "total_prompts": 24,
    "step06_min_facets": 5,
    "step06_max_pair_similarity": 0.50,
    "step09_max_pair_similarity": 0.62,
    "portfolio_max_lyric_similarity": 0.42,
    "portfolio_max_prompt_similarity": 0.58,
    "portfolio_max_ngram_jaccard": 0.18,
    "banned_imperative_openers": ["Create", "Design", "Make", "Render", "Depict"],
    "ban_words": [
        "ethereal", "dreamlike", "whimsical", "gentle light",
        "soft glow", "magical", "delicate",
    ],
    "unique_line_ratio_floor": 0.45,
    "ngram_collapse_n": 4,
    "music_prompt_hug_ceiling": 985,
    "sung_lines_floor_hug": 48,
    "music_prompt_terminal_punctuation": True,
    "max_sung_numeric_facts": 1,
    "house_lexicon": [
        "more sub and more sky", "small astonished laugh", "triple arch",
        "triple-arch panorama", "small enough to understand", "frost-air pad",
        "starfield percussion", "zodiacal glow", "clear silver tone",
        "dew-bright", "glass harmonica sheen", "crystalline arpeggios",
        "make my little fear",
    ],
}


def _repo_root() -> Path:
    # scripts/validate_step.py -> repo root is one level up.
    return Path(__file__).resolve().parent.parent


def _gates_path() -> Path:
    return _repo_root() / "vault" / "gates.yaml"


def load_gates(path: Path | None = None) -> dict:
    """Read vault/gates.yaml, FAIL-OPEN to DEFAULT_GATES on any problem.

    A missing file, a parse error, or a missing PyYAML import logs a single
    warning to stderr and returns the built-in defaults (which mirror the YAML).
    A broken gates.yaml must NEVER hard-fail an otherwise-valid run.
    """
    p = path or _gates_path()
    gates = dict(DEFAULT_GATES)
    try:
        if not p.exists():
            print(f"WARN: gates.yaml not found at {p}; using built-in defaults", file=sys.stderr)
            return gates
        raw = p.read_text(encoding="utf-8", errors="replace")
        loaded = None
        try:
            import yaml  # type: ignore
            loaded = yaml.safe_load(raw)
        except Exception:
            loaded = _tiny_yaml_parse(raw)
        if isinstance(loaded, dict):
            gates.update({k: v for k, v in loaded.items() if v is not None})
            # V6-19 (2026-11-25): a gates file that declares a line band and no
            # `length_gate` was written when lines were the gate. Read it as it
            # was written, so an older run's own RUN_GATES.yaml keeps its meaning.
            if "sung_lines" in loaded and "length_gate" not in loaded:
                gates["length_gate"] = "lines"
        else:
            print(f"WARN: gates.yaml did not parse to a mapping; using defaults", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 — fail open on ANYTHING
        print(f"WARN: could not load gates.yaml ({exc}); using built-in defaults", file=sys.stderr)
    return gates


def _tiny_yaml_parse(raw: str) -> dict:
    """Last-resort, dependency-free parser for the FLAT gates.yaml subset.

    Handles ``key: scalar``, ``key: [a, b]`` inline lists, and ``- item`` block
    lists. Only used when PyYAML is unavailable. Best-effort; on any oddity the
    caller still falls back to defaults for unparsed keys.
    """
    out: dict = {}
    cur_key = None
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith(("-", " ")) and cur_key is not None and line.lstrip().startswith("-"):
            val = line.lstrip()[1:].strip()
            out.setdefault(cur_key, [])
            if isinstance(out[cur_key], list):
                out[cur_key].append(_coerce_scalar(val))
            continue
        if ":" in line and not line.startswith(" "):
            key, _, rest = line.partition(":")
            key = key.strip()
            rest = rest.split("#", 1)[0].strip()
            if rest == "":
                out[key] = []
                cur_key = key
            elif rest.startswith("[") and rest.endswith("]"):
                items = [x.strip() for x in rest[1:-1].split(",") if x.strip()]
                out[key] = [_coerce_scalar(x) for x in items]
                cur_key = None
            else:
                out[key] = _coerce_scalar(rest)
                cur_key = None
    return out


def _coerce_scalar(s: str):
    s = s.strip().strip("'\"")
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        if re.fullmatch(r"-?\d*\.\d+", s):
            return float(s)
    except Exception:
        pass
    return s


# --------------------------------------------------------------------------
# GATE_REPORT.json — countable-subset rows {pair, step, check, expected, actual, pass}
# --------------------------------------------------------------------------

# A standalone refrain/chorus block is EXEMPT from the repeated-line/collapse
# flag. Lines belonging to a section whose header names one of these are skipped.
_CHORUS_HEADER_RE = re.compile(r"\b(chorus|refrain|hook|drop)\b", re.I)


def _pair_id_from_path(path: Path) -> str:
    m = re.search(r"pair[_-]?(\d+)", path.name.lower())
    return m.group(1).zfill(2) if m else "--"


def _current_step_segment(text: str, step: str) -> str:
    """Return the newest H1 stage section from an exact-prefix artifact."""
    number = str(int(step)) if str(step).isdigit() else re.escape(str(step))
    matches = list(re.finditer(
        rf"^#\s+.*\bstep\s*0?{number}\b.*$",
        text,
        re.I | re.M,
    ))
    return text[matches[-1].start():] if matches else text


def _complete_step_json(text: str):
    """Parse the JSON model response inside a canonical step artifact.

    Canonical artifacts wrap an otherwise JSON-only model response in provenance,
    deliberation, execution, and validation sections. The step contract's
    "JSON only" rule applies to ``## 4. Complete Step Output`` rather than to the
    whole evidence-bearing artifact, so validators must inspect that section.
    """
    match = re.search(
        r"^##\s*4\.\s*complete step output\b\s*(.*?)(?=^##\s*5\.|\Z)",
        text,
        re.I | re.M | re.S,
    )
    if not match:
        raise ValueError("missing ## 4. Complete Step Output section")
    body = "\n".join(line for line in match.group(1).splitlines() if not is_fence(line)).strip()
    start = body.find("{")
    end = body.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Complete Step Output contains no JSON object")
    trailing = body[end + 1:].strip()
    if trailing:
        raise ValueError("Complete Step Output contains text after the JSON object")
    return json.loads(body[start:end + 1])


def _step10_json_items(text: str):
    """Read the current two-revised/two-synthesized contract, or legacy prose.

    The editorial contract returns JSON. Do not require authors to duplicate
    every field as Markdown just to satisfy a reader written for old wraps.
    """
    section = re.search(
        r"^##\s*4\.\s*complete step output\b\s*(.*?)(?=^##\s*5\.|\Z)",
        text, re.I | re.M | re.S,
    )
    if not section or not re.match(r"\s*(?:```json\s*|\{)", section[1]):
        return None
    payload = _complete_step_json(text)
    if not isinstance(payload, dict) or set(payload) != {"revised_prompts", "synthesized_prompts"}:
        raise ValueError("Step10 JSON requires exactly revised_prompts and synthesized_prompts")
    items = []
    required = {"title", "music_prompt", "exclude_prompt", "lyrics_prompt"}
    for key in ("revised_prompts", "synthesized_prompts"):
        values = payload[key]
        if not isinstance(values, list) or len(values) != 2:
            raise ValueError(f"{key} requires exactly two complete objects")
        for item in values:
            if not isinstance(item, dict) or set(item) != required:
                raise ValueError(f"{key} objects require exactly {sorted(required)}")
            if any(not isinstance(value, str) or not value.strip() for value in item.values()):
                raise ValueError(f"{key} contains an empty or non-string field")
            items.append(item)
    return items


def _step10_field_view(item: dict) -> str:
    """Lossless field view for the same existing checks; never writes an artifact."""
    return ("## 1. MUSIC PROMPT\n\n```text\n" + item["music_prompt"] + "\n```\n\n"
            "## 1B. SUNO EXCLUDE PROMPT\n\n```text\n" + item["exclude_prompt"] + "\n```\n\n"
            "## 2. LYRICS\n\n```text\n" + item["lyrics_prompt"] + "\n```\n\n"
            "## 3. TITLE\n\n" + item["title"] + "\n")


def is_fence(line: str) -> bool:
    """True for a markdown code-fence line.

    THE SINGLE SOURCE OF TRUTH for fence handling in this file.

    House format wraps every paste-ready Suno field (music prompt, exclude,
    lyrics, title, dedication) in a ``` fence. A fence is DELIMITER, never
    content: it is not prompt text and it is not a sung line.

    This helper exists because that rule was re-derived independently at FOUR
    sites and got it right at only one. The consequences, all real and all
    measured on 2026-08-22:
      * the step-10 hard-fail path counted the fence as the first lyric line,
        so `[Theme:]` never matched and EVERY correctly-formatted package in
        the repo hard-failed -- which taught four pair agents to delete the
        fence to appease it;
      * the same path measured music prompts +8 chars (two fences + a join
        newline), so a true 993-char prompt failed as 1001;
      * the gate-report lyrics branch counted both fences as SUNG LINES, +2
        on every package.
    Two of those were patched separately earlier the same day and the third
    survived, which is the whole argument for a helper over an idiom: a
    duplicated rule is not fixed until every copy is gone.
    """
    return line.strip().startswith("```")


def _gate_row(rows: list, pair: str, step: str, check: str, expected, actual, passed) -> None:
    rows.append({
        "pair": pair,
        "step": step,
        "check": check,
        "expected": expected,
        "actual": actual,
        "pass": bool(passed) if passed is not None else None,
    })


def _non_chorus_lyric_lines(lyrics_body: str) -> list[str]:
    """Return sung lines OUTSIDE any chorus/refrain section (collapse-flag scope)."""
    out: list[str] = []
    in_exempt = False
    for ln in lyrics_body.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("[") and s.endswith("]"):
            in_exempt = bool(_CHORUS_HEADER_RE.search(s))
            continue
        if s.startswith("#") or is_fence(s) or (s.startswith("*") and s.endswith("*")):
            continue
        if in_exempt:
            continue
        out.append(s.lower())
    return out


def build_gate_report(step: str, path: Path, text: str, gates: dict) -> list[dict]:
    """Score ONLY the countable subset and return GATE_REPORT rows.

    Pure measurement — never decides taste. Banned-opener / artist-name rows are
    the only hard-fail-eligible rows; the collapse-ratio row is a FLAG (pass=None).
    This function NEVER raises: any internal error is swallowed so the report is
    advisory and fail-open.
    """
    rows: list[dict] = []
    pair = _pair_id_from_path(path)
    text = _current_step_segment(text, step)
    lower = text.lower()
    try:
        is_music = "music" in str(path).lower() or "song" in lower or "lyrics" in lower

        # Step 00 — taxonomy cardinality + byte floor + JSON validity.
        if step == "00":
            min_bytes = int(gates.get("step00_min_bytes", 2000))
            actual_bytes = len(text.encode("utf-8", errors="replace"))
            _gate_row(rows, pair, step, "step00_min_bytes", f">= {min_bytes}", actual_bytes, actual_bytes >= min_bytes)
            card = int(gates.get("taxonomy_cardinality", 50))
            for axis in ("aesthetic", "emotion", "frame", "genre"):
                n = len(re.findall(rf"\b{axis}s?\b", lower))
                # cardinality is best measured against a JSON payload if present; the
                # row reports the band and lets the prose §4 gate confirm the count.
                _gate_row(rows, pair, step, f"taxonomy_cardinality_{axis}", f"== {card}", None, None)

        # Image scene floor + banned imperative opener (HARD) + ban-words (FLAG).
        if not is_music and step in {"08", "09", "10"}:
            words = len(re.findall(r"\b\w+\b", text))
            min_words = int(gates.get("image_min_words", 80))
            _gate_row(rows, pair, step, "image_min_words", f">= {min_words}", words, words >= min_words)

        # Banned imperative opener — UNAMBIGUOUS HARD FAIL with offending token.
        opener_token = _banned_opener_token(text, gates)
        if not is_music:
            _gate_row(rows, pair, step, "banned_imperative_opener",
                      "no Create/Design/Make/Render/Depict opener",
                      opener_token or "none", opener_token is None)
            # ban-words: FLAG only (pass=None) — prose §4 / human decides REPAIR.
            hits = [w for w in gates.get("ban_words", []) if w.lower() in lower]
            if hits:
                _gate_row(rows, pair, step, "image_ban_words(FLAG)", "none present", ", ".join(hits), None)

        # Music prompt char band + lyrics-field cap + sung-line band.
        if is_music and step == "10":
            ps = re.search(r"^##\s*1\.\s*music prompt\b\s*(.*?)(?=^##\s*2\.\s*lyrics\b|^##\s+|\Z)",
                           text, re.I | re.M | re.S)
            lo, hi = _band(gates.get("music_prompt_chars", [850, 1000]), 850, 1000)
            if ps:
                body = "\n".join(l.strip() for l in ps.group(1).splitlines()
                                 if l.strip() and not l.strip().startswith("#")
                                 and not is_fence(l)).strip()
                # Strip "*(measured NNN chars)*" / "*(paste into Suno Style)*"
                # style annotation lines so annotations never count as prompt text.
                body = re.sub(r"^\*\([^)]*\)\*\s*$", "", body, flags=re.M).strip()
                n = len(body)
                _gate_row(rows, pair, step, "music_prompt_chars", f"{lo}-{hi}", n, lo <= n <= hi)

                # Terminal punctuation — UNAMBIGUOUS HARD FAIL (sense floor, not
                # taste). A prompt truncated mid-phrase to fit the cap shipped on
                # 2026-06-28 while the count-only check PASSed.
                if gates.get("music_prompt_terminal_punctuation", True) and body:
                    ends_clean = body[-1] in ".!?…" or body[-2:] in ('."', ".'", '!"', '?"')
                    _gate_row(rows, pair, step, "music_prompt_terminal_punctuation",
                              "prompt ends as a complete sentence (. ! ? …)",
                              body[-40:], ends_clean)

                # Boundary-hugging — FLAG only (pass=None). Gate-optimization
                # pressure pins prompts at the cap; write into the mid-band.
                hug = int(gates.get("music_prompt_hug_ceiling", 985))
                if n >= hug:
                    _gate_row(rows, pair, step, "music_prompt_boundary_hugging(FLAG)",
                              f"< {hug} (write into the mid-band, not the cap)", n, None)

                # House-lexicon self-copying guard — FLAG only. Golden references
                # teach the MOVE, never the words (2026-07-01 regression review).
                lex_hits = [p for p in gates.get("house_lexicon", [])
                            if str(p).lower() in body.lower()]
                if lex_hits:
                    _gate_row(rows, pair, step, "house_lexicon(FLAG)",
                              "no calcified golden-output phrases",
                              ", ".join(lex_hits), None)
            lyr = re.search(r"^##\s*2\.\s*lyrics\b\s*(.*?)(?=^##\s+|\Z)", text, re.I | re.M | re.S)
            if lyr:
                field = lyr.group(1)
                clean_field = "\n".join(
                    ln for ln in field.splitlines() if not is_fence(ln)
                ).strip()
                cap = int(gates.get("suno_lyrics_field_max", 5000))
                _gate_row(rows, pair, step, "suno_lyrics_field_max", f"< {cap}", len(field), len(field) < cap)
                target_ceil = int(gates.get("suno_lyrics_field_target", 3800))
                if len(clean_field) > target_ceil:
                    _gate_row(rows, pair, step, "suno_lyrics_field_target(FLAG)",
                              f"<= {target_ceil} (v6 middle-ground target 2600-3600)", len(clean_field), None)
                length_gate = str(gates.get("length_gate", "words")).strip().lower()
                slo, shi = _band(gates.get("sung_lines", [40, 100]), 40, 100)
                sung_lines_list = [s for ln in field.splitlines()
                                   if (s := ln.strip()) and not s.startswith("#")
                                   and not is_fence(s)
                                   and not (s.startswith("[") and s.endswith("]"))
                                   and not (s.startswith("*") and s.endswith("*"))]
                sung = len(sung_lines_list)
                sung_words = sum(len(s.split()) for s in sung_lines_list)
                wlo, whi = _band(gates.get("sung_words_band", [340, 460]), 340, 460)
                tlo, thi = _band(gates.get("sung_words_target", [370, 430]), 370, 430)
                if length_gate == "lines":
                    # Pre-2026-11-25 behaviour, for a gates file written that way.
                    _gate_row(rows, pair, step, "sung_lines", f"{slo}-{shi}", sung, slo <= sung <= shi)
                    floor_hug = int(gates.get("sung_lines_floor_hug", 48))
                    if slo <= sung <= floor_hug:
                        _gate_row(rows, pair, step, "sung_lines_floor_hugging(FLAG)",
                                  f"> {floor_hug} (the floor is a floor, not a target)", sung, None)
                else:
                    # V6-19: sung words are the length gate (a usual 3:30 at about 400).
                    # The unit is this reader's own: whitespace tokens on the sung lines.
                    hlo = int(gates.get("sung_words_hard_floor", 300))
                    hhi = int(gates.get("sung_words_hard_ceiling", 520))
                    _gate_row(rows, pair, step, "sung_words", f"{hlo}-{hhi}", sung_words,
                              hlo <= sung_words <= hhi)
                    if not (slo <= sung <= shi):
                        _gate_row(rows, pair, step, "sung_lines(ADVISORY)",
                                  f"{slo}-{shi} (a form choice; length is set by sung words)", sung, None)
                if sung_words < wlo or sung_words > whi:
                    _gate_row(rows, pair, step, "sung_words(FLAG)",
                              f"{wlo}-{whi} (target {tlo}-{thi}, about 3:30, unless form-declared)",
                              sung_words, None)
                if sung > 0:
                    wpl = round(sung_words / sung, 2)
                    wpl_ceil = float(gates.get("mean_words_per_line_ceiling", 6.8))
                    if wpl > wpl_ceil:
                        _gate_row(rows, pair, step, "mean_words_per_line(FLAG)",
                                  f"<= {wpl_ceil} (5.0-6.8 w/l natural v6 singing range)", wpl, None)
                sung_chars = sum(len(s) for s in sung_lines_list)
                ovh_chars = max(0, len(clean_field) - sung_chars)
                ovh_pct = round(100.0 * ovh_chars / len(clean_field), 1) if clean_field else 0.0
                ovh_flag_c = int(gates.get("lyrics_overhead_flag_chars", 1300))
                ovh_flag_p = float(gates.get("lyrics_overhead_flag_pct", 40.0))
                if ovh_chars > ovh_flag_c or ovh_pct > ovh_flag_p:
                    _gate_row(rows, pair, step, "lyrics_markup_overhead(FLAG)",
                              f"<= {ovh_flag_c}c and <= {ovh_flag_p}% (target <= 1100c)",
                              f"{ovh_chars}c ({ovh_pct}%)", None)
                if not gates.get("disc_channel_in_lyrics_box", False) and re.search(r"^\[Disc_", clean_field, re.M | re.I):
                    _gate_row(rows, pair, step, "disc_channel_in_lyrics_box(FLAG)",
                              "keep [Disc_*] in ## Production Sidecar outside pasted Suno lyrics",
                              "present in lyrics box", None)

                # One-fact rule — FLAG only. At most ONE numeric fact is sung;
                # research informs theme/form, it is not recited in meter.
                fact_re = re.compile(
                    r"\d|(?:\bhundred\b|\bthousand\b|\bmillion\b|\bpercent\b|"
                    r"\bkilometers?\b|\bkm\b|\bdegrees\b|\bmagnitude\b)", re.I)
                fact_lines = [s for s in sung_lines_list if fact_re.search(s)]
                max_facts = int(gates.get("max_sung_numeric_facts", 1))
                if len(fact_lines) > max_facts:
                    _gate_row(rows, pair, step, "max_sung_numeric_facts(FLAG)",
                              f"<= {max_facts} sung numeric-fact line(s), responded to not recited",
                              len(fact_lines), None)

                # House-lexicon guard over the lyrics field — FLAG only.
                lyr_lex_hits = [p for p in gates.get("house_lexicon", [])
                                if str(p).lower() in field.lower()]
                if lyr_lex_hits:
                    _gate_row(rows, pair, step, "house_lexicon_lyrics(FLAG)",
                              "no calcified golden-output phrases",
                              ", ".join(lyr_lex_hits), None)

                # Repeated-line collapse — FLAG ONLY, chorus/refrain EXEMPT.
                non_chorus = _non_chorus_lyric_lines(field)
                if len(non_chorus) >= 6:
                    ratio = len(set(non_chorus)) / len(non_chorus)
                    floor = float(gates.get("unique_line_ratio_floor", 0.45))
                    _gate_row(rows, pair, step, "unique_line_ratio(FLAG, chorus-exempt)",
                              f">= {floor}", round(ratio, 3), None)
    except Exception as exc:  # noqa: BLE001 — report stays advisory / fail-open
        print(f"WARN: gate-report scoring hit a snag ({exc}); rows are partial", file=sys.stderr)
    return rows


def _band(val, dlo, dhi):
    try:
        if isinstance(val, (list, tuple)) and len(val) == 2:
            return int(val[0]), int(val[1])
    except Exception:
        pass
    return dlo, dhi


def _banned_opener_token(text: str, gates: dict):
    """Return the offending banned-imperative-opener token, or None.

    Scans the first non-blank, non-heading, non-bracket content line — the place
    an image scene actually opens. UNAMBIGUOUS: an opener here is a hard fail.
    """
    openers = [str(o) for o in gates.get("banned_imperative_openers", [])]
    if not openers:
        return None
    # Narrow to the actual scene body. In a Lofn step artifact the scene prose
    # lives under the creative/output section, not at the top of the file (which
    # is provenance). Prefer that section; fall back to the whole text.
    scene = text
    sec = re.search(
        r"^##\s*\d+\.\s*(?:complete step output|model response / creative work|"
        r"creative work|step output)\b\s*(.*?)(?=^##\s+\d+\.|\Z)",
        text, re.I | re.M | re.S,
    )
    if sec and sec.group(1).strip():
        scene = sec.group(1)
    opener_set = {o.lower() for o in openers}
    for ln in scene.splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or is_fence(s) or (s.startswith("[") and s.endswith("]")):
            continue
        first = re.match(r"^([A-Za-z]+)", s)
        if first and first.group(1).lower() in opener_set:
            return first.group(1)
        break  # only the genuine first content line opens the scene
    return None


def hard_fail_from_report(rows: list[dict]) -> str | None:
    """Return a message for the FIRST unambiguous hard-fail row, else None.

    ONLY the banned-imperative-opener and terminal-punctuation rows are treated
    as deterministic hard fails here (with the offending evidence). FLAG rows
    (pass is None) and soft count rows never hard-fail from this path — the
    prose §4 gate / human owns those, keeping the script fail-open and taste-free.
    """
    for r in rows:
        if r.get("check") == "banned_imperative_opener" and r.get("pass") is False:
            return f"image scene opens with banned imperative '{r.get('actual')}' (noun-first scene required)"
        if r.get("check") == "music_prompt_terminal_punctuation" and r.get("pass") is False:
            return (f"MUSIC PROMPT is truncated / ends mid-phrase (…'{r.get('actual')}') — "
                    "a prompt must end as a complete sentence; trim a clause, don't hit the cap")
    return None


def write_gate_report(rows: list[dict], out: Path) -> None:
    try:
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 — never let report I/O break a run
        print(f"WARN: could not write GATE_REPORT.json ({exc})", file=sys.stderr)


# --------------------------------------------------------------------------
# Stage 5 meta-check: prose/YAML threshold disagreement scan (WARN only).
# --------------------------------------------------------------------------

def meta_check(root: Path | None = None) -> int:
    """Scan skill/reference files for restated numbers that DISAGREE with gates.yaml.

    WARN-only and fail-open: returns 0 even on disagreement (the harness surfaces
    it for a human to reconcile; this script never blocks a run on it). Returns
    non-zero ONLY if the scan itself cannot run.
    """
    root = root or _repo_root()
    gates = load_gates()
    # Map a small set of single-sourced numbers to the literal tokens we expect
    # to co-occur with them in prose, so we only flag a genuine contradiction.
    expectations = []
    mpc = _band(gates.get("music_prompt_chars", [850, 1000]), 850, 1000)
    mpc_target = _band(gates.get("music_prompt_chars_target", [870, 960]), 870, 960)

    def music_band_ok(match: re.Match) -> bool:
        band = (int(match.group(1)), int(match.group(2)))
        if band == mpc or not (700 <= band[0] <= 1200):
            return True
        # The target and hard band are separate YAML values. Accept the target
        # only when this particular range is explicitly labelled as a target;
        # an unlabelled replacement of the hard band must still warn.
        before = re.sub(r"[*_]", "", match.string[max(0, match.start() - 60):match.start()])
        after = re.sub(r"[*_]", "", match.string[match.end():match.end() + 30])
        labelled_target = (re.search(r"\btarget(?:\s+band)?[\s:=(]*$", before, re.I)
                           or re.match(r"[\s)]*target\b", after, re.I))
        return band == mpc_target and bool(labelled_target)

    expectations.append(("music prompt 850-1000 band", re.compile(r"music\s+(?:style\s+)?prompt", re.I),
                         re.compile(r"\b(\d{3,4})\s*[-–—to]+\s*(\d{3,4})\b"),
                         music_band_ok))
    cap = int(gates.get("suno_lyrics_field_max", 5000))
    expectations.append(("suno lyrics field cap", re.compile(r"lyrics?[\s-]*field|suno.*cap|<\s*\d{4}", re.I),
                         re.compile(r"<\s*(\d{4})\b"),
                         lambda m: int(m.group(1)) == cap or not (4000 <= int(m.group(1)) <= 6000)))

    scan_dirs = [root / ".claude" / "skills", root / ".agents" / "skills", root / "skills", root / "vault"]
    # Skiplist: paths under these prefixes are NEVER meta-checked. Studio knob
    # files (tools/explorer/flows/**) legitimately RESTATE the gate numbers (they
    # bind them via gates_key), so if the scan is ever extended to cover them they
    # must not warn. flows/** is not in scan_dirs today, but this keeps the skip
    # explicit and future-proof (plan §12: drift detection may extend to flows).
    skip_prefixes = [root / "tools" / "explorer" / "flows"]

    def _is_skipped(fp: Path) -> bool:
        rp = fp.resolve()
        for pref in skip_prefixes:
            try:
                rp.relative_to(pref.resolve())
                return True
            except ValueError:
                continue
        return False

    disagreements = 0
    files_scanned = 0
    try:
        for base in scan_dirs:
            if not base.exists():
                continue
            for fp in base.rglob("*.md"):
                if _is_skipped(fp):
                    continue
                try:
                    txt = fp.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                files_scanned += 1
                low = txt.lower()
                for label, ctx_re, num_re, ok in expectations:
                    if not ctx_re.search(low):
                        continue
                    for m in num_re.finditer(txt):
                        try:
                            if not ok(m):
                                print(f"WARN: {fp} restates '{label}' as '{m.group(0)}' "
                                      f"which disagrees with vault/gates.yaml — reconcile (prose §4 is authoritative)",
                                      file=sys.stderr)
                                disagreements += 1
                        except Exception:
                            continue
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: meta-check could not complete ({exc})", file=sys.stderr)
        return 2
    print(f"META-CHECK: scanned {files_scanned} files; {disagreements} prose/YAML disagreement(s) flagged "
          f"(WARN-only, non-blocking).")
    return 0


def _print_help() -> int:
    print(__doc__)
    return 0


def fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def main() -> int:
    argv = sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help", "help"):
        return _print_help()

    if argv[0] == "--meta-check":
        root = None
        if "--root" in argv:
            i = argv.index("--root")
            if i + 1 < len(argv):
                root = Path(argv[i + 1])
        return meta_check(root)

    emit_report = False
    out_path = None
    gates_path = None
    expected_prompts = 6
    if argv and argv[0] == "--gate-report":
        emit_report = True
        argv = argv[1:]
        if "--out" in argv:
            i = argv.index("--out")
            if i + 1 < len(argv):
                out_path = Path(argv[i + 1])
            argv = [a for j, a in enumerate(argv) if j not in (i, i + 1)]

    # --gates PATH: use a specific gates.yaml (Studio run-local, knob-adjusted).
    # Threads into the EXISTING load_gates(path). Without the flag, byte-identical
    # to before (the default vault/gates.yaml is used, exactly as always).
    if "--gates" in argv:
        i = argv.index("--gates")
        if i + 1 < len(argv):
            gates_path = Path(argv[i + 1])
            argv = [a for j, a in enumerate(argv) if j not in (i, i + 1)]
        else:
            return fail("--gates requires a PATH argument")

    if "--expected-prompts" in argv:
        i = argv.index("--expected-prompts")
        if i + 1 >= len(argv) or argv[i + 1] not in {"4", "6"}:
            return fail("--expected-prompts requires 4 or 6")
        expected_prompts = int(argv[i + 1])
        argv = [a for j, a in enumerate(argv) if j not in (i, i + 1)]

    if len(argv) != 2:
        return fail("usage: validate_step.py [--gate-report [--out PATH]] [--gates PATH] [--expected-prompts 4|6] <step> <file>  |  --meta-check  |  --help")
    step = str(argv[0]).zfill(2)
    path = Path(argv[1])
    return _validate(step, path, emit_report=emit_report, out_path=out_path, gates_path=gates_path,
                     expected_prompts=expected_prompts)


def _validate_music_step10_fields(music_text: str, gates_path: Path | None = None):
    if "lyrics" not in music_text.lower():
        return fail("music song artifact missing lyrics")
    prompt_matches = list(re.finditer(r"^##\s*1\.\s*music prompt\b|^\[suno style prompt\s*:\]", music_text, re.I | re.M))
    prompt_count = len(prompt_matches)
    lyric_count = len(re.findall(r"^##\s*2\.\s*lyrics\b", music_text, re.I | re.M))
    song_form_count = len(re.findall(r"\[song form\s*:[^\]]+\]", music_text, re.I))
    theme_count = len(re.findall(r"\[theme\s*:[^\]]+\]", music_text, re.I))
    full_emo_headers = len(re.findall(r"^\[[^\]\n]+[-–—]\s*EMO\s*:[^\]\n]+[-–—][^\]\n]+[-–—][^\]\n]+\]", music_text, re.I | re.M))
    bare_emo_headers = len(re.findall(r"^\[\s*EMO\s*:", music_text, re.I | re.M))
    if prompt_count < 1:
        return fail("music song artifact missing standalone ## 1. MUSIC PROMPT")
    # Enforce the restored legacy Suno prompt band. Extract text between the
    # prompt heading and lyrics heading (or next h2) and measure non-heading content.
    prompt_section = re.search(r"^##\s*1\.\s*music prompt\b\s*(.*?)(?=^##\s*2\.\s*lyrics\b|^##\s+|\Z)", music_text, re.I | re.M | re.S)
    if prompt_section:
        # Same extraction as the gate-report path above (step 08/10 music
        # prompt band): strip markdown code fences and "*(annotation)*"
        # lines so neither counts as prompt text. These two paths measure
        # the SAME quantity and MUST agree -- when they diverged, the
        # gate-report path read the true length while this one read it
        # +8 chars (two fences plus a newline) and hard-failed the house
        # format outright. One implementation per measurement.
        prompt_body = "\n".join(
            ln.strip() for ln in prompt_section.group(1).splitlines()
            if ln.strip() and not ln.strip().startswith("#")
            and not is_fence(ln)
        ).strip()
        prompt_body = re.sub(r"^\*\([^)]*\)\*\s*$", "", prompt_body, flags=re.M).strip()
        prompt_chars = len(prompt_body)
        if prompt_chars < 850 or prompt_chars > 1000:
            return fail(f"music prompt must be 850-1000 characters, found {prompt_chars}")
        first_prompt_line = next((ln.strip() for ln in prompt_body.splitlines() if ln.strip()), "")
        if re.match(r"^(begin\s+(in|by|with)\b|use\b|build\s+the\s+track\s+from\b|chronology\s*:|for\s+an\s+adult\s+human\s+singer\b)", first_prompt_line, re.I):
            return fail("music prompt opens with banned procedural/narrative phrasing; lead with genre/style + tempo/energy + vocalist + instrumentation")
    if lyric_count < 1:
        return fail("music song artifact missing ## 2. LYRICS section")
    if theme_count < lyric_count:
        return fail(f"music song artifact missing bracketed [Theme:] declarations ({theme_count}/{lyric_count})")
    if song_form_count < lyric_count:
        return fail(f"music song artifact missing bracketed [SONG FORM:] declarations ({song_form_count}/{lyric_count})")
    lyrics_section_for_header = re.search(r"^##\s*2\.\s*lyrics\b\s*(.*?)(?=^##\s+|\Z)", music_text, re.I | re.M | re.S)
    if lyrics_section_for_header:
        content_lines = [ln.strip() for ln in lyrics_section_for_header.group(1).splitlines() if ln.strip()]
        # Code fences are house format for a paste-ready Suno field; they are
        # not lyric content. Without this the fence IS non_title_lines[0] and
        # [Theme:] can never match -- which silently hard-failed every
        # correctly-formatted package this repo has ever produced.
        non_title_lines = [
            ln for ln in content_lines
            if not ln.startswith("#") and not is_fence(ln)
        ]
        if len(non_title_lines) < 2 or not re.match(r"^\[theme\s*:[^\]]+\]$", non_title_lines[0], re.I) or not re.match(r"^\[song form\s*:[^\]]+\]$", non_title_lines[1], re.I):
            return fail("music lyrics must begin with [Theme: ...] immediately followed by [SONG FORM: ...]")
    if full_emo_headers < max(6, lyric_count * 4):
        return fail(f"music song artifact has too few full section EMO headers ({full_emo_headers}); bare [EMO:...] is not enough")
    if bare_emo_headers:
        return fail("music song artifact uses bare [EMO:...] headers instead of [Section - EMO:... - Voice - Cue]")
    if re.search(r"^\s*EMO\s+HEADER\s*:", music_text, re.I | re.M):
        return fail("music song artifact contains prose EMO HEADER lines")
    if re.search(r"^\s*SONG FORM\s*:", music_text, re.I | re.M):
        return fail("music song artifact contains plain SONG FORM: text instead of [SONG FORM: ...]")
    if not re.search(r"^\*[^*\n]{1,50}\*\s*$", music_text, re.M):
        return fail("music song artifact missing standalone SFX cue")
    lyrics_section = re.search(r"^##\s*2\.\s*lyrics\b\s*(.*?)(?=^##\s+|\Z)", music_text, re.I | re.M | re.S)
    if lyrics_section:
        sung_lines = 0
        sung_words = 0
        for ln in lyrics_section.group(1).splitlines():
            s = ln.strip()
            if (not s or s.startswith("#") or is_fence(s)
                    or (s.startswith("[") and s.endswith("]"))
                    or (s.startswith("*") and s.endswith("*"))):
                continue
            sung_lines += 1
            sung_words += len(s.split())
        lyric_gates = load_gates(gates_path)
        if str(lyric_gates.get("length_gate", "words")).strip().lower() == "lines":
            # Pre-2026-11-25 behaviour, for a gates file written that way.
            sung_floor = int(lyric_gates.get("sung_lines_hard_floor", 45))
            sung_ceiling_raw = lyric_gates.get("sung_lines_hard_ceiling")
            sung_ceiling = int(sung_ceiling_raw) if sung_ceiling_raw is not None else None
            sung_target = lyric_gates.get("sung_lines", [52, 76])
            if sung_lines < sung_floor:
                return fail(
                    f"music lyrics have too few sung lines ({sung_lines}); "
                    f"<{sung_floor} triggers repair, target {sung_target[0]}-{sung_target[1]}"
                )
            if sung_ceiling is not None and sung_lines > sung_ceiling:
                return fail(
                    f"music lyrics have too many sung lines ({sung_lines}); "
                    f">{sung_ceiling} triggers repair, target {sung_target[0]}-{sung_target[1]}"
                )
        else:
            # V6-19: the length gate is sung words (whitespace tokens on sung lines).
            word_floor = int(lyric_gates.get("sung_words_hard_floor", 300))
            word_ceiling = int(lyric_gates.get("sung_words_hard_ceiling", 520))
            word_target = lyric_gates.get("sung_words_target", [370, 430])
            if sung_words < word_floor:
                return fail(
                    f"music lyrics have too few sung words ({sung_words} in {sung_lines} lines); "
                    f"<{word_floor} triggers repair, target {word_target[0]}-{word_target[1]} (about 3:30)"
                )
            if sung_words > word_ceiling:
                return fail(
                    f"music lyrics have too many sung words ({sung_words} in {sung_lines} lines); "
                    f">{word_ceiling} triggers repair unless the form is declared in RUN_GATES.yaml, "
                    f"target {word_target[0]}-{word_target[1]} (about 3:30)"
                )


def _validate(step: str, path: Path, emit_report: bool = False, out_path: Path | None = None,
              gates_path: Path | None = None, expected_prompts: int = 6) -> int:
    if expected_prompts not in {4, 6}:
        return fail("expected_prompts must be 4 or 6")
    if not path.exists() or not path.is_file():
        return fail(f"missing file: {path}")
    name = path.name
    canonical = (
        name.startswith("step")
        or (name.startswith("pair_") and "_step" in name)
    )
    if not canonical:
        print(f"STEP {step} SKIPPED: {path} is not a canonical step artifact")
        return 0

    text = path.read_text(encoding="utf-8", errors="replace")
    lower = text.lower()

    if len(text.strip()) < 800:
        return fail("artifact is too short to be a real Lofn step")
    # 2026-10-01 (run 2026-09-30-caterpillar-survivors): this stub-word scan ran over the
    # WHOLE file, so one word inside the frozen, preflight-checked creative-context block
    # (a root-written heading describing an omitted empty shell) failed every ICB-prefixed
    # artifact regardless of its body. The block is bounded by explicit BEGIN/END markers
    # and is validated once by check_icb_preflight.py; exclude exactly that span here.
    # The step body stays fully subject to the check. A block with no END marker is not
    # stripped, so a malformed artifact is still scanned whole (fail-closed).
    # Same day, later: the strip now goes through the shared helper (step-11 contract:
    # every reader strips the ICB through scripts/icb_span.py), not a fifth local regex.
    body_lower = strip_icb(text, load_icb(str(path.parent)))[0].lower()
    if re.search(r"\b(lorem ipsum|todo|tbd|placeholder|similar arrangement|song n|genre n)\b", body_lower):
        return fail("artifact contains placeholder/template language")
    # Catch numbered lyric/content stubs only when they are the line's actual
    # label.  The prior unanchored search also rejected legitimate provenance
    # such as "the archive begins at source line 327", which made a byte-exact
    # immutable context block impossible to carry through the pipeline.
    if re.search(
        r"^\s*(?:(?:[-*]|\d+[.)])\s*)?line\s+\d+\s*(?::|[-–—]|$)",
        text,
        re.I | re.M,
    ):
        return fail("artifact contains numbered placeholder lyric lines like 'line 1'")
    # Catch copy-paste repetition masquerading as step depth.
    nonempty = [ln.strip().lower() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith('#')]
    if len(nonempty) >= 6:
        unique_ratio = len(set(nonempty)) / len(nonempty)
        if unique_ratio < 0.45:
            return fail(f"artifact is excessively repetitive (unique line ratio {unique_ratio:.2f})")
    paras = [p.strip().lower() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 120]
    if len(paras) >= 3 and len(set(paras)) <= len(paras) // 2:
        return fail("artifact repeats large paragraph blocks instead of developing the step")
    if re.search(r"steps?[_ -]?0?6[_ -]?0?10", path.name.lower()):
        return fail("collapsed Steps 06-10 file is not a canonical step artifact")

    # Startup context is injected into the agent prompt, not proven by padding the
    # saved artifact. Rich provenance wrappers are allowed, but they are no longer
    # a hard gate; the skill progression determines which files are injected at
    # agent start. This validator should judge the artifact body and countable
    # output contract, not force boilerplate.
    has_new_contract = "## 4. complete step output" in lower or "## 5. execution log" in lower
    if has_new_contract:
        for sec in [
            "## 4. Complete Step Output",
        ]:
            if sec.lower() not in lower:
                return fail(f"artifact missing required complete-output section: {sec}")
    # Exact-prefix artifacts carry a large immutable creative-context block.  A
    # personality/library sentence can legitimately contain one of these
    # phrases, so the future-tense stub check must judge the newest step section
    # rather than the injected context.  The step body remains fully subject to
    # the check.
    current_step_lower = _current_step_segment(text, step).lower()
    if (
        "what this step would do" in current_step_lower
        or "would generate" in current_step_lower
        or "would produce" in current_step_lower
    ):
        return fail("artifact describes what the step would do instead of containing complete step output")
    if "panel / critic deliberation log" in lower:
        panel_section = re.search(r"## 3\. Panel / Critic Deliberation Log(.*?)(?=\n## 4\.)", text, re.I | re.S)
        if panel_section:
            ps = panel_section.group(1).lower()
            for marker in ["devil", "hyper-skeptic", "resolution"]:
                if marker not in ps:
                    return fail(f"panel deliberation log missing marker: {marker}")

    checks = {
        "00": ["aesthetic", "emotion", "genre"],
        "01": ["essence", "facet", "style"],
        "02": ["concept"],
        "03": ["artist", "critique"],
        "04": ["medium"],
        "05": ["pair", "concept", "medium"],
        "06": ["facet"],
        "07": ["song guide"],
        "08": ["prompt"],
        "09": ["artist", "refin"],
        "10": ["prompt"],
    }
    for needle in checks.get(step, []):
        if needle not in lower:
            return fail(f"step {step} artifact missing expected marker: {needle}")

    is_music = "music" in str(path).lower() or "song" in lower or "lyrics" in lower

    if step == "05" and is_music:
        pair_json = path.parent / "concept_medium_pairs.json"
        if not pair_json.exists():
            return fail("music Step 05 missing sibling concept_medium_pairs.json")
        try:
            data = json.loads(pair_json.read_text(encoding="utf-8", errors="replace"))
        except Exception as exc:
            return fail(f"concept_medium_pairs.json is not valid JSON: {exc}")
        if not isinstance(data, list):
            return fail("concept_medium_pairs.json must be a top-level list")
        # The concept-pair band comes from the loaded gates dict (Studio run-local
        # gates set `pair_count_band` derived from n_pairs so wide portfolios like
        # 10x2 validate). Fail-open to the canon default [4, 7] when the key is
        # absent — canon vault/gates.yaml is UNCHANGED, so no-flag behavior stays
        # byte-identical.
        pc_lo, pc_hi = _band(load_gates(gates_path).get("pair_count_band", [4, 7]), 4, 7)
        if not (pc_lo <= len(data) <= pc_hi):
            return fail(f"concept_medium_pairs.json must contain {int(pc_lo)}-{int(pc_hi)} pairs, found {len(data)}")
        for i, item in enumerate(data, 1):
            if not isinstance(item, dict):
                return fail(f"concept_medium_pairs.json item {i} is not an object")
            for key in ["pair_num", "concept", "medium"]:
                if key not in item or not str(item[key]).strip():
                    return fail(f"concept_medium_pairs.json item {i} missing required key: {key}")

    if is_music and step == "08":
        try:
            generated = _complete_step_json(_current_step_segment(text, step))
        except Exception as exc:
            return fail(f"music Step 08 Complete Step Output is not valid JSON: {exc}")
        if not isinstance(generated, dict) or set(generated) != {"music_prompts"}:
            return fail("music Step 08 JSON must have exactly one root key: music_prompts")
        prompts = generated.get("music_prompts")
        if not isinstance(prompts, list) or len(prompts) != expected_prompts:
            actual = len(prompts) if isinstance(prompts, list) else "non-list"
            return fail(f"music Step 08 must contain exactly {expected_prompts} prompt objects, found {actual}")
        required = {"music_prompt", "lyrics_prompt", "title"}
        for index, item in enumerate(prompts, 1):
            if not isinstance(item, dict) or not required <= set(item) or set(item) - required - {"exclude_prompt"}:
                return fail(
                    f"music Step 08 prompt {index} must contain "
                    "music_prompt, lyrics_prompt, and title, with only exclude_prompt optional"
                )
            for key in item:
                if not isinstance(item[key], str) or not item[key].strip():
                    return fail(f"music Step 08 prompt {index} has empty/non-string {key}")

    step10_items = None
    if is_music and step == "10":
        music_text = _current_step_segment(text, step)
        try:
            step10_items = _step10_json_items(music_text)
        except (ValueError, TypeError) as exc:
            return fail(f"music Step10 terminal JSON: {exc}")
        views = [_step10_field_view(item) for item in step10_items] if step10_items is not None else [music_text]
        for index, view in enumerate(views, 1):
            if step10_items is not None:
                item = step10_items[index - 1]
                field_gates = load_gates(gates_path)
                lo, hi = _band(field_gates.get("music_prompt_chars", [850, 1000]), 850, 1000)
                if not lo <= len(item["music_prompt"]) <= hi:
                    return fail(f"Step10 JSON candidate {index}: music prompt must be {lo}-{hi} characters, found {len(item['music_prompt'])}")
                cap = int(field_gates.get("suno_lyrics_field_max", 5000))
                if len(item["lyrics_prompt"]) >= cap:
                    return fail(f"Step10 JSON candidate {index}: lyrics field must be <{cap} characters")
            result = _validate_music_step10_fields(view, gates_path)
            if result is not None:
                if step10_items is not None:
                    print(f"Step10 JSON candidate {index} failed the field check")
                return result

    # ----------------------------------------------------------------------
    # L4 countable-subset GATE_REPORT (item 2.1). Fail-open: scoring never raises
    # and a broken gates.yaml falls back to defaults. Only the UNAMBIGUOUS
    # banned-imperative-opener row hard-fails here (with the offending token as
    # evidence); all count/FLAG rows are advisory and routed to the prose §4 gate.
    # ----------------------------------------------------------------------
    try:
        gates = load_gates(gates_path)
        if step10_items is None:
            rows = build_gate_report(step, path, text, gates)
        else:
            rows = []
            for index, item in enumerate(step10_items, 1):
                candidate_rows = build_gate_report(step, path, _step10_field_view(item), gates)
                for row in candidate_rows:
                    row["variant"] = index
                    if row["check"] == "music_prompt_chars":
                        row["actual"] = len(item["music_prompt"])
                        lo, hi = _band(gates.get("music_prompt_chars", [850, 1000]), 850, 1000)
                        row["pass"] = lo <= row["actual"] <= hi
                    if row["check"] == "suno_lyrics_field_max":
                        row["actual"] = len(item["lyrics_prompt"])
                        row["pass"] = len(item["lyrics_prompt"]) < int(gates.get("suno_lyrics_field_max", 5000))
                elo, ehi = _band(gates.get("suno_exclude_chars", [400, 900]), 400, 900)
                _gate_row(candidate_rows, _pair_id_from_path(path), step, "suno_exclude_chars",
                          f"{elo}-{ehi}", len(item["exclude_prompt"]), elo <= len(item["exclude_prompt"]) <= ehi)
                candidate_rows[-1]["variant"] = index
                rows.extend(candidate_rows)
        if emit_report:
            target = out_path or (path.parent / "GATE_REPORT.json")
            write_gate_report(rows, target)
            print(f"GATE_REPORT: {len([r for r in rows if r.get('pass') is True])} pass / "
                  f"{len([r for r in rows if r.get('pass') is False])} fail / "
                  f"{len([r for r in rows if r.get('pass') is None])} flag -> {target}")
        hard = hard_fail_from_report(rows)
        if hard is not None:
            return fail(hard)
    except Exception as exc:  # noqa: BLE001 — the report is a backstop, never a blocker
        print(f"WARN: GATE_REPORT step skipped ({exc}); core validation result stands", file=sys.stderr)

    print(f"STEP {step} PASSED: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
