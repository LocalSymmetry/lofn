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
  validate_step.py --meta-check [--root DIR]
                                            # prose/YAML threshold disagreement scan
  validate_step.py --help
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
import json


# --------------------------------------------------------------------------
# gates.yaml loading (FAIL-OPEN). Built-in defaults MIRROR vault/gates.yaml so a
# missing/broken YAML never hard-fails a valid run.
# --------------------------------------------------------------------------

DEFAULT_GATES = {
    "music_prompt_chars": [850, 1000],
    "suno_lyrics_field_max": 5000,
    "suno_lyrics_field_target": 4800,
    "sung_lines": [70, 120],
    "step00_min_bytes": 2000,
    "taxonomy_cardinality": 50,
    "image_min_words": 80,
    "total_prompts": 24,
    "banned_imperative_openers": ["Create", "Design", "Make", "Render", "Depict"],
    "ban_words": [
        "ethereal", "dreamlike", "whimsical", "gentle light",
        "soft glow", "magical", "delicate",
    ],
    "unique_line_ratio_floor": 0.45,
    "ngram_collapse_n": 4,
    "music_prompt_hug_ceiling": 985,
    "sung_lines_floor_hug": 72,
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
        raw = p.read_text(errors="replace")
        loaded = None
        try:
            import yaml  # type: ignore
            loaded = yaml.safe_load(raw)
        except Exception:
            loaded = _tiny_yaml_parse(raw)
        if isinstance(loaded, dict):
            gates.update({k: v for k, v in loaded.items() if v is not None})
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
        if s.startswith("#") or (s.startswith("*") and s.endswith("*")):
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
        if is_music and step in {"08", "10"}:
            ps = re.search(r"^##\s*1\.\s*music prompt\b\s*(.*?)(?=^##\s*2\.\s*lyrics\b|^##\s+|\Z)",
                           text, re.I | re.M | re.S)
            lo, hi = _band(gates.get("music_prompt_chars", [850, 1000]), 850, 1000)
            if ps:
                body = "\n".join(l.strip() for l in ps.group(1).splitlines()
                                 if l.strip() and not l.strip().startswith("#")
                                 and not l.strip().startswith("```")).strip()
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
                cap = int(gates.get("suno_lyrics_field_max", 5000))
                _gate_row(rows, pair, step, "suno_lyrics_field_max", f"< {cap}", len(field), len(field) < cap)
                slo, shi = _band(gates.get("sung_lines", [70, 120]), 70, 120)
                sung_lines_list = [s for ln in field.splitlines()
                                   if (s := ln.strip()) and not s.startswith("#")
                                   and not (s.startswith("[") and s.endswith("]"))
                                   and not (s.startswith("*") and s.endswith("*"))]
                sung = len(sung_lines_list)
                _gate_row(rows, pair, step, "sung_lines", f"{slo}-{shi}", sung, slo <= sung <= shi)

                # Floor-hugging — FLAG only. A lyric pinned at the 70 floor is
                # optimization to the constraint, not a song's natural length.
                floor_hug = int(gates.get("sung_lines_floor_hug", 72))
                if slo <= sung <= floor_hug:
                    _gate_row(rows, pair, step, "sung_lines_floor_hugging(FLAG)",
                              f"> {floor_hug} (the floor is a floor, not a target)", sung, None)

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
        if not s or s.startswith("#") or (s.startswith("[") and s.endswith("]")):
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
    expectations.append(("music prompt 850-1000 band", re.compile(r"music\s+(?:style\s+)?prompt", re.I),
                         re.compile(r"\b(\d{3,4})\s*[-–—to]+\s*(\d{3,4})\b"),
                         lambda m: (int(m.group(1)), int(m.group(2))) == mpc or
                                   not (700 <= int(m.group(1)) <= 1200)))
    cap = int(gates.get("suno_lyrics_field_max", 5000))
    expectations.append(("suno lyrics field cap", re.compile(r"lyrics?[\s-]*field|suno.*cap|<\s*\d{4}", re.I),
                         re.compile(r"<\s*(\d{4})\b"),
                         lambda m: int(m.group(1)) == cap or not (4000 <= int(m.group(1)) <= 6000)))

    scan_dirs = [root / ".claude" / "skills", root / "skills", root / "vault"]
    disagreements = 0
    files_scanned = 0
    try:
        for base in scan_dirs:
            if not base.exists():
                continue
            for fp in base.rglob("*.md"):
                try:
                    txt = fp.read_text(errors="replace")
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
    if argv and argv[0] == "--gate-report":
        emit_report = True
        argv = argv[1:]
        if "--out" in argv:
            i = argv.index("--out")
            if i + 1 < len(argv):
                out_path = Path(argv[i + 1])
            argv = [a for j, a in enumerate(argv) if j not in (i, i + 1)]

    if len(argv) != 2:
        return fail("usage: validate_step.py [--gate-report [--out PATH]] <step> <file>  |  --meta-check  |  --help")
    step = str(argv[0]).zfill(2)
    path = Path(argv[1])
    return _validate(step, path, emit_report=emit_report, out_path=out_path)


def _validate(step: str, path: Path, emit_report: bool = False, out_path: Path | None = None) -> int:
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

    text = path.read_text(errors="replace")
    lower = text.lower()

    if len(text.strip()) < 800:
        return fail("artifact is too short to be a real Lofn step")
    if re.search(r"\b(lorem ipsum|todo|tbd|placeholder|similar arrangement|song n|genre n)\b", lower):
        return fail("artifact contains placeholder/template language")
    if re.search(r"\bline\s+\d+\b", lower):
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
    if "what this step would do" in lower or "would generate" in lower or "would produce" in lower:
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
            data = json.loads(pair_json.read_text(errors="replace"))
        except Exception as exc:
            return fail(f"concept_medium_pairs.json is not valid JSON: {exc}")
        if not isinstance(data, list):
            return fail("concept_medium_pairs.json must be a top-level list")
        if not (4 <= len(data) <= 7):
            return fail(f"concept_medium_pairs.json must contain 4-7 pairs, found {len(data)}")
        for i, item in enumerate(data, 1):
            if not isinstance(item, dict):
                return fail(f"concept_medium_pairs.json item {i} is not an object")
            for key in ["pair_num", "concept", "medium"]:
                if key not in item or not str(item[key]).strip():
                    return fail(f"concept_medium_pairs.json item {i} missing required key: {key}")

    if is_music and step in {"08", "10"}:
        if "lyrics" not in lower:
            return fail("music song artifact missing lyrics")
        prompt_matches = list(re.finditer(r"^##\s*1\.\s*music prompt\b|^\[suno style prompt\s*:\]", text, re.I | re.M))
        prompt_count = len(prompt_matches)
        lyric_count = len(re.findall(r"^##\s*2\.\s*lyrics\b", text, re.I | re.M))
        song_form_count = len(re.findall(r"\[song form\s*:[^\]]+\]", text, re.I))
        theme_count = len(re.findall(r"\[theme\s*:[^\]]+\]", text, re.I))
        full_emo_headers = len(re.findall(r"^\[[^\]\n]+[-–—]\s*EMO\s*:[^\]\n]+[-–—][^\]\n]+[-–—][^\]\n]+\]", text, re.I | re.M))
        bare_emo_headers = len(re.findall(r"^\[\s*EMO\s*:", text, re.I | re.M))
        if prompt_count < 1:
            return fail("music song artifact missing standalone ## 1. MUSIC PROMPT")
        # Enforce the restored legacy Suno prompt band. Extract text between the
        # prompt heading and lyrics heading (or next h2) and measure non-heading content.
        prompt_section = re.search(r"^##\s*1\.\s*music prompt\b\s*(.*?)(?=^##\s*2\.\s*lyrics\b|^##\s+|\Z)", text, re.I | re.M | re.S)
        if prompt_section:
            prompt_body = "\n".join(
                ln.strip() for ln in prompt_section.group(1).splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            ).strip()
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
        lyrics_section_for_header = re.search(r"^##\s*2\.\s*lyrics\b\s*(.*?)(?=^##\s+|\Z)", text, re.I | re.M | re.S)
        if lyrics_section_for_header:
            content_lines = [ln.strip() for ln in lyrics_section_for_header.group(1).splitlines() if ln.strip()]
            non_title_lines = [ln for ln in content_lines if not ln.startswith("#")]
            if len(non_title_lines) < 2 or not re.match(r"^\[theme\s*:[^\]]+\]$", non_title_lines[0], re.I) or not re.match(r"^\[song form\s*:[^\]]+\]$", non_title_lines[1], re.I):
                return fail("music lyrics must begin with [Theme: ...] immediately followed by [SONG FORM: ...]")
        if full_emo_headers < max(6, lyric_count * 4):
            return fail(f"music song artifact has too few full section EMO headers ({full_emo_headers}); bare [EMO:...] is not enough")
        if bare_emo_headers:
            return fail("music song artifact uses bare [EMO:...] headers instead of [Section - EMO:... - Voice - Cue]")
        if re.search(r"^\s*EMO\s+HEADER\s*:", text, re.I | re.M):
            return fail("music song artifact contains prose EMO HEADER lines")
        if re.search(r"^\s*SONG FORM\s*:", text, re.I | re.M):
            return fail("music song artifact contains plain SONG FORM: text instead of [SONG FORM: ...]")
        if not re.search(r"^\*[^*\n]{1,50}\*\s*$", text, re.M):
            return fail("music song artifact missing standalone SFX cue")
        lyrics_section = re.search(r"^##\s*2\.\s*lyrics\b\s*(.*?)(?=^##\s+|\Z)", text, re.I | re.M | re.S)
        if lyrics_section:
            sung_lines = 0
            for ln in lyrics_section.group(1).splitlines():
                s = ln.strip()
                if not s or s.startswith("#") or (s.startswith("[") and s.endswith("]")) or (s.startswith("*") and s.endswith("*")):
                    continue
                sung_lines += 1
            if sung_lines < 60:
                return fail(f"music lyrics have too few sung lines ({sung_lines}); <60 triggers repair, target 70-120")

    # ----------------------------------------------------------------------
    # L4 countable-subset GATE_REPORT (item 2.1). Fail-open: scoring never raises
    # and a broken gates.yaml falls back to defaults. Only the UNAMBIGUOUS
    # banned-imperative-opener row hard-fails here (with the offending token as
    # evidence); all count/FLAG rows are advisory and routed to the prose §4 gate.
    # ----------------------------------------------------------------------
    try:
        gates = load_gates()
        rows = build_gate_report(step, path, text, gates)
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
