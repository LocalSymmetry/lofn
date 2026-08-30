#!/usr/bin/env python3
"""skadi_import_archive.py — eat what this house already knows, for free.

WHY THIS EXISTS
---------------
The atlas was seeded from ONE render audit (`2026-07-24-eight-song-deep-dive`,
the n=8 behaviour ledger). The rest of the evidence was already on disk and
already paid for: three more render-audit directories, 177 audio masters whose
style prompts are recorded in the Suno snapshots, and one refinement run whose
renders have only just arrived. Nothing here renders, publishes, spends, or
reaches the network. Every number it writes comes from a file a stranger could
re-open.

⛔ ARCHIVE EVIDENCE IS OBSERVATIONAL, NEVER A PLATE. There were no registered
predictions, no controls and no non-obvious slot behind any of it, so it may
CREATE `OBSERVED` claims and INCREMENT existing ones and it may do nothing else:
no plate status, no VOID logic, no predictions, and — by the gate
`atlas_high_requires_plate` — no HIGH confidence at any n. A corpus swept after
the fact is a prior, not an experiment.

⛔ ABSENCE OF ASK IS NOT NEGATIVE EVIDENCE. A track whose style text never asks
for a spec class contributes NOTHING to that class. The only figures computed
over every measured clip regardless of intent are the POPULATION BASELINES
(`BASELINES.md`), which are explicitly not claims: they are the denominator a
claim is read against.

⛔ n COUNTS TRIALS, NOT CONFIRMATIONS. Where an epoch's renders disagree, the row
is `PARTIAL` and its text carries `k of m`. A row that counted only the renders
which agreed with it would be a confirmation counter wearing an n.

⛔ ONE CLIP IS ONE OBSERVATION. The same render appears as a wav master, as an
mp3 inside the wide-corpus audit, and in up to nine snapshots. Evidence is keyed
by CLIP ID and deduplicated there, because n inflated with real evidence paths is
the failure this whole system exists to refuse.

⛔ IT IS IDEMPOTENT BY CONSTRUCTION. Every row it owns is rebuilt from scratch on
each run and every increment is re-applied to a RECORDED BASE (see
`ARCHIVE_EVIDENCE.json` → `increments`), so a second run reproduces the first
byte for byte instead of incrementing the increment.

WHAT IT WRITES (all under vault/music_atlas/)
  BASELINES.md                  per-epoch distributions over every measured clip — NOT claims
  KNOWN_BEHAVIOR.md             the literature lane: external, source-cited, no n, no verdict
  ATLAS.md                      new + incremented rows and one dated append block
  FRONTIER.md                   a drift-candidate note inside the CONTROL CHART section
  seeds/SEED_NOTES.md           the `ARCHIVE IMPORT` section: counts, adapters, exclusions
  seeds/MEASUREMENT_CACHE.json  the incremental measurement cache
  seeds/ARCHIVE_EVIDENCE.json   per-claim: every clip id, epoch, measured value and test

USAGE
    python3 scripts/skadi_import_archive.py --measure-only
    python3 scripts/skadi_import_archive.py --dry-run
    python3 scripts/skadi_import_archive.py

EXIT   0 imported · 1 refused · 2 input error · 3 REFUSED (the validator is red)
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_music_atlas import (  # noqa: E402
    ATLAS_COLS, ATLAS_DIR, APPEND_COLS, REPO, SPEC_CLASS_COLS,
    Report, load_gates, parse_table, read, run_all, sha12, strike,
)
from skadi_bank import cell_parts, splice, table_md  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass


IMPORT_DATE = "2026-08-17"
IMPORT_SLUG = "archive-import"
CHANNEL = "style"          # every ask read here lives in the Suno STYLE field
NOTES_SECTION = "ARCHIVE IMPORT 2026-08-17"

MASTERS = REPO / "output" / "distribution" / "wav_masters"
SNAPSHOT_DIR = REPO / "data" / "snapshots" / "suno"
LATEST = SNAPSHOT_DIR / "latest.json"
AUDITS = REPO / "output" / "render-audits"
BLEND_RUN = REPO / "output" / "2026-08-13_genre-blend-refine"

CACHE = ATLAS_DIR / "seeds" / "MEASUREMENT_CACHE.json"
EVIDENCE = ATLAS_DIR / "seeds" / "ARCHIVE_EVIDENCE.json"
NOTES = ATLAS_DIR / "seeds" / "SEED_NOTES.md"
BASELINES = ATLAS_DIR / "BASELINES.md"
KNOWN = ATLAS_DIR / "KNOWN_BEHAVIOR.md"

AUDIO_SUFFIXES = (".wav", ".mp3")
CLIP8_RE = re.compile(r"__([0-9a-f]{8})$")

DEEP_DIVE = "2026-07-24-eight-song-deep-dive"        # already eaten by the seed
HEAR_LEARN = "2026-08-02-hear-learn"
SPARE_HAIR_TIE = "2026-08-10-spare-hair-tie"
WIDE_CORPUS = "2026-08-10-wide-corpus"

# ⛔ NOT A GUESS: the coordinator supplied this binding the way `skadi_ingest.py`
# takes `--map`, because exact normalized prompt equality does not bind it. The
# audit directory holds one measurement and one mp3 and names no clip; the master
# carries the clip id. Recorded here so the join is visible rather than assumed.
AUDIT_CLIP_MAP = {SPARE_HAIR_TIE: "20ece2c5"}

# Non-Suno renderers. Nothing in the archive matched — the check is kept so a
# future lyria/minimax master cannot enter the atlas by being unnoticed.
NON_SUNO_MARKERS = ("lyria", "minimax", "udio", "musicgen", "stable-audio", "elevenlabs")


# --------------------------------------------------------------------- utilities

def rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def load_json(path: Path):
    """utf-8-sig: the deep dive's `measurements.json` carries a BOM and a plain
    utf-8 load raises on it. Found on the real artifact, not guessed."""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def clip8_of(name: str) -> str:
    m = CLIP8_RE.search(name)
    return m.group(1) if m else ""


def tick(path: str) -> str:
    return f"`{path}`"


# ------------------------------------------------------------- measurement cache

def audio_masters() -> list[Path]:
    if not MASTERS.is_dir():
        raise SystemExit(f"ERROR: {rel(MASTERS)} absent — nothing to measure")
    out = sorted(p for p in MASTERS.iterdir()
                 if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES)
    if not out:
        raise SystemExit(
            f"ERROR: 0 audio master(s) under {rel(MASTERS)} — an empty extraction is an "
            "ERROR, never a corpus of zero")
    return out


def load_cache() -> dict:
    if not CACHE.exists():
        return {}
    try:
        data = load_json(CACHE)
    except (OSError, ValueError):
        return {}
    files = data.get("files") if isinstance(data, dict) else None
    return files if isinstance(files, dict) else {}


def write_cache(files: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "writer": "scripts/skadi_import_archive.py",
        "reader": "scripts/skadi_import_archive.py — the import is incremental; a file "
                  "whose byte size still matches is never re-measured",
        "instrument": "scripts/measure_render.py :: audit()",
        "files": {k: files[k] for k in sorted(files)},
    }
    CACHE.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n",
                     encoding="utf-8", newline="\n")


def measure_all(verbose: bool = True) -> dict:
    """{rel_path: {bytes, measurements}} — incremental, keyed by path + byte size.

    `measure_render` is imported lazily exactly as the rest of the codebase does:
    numpy and soundfile are heavy and every other entry point here works without
    them. The cache is flushed periodically so a long pass that dies at file 170
    does not throw away 169 measurements.
    """
    cache = load_cache()
    files = audio_masters()
    fresh = 0
    for i, path in enumerate(files, 1):
        key = rel(path)
        size = path.stat().st_size
        hit = cache.get(key)
        if isinstance(hit, dict) and hit.get("bytes") == size \
                and isinstance(hit.get("measurements"), dict) and hit["measurements"]:
            continue
        import measure_render  # noqa: PLC0415 - lazy: numpy/soundfile are heavy
        cache[key] = {"bytes": size, "measurements": measure_render.audit(str(path))}
        fresh += 1
        if verbose:
            print(f"[MEASURE] {i:3d}/{len(files)}  {path.name}", file=sys.stderr)
        if fresh % 10 == 0:
            write_cache(cache)
    live = {rel(p) for p in files}
    for gone in [k for k in cache if k not in live]:
        cache.pop(gone)
    if verbose:
        print(f"[MEASURE] {len(files)} master(s); {fresh} newly measured, "
              f"{len(files) - fresh} from cache", file=sys.stderr)
    return cache


# ------------------------------------------------------------------- the snapshots

def snapshot_index() -> tuple[dict[str, dict], dict[str, str], list[str]]:
    """-> ({clip8: row}, {clip8: snapshot filename}, notes).

    PRIMARY IS `latest.json`, exactly as the brief specifies. The historical
    snapshots are a DOCUMENTED SECONDARY: `latest.json` is a 97-row public feed
    and most of the masters on disk are older or no longer public, so a
    latest-only join would throw away the style prompt AND the model epoch of the
    majority of the corpus — and a claim with no epoch is scoped to no version of
    the generator, which the atlas refuses. Every clip records WHICH snapshot
    supplied it.
    """
    if not SNAPSHOT_DIR.is_dir():
        raise SystemExit(f"ERROR: {rel(SNAPSHOT_DIR)} absent")
    rows: dict[str, dict] = {}
    source: dict[str, str] = {}
    history = sorted(p for p in SNAPSHOT_DIR.glob("*.json") if p.name != "latest.json")
    for path in history + [LATEST]:          # latest wins: it is the freshest statement
        if not path.exists():
            continue
        data = load_json(path)
        for r in (data.get("rows") if isinstance(data, dict) else data) or []:
            cid = str(r.get("id") or "")
            if len(cid) < 8:
                continue
            rows[cid[:8]] = r
            source[cid[:8]] = path.name
    if not rows:
        raise SystemExit("ERROR: the snapshot index is empty — an empty extraction is an "
                         "ERROR, never a corpus of zero")
    notes = [f"{len(rows)} clip(s) indexed across {len(history) + 1} snapshot file(s); "
             f"{sum(1 for v in source.values() if v == 'latest.json')} of them are in "
             "`latest.json`"]
    return rows, source, notes


# ---------------------------------------------------------------- the clip records

class Clip:
    """One RENDER — the unit of evidence. Never a file, never a row: a clip id."""

    def __init__(self, clip8: str, row: dict, snapshot: str) -> None:
        self.clip8 = clip8
        self.title = str(row.get("title") or "")
        self.tags = str(row.get("tags") or "")
        self.negative_tags = str(row.get("negative_tags") or "")
        self.epoch = str(row.get("major_model_version") or "").strip()
        self.model_name = str(row.get("model_name") or "").strip()
        self.snapshot = snapshot
        self.measurements: dict = {}
        self.audio: str = ""            # repo-relative path of the measured file
        self.origin: str = ""           # master | wide-corpus


def build_clips(cache: dict, snaps: dict[str, dict], source: dict[str, str]) -> tuple[
        dict[str, Clip], dict[str, list], list[str]]:
    """-> ({clip8: Clip}, extras, notes). `extras` carries everything refused, by reason."""
    clips: dict[str, Clip] = {}
    extras: dict[str, list] = {"no_clip_id": [], "duplicate_file": [], "unjoined": [],
                               "non_suno": [], "no_epoch": []}
    notes: list[str] = []

    # 1) the local masters — our own measurement, lossless where the CDN allowed it
    #
    # ⛔ CLIP-ID-BEARING FILES ARE READ FIRST. The album pipeline keeps a
    # plain-titled copy beside the clip-id one, and `Song.wav` sorts BEFORE
    # `Song__abcd1234.wav`. Read in plain sort order the anonymous twin claims the
    # measurement fingerprint, the real master is then discarded as its duplicate,
    # and the clip vanishes from the map — a dedup rule deleting the evidence it
    # exists to protect.
    seen_measure: dict[str, str] = {}
    for path_rel in sorted(cache, key=lambda p: (0 if clip8_of(Path(p).stem) else 1, p)):
        entry = cache[path_rel]
        meas = entry.get("measurements") or {}
        name = Path(path_rel).stem
        if any(marker in path_rel.lower() for marker in NON_SUNO_MARKERS):
            extras["non_suno"].append(path_rel)
            continue
        # A byte-identical twin of a clip-id-bearing master is one render stored
        # twice (the album pipeline copies a few under their plain title). The
        # measurement fingerprint decides, not the filename.
        finger = json.dumps({k: v for k, v in meas.items() if k != "file"}, sort_keys=True)
        if finger in seen_measure:
            extras["duplicate_file"].append(f"{path_rel} (same measurements as "
                                            f"{seen_measure[finger]})")
            continue
        seen_measure[finger] = path_rel
        cid = clip8_of(name)
        if not cid:
            extras["no_clip_id"].append(path_rel)
            continue
        row = snaps.get(cid)
        if row is None:
            extras["unjoined"].append(path_rel)
            continue
        clip = Clip(cid, row, source.get(cid, "?"))
        clip.measurements, clip.audio, clip.origin = meas, path_rel, "master"
        clips[cid] = clip

    # 2) the wide-corpus audit's own measurements, for clips with no local master
    wide = AUDITS / WIDE_CORPUS / "measurements.json"
    wide_added = 0
    if wide.exists():
        for r in load_json(wide):
            cid = str(r.get("id") or "")[:8]
            if not cid or cid in clips:
                continue
            row = snaps.get(cid)
            if row is None:
                extras["unjoined"].append(f"{rel(wide)}#{cid}")
                continue
            clip = Clip(cid, row, source.get(cid, "?"))
            clip.measurements = {k: v for k, v in r.items()
                                 if k not in ("id", "title", "tags", "play_count",
                                              "upvote_count", "comment_count",
                                              "created_at", "like_rate", "path")}
            clip.audio, clip.origin = rel(wide), "wide-corpus"
            clips[cid] = clip
            wide_added += 1
    notes.append(f"{wide_added} clip(s) contributed by {WIDE_CORPUS} that have no local master")

    for cid, clip in list(clips.items()):
        if not clip.epoch:
            extras["no_epoch"].append(cid)
            clips.pop(cid)
    if not clips:
        raise SystemExit("ERROR: 0 usable clip(s) after the join — an empty extraction is an "
                         "ERROR, never a corpus of zero")
    return clips, extras, notes


# ------------------------------------------------------------------ intent parsing
# ⛔ EXPLICIT ASKS ONLY. Each pattern below was written against the real style
# text and refuses to fire on the merely suggestive. A track that does not ask
# contributes NOTHING to the class — not a negative, not a zero.

BPM_NUM = re.compile(r"(?<![\w.])(\d{2,3})\s*(?:BPM|bpm)")

_ONES = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
         "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
         "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
         "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
BPM_WORDS = re.compile(r"((?:[a-z]+[\s-]+){1,6}?)(?:BPM|bpm|beats per minute)")


def spelled_number(words: str) -> int | None:
    """`a hundred and thirty-six` -> 136. Returns None on anything it cannot read
    exactly — an unreadable phrase is NO ASK, never a guessed tempo."""
    toks = [t for t in re.split(r"[\s-]+", words.lower().strip()) if t]
    # Keep only the trailing run of number words; the phrase is the tail of a sentence.
    keep: list[str] = []
    for tok in reversed(toks):
        if tok in ("and", "a", "at", "hundred") or tok in _ONES or tok in _TENS:
            keep.append(tok)
            continue
        break
    toks = list(reversed(keep))
    if not toks or not any(t in _ONES or t in _TENS or t == "hundred" for t in toks):
        return None
    if toks.count("hundred") > 1:
        return None          # `a hundred hundred` is not a number this house writes
    total, current = 0, 0
    for tok in toks:
        if tok in ("and", "at"):
            continue
        if tok == "a":
            current = 1
        elif tok == "hundred":
            current = (current or 1) * 100
            total += current
            current = 0
        elif tok in _TENS:
            current += _TENS[tok]
        elif tok in _ONES:
            current += _ONES[tok]
        else:
            return None
    total += current
    return total if 40 <= total <= 220 else None


def bpm_targets(text: str) -> list[int]:
    """Every distinct BPM value the style text names, numerals and words alike."""
    found = {int(v) for v in BPM_NUM.findall(text) if 40 <= int(v) <= 220}
    for m in BPM_WORDS.finditer(text):
        val = spelled_number(m.group(1))
        if val is not None:
            found.add(val)
    return sorted(found)


QUIET_OPEN = re.compile(
    r"(open(?:s|ing)?\s+(?:in|on|with|from)\s+(?:near[-\s]?)?(?:silence|black|nothing)"
    r"|(?:near[-\s]?silent|silent)\s+open(?:ing)?"
    r"|begins?\s+(?:in|on|from)\s+(?:near[-\s]?)?silence"
    r"|fades?\s+in\s+from\s+(?:near[-\s]?)?silence"
    r"|starts?\s+(?:in|from)\s+(?:near[-\s]?)?silence"
    r"|first\s+(?:\w+\s+)?seconds?\s+(?:are|is)\s+(?:near[-\s]?)?silen)", re.I)

TERMINAL_SILENCE = re.compile(
    r"((?:end|ends|ending|end\s+on|ends\s+on|ends?\s+in|then|into)\s+(?:[\w\s,'-]{0,40}\s)?"
    r"(?:absolute|total|true|pure|dead)?\s?silence\s*[.;]"
    r"|silence\s+(?:to\s+)?(?:close|finish|end)(?:es|s)?\b"
    r"|ends?\s+in\s+silence)", re.I)

INTERRUPTING_SILENCE = re.compile(
    r"((?:one|a|two|three|four)\s+(?:full\s+|single\s+)?(?:beat|bar|second)s?\s+of\s+"
    r"(?:absolute|total|true|charged|pure|dead)?\s?silence"
    r"|\d+(?:\.\d+)?\s*(?:-|\s)?second\s+(?:full\s+)?(?:stop|silence|void|gap)"
    r"|full\s+stop\b|dead\s+stop\b|everything\s+stops\b|the\s+track\s+stops\b"
    r"|clinical\s+silences?\b|silence[-\s]guillotine)", re.I)

SUB_BUILD = re.compile(
    r"(more\s+sub\b|sub\s+(?:build|grows?|accretes?|arrives?|swells?|thickens?)"
    r"|(?:build|grow|accrete)s?\s+(?:the\s+)?sub\b"
    r"|final\s+chorus[^.;]{0,60}\bsub\b"
    r"|deeper\s+sub\b|sub\s+deepens?\b)", re.I)

WIDEN = re.compile(
    r"(chorus[^.;]{0,40}\bwiden|widen(?:s|ing)?\s+(?:like|into|at|toward|through)"
    r"|blooms?\s+into\s+a\s+wide|opens?\s+up\s+(?:in|at|into)\s+the\s+chorus"
    r"|stereo\s+(?:field|image)\s+widen|widen(?:s|ing)?\s+(?:harmonies|stacks)"
    r"|mix\s+widens?\b)", re.I)

HZ_TONE = re.compile(
    r"(\d{2,4})\s*ˆ?\s*(?:Hz|hz|HZ|hertz)[^.;]{0,60}?"
    r"(hum|drone|sine|tone|pad|spine|beacon|whine)"
    r"|(hum|drone|sine|tone|spine|beacon|whine)[^.;]{0,60}?(?<![\w.])(\d{2,4})\s*(?:Hz|hz|hertz)",
    re.I)
HZ_WORDS = re.compile(r"((?:[a-z]+[\s-]+){1,6}?)hertz", re.I)


def hz_targets(text: str) -> list[int]:
    out = {int(m.group(1) or m.group(4)) for m in HZ_TONE.finditer(text)}
    for m in HZ_WORDS.finditer(text):
        val = spelled_hz(m.group(1))
        if val is not None:
            out.add(val)
    return sorted(v for v in out if v)


def spelled_hz(words: str) -> int | None:
    toks = [t for t in re.split(r"[\s-]+", words.lower().strip()) if t]
    keep: list[str] = []
    for tok in reversed(toks):
        if tok in ("and", "a", "at", "hundred") or tok in _ONES or tok in _TENS:
            keep.append(tok)
            continue
        break
    if not keep:
        return None
    total, current = 0, 0
    for tok in reversed(keep):
        if tok in ("and", "at"):
            continue
        if tok == "a":
            current = 1
        elif tok == "hundred":
            current = (current or 1) * 100
            total += current
            current = 0
        elif tok in _TENS:
            current += _TENS[tok]
        elif tok in _ONES:
            current += _ONES[tok]
        else:
            return None
    total += current
    return total or None


# --------------------------------------------------------------------- the tests
# Each test reads ONE measurement and returns (verdict, human-readable measure).
# `None` means the ask fired but the instrument cannot see it — recorded as an
# instrument gap, never silently as an absence.

def test_steady_tempo(m: dict, targets: list[int], tol: float):
    cands = m.get("tempo_candidates_bpm") or []
    if not cands:
        return None, "no tempo candidate returned"
    tgt = targets[0]
    best = min(((abs(c * mult - tgt), c, mult) for c in cands for mult in (0.5, 1, 2)))
    delta, cand, mult = best
    at = f"{cand:g}" + ("" if mult == 1 else f"×{mult:g}")
    return ("ARRIVES" if delta <= tol else "ABSENT",
            f"asked {tgt} BPM, nearest candidate {at} = {cand * mult:g} (Δ{delta:.0f})")


def test_quiet_open(m: dict, floor_db: float):
    val = m.get("opening_2s_db")
    if val is None:
        return None, "no opening level measured"
    return ("ARRIVES" if val <= floor_db else "ABSENT",
            f"opening 2 s at {val:.1f} dB below peak")


def test_terminal_silence(m: dict):
    dur, gaps = m.get("duration_s"), m.get("quiet_gaps") or []
    if dur is None:
        return None, "no duration measured"
    tail = [(t, ln) for t, ln in gaps if t + ln >= dur - 3.0]
    return ("ARRIVES" if tail else "SMOOTHED",
            f"terminal gap {tail[0][1]:.1f} s" if tail else "no gap in the final 3 s")


def test_interrupting_silence(m: dict):
    dur, gaps = m.get("duration_s"), m.get("quiet_gaps") or []
    dip = m.get("deepest_mid_dip") or {}
    if dur is None:
        return None, "no duration measured"
    body = [(t, ln) for t, ln in gaps if 0.10 * dur <= t <= 0.90 * dur]
    if body:
        return "ARRIVES", f"mid-song gap {body[0][1]:.1f} s at {body[0][0]:.0f} s"
    return "SMOOTHED", (f"no mid-song gap; deepest dip {dip.get('depth_db', 0):.1f} dB "
                        f"over {dip.get('width_s', 0):.2f} s")


def test_sub_build(m: dict, floor_db: float):
    band = (m.get("bands_by_quarter_db") or {}).get("20-80Hz")
    if not band or len(band) < 4:
        return None, "no 20–80 Hz profile measured"
    rise = band[3] - band[0]
    return ("ARRIVES" if rise >= floor_db else "SMOOTHED",
            f"20–80 Hz Q1→Q4 {rise:+.1f} dB")


def test_widen(m: dict, threshold: float):
    wd = m.get("width_dynamics") or {}
    corr = wd.get("corr_width_level")
    if corr is None:
        return None, "mono file — no width/level correlation"
    if corr >= threshold:
        return "ARRIVES", f"width/level r = {corr:+.3f}"
    if corr <= -threshold:
        return "INVERTED", f"width/level r = {corr:+.3f} — it narrows into the loud sections"
    return "ABSENT", f"width/level r = {corr:+.3f} — no width movement with level"


TONAL_LO, TONAL_HI = 70, 6000        # scripts/measure_render.py :: tonal_components()


def test_named_tone(m: dict, targets: list[int], cents: float):
    visible = [t for t in targets if TONAL_LO <= t <= TONAL_HI]
    if not visible:
        return None, (f"named {targets[0]} Hz sits outside the {TONAL_LO}–{TONAL_HI} Hz window "
                      "`tonal_components()` searches")
    tones = m.get("sustained_tones_hz") or []
    if not tones:
        return "ABSENT", f"asked {visible[0]} Hz; no sustained isolated tone at all"
    tgt = visible[0]
    best = min(tones, key=lambda f: abs(1200 * (f / tgt - 1)))
    off = abs(1200 * (best / tgt - 1))
    return ("ARRIVES" if off <= cents else "ABSENT",
            f"asked {tgt} Hz, nearest sustained tone {best:.0f} Hz")


# Each entry: spec class -> (ask detector, test). `steady-tempo` and `untuned-drone`
# take the parsed targets; the rest read the measurement alone.
def observations(clips: dict[str, Clip], gates: dict) -> tuple[list[dict], dict[str, int],
                                                               list[str]]:
    """-> (observations, ask counts, instrument-gap notes). One row per (clip, class)."""
    tol = float(gates.get("atlas_tempo_tolerance_bpm", 5))
    quiet = float(gates.get("atlas_quiet_open_db", -30))
    sub = float(gates.get("atlas_sub_build_db", 3))
    width = float(gates.get("atlas_width_corr_threshold", 0.10))
    cents = float(gates.get("atlas_tone_match_cents", 50))

    obs: list[dict] = []
    asks: dict[str, int] = {}
    gaps: list[str] = []

    def add(clip: Clip, cls: str, verdict, measure: str) -> None:
        asks[cls] = asks.get(cls, 0) + 1
        if verdict is None:
            gaps.append(f"{cls}: {clip.clip8} — {measure}")
            return
        obs.append({"clip": clip.clip8, "title": clip.title, "epoch": clip.epoch,
                    "spec_class": cls, "verdict": verdict, "measure": measure,
                    "audio": clip.audio, "snapshot": clip.snapshot, "origin": clip.origin})

    for cid in sorted(clips):
        clip = clips[cid]
        text, m = clip.tags, clip.measurements
        if not text or not m:
            continue

        targets = bpm_targets(text)
        if len(targets) == 1:
            add(clip, "steady-tempo", *test_steady_tempo(m, targets, tol))
        elif len(targets) > 1:
            # ⛔ THE INSTRUMENT CANNOT SEE THIS ONE. `audit()` reports whole-track
            # autocorrelation; two tempos in sequence and one tempo with metric
            # ambiguity produce the same four candidates. Counted as an ask and an
            # instrument gap — never adjudicated, never scored as an absence.
            asks["tempo-dramaturgy"] = asks.get("tempo-dramaturgy", 0) + 1
            gaps.append(f"tempo-dramaturgy: {clip.clip8} — asked {targets}; whole-track "
                        "autocorrelation cannot separate a programmed change from metric "
                        "ambiguity, and `audit()` runs no windowed tempo pass")

        if QUIET_OPEN.search(text):
            add(clip, "near-silent-opening", *test_quiet_open(m, quiet))
        if TERMINAL_SILENCE.search(text):
            add(clip, "terminal-silence", *test_terminal_silence(m))
        if INTERRUPTING_SILENCE.search(text):
            add(clip, "interrupting-silence", *test_interrupting_silence(m))
        if SUB_BUILD.search(text):
            add(clip, "sub-build", *test_sub_build(m, sub))
        if WIDEN.search(text):
            add(clip, "width-vs-level", *test_widen(m, width))
        hz = hz_targets(text)
        if hz:
            add(clip, "untuned-drone", *test_named_tone(m, hz, cents))

    if not obs:
        raise SystemExit("ERROR: 0 observation(s) extracted from the archive — an empty "
                         "extraction is an ERROR, never a corpus with no intents in it")
    return obs, asks, gaps


# ------------------------------------------------------------------- claim assembly

def spec_lanes() -> dict[str, str]:
    rows, problems = parse_table(read(ATLAS_DIR / "SPEC_CLASSES.md"), SPEC_CLASS_COLS)
    if problems or not rows:
        raise SystemExit(f"ERROR: SPEC_CLASSES.md unreadable: {'; '.join(problems) or 'no rows'}")
    return {r["class"]: r["lane"].upper() for r in rows}


EPOCH_ORDER = ["v3", "v3.5", "v4", "v4.5", "v4.5+", "v5", "v5.5"]


def epoch_key(name: str) -> tuple[int, str]:
    return (EPOCH_ORDER.index(name) if name in EPOCH_ORDER else len(EPOCH_ORDER), name)


CLASS_ASK = {
    "steady-tempo": "a single BPM target named in the style text",
    "near-silent-opening": "a near-silent opening named in the style text",
    "terminal-silence": "silence named as the END of the arrangement",
    "interrupting-silence": "a stop or void named INSIDE the arrangement",
    "sub-build": "low-end growth into a late section, named in the style text",
    "width-vs-level": "the image widening as the track gets louder, named in the style text",
    "untuned-drone": "a sustained tone named at a specific frequency",
}
CLASS_TEST = {
    "steady-tempo": "a measured tempo candidate within `atlas_tempo_tolerance_bpm` of the "
                    "target, octave-folded",
    "near-silent-opening": "the first 2 s at or below `atlas_quiet_open_db` under the peak",
    "terminal-silence": "a measured quiet gap in the final 3 s",
    "interrupting-silence": "a measured quiet gap between 10% and 90% of the duration",
    "sub-build": "20–80 Hz energy rising by at least `atlas_sub_build_db` from the first "
                 "quarter to the last",
    "width-vs-level": "the sign of the width/level correlation against "
                      "`atlas_width_corr_threshold`",
    "untuned-drone": "a sustained isolated tone within `atlas_tone_match_cents` of the "
                     "named frequency",
}
CLASS_FALSIFIER = {
    "steady-tempo": "A comparable set of renders in this epoch whose measured tempo lands "
                    "outside the tolerance of a single named target.",
    "near-silent-opening": "A requested near-silent opening measuring within a few dB of the "
                           "body of its track.",
    "terminal-silence": "A requested terminal silence filled with music or a fade.",
    "interrupting-silence": "A measured mid-song gap at or near the requested length.",
    "sub-build": "A requested late low-end growth whose 20–80 Hz band does not rise across "
                 "the track.",
    "width-vs-level": "A requested widening whose measured width/level correlation is "
                      "positive in this epoch.",
    "untuned-drone": "A sustained isolated tone measured at the requested frequency.",
}


def verdict_of_group(rows: list[dict]) -> tuple[str, int, int]:
    """-> (verdict, k, m). A SPLIT epoch is PARTIAL, and its text carries `k of m`.

    ⛔ n COUNTS TRIALS. Taking the majority verdict and setting n to the majority
    count would publish a row that had quietly dropped every render disagreeing
    with it — the same evidence, minus the part that hurts.
    """
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    m = len(rows)
    if len(counts) == 1:
        only = next(iter(counts))
        return only, m, m
    best = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
    return "PARTIAL", best[1], m


def claim_text(cls: str, verdict: str, k: int, m: int, epochs: list[str]) -> str:
    scope = ", ".join(epochs)
    ask, test = CLASS_ASK[cls], CLASS_TEST[cls]
    where = f"{m} `{scope}` render(s)"
    if verdict == "PARTIAL":
        return (f"Archive sweep: {ask} is honoured in {k} of {where} — measured by {test}. "
                "Mixed within the epoch, so the row is PARTIAL and the denominator is "
                "visible.")
    return (f"Archive sweep: {ask} returns {verdict} in all {where} — measured by {test}.")


def assemble(obs: list[dict], lanes: dict[str, str], atlas_rows: list[dict],
             prior: dict) -> tuple[list[dict], list[dict], list[str]]:
    """-> (new rows, increments, drift candidates).

    Per (spec class, epoch) a verdict is computed; epochs that agree are merged
    into ONE row whose `models` lists them all, because that is exactly what the
    models column means. Epochs that disagree get separate rows and a drift
    candidate naming the pair — the old row is never edited.
    """
    by_class: dict[str, dict[str, list[dict]]] = {}
    for o in obs:
        by_class.setdefault(o["spec_class"], {}).setdefault(o["epoch"], []).append(o)

    # ⛔ THE IMPORT MUST NOT FIND ITSELF. On a second run this sweep's own rows are
    # in ATLAS.md, and a row matching its own spec class would be read as a prior
    # claim and INCREMENTED — the import incrementing the import, once per run,
    # forever. Rows minted by an earlier archive sweep are excluded here and
    # rebuilt from scratch below.
    mine = {v.get("id") for v in (prior.get("claims") or {}).values()}
    existing = {}
    for r in atlas_rows:
        cls = cell_parts(r["scope"])[1]
        if r["channel"].lower() == CHANNEL and strike(r["id"])[0] not in mine:
            existing.setdefault(cls, r)

    new_rows: list[dict] = []
    increments: list[dict] = []
    drift: list[str] = []

    for cls in sorted(by_class):
        per_epoch = {ep: verdict_of_group(rows) for ep, rows in by_class[cls].items()}
        verdicts = {v for v, _, _ in per_epoch.values()}
        if len(verdicts) > 1:
            spread = "; ".join(
                f"`{ep}` {v} ({k} of {m})"
                for ep, (v, k, m) in sorted(per_epoch.items(), key=lambda kv: epoch_key(kv[0])))
            drift.append(f"**{cls}** — the epochs disagree with each other: {spread}.")

        prior_row = existing.get(cls)
        for verdict in sorted(verdicts):
            epochs = sorted((ep for ep, (v, _, _) in per_epoch.items() if v == verdict),
                            key=epoch_key)
            rows = [o for ep in epochs for o in by_class[cls][ep]]
            k = sum(per_epoch[ep][1] for ep in epochs)
            m = sum(per_epoch[ep][2] for ep in epochs)

            # ⛔ AN INCREMENT MAY NOT REWORD ITS OWN CLAIM: the append receipt is a
            # sha of the claim text, and a row silently reworded after banking
            # breaks its own receipt. So an existing row is incremented ONLY when
            # the archive agrees with it unanimously — where the archive is mixed
            # the honest artifact is a NEW row whose text carries `k of m`.
            if prior_row is not None and verdict != "PARTIAL" \
                    and prior_row["verdict"].upper() == verdict:
                increments.append({"id": strike(prior_row["id"])[0], "spec_class": cls,
                                   "add_n": m, "epochs": epochs, "rows": rows})
                continue
            if prior_row is not None and prior_row["verdict"].upper() != verdict:
                drift.append(
                    f"**{cls}** — the archive returns {verdict} on "
                    f"{', '.join('`' + e + '`' for e in epochs)} where the seeded row "
                    f"`{strike(prior_row['id'])[0]}` carries {prior_row['verdict'].upper()} "
                    f"scoped `{prior_row['models']}`. The seeded row is NOT edited: a "
                    "calibration plate settles it.")
            new_rows.append({"spec_class": cls, "verdict": verdict, "k": k, "m": m,
                             "epochs": epochs, "lane": lanes.get(cls, "NUMERIC"),
                             "rows": rows,
                             "key": f"{cls}|{CHANNEL}|{verdict}|{','.join(epochs)}"})
    return new_rows, increments, drift


def allocate_ids(new_rows: list[dict], atlas_rows: list[dict], prior: dict) -> None:
    """Stable ids across re-runs: a claim key that was minted before keeps its id."""
    prior_ids = {k: v["id"] for k, v in (prior.get("claims") or {}).items()}
    used = {strike(r["id"])[0] for r in atlas_rows} | set(prior_ids.values())
    nxt = max([int(m.group(1)) for i in used if (m := re.fullmatch(r"SK-(\d{3})", i))] or [0])
    for row in new_rows:
        if row["key"] in prior_ids:
            row["id"] = prior_ids[row["key"]]
            continue
        nxt += 1
        row["id"] = f"SK-{nxt:03d}"


# ------------------------------------------------------------------- the baselines

BASELINE_MEASURES = [
    ("stereo correlation", lambda m: (m.get("stereo") or {}).get("correlation"), "{:.3f}"),
    ("opening 2 s (dB under peak)", lambda m: m.get("opening_2s_db"), "{:.1f}"),
    ("quiet gaps (count)", lambda m: len(m.get("quiet_gaps") or []), "{:.1f}"),
    ("20–80 Hz Q1→Q4 (dB)",
     lambda m: (lambda b: b[3] - b[0] if b and len(b) >= 4 else None)(
         (m.get("bands_by_quarter_db") or {}).get("20-80Hz")), "{:+.1f}"),
    ("width/level correlation",
     lambda m: (m.get("width_dynamics") or {}).get("corr_width_level"), "{:+.3f}"),
    ("tempo candidate spread (BPM)",
     lambda m: (lambda c: max(c) - min(c) if len(c or []) >= 2 else None)(
         m.get("tempo_candidates_bpm")), "{:.0f}"),
    ("duration (s)", lambda m: m.get("duration_s"), "{:.0f}"),
]


def quantiles(values: list[float]) -> tuple[float, float, float, float, float]:
    vals = sorted(values)
    if len(vals) == 1:
        v = vals[0]
        return v, v, v, v, v
    qs = statistics.quantiles(vals, n=10)
    return vals[0], qs[0], statistics.median(vals), qs[8], vals[-1]


def render_baselines(clips: dict[str, Clip], extras: dict[str, list]) -> str:
    epochs = sorted({c.epoch for c in clips.values()}, key=epoch_key)
    out = [
        "<!-- generated by scripts/skadi_import_archive.py; read by humans and by the "
        "app's baselines view (no validator check — these are not claims) -->",
        "",
        "# BASELINES — the denominator, per model epoch",
        "",
        f"status: GENERATED {IMPORT_DATE} — pending curation by The Scientist",
        "",
        "⛔ **NOTHING ON THIS PAGE IS A CLAIM.** Every figure here is computed over EVERY "
        "measured clip regardless of what its style text asked for, which is exactly what "
        "makes it useless as evidence about an intent and essential as the denominator an "
        "intent is read against. A claim that a requested near-silent opening ARRIVES means "
        "nothing until you know how many unrequested openings are already quiet.",
        "",
        "⚠️ **The sampling frame is not the catalogue.** These clips are the ones that "
        f"survived to disk: the distribution masters, plus the `{WIDE_CORPUS}` sweep, whose "
        "own frame was *every public clip with ≥40 plays OR created since 2026-07-20* and "
        "which therefore **deliberately over-samples popular and recent work**. No rate here "
        "may be read as a catalogue-wide rate.",
        "",
        "⚠️ **Two file formats.** Masters fetched as `.wav` are lossless; six clips whose "
        "wav variant the CDN now refuses are measured from `.mp3`, as is every clip "
        f"contributed by `{WIDE_CORPUS}`. Where both existed the two agreed to the printed "
        "precision on the one clip that has both (`20ece2c5`), but the mix of formats is "
        "recorded rather than hidden.",
        "",
        "Instrument: `scripts/measure_render.py` → `audit()`. Cache: "
        f"{tick(rel(CACHE))}.",
        "",
    ]
    for ep in epochs:
        rows = [c for c in clips.values() if c.epoch == ep]
        out += [f"## {ep} — n = {len(rows)}", ""]
        table = []
        for label, get, fmt in BASELINE_MEASURES:
            vals = [v for v in (get(c.measurements) for c in rows) if v is not None]
            if not vals:
                table.append([label, "0", "—", "—", "—", "—", "—"])
                continue
            lo, p10, med, p90, hi = quantiles([float(v) for v in vals])
            table.append([label, str(len(vals)), fmt.format(lo), fmt.format(p10),
                          fmt.format(med), fmt.format(p90), fmt.format(hi)])
        out += table_md(["measure", "n", "min", "p10", "median", "p90", "max"], table)
        out.append("")
    unattributed = len(extras["no_clip_id"]) + len(extras["unjoined"])
    out += [
        "## What is not counted here",
        "",
        f"- **{len(extras['duplicate_file'])}** master file(s) are byte-identical twins of a "
        "clip-id-bearing master (the album pipeline keeps a plain-titled copy). One render is "
        "one observation, so the twin is dropped by measurement fingerprint, not by filename.",
        f"- **{unattributed}** measured file(s) carry no clip id that joins a snapshot, so "
        "they have no model epoch and cannot sit in any epoch table. They are named in "
        f"{tick(rel(NOTES))}.",
        "- **Audience numbers are absent on purpose.** Plays, likes and like-rate live in "
        "`apps/lofn-analytics` and `vault/BENCHMARKS.md`. A popularity statistic must never "
        "borrow a grain claim's evidence ladder.",
        "",
    ]
    return "\n".join(out) + "\n"


# ------------------------------------------------------------- the knowledge shelf
# ⛔ THE LITERATURE LANE. Source-cited external claims, no n, no verdict word, and
# no path onto the evidence ladder. It lands FIRST (it is the cheapest instrument
# we own) and it can never be outvoted by our n — but it never carries our
# verdicts either. A contradiction between this shelf and a measured claim is
# FLAGGED for an expedition to settle, never resolved on the page.

GUIDE = "vault/SUNO_PROMPT_CONSTRUCTION_GUIDE.md"

SHELF = [
    ("KB-01", "prompt form", f"{GUIDE} §4 Failure 1",
     "A flat list of technical specifications is reported to underperform prose in which each "
     "specification is given a job or a behaviour."),
    ("KB-02", "spatial language", f"{GUIDE} §4 Failure 2",
     "Style text with no left/right/centre/depth information is reported to leave the stereo "
     "placement of every element to the generator."),
    ("KB-03", "opening moment", f"{GUIDE} §4 Failure 3",
     "The first five seconds are reported to be the highest-leverage region of the prompt; "
     "text that does not specify them is reported to get a generic opening."),
    ("KB-04", "vocabulary without a job", f"{GUIDE} §4 Failure 4",
     "Domain or scientific vocabulary that names no sound is reported to act as decoration "
     "rather than instruction."),
    ("KB-05", "groove", f"{GUIDE} §4 Failure 5",
     "A groove described without a named timing defect or behaviour is reported to render as "
     "an interchangeable grid."),
    ("KB-06", "acoustic ban", f"{GUIDE} §4 Failure 6",
     "A ban on a palette is reported to need reinforcement at more than one position in the "
     "text to hold."),
    ("KB-07", "default warmth", f"{GUIDE} §4 Failure 7",
     "Warm or default adjectives are reported to pull the render toward the generator's own "
     "prior."),
    ("KB-08", "signature device", f"{GUIDE} §4 Failure 8",
     "Text with no single named sonic device is reported to produce a render with no "
     "identifying event."),
    ("KB-09", "artist names", f"{GUIDE} §4 Failure 9",
     "Real artist names are forbidden in this house's final prompts on policy grounds; the "
     "guide also reports them as an anchor that external sources recommend and Lofn refuses."),
    ("KB-10", "arrangement arc", f"{GUIDE} §4 Failure 10",
     "Style text carrying no narrative arc across the arrangement is reported to render as an "
     "undifferentiated single state."),
    ("KB-11", "attention weighting", f"{GUIDE} §4 (14,000-generation test corpus)",
     "The model is reported to have a centre bias: words in the middle third of the style "
     "text receive slightly more attention than words at either edge."),
    ("KB-12", "fusion operator", f"{GUIDE} §4 (14,000-generation test corpus)",
     "A slash between two genre names is reported to be the most reliable hybridisation "
     "operator, ahead of \"and\", \"meets\" and \"with elements of\"."),
    ("KB-13", "negation", f"{GUIDE} §4 (14,000-generation test corpus)",
     "Negation is reported to fail intermittently: the model attention-weights the content "
     "word, not the negation operator, so a banned element sometimes appears. Positive "
     "specification is reported to work better."),
    ("KB-14", "modal language", f"{GUIDE} §4 (14,000-generation test corpus)",
     "Modal verbs are reported to weaken the content words they precede; the imperative is "
     "recommended."),
    ("KB-15", "structural asks", f"{GUIDE} §4 (14,000-generation test corpus)",
     "A bar-counted structural instruction is reported to produce structural awareness far "
     "more often than an unbounded one, and a drop is reported as the most reliable named "
     "arrangement event."),
    ("KB-16", "named kick types", f"{GUIDE} §4 (14,000-generation test corpus)",
     "Specific kick vocabulary is reported to be recognised, while vague size adjectives are "
     "reported to be ignored and one term is reported to be confused with the bass line."),
    ("KB-17", "structured brief", f"{GUIDE} §8 (2026-06-04, v5.5)",
     "A categorised key:value production brief is reported to change audio quality relative "
     "to unbroken prose or a flat tag list. The source itself calls this directional, from "
     "roughly ten tracked generations at a self-estimated confidence under one half."),
    ("KB-18", "categorised negatives", f"{GUIDE} §8 (2026-06-04, v5.5)",
     "A negative field grouped into failure classes is reported to outperform a flat negative "
     "list, which describes only what not to emit."),
    ("KB-19", "hook layers", f"{GUIDE} §8 (2026-06-04, v5.5)",
     "Four distinct hook layers — melodic, rhythmic, instrumental, textural — are reported to "
     "be required for coherence at this model version; a lyric hook alone is reported as "
     "insufficient."),
    ("KB-20", "provenance of this shelf", f"{GUIDE} §Document Metadata",
     "The guide records that its web research attempt returned an anti-bot wall on all "
     "sixteen URLs and that no web-sourced material was included. Its authority basis is a "
     "panel of synthetic experts plus two internal reference tracks — so this shelf is "
     "house-internal secondary literature, not vendor documentation."),
]


def render_known_behavior() -> str:
    out = [
        "<!-- generated by scripts/skadi_import_archive.py; read by humans (no validator "
        "check — the literature lane carries no verdicts and no n) -->",
        "",
        "# KNOWN BEHAVIOR — the literature lane",
        "",
        f"status: GENERATED {IMPORT_DATE} — pending curation by The Scientist",
        "",
        "**THE LANE RULE.** The literature is the cheapest instrument this house owns, so it "
        "fires FIRST — before a plate is priced, before a probe is written. It **cannot be "
        "outvoted by our n**: a measured row does not delete a source-cited report, it "
        "contradicts it, and a contradiction is an expedition target. And it **never carries "
        "our verdicts**: nothing on this page has a verdict word, a confidence, an `n` or a "
        "place on the evidence ladder, because none of it was measured through our door.",
        "",
        "⚠️ **Provenance, stated once and meant.** Every row below is EXTERNAL to Skaði and "
        "**secondary**: it comes from a house guide assembled by synthetic-expert panels "
        "against a corpus we did not measure and cannot re-open. Read `KB-20` before reading "
        "any other row. These are reports, not receipts.",
        "",
        "**When a row here disagrees with a measured claim in `ATLAS.md`:** flag it in "
        "`FRONTIER.md` as an expedition target. Do not silently prefer either one. The shelf "
        "cannot be refuted by our corpus and our corpus cannot be refuted by the shelf; only "
        "a plate designed to separate them settles anything.",
        "",
    ]
    out += table_md(["id", "topic", "source", "reported behaviour (paraphrased)"],
                    [[i, topic, tick(src), text] for i, topic, src, text in SHELF])
    out += [
        "",
        "## Unmined pointers",
        "",
        "`vault/RESEARCH_SOURCES_200.md` lists two community sources for this generator "
        "(the subreddit and the vendor's public explore page). Neither has been read into "
        "this shelf. They are named here so the gap is visible rather than absent.",
        "",
    ]
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------- the atlas writes

def atlas_row_cells(row: dict, ev_paths: list[str], falsifier: str) -> list[str]:
    return [row["id"], "CELL", f"any territory × {row['spec_class']}", CHANNEL, row["lane"],
            row["claim"], row["verdict"], str(row["n"]), row["confidence"],
            ", ".join(row["epochs"]), ", ".join(tick(p) for p in ev_paths), falsifier]


def write_atlas(new_rows: list[dict], increments: list[dict], bases: dict,
                ev_paths: dict[str, list[str]], gates: dict) -> tuple[str, list[list[str]]]:
    text = read(ATLAS_DIR / "ATLAS.md")
    bank_min = int(gates.get("atlas_bank_min_n", 2))
    tiers: dict[str, list[dict]] = {}
    for t in ("BANKED", "OBSERVED", "CONJECTURED"):
        tiers[t], _ = parse_table(text, ATLAS_COLS, section=t)

    mine = {r["id"] for r in new_rows}
    inc_by_id = {i["id"]: i for i in increments}
    receipts: list[list[str]] = []

    placed: dict[str, list[list[str]]] = {"BANKED": [], "OBSERVED": [], "CONJECTURED": []}
    for t in ("BANKED", "OBSERVED", "CONJECTURED"):
        for r in tiers[t]:
            rid = strike(r["id"])[0]
            if rid in mine:
                continue                      # rebuilt below from scratch
            inc = inc_by_id.get(rid)
            if inc is not None:
                base = bases.get(rid, {})
                base_n = int(base.get("n", r["n"]) or 0)
                base_models = base.get("models", r["models"])
                base_ev = base.get("evidence", r["evidence"])
                r["n"] = str(base_n + inc["add_n"])
                spanned = [m.strip() for m in base_models.split(",") if m.strip()]
                for ep in inc["epochs"]:
                    if ep not in spanned:
                        spanned.append(ep)
                r["models"] = ", ".join(spanned)
                add = ", ".join(tick(p) for p in ev_paths[rid])
                r["evidence"] = f"{base_ev}, {add}" if base_ev.strip() else add
                # ⛔ NEVER HIGH FROM THE ARCHIVE (gates atlas_high_requires_plate): no
                # controls stood behind any of this, at any n.
                r["confidence"] = "MEDIUM" if int(r["n"]) >= bank_min else "LOW"
                receipts.append([rid, tick(sha12(r["claim"])), "N-INCREMENT",
                                 f"n {base_n} → {r['n']} from the archive sweep "
                                 f"({', '.join(inc['epochs'])}); claim text unchanged"])
            n = int(r["n"]) if r["n"].isdigit() else 0
            dest = t
            if t == "OBSERVED" and n >= bank_min:
                dest = "BANKED"
            placed[dest].append([r[c] for c in ATLAS_COLS])

    for row in new_rows:
        cells = atlas_row_cells(row, ev_paths[row["id"]], CLASS_FALSIFIER[row["spec_class"]])
        placed["BANKED" if row["n"] >= bank_min else "OBSERVED"].append(cells)
        receipts.append([row["id"], tick(sha12(row["claim"])), "NEW",
                         f"archive sweep — {row['m']} render(s), "
                         f"{', '.join(row['epochs'])}"])

    for t in ("BANKED", "OBSERVED", "CONJECTURED"):
        text = splice(text, ATLAS_COLS, placed[t], section=t)

    marker = f"### {IMPORT_DATE} {IMPORT_SLUG} additions"
    cut = text.find("\n" + marker)
    if cut >= 0:
        end = text.find("\n### ", cut + 1)
        text = text[:cut] + (text[end:] if end > 0 else "\n")
    if receipts:
        block = ["", marker, "",
                 "Observational rows, imported from work already on disk. No plate, no "
                 "controls, no registered predictions — so nothing here is HIGH at any `n` "
                 "(gates `atlas_high_requires_plate`). Per-claim clip lists, epochs and "
                 f"measured values: {tick(rel(EVIDENCE))}.", ""] \
            + table_md(APPEND_COLS, receipts) + [""]
        text = text.rstrip("\n") + "\n" + "\n".join(block) + "\n"
    return text, receipts


DRIFT_START = "<!-- archive-import: drift candidates START -->"
DRIFT_END = "<!-- archive-import: drift candidates END -->"


def write_frontier(drift: list[str], epoch_note: list[str],
                   replication: list[str] | None = None) -> str:
    text = read(ATLAS_DIR / "FRONTIER.md")
    body = [DRIFT_START, "",
            "### Drift candidates — for the calibration plate, not for the chart",
            "",
            "The chart above gains points only from CONTROL probes on real plates. The list "
            "below is what an archive sweep of work already on disk turned up that a control "
            "should be pointed at. **Nothing here edits a seeded row**; an observational "
            "sweep names a suspicion, and a plate settles it.",
            ""]
    body += [f"- {line}" for line in (drift or ["(none found)"])]
    if replication:
        body += [""] + replication
    if epoch_note:
        body += [""] + epoch_note
    body += ["", DRIFT_END]
    block = "\n".join(body)

    if DRIFT_START in text and DRIFT_END in text:
        head = text[:text.index(DRIFT_START)]
        tail = text[text.index(DRIFT_END) + len(DRIFT_END):]
        return head + block + tail
    anchor = "\n## DELTA LOG"
    at = text.find(anchor)
    if at < 0:
        raise SystemExit("ERROR: FRONTIER.md has no `## DELTA LOG` heading to insert the "
                         "drift note before — refusing to append somewhere convenient")
    return text[:at] + "\n" + block + "\n" + text[at:]


def write_notes(section: str) -> str:
    text = read(NOTES)
    marker = f"## {NOTES_SECTION}"
    at = text.find("\n" + marker)
    if at >= 0:
        end = text.find("\n## ", at + 1)
        text = text[:at] + (text[end:] if end > 0 else "\n")
    return text.rstrip("\n") + "\n\n" + section.strip("\n") + "\n"


# ------------------------------------------------------------------------- the run

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Import the archive's free evidence into the SKAÐI music atlas.")
    ap.add_argument("--measure-only", action="store_true",
                    help="fill the measurement cache and exit; writes no atlas state")
    ap.add_argument("--dry-run", action="store_true", help="report; write no atlas state")
    ap.add_argument("--quiet", action="store_true", help="no per-file measurement lines")
    ap.add_argument("--skip-validator", action="store_true",
                    help="do not refuse on a red validator (for tests only)")
    args = ap.parse_args(argv)

    if args.measure_only:
        write_cache(measure_all(verbose=not args.quiet))
        print(f"WROTE     {rel(CACHE)}")
        return 0

    gates, warnings = load_gates()
    for w in warnings:
        print(f"[GATES] {w}", file=sys.stderr)

    # ⛔ REFUSE TO IMPORT INTO A RED STATE. Adding rows on top of a failing atlas
    # buries the failure under new evidence and makes the next reader believe the
    # import caused it.
    if not args.skip_validator:
        rep = Report()
        run_all(gates, rep)
        if rep.failed:
            print("REFUSED: validate_music_atlas.py currently FAILS "
                  f"({rep.failed} check(s)). Fix the state first; an import must never "
                  "land on top of a red validator.", file=sys.stderr)
            return 3

    cache = measure_all(verbose=not args.quiet)
    snaps, source, snap_notes = snapshot_index()
    clips, extras, clip_notes = build_clips(cache, snaps, source)
    obs, asks, instrument_gaps = observations(clips, gates)
    lanes = spec_lanes()

    atlas_text = read(ATLAS_DIR / "ATLAS.md")
    atlas_rows = [r for t in ("BANKED", "OBSERVED", "CONJECTURED")
                  for r in parse_table(atlas_text, ATLAS_COLS, section=t)[0]]
    prior = load_json(EVIDENCE) if EVIDENCE.exists() else {}

    new_rows, increments, drift = assemble(obs, lanes, atlas_rows, prior)
    allocate_ids(new_rows, atlas_rows, prior)

    bank_min = int(gates.get("atlas_bank_min_n", 2))
    ev_paths: dict[str, list[str]] = {}
    for row in new_rows:
        row["n"] = row["m"]
        row["claim"] = claim_text(row["spec_class"], row["verdict"], row["k"], row["m"],
                                  row["epochs"])
        row["confidence"] = "MEDIUM" if row["n"] >= bank_min else "LOW"
        ev_paths[row["id"]] = evidence_for(row["rows"])
    for inc in increments:
        ev_paths[inc["id"]] = evidence_for(inc["rows"])

    # The recorded BASE of every incremented row, so a second run re-applies the
    # increment to the same starting point instead of incrementing the increment.
    # ⛔ Read the `base` sub-object, not the whole stored entry: the entry also
    # carries this run's `add_n` and observation list, and treating it as the base
    # would hand `write_atlas` a row with no `n` at all.
    bases = {rid: entry["base"] for rid, entry in (prior.get("increments") or {}).items()
             if isinstance(entry, dict) and isinstance(entry.get("base"), dict)}
    by_id = {strike(r["id"])[0]: r for r in atlas_rows}
    for inc in increments:
        if inc["id"] not in bases:
            r = by_id.get(inc["id"], {})
            bases[inc["id"]] = {"n": r.get("n", "0"), "models": r.get("models", ""),
                                "evidence": r.get("evidence", "")}

    adapters = audit_adapters(clips, snaps)
    new_atlas, receipts = write_atlas(new_rows, increments, bases, ev_paths, gates)
    epoch_note = epoch_provenance_note(clips, snaps)
    new_frontier = write_frontier(drift, epoch_note, replication_note(clips))
    notes_section = render_notes_section(clips, extras, asks, obs, instrument_gaps,
                                         new_rows, increments, drift, snap_notes,
                                         clip_notes, adapters, epoch_note,
                                         blend_run_matches(clips))
    new_notes = write_notes(notes_section)

    report(clips, extras, asks, obs, new_rows, increments, drift, instrument_gaps)
    if args.dry_run:
        print("\n(dry run — nothing written)")
        return 0

    write_cache(cache)
    EVIDENCE.write_text(json.dumps({
        "writer": "scripts/skadi_import_archive.py",
        "note": "Per-claim evidence for the archive import. Every row in ATLAS.md that "
                "cites this file names a claim id below; each entry lists every clip that "
                "contributed, its model epoch, the measured value and the test applied. "
                "This is the file a stranger re-opens to re-check an archive claim.",
        "date": IMPORT_DATE,
        "claims": {row["key"]: {"id": row["id"], "spec_class": row["spec_class"],
                                "verdict": row["verdict"], "k": row["k"], "m": row["m"],
                                "epochs": row["epochs"], "observations": row["rows"]}
                   for row in sorted(new_rows, key=lambda r: r["id"])},
        "increments": {i["id"]: {"spec_class": i["spec_class"], "add_n": i["add_n"],
                                 "epochs": i["epochs"], "base": bases[i["id"]],
                                 "observations": i["rows"]}
                       for i in sorted(increments, key=lambda i: i["id"])},
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    (ATLAS_DIR / "ATLAS.md").write_text(new_atlas, encoding="utf-8", newline="\n")
    (ATLAS_DIR / "FRONTIER.md").write_text(new_frontier, encoding="utf-8", newline="\n")
    NOTES.write_text(new_notes, encoding="utf-8", newline="\n")
    BASELINES.write_text(render_baselines(clips, extras), encoding="utf-8", newline="\n")
    KNOWN.write_text(render_known_behavior(), encoding="utf-8", newline="\n")

    for p in (CACHE, EVIDENCE, ATLAS_DIR / "ATLAS.md", ATLAS_DIR / "FRONTIER.md", NOTES,
              BASELINES, KNOWN):
        print(f"WROTE     {rel(p)}")
    print("\nNext: python3 scripts/validate_music_atlas.py")
    return 0


def evidence_for(rows: list[dict]) -> list[str]:
    """The paths a stranger re-opens. Deduplicated and bounded: the per-clip detail
    lives in ARCHIVE_EVIDENCE.json, which every archive row cites."""
    paths = [rel(EVIDENCE), rel(CACHE)]
    for extra in sorted({r["audio"] for r in rows}):
        if extra.endswith(".json") and extra not in paths:
            paths.append(extra)
    snapshots = sorted({r["snapshot"] for r in rows})
    for name in snapshots[:2]:
        p = f"data/snapshots/suno/{name}"
        if p not in paths:
            paths.append(p)
    return paths


# ------------------------------------------------------- the three uneaten audits
# ⛔ AN EMPTY EXTRACTION IS A HARD ERROR. Each adapter asserts what it found; a
# directory that genuinely lacks an artifact class says so in words, and a
# directory that yields nothing at all stops the import.

FINGERPRINT_KEYS = ("duration_s", "opening_2s_db", "crest_db", "peak", "loudness_spread_db")


def fingerprint(m: dict) -> str:
    return json.dumps([m.get(k) for k in FINGERPRINT_KEYS] + [m.get("stereo")],
                      sort_keys=True)


def deep_dive_join(snaps: dict[str, dict]) -> list[dict]:
    """⭐ WHOSE EPOCH IS THE SEED SCOPED TO? Matches the deep dive's eight rows to
    clip ids THROUGH the wide-corpus sweep — both measured the same mp3s with the
    same script, so an identical measurement fingerprint is an identity, not a
    title guess. Returns one entry per deep-dive row, matched or not."""
    dd = AUDITS / DEEP_DIVE / "measurements.json"
    wide = AUDITS / WIDE_CORPUS / "measurements.json"
    if not dd.exists() or not wide.exists():
        return []
    by_print = {}
    for r in load_json(wide):
        if r.get("id"):
            by_print[fingerprint(r)] = r["id"]
    out = []
    for r in load_json(dd):
        cid = by_print.get(fingerprint(r), "")
        row = snaps.get(cid[:8]) if cid else None
        out.append({"file": r.get("file", ""), "clip": cid[:8],
                    "epoch": str((row or {}).get("major_model_version") or ""),
                    "model_name": str((row or {}).get("model_name") or ""),
                    "created": str((row or {}).get("created_at") or "")[:10]})
    return out


def audit_adapters(clips: dict[str, Clip], snaps: dict[str, dict]) -> list[dict]:
    """One entry per uneaten render-audit directory: what it holds, what it gave."""
    out: list[dict] = []

    # --- 2026-08-02-hear-learn -------------------------------------------------
    hl = AUDITS / HEAR_LEARN
    blind = sorted(hl.rglob("song_summaries.json"))
    windows = sorted(hl.rglob("window_reports.raw.json"))
    ident = sorted(hl.rglob("source_identity_map.json"))
    meas = hl / "calibration" / "measurements.json"
    rows = load_json(meas) if meas.exists() else []
    dupes = 0
    dd_prints = set()
    dd = AUDITS / DEEP_DIVE / "measurements.json"
    if dd.exists():
        dd_prints = {fingerprint(r) for r in load_json(dd)}
    for r in rows:
        if fingerprint(r) in dd_prints:
            dupes += 1
    found = len(blind) + len(windows) + len(ident) + len(rows)
    if not found:
        raise SystemExit(f"ERROR: {rel(hl)} yielded 0 artifacts — an empty extraction is an "
                         "ERROR, never a silent skip. Check the directory's schema.")
    out.append({
        "dir": rel(hl), "artifacts": found,
        "claims": 0, "reason": "no intent turn exists, and the numbers are a duplicate set",
        "adapter": (
            f"{len(blind)} `song_summaries.json` + {len(windows)} `window_reports.raw.json` "
            f"+ {len(ident)} private source map(s) + {len(rows)} measurement row(s). "
            "**Schema predates the render-audit instrument**: the blind artifacts are named "
            "`song_summaries.json`, not `blind_listener.json`, and there is **no intent "
            "comparison of any kind** — the second turn of THE BLIND RULE was never run "
            "here, because this directory is a CALIBRATION of a local listening model "
            "(Qwen2.5-Omni-3B, 4-bit) on three smoke passes over ONE song, not an audit of "
            "renders. `check_lanes` would refuse a LISTENER row citing it, and it would be "
            f"right to. Its measurement rows are the SAME renders the seed already banked: "
            f"{dupes} of {len(rows)} carry a measurement fingerprint identical to a "
            f"`{DEEP_DIVE}` row, so importing them would have been double counting. "
            "What it does carry — and what is worth more than a claim — is the calibration "
            "itself: that listener is reported reliable on voice, repeated words, broad "
            "instrumentation, energy and emotional colour, and UNRELIABLE on exact "
            "timestamps, fine instrument identity, stereo and silence."),
    })

    # --- 2026-08-10-spare-hair-tie --------------------------------------------
    sht = AUDITS / SPARE_HAIR_TIE
    sm = sht / "measurements.json"
    mapped = AUDIT_CLIP_MAP[SPARE_HAIR_TIE]
    agree = "not comparable"
    if sm.exists() and mapped in clips:
        a, b = load_json(sm), clips[mapped].measurements
        same = [k for k in ("duration_s", "opening_2s_db", "tempo_candidates_bpm")
                if a.get(k) == b.get(k)]
        agree = (f"{len(same)} of 3 spot values reproduce exactly on a fresh measurement "
                 f"of the same mp3 ({', '.join(same) or 'none'})")
    if not sm.exists():
        raise SystemExit(f"ERROR: {rel(sht)} holds no measurements.json — an empty "
                         "extraction is an ERROR, never a silent skip")
    out.append({
        "dir": rel(sht), "artifacts": 1,
        "claims": 1, "reason": "one render, one intent, one epoch — it joins",
        "adapter": (
            "A single-object directory: one `measurements.json` (a bare object, **not** the "
            "list every other audit writes) and one mp3, and **it names no clip id at all**. "
            f"It is bound to `{mapped}` by the map the coordinator supplied, the way "
            "`skadi_ingest.py` takes `--map`, because normalized prompt equality cannot bind "
            "a directory that carries no prompt. With the clip id the style text and the "
            "model epoch arrive from the snapshot and the render becomes ordinary evidence. "
            f"Instrument agreement: {agree}. It carries **no blind listening artifact**, so "
            "it is a NUMERIC-lane contribution only."),
    })

    # --- 2026-08-10-wide-corpus -----------------------------------------------
    wc = AUDITS / WIDE_CORPUS
    wm, wman = wc / "measurements.json", wc / "manifest.json"
    if not wm.exists():
        raise SystemExit(f"ERROR: {rel(wc)} holds no measurements.json")
    wrows = load_json(wm)
    with_id = [r for r in wrows if r.get("id")]
    contributed = sum(1 for c in clips.values() if c.origin == "wide-corpus")
    out.append({
        "dir": rel(wc), "artifacts": len(wrows),
        "claims": contributed, "reason": "measured renders with their style text attached",
        "adapter": (
            f"{len(wrows)} measurement row(s), {len(with_id)} carrying a clip id, plus "
            f"`{wman.name}` ({len(load_json(wman)) if wman.exists() else 0} rows). ⚠️ Its "
            "measurement rows are the manifest MERGED INTO them, so each row also carries "
            "`play_count`, `upvote_count` and `like_rate` — **audience columns, stripped on "
            "import**: they belong to `apps/lofn-analytics`, and a popularity statistic must "
            "never borrow a grain claim's evidence ladder. ⚠️ Its sampling frame is stated "
            "in its own `WIDE_ANALYSIS.md` — *every public clip with ≥40 plays OR created "
            "since 2026-07-20* — which deliberately over-samples popular and recent work; "
            "that caveat is carried into `BASELINES.md` and must be carried into any reading "
            f"of these rows. {contributed} clip(s) reached the atlas through this directory "
            "alone (the rest had a local master, which is measured losslessly and wins). It "
            "carries **no blind listening artifact**, so it is NUMERIC-lane only."),
    })
    return out


# ----------------------------------------------------- SKOP-01's run, measured at last

STYLE_HEADING = re.compile(r"^##\s*(?:1\.\s*MUSIC PROMPT|SUNO STYLE PROMPT)\b.*?$(.*?)(?=^##\s)",
                           re.M | re.S)


def blend_run_matches(clips: dict[str, Clip]) -> list[dict]:
    """Which `2026-08-13_genre-blend-refine` package does each render carry?

    ⛔ MEASURED, NOT ASSUMED. `skadi_ingest.py`'s own `normalize()` is imported
    rather than re-implemented, and the score is `difflib` over the normalized
    text — the same comparison the ingest matcher shows a human when it refuses to
    bind. A high score says the render carries that package's ask; a low one says
    it does not, and that is a finding about the evidence, not a matching failure.
    """
    if not BLEND_RUN.is_dir():
        return []
    import difflib  # noqa: PLC0415 - stdlib, only needed here
    from skadi_ingest import normalize  # noqa: PLC0415 - reuse the repaired matcher

    out: list[dict] = []
    for pkg in sorted(BLEND_RUN.glob("*/refined_package.md")):
        m = STYLE_HEADING.search(read(pkg))
        if not m:
            out.append({"package": pkg.parent.name, "clip": "", "score": None,
                        "epoch": "", "note": "no style-prompt heading this reader knows"})
            continue
        want = normalize(re.sub(r"```\w*|<!--.*?-->", " ", m.group(1), flags=re.S))
        if not want:
            out.append({"package": pkg.parent.name, "clip": "", "score": None,
                        "epoch": "", "note": "style prompt normalized to the empty string"})
            continue
        best = max(((difflib.SequenceMatcher(None, want, normalize(c.tags)).ratio(), cid)
                    for cid, c in clips.items()), default=(0.0, ""))
        out.append({"package": pkg.parent.name, "clip": best[1], "score": round(best[0], 3),
                    "epoch": clips[best[1]].epoch if best[1] else "",
                    "title": clips[best[1]].title if best[1] else "", "note": ""})
    if not out:
        raise SystemExit(f"ERROR: {rel(BLEND_RUN)} yielded 0 refined packages — an empty "
                         "extraction is an ERROR, never a silent skip")
    return out


BLEND_BIND_MIN = 0.90     # above this the render carries that package's ask


def replication_note(clips: dict[str, Clip]) -> list[str]:
    matches = [m for m in blend_run_matches(clips) if m.get("score") is not None]
    bound = [m for m in matches if m["score"] >= BLEND_BIND_MIN]
    if not matches:
        return []
    return [
        "### Replication queue — `SKOP-01` is cheaper than it was, and still may not move",
        "",
        f"Menu slot P3 replicates THE INVERSION. {len(bound)} of {len(matches)} songs from "
        "`output/2026-08-13_genre-blend-refine/` now exist as measured masters, so the "
        "replication no longer has to buy its renders: "
        + "; ".join(f"`{m['package']}` → clip `{m['clip']}` (`{m['epoch']}`, normalized "
                    f"style-text similarity {m['score']:.2f})" for m in matches)
        + ".",
        "",
        "⛔ **The row does not move.** `SKOP-01` is `LISTENER`-lane: its question is whether "
        "a blind listener names both parents, and no measurement answers that. It stays on "
        "the CONJECTURED shelf at `n = 0` with no confidence until a blind pass lands. The "
        "audio being free changes the PRICE of the probe, never its tier.",
    ]


# ------------------------------------------------------------- the epoch provenance

def epoch_provenance_note(clips: dict[str, Clip], snaps: dict[str, dict]) -> list[str]:
    """⭐ The seed scoped all thirteen of its rows to one epoch. Does the evidence?"""
    join = deep_dive_join(snaps)
    matched = [j for j in join if j["clip"] and j["epoch"]]
    if not matched:
        return []
    epochs = sorted({j["epoch"] for j in matched}, key=epoch_key)
    return [
        "### ⭐ Epoch provenance — a question the sweep raised about the SEED, not about a "
        "render",
        "",
        f"The seeded rows `SK-001`…`SK-013` are all scoped `v4.5`. Their source — the "
        "behaviour ledger in `.claude/skills/lofn-render-audit/SKILL.md` — **names no model "
        "version anywhere**; the epoch was supplied by the seeding script. Matching the deep "
        "dive's measurement rows to clip ids by identical measurement fingerprint (not by "
        f"title) resolves {len(matched)} of {len(join)} of them, and every one lands on "
        f"{', '.join('`' + e + '`' for e in epochs)}: "
        + "; ".join(f"`{j['clip']}` {j['epoch']}/{j['model_name']} created {j['created']}"
                    for j in matched)
        + ".",
        "",
        "⛔ **Nothing here edits those rows** — an observational sweep may not rescope a "
        "banked claim, and the correction is The Scientist's. It is recorded as the single "
        "highest-value target a calibration plate could take: if the seeded epoch is wrong, "
        "every drift candidate above is measuring drift against a label rather than against "
        "a generator.",
    ]


# ------------------------------------------------------------------ the seed notes

def render_notes_section(clips, extras, asks, obs, gaps, new_rows, increments, drift,
                         snap_notes, clip_notes, adapters, epochs_note, blend) -> str:
    epochs = sorted({c.epoch for c in clips.values()}, key=epoch_key)
    per_epoch = {ep: sum(1 for c in clips.values() if c.epoch == ep) for ep in epochs}
    origins = {}
    for c in clips.values():
        origins[c.origin] = origins.get(c.origin, 0) + 1

    out = [f"## {NOTES_SECTION}", "",
           "Written by `scripts/skadi_import_archive.py`. **Everything in this section "
           "carries `source: archive`** — observational evidence swept from work already on "
           "disk, with no plate, no controls, no registered predictions and no non-obvious "
           "slot behind it. It may open a cell and it may raise an `n`; it may never reach "
           "HIGH confidence (gates `atlas_high_requires_plate`), and it never scores a "
           "prediction, because none was made.", "",
           "### A1. What was measured", "",
           f"- **{len(clips)}** distinct RENDERS reached the atlas, keyed by clip id: "
           + ", ".join(f"`{ep}` {n}" for ep, n in per_epoch.items()) + ".",
           f"- By origin: " + ", ".join(f"{k} {v}" for k, v in sorted(origins.items()))
           + ". A local master is measured losslessly by us and always wins over the "
             "wide-corpus mp3 of the same clip.",
           "- " + "; ".join(snap_notes + clip_notes) + ".",
           "", "### A2. The three uneaten render audits — per-directory adapter notes", ""]
    for a in adapters:
        out += [f"**{tick(a['dir'])}** — {a['artifacts']} artifact(s) read, "
                f"{a['claims']} contribution(s) to claims ({a['reason']}).", "",
                a["adapter"], ""]

    out += ["### A3. Asks found, and what each one could be adjudicated with", "",
            "An ask is an EXPLICIT instruction in the style text. A track that does not ask "
            "contributes nothing to a class — not a zero, not a negative.", ""]
    rows = []
    for cls in sorted(asks):
        judged = sum(1 for o in obs if o["spec_class"] == cls)
        rows.append([cls, str(asks[cls]), str(judged), str(asks[cls] - judged)])
    out += table_md(["spec class", "asks found", "adjudicated", "instrument could not see"],
                    rows)
    out += ["",
            "⛔ **The largest hole is `tempo-dramaturgy`, and it is an INSTRUMENT hole, not a "
            "generator finding.** `measure_render.audit()` reports whole-track "
            "autocorrelation, and two tempos in sequence produce the same four candidates as "
            "one tempo with metric ambiguity. Every clip asking for a programmed tempo change "
            "was therefore counted as an ask and adjudicated by nothing. A windowed tempo "
            "pass is the single highest-yield instrument upgrade this map can buy, and it "
            "costs no renders. (`SPEC_CLASSES.md` already promises `tempo_candidates() over "
            "windows`; `audit()` does not do it.)",
            "",
            "⛔ **The `exclude` channel is still at zero claims, and that is honest.** Every "
            "clip on disk carries a populated negative field, but what those fields exclude "
            "is overwhelmingly timbral and vocal — voices, instruments, production clichés — "
            "and none of it is visible to a waveform measure. The exclude channel needs a "
            "blind listener, not a bigger corpus.",
            ""]

    if gaps:
        shown = gaps[:6]
        out += ["<details><summary>Instrument gaps, first few verbatim</summary>", ""]
        out += [f"- {g}" for g in shown]
        out += ["", f"({len(gaps)} in total.)", "", "</details>", ""]

    out += ["### A4. What was refused, and why", "",
            f"- **{len(extras['duplicate_file'])}** duplicate master file(s) — a byte-"
            "identical twin under a plain title. Detected by measurement fingerprint, not by "
            "filename: "
            + ("; ".join(tick(x) for x in extras["duplicate_file"]) or "none") + ".",
            f"- **{len(extras['no_clip_id'])}** measured file(s) with no clip id in the "
            "filename, so no style text and no epoch: "
            + ("; ".join(tick(x) for x in extras["no_clip_id"]) or "none")
            + ". A title match against a snapshot row was available and was **not** taken — "
              "binding a render by its title is exactly the guess this system refuses.",
            f"- **{len(extras['unjoined'])}** measured file(s) whose clip id is in no "
            "snapshot: " + ("; ".join(tick(x) for x in extras["unjoined"]) or "none") + ".",
            f"- **{len(extras['no_epoch'])}** clip(s) joined but carrying no "
            "`major_model_version`: " + (", ".join(extras["no_epoch"]) or "none")
            + ". A claim with no epoch is scoped to no version of the generator.",
            f"- **{len(extras['non_suno'])}** non-Suno render(s) "
            f"({', '.join(NON_SUNO_MARKERS)}): "
            + ("; ".join(tick(x) for x in extras["non_suno"])
               or "none found — the check is kept so a future one cannot enter unnoticed")
            + ".",
            "- **Audience and engagement: excluded entirely.** The wide-corpus rows arrive "
            "with `play_count`, `upvote_count`, `comment_count` and `like_rate` merged in; "
            "all four are stripped at the door. That lane is `apps/lofn-analytics` and "
            "`vault/BENCHMARKS.md`, and it has already burned this house once — see §6.",
            ""]

    out += ["### A5. What the sweep banked", "",
            f"- **{len(new_rows)}** new claim row(s): "
            + ("; ".join(f"`{r['id']}` {r['spec_class']} {r['verdict']} "
                         f"({r['k']} of {r['m']}, {', '.join(r['epochs'])})"
                         for r in new_rows) or "none") + ".",
            f"- **{len(increments)}** increment(s) to existing rows: "
            + ("; ".join(f"`{i['id']}` +{i['add_n']} ({', '.join(i['epochs'])})"
                         for i in increments) or "none")
            + ". An increment never rewords its claim — the append receipt is a sha of that "
              "text.",
            f"- **{len(drift)}** drift candidate(s), listed in `FRONTIER.md` under the "
            "control chart. A drift candidate is a suspicion for a plate to settle; it edits "
            "nothing.",
            ""]

    scored = [m for m in blend if m.get("score") is not None]
    bound = [m for m in scored if m["score"] >= BLEND_BIND_MIN]
    loose = [m for m in scored if m["score"] < BLEND_BIND_MIN]
    out += ["### A6. `SKOP-01` — the renders exist now, and it still may not move", "",
            "`SKOP-01` is a `LISTENER`-lane claim: its question is whether a blind listener "
            "names both parents, and **a measurement cannot answer it**. The row stays on "
            "the CONJECTURED shelf at `n = 0` with no confidence, exactly where the seed put "
            "it. What changed is the PRICE: masters from that run are on disk and measured, "
            "so the replication no longer has to buy its renders.", ""]
    out += table_md(["refined package", "closest clip", "style-text similarity", "epoch"],
                    [[m["package"], f"`{m['clip']}` {m.get('title', '')}"[:44]
                      if m["clip"] else "—",
                      f"{m['score']:.2f}" if m["score"] is not None else "—",
                      m["epoch"] or "—"] for m in scored])
    out += ["",
            "Similarity is `difflib` over the normalized style text, using "
            "`skadi_ingest.py`'s own `normalize()` — the same comparison the ingest matcher "
            "shows a human when it refuses to bind.",
            ""]
    if loose:
        out += ["⚠️ **Not every render carries the prompt the run wrote.** "
                + "; ".join(f"`{m['package']}` sits at {m['score']:.2f}" for m in loose)
                + f", against {len(bound)} that bind above {BLEND_BIND_MIN:.2f}. On the "
                  "lowest of them the rendered style text names the HISTORICAL parent first "
                  "and the modern genre as a modifier, where the refined package names the "
                  "modern one as the world — **the inverse of the order `SKOP-01` "
                  "predicts**. That makes those renders a natural counterfactual rather than "
                  "a confirmation, and makes a blind pass across the pair worth more than "
                  "either alone. It changes nothing about the row; it changes what the plate "
                  "should ask.",
                ""]

    if epochs_note:
        out += epochs_note + [""]
    return "\n".join(out)


# ---------------------------------------------------------------------- the report

def report(clips, extras, asks, obs, new_rows, increments, drift, gaps) -> None:
    epochs: dict[str, int] = {}
    for c in clips.values():
        epochs[c.epoch] = epochs.get(c.epoch, 0) + 1
    print(f"\nARCHIVE IMPORT {IMPORT_DATE}")
    print(f"  clips (deduplicated by clip id): {len(clips)}  "
          + " · ".join(f"{k} {v}" for k, v in sorted(epochs.items(), key=lambda kv: epoch_key(kv[0]))))
    print(f"  asks found: " + ", ".join(f"{k}={v}" for k, v in sorted(asks.items())))
    print(f"  observations adjudicated: {len(obs)}   instrument gaps: {len(gaps)}")
    for r in new_rows:
        print(f"  NEW  {r['id']}  {r['spec_class']:22s} {r['verdict']:9s} "
              f"{r['k']}/{r['m']}  {', '.join(r['epochs'])}")
    for i in increments:
        print(f"  INC  {i['id']}  {i['spec_class']:22s} +{i['add_n']:<3d} "
              f"{', '.join(i['epochs'])}")
    for d in drift:
        print(f"  DRIFT-CANDIDATE: {re.sub(r'[*`]', '', d)[:150]}")
    for key, vals in sorted(extras.items()):
        if vals:
            print(f"  refused[{key}]: {len(vals)}")


if __name__ == "__main__":
    sys.exit(main())
