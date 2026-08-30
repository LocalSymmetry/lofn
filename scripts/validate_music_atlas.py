#!/usr/bin/env python3
"""validate_music_atlas.py — the machine reader for the SKAÐI music atlas.

WHY THIS EXISTS
---------------
Skaði maps what Suno does through our door. Every number it obeys lives in
`vault/gates.yaml` under the MUSIC ATLAS section; every claim it makes names a
file on disk a stranger could re-open. This script is the reader that makes both
of those true rather than aspirational — a lesson paid for when an earlier gates
file had zero machine readers and every threshold in it could be violated silently.

It is also the single source of truth for the STATE FILE CONTRACT. The column
lists below are the contract; the writer scripts (`scripts/skadi_*.py`) import
them from here. An output contract that does not pin its heading convention gets
reinvented by every agent that touches it (COMPETITION_LEARNINGS L28), and a
validator keyed to a shape the artifacts do not have extracts nothing and prints
CLEAN (L25).

CONTRACT
--------
  * FAIL-OPEN on a missing/unparseable gates file: warn, use defaults, never
    hard-fail. Numbers and unambiguous string lists only.
  * An EMPTY EXTRACTION IS A HARD ERROR, never a score, never a silent skip.
    A malformed table row FAILS loudly; it is never dropped.
  * A tier is NEVER promoted here. This script can only ever LOWER or REFUSE.
  * Nothing here renders, publishes, spends, or reaches the network.

USAGE
    python3 scripts/validate_music_atlas.py
    python3 scripts/validate_music_atlas.py --doctrine
    python3 scripts/validate_music_atlas.py --preflight output/expeditions/<dir>
    python3 scripts/validate_music_atlas.py --json

EXIT   0 all checks pass · 1 one or more FAIL · 2 usage/config error
       3 PREFLIGHT REFUSED — the expedition may not open
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GATES = REPO / "vault" / "gates.yaml"
ATLAS_DIR = REPO / "vault" / "music_atlas"

# Windows consoles default to cp1252 and this repo's doctrine is full of ⛔/⭐/≥.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001 - older streams simply keep their encoding
        pass


# --------------------------------------------------------------------------- gates

# Mirrors the MUSIC ATLAS section of vault/gates.yaml. Kept in sync by hand; the
# doctrine check reports drift between the two.
DEFAULTS = {
    "atlas_probe_slots": 6,
    "atlas_probe_slots_min": 2,
    "atlas_min_positive_controls": 1,
    "atlas_min_negative_controls": 1,
    "atlas_non_obvious_quota": 1,
    "atlas_non_obvious_min_slots": 4,
    "atlas_confidence_cap_n": 4,
    "atlas_bank_min_n": 2,
    "atlas_verdicts": ["ARRIVES", "PARTIAL", "SMOOTHED", "ABSENT", "INVERTED", "TRANSFORMS"],
    "atlas_channels": ["style", "exclude", "lyric_form", "emo", "section"],
    "atlas_lanes": ["NUMERIC", "LISTENER", "BOTH"],
    "atlas_delta_moves": ["NEW-BEHAVIOR", "BEHAVIOR-CONFIRMED", "SCOPE-NARROWED",
                          "OPERATOR-LAW", "INSTRUMENT-ONLY", "NONE", "RE-AUTHORIZED"],
    "atlas_streak_reauthorize": 3,
    "atlas_scruff_rift_sightings": 3,
    "atlas_expedition_prefix": "output/expeditions/",
    "atlas_high_requires_plate": True,
}


def load_gates() -> tuple[dict, list[str]]:
    """FAIL-OPEN: a broken gates file must never hard-fail an otherwise-valid run."""
    warnings: list[str] = []
    if not GATES.exists():
        return dict(DEFAULTS), [f"gates file absent at {GATES}; using built-in defaults"]
    text = GATES.read_text(encoding="utf-8", errors="replace")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
    except Exception as exc:  # noqa: BLE001 - any parse failure falls back
        warnings.append(
            f"gates file unparseable ({exc.__class__.__name__}); using scalar fallback")
        data = _fallback_parse(text)
    if not isinstance(data, dict):
        warnings.append("gates file did not parse to a mapping; using scalar fallback")
        data = _fallback_parse(text)
    merged = dict(DEFAULTS)
    merged.update({k: v for k, v in data.items() if v is not None})
    return merged, warnings


def _fallback_parse(text: str) -> dict:
    """Minimal scalar/list reader, used only when PyYAML is unavailable or angry.

    Handles both YAML list forms the gates file actually uses: block lists
    (`key:` then `  - item`) and inline flow lists (`key: [a, b, c]`). The MUSIC
    ATLAS section is written entirely in the flow form, so a fallback that could
    only read block lists would silently hand every vocabulary check a string.

    LIST ITEMS ARE TYPED THE SAME WAY SCALARS ARE. `[850, 1000]` is two integers
    under PyYAML; a fallback that returned `['850', '1000']` would make the two
    loaders disagree about the TYPE of half the file, and the first reader to
    compare a gate numerically against a fallback-parsed value would silently
    compare a string. Found by the tester diffing the two parsers over the real
    `vault/gates.yaml` (4 keys differed).
    """
    out: dict = {}
    key = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith("  - ") and key:
            out.setdefault(key, []).append(_scalar(line[4:].strip()))
            continue
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$", line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if val == "":
            out[key] = []
        elif val.startswith("[") and val.endswith("]"):
            body = val[1:-1].strip()
            out[key] = [_scalar(p.strip()) for p in body.split(",") if p.strip()] if body else []
        else:
            out[key] = _scalar(val)
    return out


def _scalar(val: str):
    """One YAML scalar, typed the way PyYAML types it. Used for values AND list items."""
    val = val.strip()
    if val.lower() in ("true", "false"):
        return val.lower() == "true"
    if re.fullmatch(r"-?\d+", val):
        return int(val)
    if re.fullmatch(r"-?\d*\.\d+", val):
        return float(val)
    return val.strip("'\"")


# --------------------------------------------------------------------------- report

class Report:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, section: str, name: str, ok: bool | None, detail: str) -> None:
        self.rows.append({"section": section, "check": name, "ok": ok, "detail": detail})

    @property
    def failed(self) -> int:
        return sum(1 for r in self.rows if r["ok"] is False)

    def render(self) -> str:
        out, section = [], None
        for r in self.rows:
            if r["section"] != section:
                section = r["section"]
                out.append(f"\n=== {section} ===")
            mark = {True: "PASS", False: "FAIL", None: "SKIP"}[r["ok"]]
            out.append(f"  [{mark}] {r['check']}: {r['detail']}")
        n_fail = self.failed
        n_skip = sum(1 for r in self.rows if r["ok"] is None)
        n_pass = sum(1 for r in self.rows if r["ok"] is True)
        out.append(f"\n{n_pass} passed · {n_fail} FAILED · {n_skip} skipped")
        return "\n".join(out)


# ------------------------------------------------------------------- the contract
# THE STATE FILE CONTRACT. The validator, never the artifacts, is the source of
# truth for these shapes (L28). Writer scripts import them from here.

ATLAS_COLS = ["id", "kind", "scope", "channel", "lane", "claim", "verdict",
              "n", "confidence", "models", "evidence", "falsifier"]
APPEND_COLS = ["id", "sha12", "action", "note"]
TERRITORY_COLS = ["territory", "base_genres", "adjacent", "state", "footprint", "note"]
SPEC_CLASS_COLS = ["class", "lane", "definition", "measurable_by", "seed_evidence"]
COVERAGE_COLS = ["territory", "state", "probes", "claims", "last_expedition", "saturation_note"]
MENU_COLS = ["slot", "cell", "channel", "hypothesis", "price", "status"]
SCRUFF_COLS = ["scruff_id", "first_seen", "description", "sightings", "status"]
CONTROL_CHART_COLS = ["control", "expedition", "verdict", "measure", "model_epoch"]
DELTA_COLS = ["#", "date", "expedition", "move", "rows", "evidence"]
CONSUMPTION_COLS = ["date", "run_dir", "rows_cited"]
LEDGER_COLS = ["date", "expedition", "probe_id", "cell", "channel", "prediction",
               "rendered", "verdict", "tier", "harvest"]
PLAN_COLS = ["probe_id", "kind", "cell", "channel", "hypothesis", "expected", "price"]
PREDICTION_COLS = ["probe_id", "cell", "channel", "prediction", "registered_utc"]

CLAIM_KINDS = ["CELL", "OPERATOR"]
CONFIDENCES = ["LOW", "MEDIUM", "HIGH"]
TIERS = ["SPECIFIED", "RENDERED", "MEASURED", "HEARD"]
PROBE_KINDS = ["POSITIVE-CONTROL", "NEGATIVE-CONTROL", "PROBE", "NON-OBVIOUS", "SCRUFF-CHASE"]
SCRUFF_STATES = ["WATCH", "CHASED", "PROMOTED", "RETIRED"]
TERRITORY_STATES = ["UNEXPLORED", "PROBED", "SATURATED"]
MENU_STATES = ["PROPOSED", "QUEUED", "SPECIFIED", "RENDERED", "RETIRED"]
ATLAS_TIERS = ["BANKED", "OBSERVED", "CONJECTURED"]
APPEND_ACTIONS = ["SEED", "NEW", "N-INCREMENT", "SUPERSEDES", "CORRECTION"]
VOID_PLATE = "VOID-PLATE"
PENDING = "PENDING"
NON_OBVIOUS_PLACEHOLDER = "(left for a mind)"
NO_REASON_YET = "(none yet)"

# ⛔ PLACEHOLDERS SATISFY EVERY NON-EMPTINESS GATE EVER WRITTEN. A cell reading
# "(declare the model epoch)" is not empty, so `models non-empty` passes, and the
# claim ships scoped to nothing. These strings are the ones the writer scripts
# used to emit; they are now refused wherever a real value belongs, so a
# regression in a writer surfaces here instead of in the catalog.
#
# `(left for a mind)` is LEGAL — and required — in FRONTIER's MENU (DESIGN §2
# prints the non-obvious slot literally). It is a placeholder everywhere else,
# which is exactly why the scan below is scoped to ATLAS rows and control-chart
# points rather than run over every table in the vault.
PLACEHOLDER_STRINGS = [
    "(declare the model epoch)",
    "(name the observation that would refute this)",
    NON_OBVIOUS_PLACEHOLDER,
]

# CONJECTURED rows are PREDICTIONS, not observations: the `verdict` column holds
# the verdict we EXPECT, `n` is 0 (nothing has been observed) and `confidence` is
# empty (a prediction has no confidence — it has a prior and a falsifier). A
# CONJECTURED row carrying n>0 or a confidence word is an observation shelved
# where no lane rule reaches it, which is how an unheard LISTENER claim gets to
# look banked.
DASHES = {"", "-", "—", "–"}

# A LISTENER-lane claim must cite a blind-pass artifact. These are the filename
# markers a HEARD artifact carries: the render-audit instrument writes
# `blind_listener.json` / `intent_comparison.json`; skadi_listen.py writes
# `<probe>.blind.json` / `<probe>.intent.json`.
HEARD_MARKERS = ("blind_listener.json", "intent_comparison.json",
                 ".blind.json", ".intent.json")

EMPTY_CELL = {"", "-", "—", "–", "n/a", "none", "tbd", "pending"}


# --------------------------------------------------------------------- table parse

def table_cells(line: str) -> list[str]:
    """Markdown row -> cells, preserving escaped pipes and pipes inside code spans.

    A plain `str.split('|')` invents a column wherever a claim contains `a|b` in
    backticks and then drops the row as malformed. Lifted in spirit from the
    repository's committed-ledger validators, which require exactly this.
    """
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return []
    cells: list[str] = []
    cell: list[str] = []
    code_ticks = 0
    body = stripped[1:-1]
    i = 0
    while i < len(body):
        char = body[i]
        if char == "`":
            run = 1
            while i + run < len(body) and body[i + run] == "`":
                run += 1
            cell.append("`" * run)
            if code_ticks == 0:
                code_ticks = run
            elif code_ticks == run:
                code_ticks = 0
            i += run
            continue
        backslashes = 0
        j = i - 1
        while j >= 0 and body[j] == "\\":
            backslashes += 1
            j -= 1
        if char == "|" and code_ticks == 0 and backslashes % 2 == 0:
            cells.append("".join(cell).strip())
            cell = []
        else:
            cell.append(char)
        i += 1
    cells.append("".join(cell).strip())
    return cells


def _is_rule(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", c or "") for c in cells)


def _section_bounds(lines: list[str], section: str | None) -> tuple[int, int] | None:
    if section is None:
        return 0, len(lines)
    head = next((i for i, ln in enumerate(lines)
                 if re.match(r"^#{2,4}\s+", ln)
                 and ln.split(None, 1)[-1].strip() == section), None)
    if head is None:
        return None
    start = head + 1
    end = next((i for i in range(start, len(lines)) if re.match(r"^#{2,4}\s+", lines[i])),
               len(lines))
    return start, end


def _locate(text: str, cols: list[str], section: str | None) -> tuple[list[str], int, int] | None:
    """-> (lines, header_index, section_end) or None when the header is absent."""
    lines = text.splitlines()
    bounds = _section_bounds(lines, section)
    if bounds is None:
        return None
    start, end = bounds
    want = [c.lower() for c in cols]
    for i in range(start, end):
        cells = table_cells(lines[i])
        if cells and [c.lower() for c in cells] == want:
            return lines, i, end
    return None


def parse_table(text: str, cols: list[str], section: str | None = None) -> tuple[list[dict], list[str]]:
    """Find the table whose header is exactly `cols` and return (rows, problems).

    Never raises and never silently drops a row: a row whose cell count does not
    match the header lands in `problems` with its line number. An empty table is
    a problem, not an empty result — an empty extraction is an ERROR (L25/L28).

    `section` restricts the search to the body of that `## `/`### ` heading.
    """
    problems: list[str] = []
    located = _locate(text, cols, section)
    if located is None:
        lines = text.splitlines()
        if section is not None and _section_bounds(lines, section) is None:
            return [], [f"section {section!r} not found"]
        return [], [f"no table header matching {cols}"
                    + (f" in section {section!r}" if section else "")]
    lines, header_at, end = located

    rows: list[dict] = []
    for i in range(header_at + 1, end):
        cells = table_cells(lines[i])
        if not cells:
            if rows or i > header_at + 2:
                break
            continue
        if _is_rule(cells):
            continue
        if len(cells) != len(cols):
            problems.append(f"malformed row at line {i + 1}: "
                            f"{len(cells)} cell(s), header has {len(cols)}")
            continue
        row = {c: cells[k] for k, c in enumerate(cols)}
        row["_line"] = str(i + 1)
        rows.append(row)
    if not rows:
        problems.append(f"table {cols[0]!r}…{cols[-1]!r}"
                        + (f" in section {section!r}" if section else "")
                        + " has a header but no rows")
    return rows, problems


def table_span(text: str, cols: list[str], section: str | None = None) -> tuple[int, int] | None:
    """-> (first_line_index, end_line_index_exclusive) of the whole table block.

    Used by the writer scripts to splice a rebuilt table back in place without
    disturbing a single line of the surrounding prose. Returns None when the
    header is absent, which the caller must treat as an ERROR rather than as an
    excuse to append somewhere convenient.
    """
    located = _locate(text, cols, section)
    if located is None:
        return None
    lines, header_at, end = located
    last = header_at
    for i in range(header_at + 1, end):
        cells = table_cells(lines[i])
        if not cells:
            break
        last = i
    return header_at, last + 1


def strike(value: str) -> tuple[str, bool]:
    """`~~SK-001~~` -> ('SK-001', True). A superseded row is kept, never deleted."""
    m = re.fullmatch(r"~~\s*(.*?)\s*~~", value.strip())
    return (m.group(1).strip(), True) if m else (value.strip(), False)


def sha12(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:12]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def expedition_dirs(gates: dict) -> list[Path]:
    root = REPO / str(gates.get("atlas_expedition_prefix", "output/expeditions/")).strip("/")
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))


def is_closed(expedition: Path) -> bool:
    """An expedition is CLOSED when it has written its INDEX."""
    return (expedition / "INDEX.md").exists()


def _evidence_paths(cell: str) -> list[str]:
    return [p.strip().strip("`") for p in cell.split(",") if p.strip().strip("`")]


def _resolves(rel: str) -> bool:
    """Is this a repo-relative path to a FILE a stranger could re-open, inside the repo?

    ⛔ Four refusals, each one a way a citation can look resolvable while being
    unre-checkable:

      * `..` anywhere — `vault/../MEMORY.md` reads as a vault path and is not one.
      * absolute or drive-prefixed — `/etc/passwd`, `C:\\...`, `\\\\host\\share`
        are not repo-relative, and a claim that cites one cites a machine.
      * a DIRECTORY — a directory is not a receipt. `output/render-audits/`
        exists on every checkout and proves nothing about what was measured.
      * anything that resolves outside the repo — symlinks included.
    """
    rel = rel.strip().strip("`")
    if not rel or rel in EMPTY_CELL:
        return False
    if "\\" in rel:
        rel = rel.replace("\\", "/")
    if ".." in [p for p in rel.split("/") if p]:
        return False
    candidate = Path(rel)
    if candidate.is_absolute() or candidate.drive or rel.startswith("/") or rel.startswith("~"):
        return False
    target = REPO / candidate
    if not target.is_file():
        return False
    try:
        return target.resolve().is_relative_to(REPO.resolve())
    except (OSError, ValueError):  # unresolvable path, or a cross-drive comparison
        return False


# ------------------------------------------------------------------- atlas checks

def _atlas_rows(rep: Report) -> tuple[dict[str, list[dict]], list[str]]:
    """-> ({tier: rows}, problems). Tier sections are BANKED / OBSERVED / CONJECTURED."""
    path = ATLAS_DIR / "ATLAS.md"
    text = read(path)
    if not text:
        return {}, [f"{path.relative_to(REPO).as_posix()} absent or empty"]
    tiers: dict[str, list[dict]] = {}
    problems: list[str] = []
    for tier in ("BANKED", "OBSERVED", "CONJECTURED"):
        rows, probs = parse_table(text, ATLAS_COLS, section=tier)
        # An empty tier section is legitimate (nothing banked yet); an absent
        # header is not.
        probs = [p for p in probs if "has a header but no rows" not in p]
        problems.extend(f"{tier}: {p}" for p in probs)
        tiers[tier] = rows
    return tiers, problems


def check_atlas_schema(root: Path, gates: dict, rep: Report) -> None:
    sec = "ATLAS — the catalog"
    tiers, problems = _atlas_rows(rep)
    if problems:
        rep.add(sec, "ATLAS.md parses against the row schema", False, "; ".join(problems))
        if not tiers:
            return
    else:
        total = sum(len(v) for v in tiers.values())
        rep.add(sec, "ATLAS.md parses against the row schema", True,
                f"{total} row(s): " + " · ".join(f"{k}={len(v)}" for k, v in tiers.items()))

    verdicts = {str(v).upper() for v in gates.get("atlas_verdicts", [])}
    channels = {str(v).lower() for v in gates.get("atlas_channels", [])}
    lanes = {str(v).upper() for v in gates.get("atlas_lanes", [])}

    bad: list[str] = []
    dangling: list[str] = []
    thin: list[str] = []
    planted: list[str] = []
    ids: dict[str, str] = {}
    for tier, rows in tiers.items():
        for r in rows:
            rid, struck = strike(r["id"])
            tag = f"{tier}/{rid}"
            prediction = tier == "CONJECTURED"
            if rid in ids:
                bad.append(f"{tag} duplicate id (also in {ids[rid]})")
            ids[rid] = tier
            if not re.fullmatch(r"SK-\d{3}|SKOP-\d{2}", rid):
                bad.append(f"{tag} id is not SK-NNN / SKOP-NN")
            if r["kind"].upper() not in CLAIM_KINDS:
                bad.append(f"{tag} kind={r['kind']!r}")
            if r["channel"].lower() not in channels:
                bad.append(f"{tag} channel={r['channel']!r}")
            if r["lane"].upper() not in lanes:
                bad.append(f"{tag} lane={r['lane']!r}")
            if r["verdict"].upper() not in verdicts:
                bad.append(f"{tag} verdict={r['verdict']!r}")
            # A prediction carries no confidence and no n; both are checked, as
            # the shape they must have, by check_conjectured_shelf.
            if not prediction and r["confidence"].upper() not in CONFIDENCES:
                bad.append(f"{tag} confidence={r['confidence']!r}")
            if not prediction and not re.fullmatch(r"\d+", r["n"]):
                bad.append(f"{tag} n={r['n']!r} is not an integer")
            for column in ATLAS_COLS:
                for ghost in PLACEHOLDER_STRINGS:
                    if ghost in r[column]:
                        planted.append(f"{tag} {column}={ghost!r}")
            if r["scope"].lower() in EMPTY_CELL:
                thin.append(f"{tag} empty scope")
            if r["claim"].lower() in EMPTY_CELL:
                thin.append(f"{tag} empty claim")
            if r["models"].lower() in EMPTY_CELL:
                thin.append(f"{tag} empty models")
            if r["falsifier"].lower() in EMPTY_CELL:
                thin.append(f"{tag} empty falsifier")
            paths = _evidence_paths(r["evidence"])
            if not paths and not struck:
                thin.append(f"{tag} empty evidence")
            dangling.extend(f"{tag} -> {p}" for p in paths if not _resolves(p))

    rep.add(sec, "row values use the canonical vocabulary", not bad,
            "clean" if not bad else f"{len(bad)}: " + "; ".join(bad[:10]))
    rep.add(sec, "scope, claim, models and falsifier are non-empty", not thin,
            "clean" if not thin else f"{len(thin)}: " + "; ".join(thin[:10]))
    rep.add(sec, "no cell carries a writer's placeholder text", not planted,
            "clean" if not planted else f"{len(planted)}: " + "; ".join(planted[:10])
            + " — a placeholder is not empty, so every non-emptiness gate passes it; "
              "the value it stands in for was never supplied")
    rep.add(sec, "every evidence path exists on disk", not dangling,
            f"{sum(len(_evidence_paths(r['evidence'])) for v in tiers.values() for r in v)} "
            f"path(s), {len(dangling)} dangling"
            + (": " + "; ".join(dangling[:8]) if dangling else ""))


def check_append_blocks(root: Path, gates: dict, rep: Report) -> None:
    """The dated append blocks are the atlas's receipts: id + sha12 of the claim
    text. A claim silently reworded after banking breaks its own sha and says so."""
    sec = "ATLAS — the catalog"
    text = read(ATLAS_DIR / "ATLAS.md")
    if not text:
        rep.add(sec, "every row has a dated append receipt", False, "ATLAS.md absent")
        return
    blocks = re.findall(r"^###\s+(\d{4}-\d{2}-\d{2})\s+(\S+)\s+additions\s*$", text, re.MULTILINE)
    receipts: dict[str, str] = {}
    dup: list[str] = []
    malformed: list[str] = []
    for m in re.finditer(r"^###\s+\d{4}-\d{2}-\d{2}\s+\S+\s+additions\s*$", text, re.MULTILINE):
        end = text.find("\n### ", m.end())
        body = text[m.end(): end if end > 0 else len(text)]
        rows, probs = parse_table(body, APPEND_COLS)
        malformed.extend(p for p in probs if "malformed row" in p)
        # An id may appear in SEVERAL blocks — that is the audit trail of an
        # n-increment. It may not appear twice in the SAME block. The last
        # receipt is the live one, and every receipt for an id must agree,
        # because an n-increment never changes the claim text.
        seen_here: set[str] = set()
        for r in rows:
            rid, _ = strike(r["id"])
            if rid in seen_here:
                dup.append(rid)
            seen_here.add(rid)
            if rid in receipts and receipts[rid] != r["sha12"].strip().strip("`"):
                dup.append(f"{rid} (receipt sha changed between blocks)")
            receipts[rid] = r["sha12"].strip().strip("`")

    rep.add(sec, "dated append blocks parse", not malformed and bool(blocks),
            f"{len(blocks)} block(s), {len(receipts)} receipt(s)"
            + ("; " + "; ".join(malformed[:5]) if malformed else "")
            + ("" if blocks else " — no `### <date> <slug> additions` block found"))
    if dup:
        rep.add(sec, "append receipts are consistent", False,
                f"{len(dup)}: {', '.join(sorted(set(dup))[:8])}")
    else:
        rep.add(sec, "append receipts are consistent", True,
                "no id repeated inside a block; no receipt sha changed between blocks")

    tiers, _ = _atlas_rows(rep)
    missing: list[str] = []
    drifted: list[str] = []
    for tier, rows in tiers.items():
        for r in rows:
            rid, _ = strike(r["id"])
            if rid not in receipts:
                missing.append(f"{tier}/{rid}")
                continue
            actual = sha12(r["claim"])
            if receipts[rid] != actual:
                drifted.append(f"{tier}/{rid} receipt={receipts[rid]} actual={actual}")
    rep.add(sec, "every claim row has an append receipt", not missing,
            "clean" if not missing else f"{len(missing)} without a receipt: "
            + ", ".join(missing[:8]))
    rep.add(sec, "claim text matches its recorded sha12", not drifted,
            "clean" if not drifted else f"{len(drifted)} drifted: " + "; ".join(drifted[:6])
            + " — a claim was reworded after it was banked")


def check_confidence_vs_n(root: Path, gates: dict, rep: Report) -> None:
    sec = "ATLAS — the catalog"
    cap = int(gates.get("atlas_confidence_cap_n", 4))
    tiers, _ = _atlas_rows(rep)
    rows = [r for v in tiers.values() for r in v]
    if not rows:
        rep.add(sec, "confidence capped by n", None, "no claim rows to check")
        return
    over = []
    for r in rows:
        rid, _ = strike(r["id"])
        if not re.fullmatch(r"\d+", r["n"]):
            continue
        if int(r["n"]) < cap and r["confidence"].upper() == "HIGH":
            over.append(f"{rid} n={r['n']} confidence=HIGH")
    rep.add(sec, f"confidence capped by n (gates atlas_confidence_cap_n)", not over,
            f"{len(rows)} row(s) checked; {len(over)} over cap"
            + (": " + "; ".join(over[:8]) if over else ""))


def check_banked_min_n(root: Path, gates: dict, rep: Report) -> None:
    sec = "ATLAS — the catalog"
    floor = int(gates.get("atlas_bank_min_n", 2))
    tiers, _ = _atlas_rows(rep)
    banked = tiers.get("BANKED", [])
    observed = tiers.get("OBSERVED", [])
    if not banked and not observed:
        rep.add(sec, "BANKED respects the n floor", None, "no banked or observed rows")
        return
    under = [f"{strike(r['id'])[0]} n={r['n']}" for r in banked
             if re.fullmatch(r"\d+", r["n"]) and int(r["n"]) < floor]
    rep.add(sec, "BANKED respects the n floor (gates atlas_bank_min_n)", not under,
            f"{len(banked)} banked row(s); {len(under)} under floor"
            + (": " + "; ".join(under[:8]) if under else ""))
    # OBSERVED is the n=1 shelf; a row that has grown past the floor belongs in
    # BANKED and its presence here is a promotion the banker failed to make.
    stale = [f"{strike(r['id'])[0]} n={r['n']}" for r in observed
             if re.fullmatch(r"\d+", r["n"]) and int(r["n"]) > 1]
    rep.add(sec, "OBSERVED holds only n=1 rows", not stale,
            f"{len(observed)} observed row(s)"
            + ("" if not stale else f"; {len(stale)} above n=1: " + "; ".join(stale[:8])))


def check_conjectured_shelf(root: Path, gates: dict, rep: Report) -> None:
    """⛔ CONJECTURED IS A PREDICTION SHELF, NOT A PARKING SHELF.

    DESIGN §2: "`## CONJECTURED` (registered predictions not yet rendered)". A
    prediction has no observations behind it, so its `n` is 0 (or `—`) and its
    `confidence` is empty — and `check_lanes` deliberately does not reach this
    tier, because a blind pass on an unrendered probe is impossible.

    Both facts together are the hole: a row with n=9 and HIGH confidence parked
    in CONJECTURED reads to every other check as a registered prediction and to
    every human as a banked finding, and no lane rule touches it. Definitional
    drift, unpoliced, was the tester's finding here.
    """
    sec = "CONJECTURED — a prediction carries no n and no confidence"
    tiers, _ = _atlas_rows(rep)
    rows = tiers.get("CONJECTURED", [])
    if not rows:
        rep.add(sec, "CONJECTURED rows are predictions, not observations", None,
                "no registered predictions on the shelf")
        return
    offenders: list[str] = []
    for r in rows:
        rid, _ = strike(r["id"])
        n_cell = r["n"].strip()
        conf = r["confidence"].strip()
        if n_cell not in DASHES and not (re.fullmatch(r"\d+", n_cell) and int(n_cell) == 0):
            offenders.append(f"{rid} n={n_cell!r} — a prediction has observed nothing; "
                             "n must be 0 or `—`")
        if conf not in DASHES:
            offenders.append(f"{rid} confidence={conf!r} — a prediction carries a falsifier, "
                             "not a confidence")
    rep.add(sec, "CONJECTURED rows are predictions, not observations", not offenders,
            f"{len(rows)} registered prediction(s)" if not offenders else
            f"{len(offenders)}: " + "; ".join(offenders[:8])
            + " — evidence parked in CONJECTURED is out of reach of every lane rule")


def check_archive_provenance(root: Path, gates: dict, rep: Report) -> None:
    """⛔ HIGH CONFIDENCE REQUIRES A PLATE — n IS NOT PROVENANCE.

    An archive sweep can reach an `n` a plate never will: the catalogue is
    hundreds of renders deep and every one of them is free to re-measure. None of
    it had a registered prediction, a positive control, a negative control or a
    non-obvious slot behind it, and a corpus swept after the fact answers the
    question it was searched with. Left alone, `check_confidence_vs_n` would have
    waved a HIGH row through on n=46 of exactly that evidence — the cap reads the
    count and cannot see where the count came from.

    So HIGH additionally requires at least one evidence path from a real
    expedition plate (under gates `atlas_expedition_prefix`). Archive-only claims
    cap at MEDIUM no matter how large n grows. This check only ever LOWERS.
    """
    sec = "PROVENANCE — HIGH confidence requires a plate, not a big n"
    name = ("HIGH-confidence claims cite an expedition plate "
            "(gates atlas_high_requires_plate)")
    tiers, _ = _atlas_rows(rep)
    rows = tiers.get("BANKED", []) + tiers.get("OBSERVED", [])
    if not rows:
        rep.add(sec, name, None, "no banked or observed rows")
        return
    if not gates.get("atlas_high_requires_plate", True):
        rep.add(sec, name, None,
                "gates atlas_high_requires_plate is false — the plate requirement is "
                "switched off, and confidence rests on n alone")
        return
    prefix = str(gates.get("atlas_expedition_prefix", "output/expeditions/")).strip("/") + "/"
    offenders: list[str] = []
    high = 0
    for r in rows:
        if r["confidence"].strip().upper() != "HIGH":
            continue
        high += 1
        plate = [p for p in _evidence_paths(r["evidence"])
                 if p.replace("\\", "/").startswith(prefix) and _resolves(p)]
        if not plate:
            offenders.append(f"{strike(r['id'])[0]} n={r['n']}")
    rep.add(sec, name, not offenders,
            f"{high} HIGH row(s) of {len(rows)} checked"
            + ("; all cite a plate" if high and not offenders else "")
            + ("" if not offenders else
               f"; {len(offenders)} archive-only: " + "; ".join(offenders[:8])
               + f" — no evidence under {prefix}, so no controls and no registered "
                 "prediction stood behind it. An archive sweep caps at MEDIUM at any n."))


def check_lanes(root: Path, gates: dict, rep: Report) -> None:
    """THE BLIND RULE, made machine-checkable. A LISTENER-lane claim is a claim
    about what a listener heard; if no blind artifact is cited, nobody heard it."""
    sec = "LANES — a claim about hearing must cite a hearing"
    tiers, _ = _atlas_rows(rep)
    # CONJECTURED rows are registered predictions not yet rendered; a blind pass
    # is impossible by construction, so the lane rule does not reach them.
    rows = tiers.get("BANKED", []) + tiers.get("OBSERVED", [])
    if not rows:
        rep.add(sec, "LISTENER-lane claims cite HEARD evidence", None,
                "no banked or observed rows")
        return
    offenders = []
    checked = 0
    for r in rows:
        if r["lane"].upper() not in ("LISTENER", "BOTH"):
            continue
        checked += 1
        paths = _evidence_paths(r["evidence"])
        heard = [p for p in paths
                 if any(marker in Path(p).name for marker in HEARD_MARKERS) and _resolves(p)]
        if not heard:
            offenders.append(f"{strike(r['id'])[0]} ({r['lane']})")
    rep.add(sec, "LISTENER-lane claims cite HEARD evidence", not offenders,
            f"{checked} listener-lane row(s); {len(offenders)} without a blind artifact"
            + (": " + ", ".join(offenders[:8]) if offenders else "")
            + (" — the blind pass is the only instrument that can fail here" if offenders else ""))


# ---------------------------------------------------------------- territory/spec

def check_territory_states(root: Path, gates: dict, rep: Report) -> None:
    sec = "TERRITORIES — unexplored is never absent"
    path = ATLAS_DIR / "TERRITORIES.md"
    text = read(path)
    if not text:
        rep.add(sec, "TERRITORIES.md present", False, f"absent or empty at {path}")
        return
    rows, problems = parse_table(text, TERRITORY_COLS)
    rep.add(sec, "TERRITORIES.md parses", not problems, "; ".join(problems)
            if problems else f"{len(rows)} territor(ies)")
    if not rows:
        return
    bad_state = [f"{r['territory']}={r['state']!r}" for r in rows
                 if r["state"].upper() not in TERRITORY_STATES]
    rep.add(sec, "every territory carries a canonical state", not bad_state,
            "clean" if not bad_state else f"{len(bad_state)}: " + "; ".join(bad_state[:8]))
    # UNEXPLORED must be DECLARED: either a deliberate reason or the honest
    # `(none yet)`. A blank cell is the one thing it may not be — that is
    # shading the unsounded.
    silent = [r["territory"] for r in rows
              if r["state"].upper() == "UNEXPLORED" and r["note"].strip().lower() in EMPTY_CELL]
    rep.add(sec, f"UNEXPLORED carries a reason or {NO_REASON_YET!r}", not silent,
            "clean" if not silent else
            f"{len(silent)} silent unexplored territor(ies): " + ", ".join(silent[:8]))
    empty_bases = [r["territory"] for r in rows if r["base_genres"].strip().lower() in EMPTY_CELL]
    rep.add(sec, "every territory names its base genres", not empty_bases,
            "clean" if not empty_bases else f"{len(empty_bases)}: " + ", ".join(empty_bases[:8]))


def check_spec_classes(root: Path, gates: dict, rep: Report) -> None:
    sec = "SPEC CLASSES — every class declares its lane"
    path = ATLAS_DIR / "SPEC_CLASSES.md"
    text = read(path)
    if not text:
        rep.add(sec, "SPEC_CLASSES.md present", False, f"absent or empty at {path}")
        return
    rows, problems = parse_table(text, SPEC_CLASS_COLS)
    rep.add(sec, "SPEC_CLASSES.md parses", not problems, "; ".join(problems)
            if problems else f"{len(rows)} spec class(es)")
    if not rows:
        return
    lanes = {str(v).upper() for v in gates.get("atlas_lanes", [])}
    bad = [f"{r['class']}={r['lane']!r}" for r in rows if r["lane"].upper() not in lanes]
    rep.add(sec, "every spec class declares a canonical lane", not bad,
            "clean" if not bad else f"{len(bad)}: " + "; ".join(bad[:8]))
    thin = [r["class"] for r in rows
            if r["definition"].strip().lower() in EMPTY_CELL
            or r["measurable_by"].strip().lower() in EMPTY_CELL]
    rep.add(sec, "every spec class says how it is measured", not thin,
            "clean" if not thin else f"{len(thin)}: " + ", ".join(thin[:8]))
    # A NUMERIC-lane class whose measurable_by names no instrument is a lane
    # label with nothing behind it.
    unarmed = [r["class"] for r in rows if r["lane"].upper() in ("NUMERIC", "BOTH")
               and "measure_render" not in r["measurable_by"]
               and "measure_soundcraft" not in r["measurable_by"]
               and "snapshot" not in r["measurable_by"]]
    rep.add(sec, "NUMERIC/BOTH classes name a measuring instrument", not unarmed,
            "clean" if not unarmed else
            f"{len(unarmed)} class(es) with no instrument: " + ", ".join(unarmed[:8]))


# ------------------------------------------------------------------- frontier

def check_scruff(root: Path, gates: dict, rep: Report) -> None:
    sec = "SCRUFF SHELF — repetition is the rift"
    text = read(ATLAS_DIR / "FRONTIER.md")
    if not text:
        rep.add(sec, "FRONTIER.md present", False, "absent or empty")
        return
    rows, problems = parse_table(text, SCRUFF_COLS, section="SCRUFF SHELF")
    problems = [p for p in problems if "has a header but no rows" not in p]
    rep.add(sec, "scruff shelf parses", not problems,
            "; ".join(problems) if problems else f"{len(rows)} scruff row(s)")
    if not rows:
        return
    bad = [f"{r['scruff_id']}={r['status']!r}" for r in rows
           if r["status"].upper() not in SCRUFF_STATES]
    rep.add(sec, "scruff status uses the canonical vocabulary", not bad,
            "clean" if not bad else f"{len(bad)}: " + "; ".join(bad[:8]))
    threshold = int(gates.get("atlas_scruff_rift_sightings", 3))
    rifts = []
    for r in rows:
        sightings = [s for s in re.split(r"[,\s]+", r["sightings"]) if s and s not in EMPTY_CELL]
        if len(sightings) >= threshold:
            rifts.append(f"{r['scruff_id']} ({len(sightings)} sightings)")
    # A rift is the map's most valuable object, not a defect. FLAG, never FAIL.
    rep.add(sec, "rift detector (gates atlas_scruff_rift_sightings)",
            None if rifts else True,
            f"{len(rows)} scruff row(s); " + (
                f"RIFT: {', '.join(rifts)} — the same deviation keeps arriving unasked; "
                "this is survey material, not a defect" if rifts else "no rift at threshold"))


def check_delta_log(root: Path, gates: dict, rep: Report) -> None:
    sec = "DELTA LOG — did the map move"
    text = read(ATLAS_DIR / "FRONTIER.md")
    if not text:
        rep.add(sec, "FRONTIER.md present", False, "absent or empty")
        return
    rows, problems = parse_table(text, DELTA_COLS, section="DELTA LOG")
    problems = [p for p in problems if "has a header but no rows" not in p]
    rep.add(sec, "delta log parses", not problems,
            "; ".join(problems) if problems else f"{len(rows)} delta row(s)")

    moves = {str(m).upper() for m in gates.get("atlas_delta_moves", [])}
    bad = [f"#{r['#']}={r['move']!r}" for r in rows if r["move"].upper() not in moves]
    rep.add(sec, "delta moves use the canonical vocabulary", not bad,
            "clean" if not bad else f"{len(bad)}: " + "; ".join(bad[:8]))

    closed = [e for e in expedition_dirs(gates) if is_closed(e)]
    logged = [r["expedition"].strip().strip("`") for r in rows]
    missing = [e.name for e in closed if e.name not in logged]
    extra = [n for n in logged if n and n not in {e.name for e in closed}]
    dupes = sorted({n for n in logged if logged.count(n) > 1})
    ok = not missing and not extra and not dupes
    rep.add(sec, "exactly one delta row per closed expedition", ok,
            f"{len(closed)} closed expedition(s), {len(rows)} delta row(s)"
            + (f"; missing: {', '.join(missing[:6])}" if missing else "")
            + (f"; unknown expedition: {', '.join(extra[:6])}" if extra else "")
            + (f"; duplicated: {', '.join(dupes[:6])}" if dupes else ""))

    dangling = [f"#{r['#']} -> {p}" for r in rows for p in _evidence_paths(r["evidence"])
                if not _resolves(p)]
    rep.add(sec, "delta evidence resolves on disk", not dangling,
            "clean" if not dangling else f"{len(dangling)}: " + "; ".join(dangling[:6]))

    # THE TRIPWIRE. INSTRUMENT-ONLY and NONE are blameless — a wall-map is a
    # result. Only a STREAK is a signal, and what it trips is a human
    # re-authorization, never a score.
    cap = int(gates.get("atlas_streak_reauthorize", 3))
    tail = [r["move"].upper() for r in rows][-cap:]
    streak = len(tail) == cap and all(m in ("INSTRUMENT-ONLY", "NONE") for m in tail)
    rep.add(sec, f"no streak of {cap} non-moving expeditions", None if streak else True,
            f"last {len(tail)} move(s): {', '.join(tail) or '(none yet)'}"
            + (f" — RE-AUTHORIZE: the last {cap} expeditions moved nothing on the map. "
               "This is not a verdict on the music; it asks The Scientist for a recorded "
               "decision (a RE-AUTHORIZED row) before the next plate." if streak else ""))


def check_coverage(root: Path, gates: dict, rep: Report) -> None:
    sec = "FRONTIER — the coverage face"
    text = read(ATLAS_DIR / "FRONTIER.md")
    if not text:
        rep.add(sec, "FRONTIER.md present", False, "absent or empty")
        return
    rows, problems = parse_table(text, COVERAGE_COLS, section="COVERAGE")
    rep.add(sec, "coverage table parses", not problems,
            "; ".join(problems) if problems else f"{len(rows)} coverage row(s)")
    if not rows:
        return
    terr_rows, _ = parse_table(read(ATLAS_DIR / "TERRITORIES.md"), TERRITORY_COLS)
    names = {r["territory"] for r in terr_rows}
    if names:
        missing = sorted(names - {r["territory"] for r in rows})
        extra = sorted({r["territory"] for r in rows} - names)
        rep.add(sec, "coverage names exactly the declared territories",
                not missing and not extra,
                f"{len(rows)} coverage row(s) vs {len(names)} territor(ies)"
                + (f"; missing: {', '.join(missing[:6])}" if missing else "")
                + (f"; unknown: {', '.join(extra[:6])}" if extra else ""))
    bad = [f"{r['territory']}={r['state']!r}" for r in rows
           if r["state"].upper() not in TERRITORY_STATES]
    rep.add(sec, "coverage states are canonical", not bad,
            "clean" if not bad else f"{len(bad)}: " + "; ".join(bad[:8]))

    menu, menu_problems = parse_table(text, MENU_COLS, section="MENU")
    menu_problems = [p for p in menu_problems if "has a header but no rows" not in p]
    rep.add(sec, "menu parses", not menu_problems,
            "; ".join(menu_problems) if menu_problems else
            f"{len(menu)} menu slot(s) — a menu, never an order; renders are The Scientist's")


def check_control_chart(root: Path, gates: dict, rep: Report) -> None:
    sec = "CONTROL CHART — the instrument, watched"
    text = read(ATLAS_DIR / "FRONTIER.md")
    if not text:
        rep.add(sec, "FRONTIER.md present", False, "absent or empty")
        return
    rows, problems = parse_table(text, CONTROL_CHART_COLS, section="CONTROL CHART")
    problems = [p for p in problems if "has a header but no rows" not in p]
    rep.add(sec, "control chart parses", not problems,
            "; ".join(problems) if problems else f"{len(rows)} chart point(s)")
    if not rows:
        return
    verdicts = {str(v).upper() for v in gates.get("atlas_verdicts", [])} | {VOID_PLATE}
    bad = [f"{r['control']}@{r['expedition']}={r['verdict']!r}" for r in rows
           if r["verdict"].upper() not in verdicts]
    rep.add(sec, "chart verdicts are canonical", not bad,
            "clean" if not bad else f"{len(bad)}: " + "; ".join(bad[:8]))
    planted = [f"{r['control']}@{r['expedition']} {col}={ghost!r}"
               for r in rows for col in CONTROL_CHART_COLS for ghost in PLACEHOLDER_STRINGS
               if ghost in r[col]]
    epochs = sorted({r["model_epoch"] for r in rows
                     if r["model_epoch"] not in EMPTY_CELL
                     and not any(g in r["model_epoch"] for g in PLACEHOLDER_STRINGS)})
    rep.add(sec, "every chart point names its model epoch",
            all(r["model_epoch"] not in EMPTY_CELL for r in rows) and not planted,
            f"epochs on the chart: {', '.join(epochs) or '(none)'} — a version string is a "
            "claim, not a fact"
            + (f"; {len(planted)} placeholder(s): " + "; ".join(planted[:6]) if planted else ""))


def check_consumption(root: Path, gates: dict, rep: Report) -> None:
    """Does the atlas serve the ear? INFO level — this measures direction, never truth."""
    sec = "CONSUMPTION — the atlas serves the ear"
    text = read(ATLAS_DIR / "FRONTIER.md")
    if not text:
        rep.add(sec, "consumption log present", False, "FRONTIER.md absent or empty")
        return
    rows, problems = parse_table(text, CONSUMPTION_COLS, section="CONSUMPTION LOG")
    problems = [p for p in problems if "has a header but no rows" not in p]
    if problems:
        rep.add(sec, "consumption log parses", False, "; ".join(problems))
        return
    logged_runs = {r["run_dir"].strip().strip("`") for r in rows}
    tiers, _ = _atlas_rows(rep)
    known = {strike(r["id"])[0] for v in tiers.values() for r in v}
    citing: dict[str, set[str]] = {}
    out = REPO / "output"
    if out.is_dir() and known:
        for run in sorted(p for p in out.iterdir() if p.is_dir())[-40:]:
            if run.name == "expeditions":
                continue
            for md in list(run.glob("*.md"))[:60]:
                hits = {m for m in re.findall(r"\bSK(?:OP)?-\d{2,3}\b", read(md)) if m in known}
                if hits:
                    citing.setdefault(run.relative_to(REPO).as_posix(), set()).update(hits)
    unlogged = sorted(set(citing) - logged_runs)
    rep.add(sec, "runs citing atlas rows are logged", None,
            f"{len(rows)} logged consumption row(s); {len(citing)} recent run(s) cite atlas ids"
            + (f"; not yet logged: {', '.join(unlogged[:6])}" if unlogged else "")
            + " — INFO only: a wall-map is a result and a quiet atlas is a question for "
              "The Scientist, never a failure here")


# -------------------------------------------------------------------- expeditions

def check_ledger_schema(root: Path, gates: dict, rep: Report) -> None:
    sec = "EXPEDITION LEDGER — append-only, harvest mandatory"
    path = ATLAS_DIR / "EXPEDITION_LEDGER.md"
    text = read(path)
    if not text:
        rep.add(sec, "EXPEDITION_LEDGER.md present", False, f"absent or empty at {path}")
        return
    rows, problems = parse_table(text, LEDGER_COLS)
    problems = [p for p in problems if "has a header but no rows" not in p]
    rep.add(sec, "ledger parses against the row schema", not problems,
            "; ".join(problems) if problems else f"{len(rows)} probe row(s)")
    if not rows:
        rep.add(sec, "harvest column is filled on every row", None, "no probe rows yet")
        return

    verdicts = {str(v).upper() for v in gates.get("atlas_verdicts", [])} | {VOID_PLATE, PENDING}
    channels = {str(v).lower() for v in gates.get("atlas_channels", [])}
    bad, no_harvest = [], []
    for r in rows:
        tag = r["probe_id"]
        if r["verdict"].upper() not in verdicts:
            bad.append(f"{tag} verdict={r['verdict']!r}")
        if r["channel"].lower() not in channels:
            bad.append(f"{tag} channel={r['channel']!r}")
        if r["tier"].upper() not in TIERS:
            bad.append(f"{tag} tier={r['tier']!r}")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", r["date"]):
            bad.append(f"{tag} date={r['date']!r}")
        # `harvest` may be `—`; it may not be missing. Deviation capture is
        # mandatory and an empty cell is how it stops happening.
        if not r["harvest"].strip():
            no_harvest.append(tag)
    rep.add(sec, "ledger values are canonical", not bad,
            "clean" if not bad else f"{len(bad)}: " + "; ".join(bad[:10]))
    rep.add(sec, "harvest column is filled on every row", not no_harvest,
            f"{len(rows)} row(s); {len(no_harvest)} with an empty harvest cell"
            + (": " + ", ".join(no_harvest[:8]) if no_harvest else ""))

    dupes = sorted({r["probe_id"] for r in rows
                    if [x["probe_id"] for x in rows].count(r["probe_id"]) > 1})
    rep.add(sec, "probe ids are unique (the ledger is never rewritten)", not dupes,
            "clean" if not dupes else f"{len(dupes)} duplicated: {', '.join(dupes[:8])}")

    # ⛔ ONE RENDER ANSWERS ONE ASK. The `rendered` column holds the clip id, and
    # the clip id is the only durable key in this system. Two probes carrying the
    # same clip id means one audio file was counted twice — the downstream shape
    # of the matcher's double-binding bug, and the shape n-inflation takes when
    # it is honest-looking: every row has evidence, and the evidence is the same
    # file. Caught here as well as at ingest and at bank, because this is the
    # layer a hand-edited ledger reaches.
    by_clip: dict[str, list[str]] = {}
    for r in rows:
        cid = r["rendered"].strip().strip("`")
        if not cid or cid.lower() in EMPTY_CELL:
            continue
        by_clip.setdefault(cid, []).append(f"{r['expedition']}/{r['probe_id']}")
    collisions = {cid: probes for cid, probes in by_clip.items() if len(probes) > 1}
    rep.add(sec, "one clip id answers at most one probe", not collisions,
            f"{len(by_clip)} distinct clip id(s) across {len(rows)} row(s)" if not collisions
            else f"{len(collisions)} clip(s) bound to several probes: "
                 + "; ".join(f"{cid} -> {', '.join(p)}" for cid, p in
                             list(collisions.items())[:5])
                 + " — one render counted twice is n inflated with real evidence paths")


def check_tier_rederivation(root: Path, gates: dict, rep: Report) -> None:
    """⛔ THE TIER IS STAMPED BY THE TOOL — AND RE-DERIVED HERE, FROM THE SAME FUNCTION.

    SKILL.md promises it in as many words: "derived from artifacts on disk by
    `skadi_bank.py` **and re-derived by the validator**". Without this the ledger
    is the only witness to its own tier, and a hand-written `HEARD` on a row with
    no blind artifact, no wav and no RENDERS.json passes every other check in
    this file — which is exactly what the tester demonstrated.

    ONE SOURCE OF DERIVATION, TWO READERS. The function is imported from
    `skadi_bank`, never re-implemented: two implementations of a ladder drift,
    and the drift is invisible precisely because both sides "check" it. The
    import is lazy because `skadi_bank` imports this module at its own top.

    This check only ever LOWERS: a ledger tier BELOW the derived one is fine (a
    row banked before the listening pass landed is honest, not inflated). A row
    whose expedition directory does not exist is SKIPPED and named — pre-Skaði
    seed rows cite render-audits that were never expeditions.
    """
    sec = "TIERS — re-derived from the artifacts, never read off the row"
    rows, problems = parse_table(read(ATLAS_DIR / "EXPEDITION_LEDGER.md"), LEDGER_COLS)
    if problems and not rows:
        rep.add(sec, "every ledger tier is re-derived from artifacts", None,
                "ledger unreadable or empty; check_ledger_schema owns that failure")
        return

    try:
        import skadi_bank  # noqa: PLC0415 - lazy: skadi_bank imports this module
    except Exception as err:  # noqa: BLE001
        rep.add(sec, "every ledger tier is re-derived from artifacts", False,
                f"could not import the derivation function from skadi_bank: "
                f"{type(err).__name__}: {err} — the tier would be taken on the ledger's word")
        return

    prefix = str(gates.get("atlas_expedition_prefix", "output/expeditions/")).strip("/")
    inflated: list[str] = []
    skipped: list[str] = []
    checked = 0
    cache: dict[str, dict] = {}
    for r in rows:
        exp_name = r["expedition"].strip().strip("`")
        # An expedition NAME is a directory name, never a path: a ledger row is
        # data, and `../..` in it must not send a reader walking the disk.
        if not exp_name or "/" in exp_name or "\\" in exp_name or exp_name.startswith("."):
            skipped.append(f"{exp_name!r}/{r['probe_id']} (not a plain directory name)")
            continue
        exp = REPO / prefix / exp_name
        if not exp.is_dir():
            skipped.append(f"{exp_name}/{r['probe_id']}")
            continue
        if exp_name not in cache:
            path = exp / "RENDERS.json"
            try:
                loaded = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            except (OSError, ValueError):
                loaded = {}
            cache[exp_name] = loaded if isinstance(loaded, dict) else {}
        derived, _, _ = skadi_bank.stamp_tier(exp, r["probe_id"], cache[exp_name], repo=REPO)
        checked += 1
        claimed = r["tier"].strip().upper()
        if claimed not in TIERS:
            continue  # check_ledger_schema owns the vocabulary failure
        rung = TIERS.index(claimed)
        floor = TIERS.index(derived) if derived in TIERS else -1
        if rung > floor:
            inflated.append(f"{exp_name}/{r['probe_id']} claims {claimed}, artifacts support "
                            f"{derived or 'nothing — not even a package on disk'}")
    rep.add(sec, "every ledger tier is re-derived from artifacts", not inflated,
            f"{checked} row(s) re-derived"
            + (f", {len(skipped)} skipped (no expedition directory: "
               f"{', '.join(skipped[:4])})" if skipped else "")
            + ("" if not inflated else f"; {len(inflated)} INFLATED: " + "; ".join(inflated[:6])
               + " — a tier asserted by a row is not a tier"))


def _plan_rows(expedition: Path) -> tuple[list[dict], list[str]]:
    return parse_table(read(expedition / "PLAN.md"), PLAN_COLS)


def check_controls(root: Path, gates: dict, rep: Report) -> None:
    """A plate with a failed control proves nothing. VOID-PLATE must propagate."""
    sec = "CONTROLS — a plate whose control failed banks nothing"
    exps = expedition_dirs(gates)
    if not exps:
        rep.add(sec, "expeditions have declared controls", None,
                f"no expedition directories under {gates.get('atlas_expedition_prefix')}")
        return
    min_pos = int(gates.get("atlas_min_positive_controls", 1))
    min_neg = int(gates.get("atlas_min_negative_controls", 1))
    ledger_rows, _ = parse_table(read(ATLAS_DIR / "EXPEDITION_LEDGER.md"), LEDGER_COLS)

    short, unverdicted, unvoided = [], [], []
    for exp in exps:
        rows, problems = _plan_rows(exp)
        if problems:
            short.append(f"{exp.name}: PLAN.md unreadable ({problems[0]})")
            continue
        pos = [r for r in rows if r["kind"].upper() == "POSITIVE-CONTROL"]
        neg = [r for r in rows if r["kind"].upper() == "NEGATIVE-CONTROL"]
        if len(pos) < min_pos or len(neg) < min_neg:
            short.append(f"{exp.name}: {len(pos)} positive / {len(neg)} negative control(s)")
        if not is_closed(exp):
            continue
        mine = {r["probe_id"]: r for r in ledger_rows if r["expedition"] == exp.name}
        failed_control = False
        for c in pos + neg:
            row = mine.get(c["probe_id"])
            if row is None or row["verdict"].upper() in ("", PENDING):
                unverdicted.append(f"{exp.name}/{c['probe_id']}")
                continue
            if row["verdict"].upper() == VOID_PLATE:
                failed_control = True
                continue
            if c["expected"].upper() not in EMPTY_CELL and \
                    row["verdict"].upper() != c["expected"].upper():
                failed_control = True
        if failed_control:
            not_void = [pid for pid, r in mine.items() if r["verdict"].upper() != VOID_PLATE]
            if not_void:
                unvoided.append(f"{exp.name}: {len(not_void)} row(s) still carry a verdict "
                                f"({', '.join(sorted(not_void)[:5])})")

    rep.add(sec, "every expedition declares its controls "
                 "(gates atlas_min_positive_controls / atlas_min_negative_controls)",
            not short, f"{len(exps)} expedition(s)"
            + ("; " + "; ".join(short[:6]) if short else "; all plates carry both controls"))
    rep.add(sec, "closed plates record a verdict for every control", not unverdicted,
            "clean" if not unverdicted else
            f"{len(unverdicted)} control(s) with no verdict: " + ", ".join(unverdicted[:8]))
    rep.add(sec, "a failed control voids the whole plate", not unvoided,
            "clean" if not unvoided else "; ".join(unvoided[:5]))


# ⛔ A REGISTRATION TIMESTAMP IS A TIMESTAMP. `registered_utc` was free text, so
# "after we heard it" registered a prediction — the one field whose entire job is
# to prove the order of two events accepted a sentence that admits the order was
# wrong. Strict ISO-8601 UTC only: date, `T`, time, `Z`.
ISO_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d{1,6})?)?Z$")

# The literal `skadi_plan.py` writes into the template. It is deliberately NOT a
# valid timestamp: an unfilled template must be refused by the ISO check above,
# not quietly accepted because the planner helpfully stamped a clock.
REGISTER_BEFORE_RENDER = "REGISTER-BEFORE-RENDER"


def parse_iso_utc(value: str) -> datetime | None:
    """-> aware datetime, or None when the cell is not strict ISO-8601 UTC."""
    raw = (value or "").strip().strip("`")
    if not ISO_UTC_RE.match(raw):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _renders_written_at(expedition: Path) -> dict[str, datetime]:
    """`{probe_id: when ingest bound its clip}` from RENDERS.json, where recorded."""
    path = expedition / "RENDERS.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, datetime] = {}
    for pid, entry in data.items():
        if not isinstance(entry, dict):
            continue
        when = parse_iso_utc(str(entry.get("written_at_utc") or ""))
        if when is not None:
            out[pid] = when
    return out


def registration_problems(expedition: Path) -> tuple[list[str], list[str]]:
    """-> (cells that are not ISO-8601 UTC, predictions registered after the render)."""
    preds, problems = parse_table(read(expedition / "PREDICTIONS.md"), PREDICTION_COLS)
    if problems and not preds:
        return [], []
    written = _renders_written_at(expedition)
    unparseable: list[str] = []
    late: list[str] = []
    for r in preds:
        pid = r["probe_id"]
        when = parse_iso_utc(r["registered_utc"])
        if when is None:
            unparseable.append(f"{pid} registered_utc={r['registered_utc'].strip()!r}")
            continue
        bound = written.get(pid)
        if bound is not None and when > bound:
            late.append(f"{pid} registered {r['registered_utc'].strip()} but its clip was "
                        f"bound at {bound.isoformat().replace('+00:00', 'Z')}")
    return unparseable, late


def check_predictions(root: Path, gates: dict, rep: Report) -> None:
    """A prediction registered after the render is not a prediction."""
    sec = "PREDICTIONS — registered before the render, scored after"
    exps = expedition_dirs(gates)
    if not exps:
        rep.add(sec, "every probe has a registered prediction", None,
                "no expedition directories")
        return
    missing, unscored, malformed, late = [], [], [], []
    for exp in exps:
        plan, plan_problems = _plan_rows(exp)
        if plan_problems:
            malformed.append(f"{exp.name}: PLAN.md {plan_problems[0]}")
            continue
        preds, pred_problems = parse_table(read(exp / "PREDICTIONS.md"), PREDICTION_COLS)
        if pred_problems:
            malformed.append(f"{exp.name}: PREDICTIONS.md {pred_problems[0]}")
            continue
        registered = {r["probe_id"] for r in preds if r["prediction"].strip().lower() not in EMPTY_CELL}
        for p in plan:
            if p["probe_id"] not in registered:
                missing.append(f"{exp.name}/{p['probe_id']}")
        if is_closed(exp) and not (exp / "PREDICTION_SCORE.md").exists():
            unscored.append(exp.name)
        # The order is the whole value of the protocol, and it is checkable:
        # ingest stamps `written_at_utc` when it binds a clip, so a registration
        # dated after that stamp is a prediction written with the answer in hand.
        # (Preflight owns the "is it even a timestamp" refusal.)
        late.extend(f"{exp.name}/{p}" for p in registration_problems(exp)[1])
    rep.add(sec, "PLAN and PREDICTIONS parse", not malformed,
            "clean" if not malformed else "; ".join(malformed[:6]))
    rep.add(sec, "every probe has a registered prediction", not missing,
            "clean" if not missing else
            f"{len(missing)} probe(s) with no prediction: " + ", ".join(missing[:8]))
    rep.add(sec, "closed expeditions carry PREDICTION_SCORE.md", not unscored,
            "clean" if not unscored else
            f"{len(unscored)} closed without a score: " + ", ".join(unscored[:6]))
    rep.add(sec, "no prediction is registered after its render", not late,
            "clean" if not late else f"{len(late)}: " + "; ".join(late[:6])
            + " — a prediction registered after the render is not a prediction")


# --------------------------------------------------------------------- preflight

def preflight(expedition: Path, gates: dict, rep: Report) -> None:
    """Phase-0 gate. Refuses the plate BEFORE any render is priced or asked for."""
    sec = "PREFLIGHT — the plate may not open until this passes"
    try:
        label = str(expedition.relative_to(REPO))
    except ValueError:
        label = str(expedition)
    rep.add(sec, "expedition directory", True, label)

    plan_path = expedition / "PLAN.md"
    if not plan_path.exists():
        rep.add(sec, "PLAN.md present", False, "absent — run scripts/skadi_plan.py first")
        return
    rows, problems = _plan_rows(expedition)
    rep.add(sec, "PLAN.md parses against the slot schema", not problems,
            "; ".join(problems) if problems else f"{len(rows)} slot(s)")
    if not rows:
        return

    # A DOWN-SCALED SLICE IS LEGAL. SKILL.md Phase 1: "A down-scaled slice (2
    # pairs × 2 variations, the lofn-daily test-slice precedent) is legal for
    # probe waves." A strict `== atlas_probe_slots` refused the plate doctrine
    # explicitly permits, and a gate that refuses legal work gets bypassed rather
    # than obeyed. The floor is the two CONTROLS: the smallest legal plate is a
    # calibration plate that measures the instrument and claims nothing.
    slots = int(gates.get("atlas_probe_slots", 6))
    floor = int(gates.get("atlas_probe_slots_min", 2))
    rep.add(sec, "slot count within gates atlas_probe_slots_min..atlas_probe_slots",
            floor <= len(rows) <= slots,
            f"{len(rows)} slot(s) planned; gates allow {floor}–{slots}"
            + ("" if floor <= len(rows) <= slots else
               " — below the floor there is no control pair; above the ceiling the plate is "
               "not the plate the gates priced"))

    bad_kind = [f"{r['probe_id']}={r['kind']!r}" for r in rows
                if r["kind"].upper() not in PROBE_KINDS]
    rep.add(sec, "slot kinds are canonical", not bad_kind,
            "clean" if not bad_kind else "; ".join(bad_kind[:8]))

    min_pos = int(gates.get("atlas_min_positive_controls", 1))
    min_neg = int(gates.get("atlas_min_negative_controls", 1))
    quota = int(gates.get("atlas_non_obvious_quota", 1))
    pos = [r for r in rows if r["kind"].upper() == "POSITIVE-CONTROL"]
    neg = [r for r in rows if r["kind"].upper() == "NEGATIVE-CONTROL"]
    non = [r for r in rows if r["kind"].upper() == "NON-OBVIOUS"]
    rep.add(sec, "controls assigned (gates atlas_min_positive_controls / "
                 "atlas_min_negative_controls)",
            len(pos) >= min_pos and len(neg) >= min_neg,
            f"{len(pos)} positive · {len(neg)} negative")
    no_expected = [r["probe_id"] for r in pos + neg if r["expected"].upper() in EMPTY_CELL]
    rep.add(sec, "every control declares the verdict it expects", not no_expected,
            "clean" if not no_expected else
            f"{len(no_expected)} without an expected verdict: {', '.join(no_expected)}"
            + " — a control with no expectation cannot fail, and a gate that cannot fail "
              "is a sentence")
    # THE NON-OBVIOUSNESS QUOTA IS A QUOTA ON A FULL PLATE. On a calibration
    # slice (controls plus a probe or two) there is no room for a survivor nobody
    # can locally justify, and demanding one there would either refuse the slice
    # or turn the non-obvious slot into a formality — which is worse, because the
    # rule's whole purpose is that the slot is real. Below
    # `atlas_non_obvious_min_slots` it is a FLAG that says the quota is waived
    # and why; at or above it, it is a refusal.
    small = len(rows) < int(gates.get("atlas_non_obvious_min_slots", 4))
    rep.add(sec, "non-obvious slot present (gates atlas_non_obvious_quota)",
            None if (small and len(non) < quota) else len(non) >= quota,
            f"{len(non)} non-obvious slot(s)"
            + (f" — quota WAIVED on a {len(rows)}-slot calibration slice (gates "
               "atlas_non_obvious_min_slots); the slice claims nothing and is not a search"
               if small and len(non) < quota else ""))

    # ⛔ THE ONE THE MACHINE MAY NOT FILL. skadi_plan.py writes the literal
    # placeholder and stops; a mind replaces it. If the placeholder survives to
    # preflight, no mind was here.
    unfilled = [r["probe_id"] for r in non
                if NON_OBVIOUS_PLACEHOLDER in (r["hypothesis"] + r["cell"])
                or r["hypothesis"].strip().lower() in EMPTY_CELL]
    rep.add(sec, "non-obvious slot was filled by a mind, not by the planner", not unfilled,
            "clean" if not unfilled else
            f"{len(unfilled)} slot(s) still carry {NON_OBVIOUS_PLACEHOLDER!r}: "
            + ", ".join(unfilled)
            + " — the planner does arithmetic; without this the search collapses onto the prior")

    preds, pred_problems = parse_table(read(expedition / "PREDICTIONS.md"), PREDICTION_COLS)
    rep.add(sec, "PREDICTIONS.md present and parses", not pred_problems,
            "; ".join(pred_problems) if pred_problems else f"{len(preds)} registered prediction(s)")
    registered = {r["probe_id"] for r in preds
                  if r["prediction"].strip().lower() not in EMPTY_CELL
                  and r["registered_utc"].strip().lower() not in EMPTY_CELL}
    missing = [r["probe_id"] for r in rows if r["probe_id"] not in registered]
    rep.add(sec, "every slot has a prediction registered before the render", not missing,
            "clean" if not missing else
            f"{len(missing)} unregistered: " + ", ".join(missing[:8]))

    # ⛔ THE TIMESTAMP IS THE PROOF OF ORDER, SO IT MUST BE A TIMESTAMP. The
    # planner writes REGISTER-BEFORE-RENDER here precisely so an unfilled
    # template cannot pass: a placeholder that parsed as a time would be a gate
    # that stamps its own alibi.
    unparseable, late = registration_problems(expedition)
    rep.add(sec, "registered_utc is strict ISO-8601 UTC (2026-08-17T21:04:00Z)",
            not unparseable,
            "clean" if not unparseable else f"{len(unparseable)}: "
            + "; ".join(unparseable[:8])
            + f" — an unfilled template carries {REGISTER_BEFORE_RENDER!r}, which is refused "
              "here by design")
    rep.add(sec, "no prediction is registered after its render", not late,
            "clean — nothing is bound yet" if not late else
            f"{len(late)}: " + "; ".join(late[:6]))

    # The frontier read receipt: consumption is wired, and the wiring is visible.
    text = read(plan_path)
    receipt = re.search(r"^\s*frontier_read:\s*`?([^`\s]+)`?\s*$", text, re.MULTILINE)
    ok = bool(receipt) and _resolves(receipt.group(1))
    rep.add(sec, "FRONTIER read receipt present in PLAN.md", ok,
            f"frontier_read: {receipt.group(1) if receipt else '(absent)'}"
            + ("" if ok else " — Phase 0 reads the frontier; the receipt is how we know it did"))


# --------------------------------------------------------------------- doctrine

DOCTRINE_FILES = [
    ".claude/skills/skadi/SKILL.md",
    ".claude/skills/skadi/EXECUTION.md",
    ".agents/skills/skadi/SKILL.md",
    ".agents/skills/skadi/EXECUTION.md",
    "vault/music_atlas/ATLAS.md",
    "vault/music_atlas/FRONTIER.md",
    "vault/music_atlas/TERRITORIES.md",
    "vault/music_atlas/SPEC_CLASSES.md",
    "vault/music_atlas/seeds/SEED_NOTES.md",
]

# Prose claims that must agree with the YAML. This is the drift class a careful
# read does not catch.
DOCTRINE_PATTERNS = [
    ("atlas_probe_slots", r"\b(\d+)\s+probe[\s-]slots?\b"),
    ("atlas_probe_slots", r"\bplates?\s+of\s+(\d+)\s+probes?\b"),
    ("atlas_confidence_cap_n", r"\bn\s*<\s*(\d+)\b"),
    ("atlas_bank_min_n", r"\bBANKED\s+requires\s+n\s*[>≥]=?\s*(\d+)\b"),
    ("atlas_bank_min_n", r"\bn\s*[≥>]=?\s*(\d+)\s*,?\s*controls\s+valid\b"),
    ("atlas_streak_reauthorize", r"\b(\d+)\s+consecutive\b"),
    ("atlas_scruff_rift_sightings", r"\bin\s*[≥>]=?\s*(\d+)\s+probes?\b"),
]
EMPHASIS = re.compile(r"[*`_]+")


def check_doctrine(gates: dict, rep: Report) -> None:
    sec = "DOCTRINE — prose vs vault/gates.yaml"
    present = [rel for rel in DOCTRINE_FILES if (REPO / rel).exists()]
    if not present:
        rep.add(sec, "no prose/YAML drift", None,
                "no Skaði doctrine or state files on disk yet")
    else:
        drift: list[str] = []
        for rel in present:
            for lineno, raw in enumerate(read(REPO / rel).splitlines(), 1):
                # Historical narrative about a past run is not a spec claim.
                if re.search(r"\b(on|in)\s+`?20\d\d-", raw):
                    continue
                line = EMPHASIS.sub("", raw)
                for key, pat in DOCTRINE_PATTERNS:
                    m = re.search(pat, line, re.IGNORECASE)
                    if not m:
                        continue
                    try:
                        claimed = int([g for g in m.groups() if g][-1])
                    except (ValueError, IndexError):
                        continue
                    expected = gates.get(key)
                    if isinstance(expected, int) and claimed != expected:
                        drift.append(f"{rel}:{lineno} says {claimed}, {key}={expected} "
                                     f"— {raw.strip()[:80]}")
        rep.add(sec, "no prose/YAML drift", not drift,
                f"{len(present)} file(s) scanned"
                + ("" if not drift else f"; {len(drift)} contradiction(s):\n      "
                                        + "\n      ".join(drift[:10])))

        # A vocabulary is a gate too. A state file that invents a verdict word is
        # the same drift class as a wrong number, and it is the one an app will
        # render as a blank cell.
        # Scanned per CELL, and only cells that are a single ALL-CAPS token — those
        # are exactly the vocabulary cells. Scanning whole rows instead flags
        # `RENDER_AUDIT.md` inside an evidence path, which trains everyone to
        # ignore the detector.
        vocab_keys = ("atlas_verdicts", "atlas_channels", "atlas_lanes", "atlas_delta_moves")
        listed = {str(v).upper() for k in vocab_keys for v in gates.get(k, [])}
        listed |= {VOID_PLATE, PENDING} | set(CLAIM_KINDS) | set(CONFIDENCES) | set(TIERS)
        listed |= set(PROBE_KINDS) | set(SCRUFF_STATES) | set(TERRITORY_STATES)
        listed |= set(MENU_STATES) | set(ATLAS_TIERS) | set(APPEND_ACTIONS)
        listed |= {"GENERATED", "INFO", "MPB", "IDM", "EDM", "EBM"}
        stray: list[str] = []
        for rel in present:
            for lineno, raw in enumerate(read(REPO / rel).splitlines(), 1):
                if not raw.lstrip().startswith("|"):
                    continue
                for cell in table_cells(raw):
                    tok = EMPHASIS.sub("", cell).strip()
                    # Identifiers (SK-001, SKOP-01, SCR-04) are names, not vocabulary.
                    if re.fullmatch(r"[A-Z]{2,6}-\d{1,4}", tok):
                        continue
                    if re.fullmatch(r"[A-Z][A-Z0-9-]{3,}", tok) and tok not in listed:
                        stray.append(f"{rel}:{lineno} {tok}")
        rep.add(sec, "state tables invent no vocabulary", not stray,
                "clean" if not stray else
                f"{len(stray)} token(s) not in gates.yaml or the contract: "
                + "; ".join(stray[:10]))

    # A READER OPENS THE FILE. A docstring that merely names gates.yaml is not a
    # reader — that distinction previously caused a whole class of silent violations.
    readers = []
    for p in sorted((REPO / "scripts").glob("*.py")):
        text = read(p)
        if "gates.yaml" not in text and "validate_music_atlas" not in text:
            continue
        opens = any("gates.yaml" in ln and re.search(r"\bopen\(|read_text|read_bytes|Path\(|/ *[\"']gates", ln)
                    for ln in text.splitlines() if not ln.lstrip().startswith("#"))
        imports = re.search(r"^\s*from\s+validate_music_atlas\s+import\b", text, re.MULTILINE)
        if p.name == "validate_music_atlas.py" or opens or imports:
            readers.append(p.name)
    rep.add(sec, "the MUSIC ATLAS gates have machine readers", bool(readers),
            f"{len(readers)} script(s) load vault/gates.yaml: {', '.join(readers) or 'NONE'}")


# ------------------------------------------------------------------------- main

CHECKS = [
    check_atlas_schema,
    check_append_blocks,
    check_confidence_vs_n,
    check_banked_min_n,
    check_conjectured_shelf,
    check_archive_provenance,
    check_lanes,
    check_territory_states,
    check_spec_classes,
    check_coverage,
    check_scruff,
    check_control_chart,
    check_delta_log,
    check_ledger_schema,
    check_tier_rederivation,
    check_controls,
    check_predictions,
    check_consumption,
]


def run_all(gates: dict, rep: Report) -> None:
    if not ATLAS_DIR.is_dir():
        rep.add("ATLAS", "state home present", False,
                f"{ATLAS_DIR} absent — run scripts/skadi_seed.py")
        return
    for fn in CHECKS:
        fn(ATLAS_DIR, gates, rep)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Machine reader for the SKAÐI music atlas and the MUSIC ATLAS "
                    "section of vault/gates.yaml.")
    ap.add_argument("--preflight", metavar="EXPEDITION_DIR",
                    help="Phase-0 gate: refuse a plate whose PLAN/PREDICTIONS are "
                         "incomplete; exit 3 refuses BEFORE anything is rendered")
    ap.add_argument("--doctrine", action="store_true",
                    help="check prose vs gates.yaml across the Skaði skill and state files")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = ap.parse_args(argv)

    gates, warnings = load_gates()
    rep = Report()
    for w in warnings:
        rep.add("GATES", "load", None, w)

    if args.preflight:
        exp = Path(args.preflight)
        if not exp.is_absolute():
            exp = REPO / exp
        if not exp.is_dir():
            print(f"error: no such expedition directory: {exp}", file=sys.stderr)
            return 2
        preflight(exp, gates, rep)
        print(json.dumps({"failed": rep.failed, "rows": rep.rows}, indent=2)
              if args.json else rep.render())
        return 3 if rep.failed else 0

    if args.doctrine:
        check_doctrine(gates, rep)
    else:
        run_all(gates, rep)

    print(json.dumps({"failed": rep.failed, "rows": rep.rows}, indent=2)
          if args.json else rep.render())
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
