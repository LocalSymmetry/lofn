#!/usr/bin/env python3
"""skadi_bank.py — close a plate: stamp tiers, score predictions, bank claims.

⛔ THE TIER IS STAMPED BY THE TOOL, NEVER BY THE AGENT. Every probe's tier here
is a pure function of WHICH FILES EXIST:

    SPECIFIED  the package parses against the canonical Suno heading contract
    RENDERED   RENDERS.json binds a clip id AND the audio is on disk
    MEASURED   RENDERS.json carries a measurements object for it
    HEARD      listen/<probe>.blind.json AND listen/<probe>.intent.json exist

Nothing in this file can raise a tier by believing something. "I checked it, it
is in my context" is the sound a gate makes while it fails.

⛔ A PLATE WHOSE CONTROL FAILED BANKS NOTHING. If any declared control returns a
verdict other than the one PLAN.md registered for it, the plate is VOID: every
ledger row records `VOID-PLATE`, no claim is proposed, and the DELTA move is
INSTRUMENT-ONLY. The control chart still receives the control's ACTUAL verdict —
an out-of-limits point is the most informative thing a plate can produce.

⛔ IT NEVER WRITES HIGH. Confidence is capped by `n` (gates
`atlas_confidence_cap_n`) and the banker's ceiling is MEDIUM regardless: only The
Scientist promotes a claim to doctrine.

It writes: EXPEDITION_LEDGER rows · a dated ATLAS append block (+ tier tables) ·
FRONTIER coverage/scruff/control-chart updates · `PREDICTION_SCORE.md` ·
`DELTA_DRAFT.md`. It touches `vault/COMPETITION_LEARNINGS.md` and
`vault/RUN_LEDGER.md` never.

⛔ IT NEVER WRITES A PLACEHOLDER. Every value on a banked row is supplied or
derived from an artifact; where one is missing the CLAIM IS REFUSED and the
missing value is named. The previous version wrote `(declare the model epoch)`
and `(name the observation that would refute this)` into live rows, and a
placeholder is not empty — so `models non-empty` and `falsifier non-empty` both
passed while the claim was scoped to nothing and refutable by nothing. A gate
that a placeholder satisfies is a gate that cannot fail.

  * `models` comes from the clip's `major_model_version`, recorded per probe by
    `skadi_ingest.py` in RENDERS.json. No epoch on the render → no claim.
  * `falsifier` comes from VERDICTS.md, per claim, from the mind that judged it.

SPEC-GAP: DESIGN v1 §4 says the banker derives TIERS from artifacts but does not
name the artifact that carries a probe's VERDICT and its harvest — and a verdict
is a judgement, so the banker must not invent one. This script reads
`<expedition-dir>/VERDICTS.md`:

    | probe_id | verdict | claim | harvest | falsifier | scruff_id | note |

`claim` may be `—` (a mechanical sentence is then composed from the plan cell),
`harvest` is mandatory and may be `—`, `falsifier` is mandatory on any row that
will propose a claim (it is the observation that would refute it, and a claim
without one is not a claim), and `scruff_id` attaches a harvest to an existing
shelf row so sightings ACCUMULATE toward the rift threshold instead of minting a
fresh id every time.

USAGE
    python3 scripts/skadi_bank.py output/expeditions/<dir> [--dry-run]

EXIT   0 banked · 1 refused (missing artifacts / unbankable claims) · 2 input error
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_music_atlas import (  # noqa: E402
    ATLAS_COLS, APPEND_COLS, ATLAS_DIR, CONTROL_CHART_COLS, COVERAGE_COLS, DELTA_COLS,
    EMPTY_CELL, LEDGER_COLS, PLAN_COLS, PREDICTION_COLS, REPO, SCRUFF_COLS,
    SPEC_CLASS_COLS, VOID_PLATE,
    load_gates, parse_table, read, sha12, strike, table_span,
)

VERDICT_COLS = ["probe_id", "verdict", "claim", "harvest", "falsifier", "scruff_id", "note"]
PACKAGE_HEADINGS = [r"^## 1\. MUSIC PROMPT\s*$", r"^## 2\. LYRICS\s*$", r"^## 3\. TITLE\s*$"]


# ---------------------------------------------------------------- markdown writing

def row_md(cells: list) -> str:
    return "| " + " | ".join(str(c).replace("|", "\\|") for c in cells) + " |"


def table_md(cols: list[str], rows: list[list]) -> str:
    return [row_md(cols), row_md(["---"] * len(cols))] + [row_md(r) for r in rows]


def splice(text: str, cols: list[str], rows: list[list], section: str | None = None) -> str:
    """Rebuild one table in place. A missing header is an ERROR, never an append."""
    span = table_span(text, cols, section)
    if span is None:
        raise SystemExit(f"ERROR: no table with header {cols} "
                         + (f"in section {section!r} " if section else "")
                         + "— refusing to append a table somewhere convenient")
    start, end = span
    lines = text.splitlines()
    out = lines[:start] + table_md(cols, rows) + lines[end:]
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def cells_of(row: dict, cols: list[str]) -> list[str]:
    return [row[c] for c in cols]


# ------------------------------------------------------------------- tier stamping

def package_of(expedition: Path, pid: str) -> Path | None:
    for p in sorted(expedition.rglob("*.md")):
        if re.match(rf"^{re.escape(pid)}(?![0-9A-Za-z])", p.name) \
                and re.search(PACKAGE_HEADINGS[0], read(p), re.M):
            return p
    return None


def stamp_tier(expedition: Path, pid: str, renders: dict,
               repo: Path | None = None) -> tuple[str, list[str], list[str]]:
    """-> (tier, evidence paths, notes). Artifact presence only, in ladder order.

    ⛔ ONE SOURCE OF DERIVATION, TWO READERS. `validate_music_atlas.py`
    (`check_tier_rederivation`) imports THIS function to re-derive every ledger
    row's tier, which is what SKILL.md means by "re-derived by the validator". It
    takes `repo` explicitly so the reader can point it at a fixture root without
    reaching into this module's globals — two copies of a ladder drift, and the
    drift is invisible because both sides believe they are checking it.
    """
    repo = repo or REPO
    notes: list[str] = []
    evidence: list[str] = []

    pkg = package_of(expedition, pid)
    if pkg is None:
        return "", [], [f"{pid}: no package on disk — not even SPECIFIED"]
    text = read(pkg)
    missing = [h for h in PACKAGE_HEADINGS if not re.search(h, text, re.M)]
    if missing:
        return "", [], [f"{pid}: {pkg.name} lacks {len(missing)} canonical heading(s) "
                        f"({', '.join(h.strip('^$').replace(chr(92) + 's*', '') for h in missing)})"
                        " — the contract is pinned by "
                        "skills/music/scripts/validate_suno_packages.py"]
    tier = "SPECIFIED"
    evidence.append(pkg.relative_to(repo).as_posix())

    entry = renders.get(pid) or {}
    wav_rel = entry.get("wav")
    wav = (repo / wav_rel) if wav_rel else None
    if entry.get("clip_id") and wav is not None and wav.exists():
        tier = "RENDERED"
        evidence.append((expedition / "RENDERS.json").relative_to(repo).as_posix())
    elif entry.get("clip_id"):
        notes.append(f"{pid}: bound to clip {entry['clip_id']} but no audio on disk — "
                     "stays SPECIFIED")

    if tier == "RENDERED" and isinstance(entry.get("measurements"), dict) \
            and entry["measurements"]:
        tier = "MEASURED"

    blind = expedition / "listen" / f"{pid}.blind.json"
    intent = expedition / "listen" / f"{pid}.intent.json"
    if tier in ("RENDERED", "MEASURED") and blind.exists() and intent.exists():
        tier = "HEARD"
        evidence.append(blind.relative_to(repo).as_posix())
        evidence.append(intent.relative_to(repo).as_posix())
    elif blind.exists() != intent.exists():
        notes.append(f"{pid}: only one of blind/intent exists — THE BLIND RULE needs both "
                     "turns, so this is not HEARD")
    return tier, evidence, notes


# ------------------------------------------------------------------------- loading

def load_plate(expedition: Path) -> dict:
    plan, probs = parse_table(read(expedition / "PLAN.md"), PLAN_COLS)
    if probs or not plan:
        raise SystemExit(f"ERROR: PLAN.md unreadable: {'; '.join(probs) or 'no slots'}")
    preds, probs = parse_table(read(expedition / "PREDICTIONS.md"), PREDICTION_COLS)
    if probs or not preds:
        raise SystemExit(f"ERROR: PREDICTIONS.md unreadable: {'; '.join(probs) or 'no rows'}")
    verdicts, probs = parse_table(read(expedition / "VERDICTS.md"), VERDICT_COLS)
    if probs or not verdicts:
        raise SystemExit(
            f"ERROR: {expedition.name}/VERDICTS.md unreadable: "
            f"{'; '.join(probs) or 'no rows'}\n"
            "  The banker stamps TIERS from artifacts but never invents a VERDICT. Write "
            "VERDICTS.md with columns: " + " | ".join(VERDICT_COLS))
    renders_path = expedition / "RENDERS.json"
    renders = json.loads(renders_path.read_text(encoding="utf-8")) if renders_path.exists() else {}

    date = re.match(r"(\d{4}-\d{2}-\d{2})", expedition.name)
    if not date:
        raise SystemExit(f"ERROR: expedition directory {expedition.name!r} carries no "
                         "YYYY-MM-DD prefix; the ledger date is read from it, never from a "
                         "clock, so the close is reproducible")
    return {"plan": plan, "preds": preds, "verdicts": verdicts, "renders": renders,
            "date": date.group(1), "name": expedition.name}


def spec_lanes() -> dict[str, str]:
    rows, probs = parse_table(read(ATLAS_DIR / "SPEC_CLASSES.md"), SPEC_CLASS_COLS)
    if probs or not rows:
        raise SystemExit(f"ERROR: SPEC_CLASSES.md unreadable: {'; '.join(probs) or 'no rows'}")
    return {r["class"]: r["lane"].upper() for r in rows}


def cell_parts(scope: str) -> tuple[str, str]:
    parts = [p.strip() for p in re.split(r"[×x]", scope) if p.strip()]
    return (parts[0], parts[-1]) if len(parts) >= 2 else ("", scope.strip())


# ------------------------------------------------------------------------ the close

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Close a SKAÐI plate and bank what it earned.")
    ap.add_argument("expedition")
    ap.add_argument("--dry-run", action="store_true", help="report; write nothing")
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass

    gates, warnings = load_gates()
    for w in warnings:
        print(f"[GATES] {w}", file=sys.stderr)
    cap = int(gates.get("atlas_confidence_cap_n", 4))
    bank_min = int(gates.get("atlas_bank_min_n", 2))
    verdict_vocab = {str(v).upper() for v in gates.get("atlas_verdicts", [])}

    expedition = Path(args.expedition)
    if not expedition.is_absolute():
        expedition = REPO / expedition
    if not expedition.is_dir():
        print(f"error: no such expedition directory: {expedition}", file=sys.stderr)
        return 2

    plate = load_plate(expedition)
    lanes = spec_lanes()
    by_id = {r["probe_id"]: r for r in plate["plan"]}
    verdicts = {r["probe_id"]: r for r in plate["verdicts"]}
    preds = {r["probe_id"]: r for r in plate["preds"]}

    print(f"[EXTRACT] {len(plate['plan'])} slot(s) · {len(verdicts)} verdict(s) · "
          f"{len(plate['renders'])} bound clip(s)", file=sys.stderr)

    bad_vocab = [f"{p}={v['verdict']!r}" for p, v in verdicts.items()
                 if v["verdict"].upper() not in verdict_vocab]
    if bad_vocab:
        print("ERROR: VERDICTS.md uses words that are not in gates atlas_verdicts: "
              + "; ".join(bad_vocab), file=sys.stderr)
        return 2
    no_harvest = [p for p, v in verdicts.items() if not v["harvest"].strip()]
    if no_harvest:
        print("ERROR: harvest is mandatory (it may be `—`; it may not be missing). Empty for: "
              + ", ".join(no_harvest), file=sys.stderr)
        return 2

    # ⛔ ONE CLIP ANSWERS ONE ASK — the second of three layers (the first is
    # `skadi_ingest.py`'s matcher, the third is `check_ledger_schema`). A plate
    # that binds one render to two probes would bank two observations of one
    # audio file, with every evidence path resolving. Checked WITHIN this plate
    # and ACROSS the ledger, because a re-run that re-binds an earlier plate's
    # clip is the same double count wearing a later date.
    holders: dict[str, list[str]] = {}
    for pid, entry in (plate["renders"] or {}).items():
        cid = (entry or {}).get("clip_id")
        if cid:
            holders.setdefault(str(cid), []).append(pid)
    collisions = {cid: pids for cid, pids in holders.items() if len(pids) > 1}
    if collisions:
        print("ERROR: RENDERS.json binds one clip to several probes — one render answers one "
              "ask:\n  " + "\n  ".join(f"{cid} -> {', '.join(sorted(p))}"
                                       for cid, p in sorted(collisions.items()))
              + "\n  Re-run scripts/skadi_ingest.py and resolve with --map.", file=sys.stderr)
        return 2
    prior_ledger, _ = parse_table(read(ATLAS_DIR / "EXPEDITION_LEDGER.md"), LEDGER_COLS)
    # Rows from THIS expedition are not a reuse — they are a re-run, which the
    # append-only refusal below the report diagnoses far more usefully.
    seen_before = {r["rendered"].strip().strip("`"): f"{r['expedition']}/{r['probe_id']}"
                   for r in prior_ledger
                   if r["expedition"].strip() != plate["name"]
                   and r["rendered"].strip().strip("`").lower() not in EMPTY_CELL}
    reused = [f"{pid} -> {cid} (already banked as {seen_before[cid]})"
              for cid, pids in sorted(holders.items()) for pid in pids if cid in seen_before]
    if reused:
        print("ERROR: these clips are already bound to a probe in EXPEDITION_LEDGER.md:\n  "
              + "\n  ".join(reused)
              + "\n  A render re-counted under a new probe id is n inflated with real "
                "evidence paths.", file=sys.stderr)
        return 2

    # ---- tiers, stamped from artifacts only
    tiers: dict[str, str] = {}
    evidence: dict[str, list[str]] = {}
    notes: list[str] = []
    for pid in by_id:
        tier, ev, why = stamp_tier(expedition, pid, plate["renders"])
        notes.extend(why)
        if tier:
            tiers[pid], evidence[pid] = tier, ev

    def epoch_of(pid: str) -> str:
        """The model epoch behind a probe, from the render it is bound to.

        ⛔ DERIVED, NEVER DECLARED BY THIS SCRIPT. `skadi_ingest.py` copies the
        clip's `major_model_version` into RENDERS.json at bind time; if it is not
        there, this returns "" and the caller REFUSES rather than writing a
        placeholder. A version string is a claim, not a fact — but an absent one
        is not a claim at all.
        """
        return str((plate["renders"].get(pid) or {}).get("model") or "").strip()

    # ---- control validity
    void_reasons: list[str] = []
    control_points: list[list[str]] = []
    for pid, slot in by_id.items():
        if slot["kind"].upper() not in ("POSITIVE-CONTROL", "NEGATIVE-CONTROL"):
            continue
        _, cls = cell_parts(slot["cell"])
        actual = verdicts.get(pid, {}).get("verdict", "").upper()
        expected = slot["expected"].upper()
        # An out-of-limits control point is the chart's whole purpose: record it
        # even — especially — when it voids the plate. But a chart point with no
        # epoch is unreadable as a series (the epoch rule-offs are what make the
        # points comparable), so a control with no render behind it produces a
        # NOTE, not a point with an invented epoch.
        epoch = epoch_of(pid)
        if epoch:
            control_points.append([cls, plate["name"], actual or VOID_PLATE,
                                   verdicts.get(pid, {}).get("note", "—") or "—", epoch])
        else:
            notes.append(f"{pid} ({cls}): no model epoch on its render (RENDERS.json carries "
                         "no `model` for it), so no control-chart point was written — a point "
                         "with no epoch cannot be placed against the rule-offs")
        if not actual:
            void_reasons.append(f"{pid} ({cls}) has no recorded verdict")
        elif expected not in EMPTY_CELL and actual != expected:
            void_reasons.append(f"{pid} ({cls}) returned {actual}, expected {expected}")
    plate_void = bool(void_reasons)

    # ---- ledger rows
    ledger_rows: list[list[str]] = []
    for pid, slot in by_id.items():
        v = verdicts.get(pid, {})
        entry = plate["renders"].get(pid, {})
        ledger_rows.append([
            plate["date"], plate["name"], pid, slot["cell"], slot["channel"],
            preds.get(pid, {}).get("prediction", "—") or "—",
            entry.get("clip_id", "—") or "—",
            VOID_PLATE if plate_void else (v.get("verdict", "—") or "—"),
            tiers.get(pid, "SPECIFIED"),
            v.get("harvest", "—") or "—",
        ])

    # ---- prediction score
    hits = 0
    score_rows: list[list[str]] = []
    for pid in by_id:
        predicted = (preds.get(pid, {}).get("prediction", "") or "").strip()
        actual = (verdicts.get(pid, {}).get("verdict", "") or "").strip()
        hit = ("—" if plate_void or predicted.lower() in EMPTY_CELL
               else ("HIT" if predicted.upper() == actual.upper() else "MISS"))
        hits += hit == "HIT"
        score_rows.append([pid, predicted or "—", actual or "—", hit])
    scored = sum(1 for r in score_rows if r[3] in ("HIT", "MISS"))

    # ---- claim proposals
    atlas_text = read(ATLAS_DIR / "ATLAS.md")
    tier_rows = {}
    for t in ("BANKED", "OBSERVED", "CONJECTURED"):
        rows, _ = parse_table(atlas_text, ATLAS_COLS, section=t)
        tier_rows[t] = rows
    all_rows = [r for rows in tier_rows.values() for r in rows]
    next_cell = max([int(m.group(1)) for r in all_rows
                     if (m := re.match(r"SK-(\d{3})$", strike(r["id"])[0]))] or [0]) + 1
    next_op = max([int(m.group(1)) for r in all_rows
                   if (m := re.match(r"SKOP-(\d{2})$", strike(r["id"])[0]))] or [0]) + 1

    proposals: list[dict] = []
    increments: list[tuple[dict, str]] = []
    refused: list[str] = []
    conflicts = 0
    if plate_void:
        refused.append("PLATE VOID — no claim banks from it: " + "; ".join(void_reasons))
    else:
        for pid, slot in by_id.items():
            if slot["kind"].upper() in ("POSITIVE-CONTROL", "NEGATIVE-CONTROL"):
                continue
            v = verdicts.get(pid)
            if not v:
                refused.append(f"{pid}: no verdict recorded")
                continue
            tier = tiers.get(pid, "")
            if tier not in ("MEASURED", "HEARD"):
                refused.append(f"{pid}: tier is {tier or 'unstamped'}; a claim needs MEASURED "
                               "or HEARD evidence")
                continue
            territory, cls = cell_parts(slot["cell"])
            lane = lanes.get(cls, "LISTENER")
            if lane in ("LISTENER", "BOTH") and tier != "HEARD":
                refused.append(f"{pid}: spec class `{cls}` is lane {lane}; a claim about "
                               "hearing must cite a hearing, and this probe was never heard")
                continue
            # ⛔ NO PLACEHOLDERS: a value that is missing REFUSES the claim and
            # says which value it was. The row is not written half-finished and
            # left for a later pass to "fill in", because nothing downstream can
            # tell a filled cell from a stood-in one.
            epoch = epoch_of(pid)
            if not epoch:
                refused.append(f"{pid}: no model epoch — RENDERS.json carries no `model` for "
                               "its clip, so the claim would be scoped to no version of the "
                               "generator. Re-run skadi_ingest.py against a snapshot that "
                               "carries `major_model_version`")
                continue
            falsifier = (v.get("falsifier") or "").strip()
            if falsifier.lower() in EMPTY_CELL:
                refused.append(f"{pid}: no falsifier — VERDICTS.md must name the observation "
                               "that would refute this claim; a claim nothing could refute is "
                               "not a claim")
                continue
            verdict = v["verdict"].upper()
            match = next((r for r in all_rows
                          if cell_parts(r["scope"])[1] == cls
                          and r["channel"].lower() == slot["channel"].lower()), None)
            ev = list(evidence.get(pid, [])) + [
                (expedition / "VERDICTS.md").relative_to(REPO).as_posix()]
            if match is not None and match["verdict"].upper() == verdict:
                increments.append((match, ", ".join(f"`{p}`" for p in ev), epoch))
                continue
            if match is not None:
                conflicts += 1
            kind = "OPERATOR" if cls in ("blend-of-two", "modifier-application") else "CELL"
            if kind == "OPERATOR":
                cid, next_op = f"SKOP-{next_op:02d}", next_op + 1
            else:
                cid, next_cell = f"SK-{next_cell:03d}", next_cell + 1
            claim = v["claim"].strip()
            if claim.lower() in EMPTY_CELL:
                claim = (f"In `{territory or 'any territory'}`, `{cls}` asked through the "
                         f"`{slot['channel']}` channel: {verdict}.")
            proposals.append({
                "id": cid, "kind": kind, "scope": slot["cell"], "channel": slot["channel"],
                "lane": lane, "claim": claim, "verdict": verdict, "n": 1,
                "confidence": "LOW", "models": epoch,
                "evidence": ", ".join(f"`{p}`" for p in ev),
                "falsifier": falsifier,
                "superseded": match["id"] if match is not None else None,
            })

    # ---- apply increments and placement
    receipts: list[list[str]] = []
    for match, ev, epoch in increments:
        n = int(match["n"]) + 1 if match["n"].isdigit() else 1
        match["n"] = str(n)
        # ⛔ The banker's ceiling is MEDIUM regardless of n. Only The Scientist
        # promotes a claim to doctrine.
        match["confidence"] = "LOW" if n == 1 else "MEDIUM"
        if match["evidence"] and ev not in match["evidence"]:
            match["evidence"] = match["evidence"] + ", " + ev
        # `models` = the epochs the evidence spans (DESIGN §2). An increment from
        # a later epoch WIDENS the scope; it never silently replaces it.
        spanned = [m.strip() for m in match["models"].split(",") if m.strip()]
        if epoch not in spanned:
            match["models"] = ", ".join(spanned + [epoch]) if spanned else epoch
        receipts.append([strike(match["id"])[0], f"`{sha12(match['claim'])}`", "N-INCREMENT",
                         f"n -> {n} from {plate['name']}"])
    for p in proposals:
        p["confidence"] = "LOW"
        receipts.append([p["id"], f"`{sha12(p['claim'])}`",
                         "SUPERSEDES" if p["superseded"] else "NEW",
                         (f"supersedes {p['superseded']} — same cell and channel, different "
                          f"verdict" if p["superseded"] else f"new from {plate['name']}")])

    # rebuild the three tier tables: a row that has grown past the floor moves up
    placed = {"BANKED": [], "OBSERVED": [], "CONJECTURED": []}
    for t in ("BANKED", "OBSERVED", "CONJECTURED"):
        for r in tier_rows[t]:
            n = int(r["n"]) if r["n"].isdigit() else 0
            dest = t
            if t == "OBSERVED" and n >= bank_min:
                dest = "BANKED"
            placed[dest].append(cells_of(r, ATLAS_COLS))
    for p in proposals:
        target = "BANKED" if p["n"] >= bank_min else "OBSERVED"
        placed[target].append([p["id"], p["kind"], p["scope"], p["channel"], p["lane"],
                               p["claim"], p["verdict"], str(p["n"]), p["confidence"],
                               p["models"], p["evidence"], p["falsifier"]])

    # ---- DELTA move, mechanically
    if plate_void:
        move = "INSTRUMENT-ONLY"
    elif conflicts:
        move = "SCOPE-NARROWED"
    elif any(p["kind"] == "OPERATOR" for p in proposals):
        move = "OPERATOR-LAW"
    elif proposals:
        move = "NEW-BEHAVIOR"
    elif increments:
        move = "BEHAVIOR-CONFIRMED"
    else:
        move = "NONE"

    # ---- report
    print(f"\nPLATE {plate['name']}  ({'VOID' if plate_void else 'valid'})")
    for pid in sorted(by_id):
        print(f"  {pid}  tier={tiers.get(pid, 'UNSTAMPED'):9s} "
              f"verdict={verdicts.get(pid, {}).get('verdict', '—')}")
    print(f"\n  {len(proposals)} new claim(s) · {len(increments)} n-increment(s) · "
          f"delta move {move}")
    print(f"  predictions: {hits} hit of {scored} scored")
    for n in notes:
        print(f"  note: {n}")
    for r in refused:
        print(f"  REFUSED: {r}")
    if args.dry_run:
        print("\n(dry run — nothing written)")
        return 0

    # ---- write EXPEDITION_LEDGER (append-only)
    led_path = ATLAS_DIR / "EXPEDITION_LEDGER.md"
    led_text = read(led_path)
    existing, _ = parse_table(led_text, LEDGER_COLS)
    if any(r["expedition"] == plate["name"] for r in existing):
        print(f"REFUSED: {led_path.name} already carries rows for {plate['name']} — the "
              "ledger is append-only and is never rewritten by a summarizer.", file=sys.stderr)
        return 1
    led_path.write_text(
        splice(led_text, LEDGER_COLS,
               [cells_of(r, LEDGER_COLS) for r in existing] + ledger_rows),
        encoding="utf-8", newline="\n")

    # ---- write ATLAS (tier tables + a dated append block)
    for t in ("BANKED", "OBSERVED", "CONJECTURED"):
        atlas_text = splice(atlas_text, ATLAS_COLS, placed[t], section=t)
    if receipts:
        block = ["", f"### {plate['date']} {plate['name']} additions", ""] \
            + table_md(APPEND_COLS, receipts) + [""]
        atlas_text = atlas_text.rstrip("\n") + "\n" + "\n".join(block) + "\n"
    (ATLAS_DIR / "ATLAS.md").write_text(atlas_text, encoding="utf-8", newline="\n")

    # ---- write FRONTIER (coverage · scruff · control chart)
    front = read(ATLAS_DIR / "FRONTIER.md")
    coverage, _ = parse_table(front, COVERAGE_COLS, section="COVERAGE")
    touched: dict[str, int] = {}
    for pid, slot in by_id.items():
        terr, _ = cell_parts(slot["cell"])
        if terr and terr.lower() != "any territory":
            touched[terr] = touched.get(terr, 0) + 1
    cov_rows = []
    for r in coverage:
        if r["territory"] in touched:
            r["state"] = "PROBED"
            r["probes"] = str((int(r["probes"]) if r["probes"].isdigit() else 0)
                              + touched[r["territory"]])
            r["last_expedition"] = plate["name"]
            if r["saturation_note"].strip() == "(none yet)":
                r["saturation_note"] = "—"
        cov_rows.append(cells_of(r, COVERAGE_COLS))
    front = splice(front, COVERAGE_COLS, cov_rows, section="COVERAGE")

    scruff, _ = parse_table(front, SCRUFF_COLS, section="SCRUFF SHELF")
    by_scruff = {r["scruff_id"]: r for r in scruff}
    next_scr = max([int(m.group(1)) for r in scruff
                    if (m := re.match(r"SCR-(\d+)$", r["scruff_id"]))] or [0]) + 1
    for pid in sorted(by_id):
        v = verdicts.get(pid, {})
        harvest = (v.get("harvest") or "").strip()
        if harvest.lower() in EMPTY_CELL:
            continue
        sid = (v.get("scruff_id") or "").strip()
        if sid in by_scruff:
            row = by_scruff[sid]
            if pid not in row["sightings"]:
                row["sightings"] = (row["sightings"] + ", " + pid
                                    if row["sightings"].strip() not in EMPTY_CELL else pid)
        else:
            sid = f"SCR-{next_scr:02d}"
            next_scr += 1
            by_scruff[sid] = {"scruff_id": sid, "first_seen": plate["date"],
                              "description": harvest, "sightings": pid, "status": "WATCH"}
            scruff.append(by_scruff[sid])
    front = splice(front, SCRUFF_COLS, [cells_of(r, SCRUFF_COLS) for r in scruff],
                   section="SCRUFF SHELF")

    chart, _ = parse_table(front, CONTROL_CHART_COLS, section="CONTROL CHART")
    front = splice(front, CONTROL_CHART_COLS,
                   [cells_of(r, CONTROL_CHART_COLS) for r in chart] + control_points,
                   section="CONTROL CHART")
    (ATLAS_DIR / "FRONTIER.md").write_text(front, encoding="utf-8", newline="\n")

    # ---- write PREDICTION_SCORE.md
    (expedition / "PREDICTION_SCORE.md").write_text("\n".join([
        "<!-- generated by scripts/skadi_bank.py; read by scripts/validate_music_atlas.py "
        "(check_predictions) -->",
        "",
        f"# PREDICTION SCORE — {plate['name']}",
        "",
        f"**{hits} hit of {scored} scored.** This is the falsifiable record of whether the rig "
        "earns itself — predictions registered before the render, scored against what the "
        "render actually did. A low score is information about the map, not a failure of the "
        "plate: a plate that only ever confirms what was already expected is not exploring.",
        "" if not plate_void else
        "\n⛔ **The plate is VOID** — a control did not return its expected verdict, so no "
        "prediction is scored here and nothing banks.\n",
        "",
        *table_md(["probe_id", "predicted", "actual", "hit"], score_rows),
        "",
    ]), encoding="utf-8", newline="\n")

    # ---- write DELTA_DRAFT.md (the FRONTIER row stays a human's to append)
    delta_row = [["(next)", plate["date"], plate["name"], move,
                  f"{len(proposals)} new, {len(increments)} incremented",
                  f"`{(expedition / 'PREDICTION_SCORE.md').relative_to(REPO).as_posix()}`"]]
    (expedition / "DELTA_DRAFT.md").write_text("\n".join([
        "<!-- generated by scripts/skadi_bank.py; the row below is appended to the FRONTIER "
        "DELTA LOG by a human at close -->",
        "",
        f"# DELTA DRAFT — {plate['name']}",
        "",
        f"Computed move: **{move}**. `INSTRUMENT-ONLY` and `NONE` are blameless — a wall-map "
        "is a result, and only a STREAK trips anything. What a streak trips is a recorded "
        "re-authorization by The Scientist, never a score.",
        "",
        *table_md(DELTA_COLS, delta_row),
        "",
        "Append this row to `## DELTA LOG` in `vault/music_atlas/FRONTIER.md`, numbered in "
        "sequence, then write `INDEX.md` to close the expedition. "
        "`validate_music_atlas.py` FAILs on a closed expedition with no delta row.",
        "",
    ]), encoding="utf-8", newline="\n")

    print(f"\nWROTE     {led_path.relative_to(REPO).as_posix()} (+{len(ledger_rows)} row(s))")
    print(f"WROTE     vault/music_atlas/ATLAS.md (+{len(receipts)} receipt(s))")
    print("WROTE     vault/music_atlas/FRONTIER.md (coverage · scruff · control chart)")
    print(f"WROTE     {(expedition / 'PREDICTION_SCORE.md').name}")
    print(f"WROTE     {(expedition / 'DELTA_DRAFT.md').name}")
    print("\nNext: append the DELTA row, write INDEX.md, then\n"
          "  python3 scripts/validate_music_atlas.py")
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
