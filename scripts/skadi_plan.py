#!/usr/bin/env python3
"""skadi_plan.py — lay out one expedition (a plate of probe slots).

Reads the atlas state and `vault/gates.yaml`; writes `PLAN.md` and a
`PREDICTIONS.md` template into an expedition directory.

⛔ THE PLANNER DOES ARITHMETIC. It never ranks by taste and it never fills the
non-obvious slot — it writes the literal placeholder and stops, and
`validate_music_atlas.py --preflight` refuses the plate while that placeholder
survives. Without a slot no machine can justify, the search collapses onto the
prior and the expedition becomes an expensive lookup that looks thorough.

⛔ IT ALSO WRITES NO PREDICTIONS. A prediction registered after the render is not
a prediction; the template leaves both `prediction` and `registered_utc` empty
and preflight refuses until a mind has filled them.

Nothing here renders, publishes, spends, or touches the network. The plan is a
menu; the renders are The Scientist's.

SPEC-GAP: DESIGN v1 §4 is silent on whether an expedition directory takes a
`scripts/run_lock.py` lock the way a Lofn run does. Not wired here — the
expedition rides `lofn-music`, which holds its own lock, and the skill (DESIGN §7,
Lofn's to write) is the right place to decide. Flagged rather than invented.

USAGE
    python3 scripts/skadi_plan.py --out output/expeditions/2026-08-20_first-plate
    python3 scripts/skadi_plan.py --out <dir> --force

EXIT   0 written · 1 refused (files exist; --force to overwrite) · 2 input error
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_music_atlas import (  # noqa: E402
    ATLAS_COLS, CONTROL_CHART_COLS, COVERAGE_COLS, PLAN_COLS, PREDICTION_COLS,
    SCRUFF_COLS, SPEC_CLASS_COLS, TERRITORY_COLS,
    ATLAS_DIR, EMPTY_CELL, NON_OBVIOUS_PLACEHOLDER, REGISTER_BEFORE_RENDER, REPO,
    load_gates, parse_table, read, strike,
)

FRONTIER_REL = "vault/music_atlas/FRONTIER.md"


def _fail(msg: str) -> "SystemExit":
    return SystemExit(f"ERROR: {msg}")


def cell_parts(scope: str) -> tuple[str, str]:
    """`Jazz × interrupting-silence` -> ('Jazz', 'interrupting-silence')."""
    parts = [p.strip() for p in re.split(r"[×x]", scope) if p.strip()]
    if len(parts) >= 2:
        return parts[0], parts[-1]
    return "", scope.strip()


def load_state(gates: dict) -> dict:
    """Every parse asserts its cardinality: an empty extraction is an ERROR."""
    atlas_text = read(ATLAS_DIR / "ATLAS.md")
    if not atlas_text:
        raise _fail("vault/music_atlas/ATLAS.md absent — run scripts/skadi_seed.py first")
    claims: list[dict] = []
    for tier in ("BANKED", "OBSERVED", "CONJECTURED"):
        rows, _ = parse_table(atlas_text, ATLAS_COLS, section=tier)
        for r in rows:
            r["_tier"] = tier
        claims.extend(rows)
    if not claims:
        raise _fail("ATLAS.md yielded 0 claim rows — an empty extraction is an ERROR, "
                    "never an empty plan")

    classes, probs = parse_table(read(ATLAS_DIR / "SPEC_CLASSES.md"), SPEC_CLASS_COLS)
    if probs or not classes:
        raise _fail(f"SPEC_CLASSES.md unreadable: {'; '.join(probs) or 'no rows'}")

    territories, probs = parse_table(read(ATLAS_DIR / "TERRITORIES.md"), TERRITORY_COLS)
    if probs or not territories:
        raise _fail(f"TERRITORIES.md unreadable: {'; '.join(probs) or 'no rows'}")

    frontier_text = read(ATLAS_DIR / "FRONTIER.md")
    if not frontier_text:
        raise _fail("FRONTIER.md absent — Phase 0 reads the frontier")
    coverage, probs = parse_table(frontier_text, COVERAGE_COLS, section="COVERAGE")
    if probs or not coverage:
        raise _fail(f"FRONTIER COVERAGE unreadable: {'; '.join(probs) or 'no rows'}")
    chart, _ = parse_table(frontier_text, CONTROL_CHART_COLS, section="CONTROL CHART")
    scruff, _ = parse_table(frontier_text, SCRUFF_COLS, section="SCRUFF SHELF")

    return {"claims": claims, "classes": classes, "territories": territories,
            "coverage": coverage, "chart": chart, "scruff": scruff}


def standing_controls(chart: list[dict]) -> list[tuple[str, str, str]]:
    """-> [(kind, spec_class, expected_verdict)] in a stable order.

    The CONTROL CHART is the machine surface for which controls are standing and
    what each is expected to return: the LAST recorded point per control is the
    expectation. A control expecting ARRIVES is the positive one; a control
    expecting a smoothing/absence is the negative one. No taste, no second file.
    """
    latest: dict[str, str] = {}
    for r in chart:
        latest[r["control"]] = r["verdict"].upper()
    out = []
    for name in sorted(latest):
        verdict = latest[name]
        if verdict == "ARRIVES":
            out.append(("POSITIVE-CONTROL", name, verdict))
        elif verdict in ("SMOOTHED", "ABSENT", "INVERTED"):
            out.append(("NEGATIVE-CONTROL", name, verdict))
    out.sort(key=lambda t: (0 if t[0] == "POSITIVE-CONTROL" else 1, t[1]))
    return out


def _consumption_counts() -> dict[str, int]:
    rows, _ = parse_table(read(ATLAS_DIR / "FRONTIER.md"),
                          ["date", "run_dir", "rows_cited"], section="CONSUMPTION LOG")
    counts: dict[str, int] = {}
    for r in rows:
        for rid in re.findall(r"\bSK(?:OP)?-\d{2,3}\b", r["rows_cited"]):
            counts[rid] = counts.get(rid, 0) + 1
    return counts


def rank_candidates(state: dict, gates: dict) -> dict[str, list[tuple[str, str, str, str]]]:
    """-> {generator: [(territory, spec_class, channel, price)]}, all arithmetic.

    Four generators, each deterministically ordered. The plan round-robins them
    so no single kind of arithmetic can eat the whole plate.
    """
    claims, classes = state["claims"], state["classes"]
    channels = [str(c) for c in gates.get("atlas_channels", [])]
    bank_min = int(gates.get("atlas_bank_min_n", 2))
    consumption = _consumption_counts()

    class_names = [c["class"] for c in classes]
    claims_per_class: dict[str, int] = {c: 0 for c in class_names}
    claims_per_channel: dict[str, int] = {c: 0 for c in channels}
    for r in claims:
        _, cls = cell_parts(r["scope"])
        if cls in claims_per_class:
            claims_per_class[cls] += 1
        ch = r["channel"].lower()
        if ch in claims_per_channel:
            claims_per_channel[ch] += 1

    probes_per_territory = {r["territory"]: int(r["probes"]) if r["probes"].isdigit() else 0
                            for r in state["coverage"]}
    footprint = {}
    for r in state["territories"]:
        nums = [int(x) for x in re.findall(r"\d+", r["footprint"])]
        footprint[r["territory"]] = sum(nums)

    # Territories ordered by: fewest probes, then most prior material (work where
    # we already have samples), then name.
    territory_order = sorted(probes_per_territory,
                             key=lambda t: (probes_per_territory[t], -footprint.get(t, 0), t))
    home = territory_order[0] if territory_order else "(no territory)"

    # A. REPLICATION — rows that are cited but under-evidenced. Replication
    #    priority follows CONSUMPTION: confirm the hits that are already steering
    #    runs, not the ones that are merely old.
    replication = []
    for r in claims:
        rid, _ = strike(r["id"])
        n = int(r["n"]) if r["n"].isdigit() else 0
        needs = r["_tier"] == "CONJECTURED" or n < bank_min
        if not needs:
            continue
        terr, cls = cell_parts(r["scope"])
        price = (f"moves {rid} to n={n + 1}"
                 + (f" — cited by {consumption.get(rid, 0)} run(s)" if consumption.get(rid) else "")
                 + (" — leaves CONJECTURED once a blind pass hears it"
                    if r["_tier"] == "CONJECTURED" else ""))
        replication.append((-consumption.get(rid, 0), n, rid,
                            (terr if terr and terr != "any territory" else home,
                             cls, r["channel"].lower(), price)))
    replication.sort(key=lambda t: (t[0], t[1], t[2]))

    # B. OPEN-CHANNEL — a channel with zero claims is a whole face of the map
    #    nobody has touched. Paired with the least-claimed spec class.
    thin_class = sorted(class_names, key=lambda c: (claims_per_class[c], c))
    open_channel = [(home, thin_class[0] if thin_class else "(no class)", ch,
                     f"opens the `{ch}` channel — it carries zero claims")
                    for ch in channels if claims_per_channel.get(ch, 0) == 0]

    # C. OPEN-CLASS — a spec class with zero claims.
    open_class = [(home, c, "style", f"opens the `{c}` spec class — it carries zero claims")
                  for c in sorted(class_names, key=lambda c: (claims_per_class[c], c))
                  if claims_per_class[c] == 0]

    # D. OPEN-TERRITORY — a territory with zero probes. Territory coverage is
    #    accounted through genre-arrival: the blind listener names what it heard.
    open_territory = [(t, "genre-arrival", "style",
                       f"opens the `{t}` territory — 0 probes, footprint {footprint.get(t, 0)}")
                      for t in territory_order if probes_per_territory[t] == 0]

    return {"replication": [c for *_, c in replication],
            "open-channel": open_channel,
            "open-class": open_class,
            "open-territory": open_territory}


def scruff_due(state: dict, gates: dict) -> tuple[str, str] | None:
    """A scruff at or over the rift threshold and still on WATCH earns a slot."""
    threshold = int(gates.get("atlas_scruff_rift_sightings", 3))
    for r in state["scruff"]:
        sightings = [s for s in re.split(r"[,\s]+", r["sightings"]) if s and s not in EMPTY_CELL]
        if len(sightings) >= threshold and r["status"].upper() == "WATCH":
            return r["scruff_id"], (f"chases {r['scruff_id']} — {len(sightings)} sightings, "
                                    f"at or over the rift threshold")
    return None


def build_slots(state: dict, gates: dict) -> list[list[str]]:
    slots = int(gates.get("atlas_probe_slots", 6))
    quota = int(gates.get("atlas_non_obvious_quota", 1))
    min_pos = int(gates.get("atlas_min_positive_controls", 1))
    min_neg = int(gates.get("atlas_min_negative_controls", 1))

    controls = standing_controls(state["chart"])
    pos = [c for c in controls if c[0] == "POSITIVE-CONTROL"][:max(min_pos, 1)]
    neg = [c for c in controls if c[0] == "NEGATIVE-CONTROL"][:max(min_neg, 1)]
    if len(pos) < min_pos or len(neg) < min_neg:
        raise _fail(
            f"the CONTROL CHART declares {len(pos)} positive and {len(neg)} negative standing "
            f"control(s); gates require {min_pos}/{min_neg}. A plate with no control proves "
            "nothing — add the control to FRONTIER.md before planning.")

    chase = scruff_due(state, gates)
    body = slots - len(pos) - len(neg) - quota - (1 if chase else 0)
    if body < 0:
        raise _fail(f"gates atlas_probe_slots leaves no room: {slots} slot(s) cannot hold "
                    f"{len(pos)}+{len(neg)} controls, {quota} non-obvious"
                    + (" and a scruff chase" if chase else ""))

    ranked = rank_candidates(state, gates)
    order = ["replication", "open-channel", "open-class", "open-territory"]
    cursors = {k: 0 for k in order}
    picked: list[tuple[str, str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    while len(picked) < body:
        progressed = False
        for key in order:
            if len(picked) >= body:
                break
            pool = ranked[key]
            while cursors[key] < len(pool):
                cand = pool[cursors[key]]
                cursors[key] += 1
                sig = (cand[0], cand[1], cand[2])
                if sig in seen:
                    continue
                seen.add(sig)
                picked.append(cand)
                progressed = True
                break
        if not progressed:
            break
    if len(picked) < body:
        raise _fail(f"only {len(picked)} of {body} probe slot(s) could be justified by "
                    "arithmetic. The planner refuses to pad a plate — widen the atlas or "
                    "shrink gates atlas_probe_slots.")

    rows: list[list[str]] = []
    i = 1
    for kind, cls, expected in pos + neg:
        rows.append([f"P{i}", kind, f"any territory × {cls}", "style",
                     f"Standing control: ask for `{cls}` and expect `{expected}`.",
                     expected,
                     "control due — extends the control chart and re-validates this epoch"])
        i += 1
    for terr, cls, channel, price in picked:
        rows.append([f"P{i}", "PROBE", f"{terr} × {cls}", channel,
                     f"Does `{cls}` arrive in `{terr}` when asked through the `{channel}` "
                     "channel?", "—", price])
        i += 1
    if chase:
        rows.append([f"P{i}", "SCRUFF-CHASE", f"(scruff {chase[0]})", "style",
                     f"Chase {chase[0]}: reproduce the deviation deliberately and see whether "
                     "it is a law or a coincidence.", "—", chase[1]])
        i += 1
    for _ in range(quota):
        rows.append([f"P{i}", "NON-OBVIOUS", NON_OBVIOUS_PLACEHOLDER, "style",
                     NON_OBVIOUS_PLACEHOLDER, "—",
                     "the non-obvious slot: the one probe no arithmetic can justify"])
        i += 1
    return rows


def row_md(cells: list) -> str:
    return "| " + " | ".join(str(c).replace("|", "\\|") for c in cells) + " |"


def table_md(cols: list[str], rows: list[list]) -> str:
    return "\n".join([row_md(cols), row_md(["---"] * len(cols))] + [row_md(r) for r in rows])


def render_plan(expedition: str, rows: list[list[str]], gates: dict) -> str:
    return "\n".join([
        "<!-- generated by scripts/skadi_plan.py; read by scripts/validate_music_atlas.py "
        "(--preflight, check_controls, check_predictions) -->",
        "",
        f"# PLAN — {expedition}",
        "",
        f"expedition: `{expedition}`",
        f"frontier_read: `{FRONTIER_REL}`",
        f"gates_read: `vault/gates.yaml` (atlas_probe_slots={gates.get('atlas_probe_slots')}, "
        f"atlas_min_positive_controls={gates.get('atlas_min_positive_controls')}, "
        f"atlas_min_negative_controls={gates.get('atlas_min_negative_controls')}, "
        f"atlas_non_obvious_quota={gates.get('atlas_non_obvious_quota')})",
        "",
        "> **A menu, never an order. Renders are The Scientist's.** Nothing in this system "
        "renders, publishes, or spends.",
        "",
        "## SLOTS",
        "",
        table_md(PLAN_COLS, rows),
        "",
        "## HOW THESE WERE PRICED",
        "",
        "Every `price` above is map arithmetic and nothing else: *control due* · *opens a "
        "cell/channel/class/territory* · *moves an existing claim to n+1* · *chases a scruff at "
        "the rift threshold*. Candidates are generated by four independent rules and taken "
        "round-robin, so no single kind of arithmetic can eat the plate. Replication is ordered "
        "by CONSUMPTION first — confirm the claims that are already steering runs, not the ones "
        "that are merely old.",
        "",
        "⛔ **Two cells on this plate were left for a mind.**",
        "",
        f"1. The `NON-OBVIOUS` slot carries the literal placeholder `{NON_OBVIOUS_PLACEHOLDER}`. "
        "The planner may not fill it, and `validate_music_atlas.py --preflight` refuses the "
        "plate while the placeholder survives. Every prune must keep at least one survivor "
        "nobody can locally justify, with a recorded one-line reason — without it the search "
        "collapses onto the prior.",
        "2. **PREDICTIONS.md is empty.** A prediction registered after the render is not a "
        "prediction. Fill `prediction` and `registered_utc` for every slot before anything is "
        "rendered; preflight refuses the plate until they are there.",
        "",
        "## CLOSE PROTOCOL",
        "",
        "`skadi_ingest.py` → `skadi_listen.py` → `skadi_bank.py` → "
        "`validate_music_atlas.py` → one DELTA row → INDEX.md.",
        "",
    ])


def render_predictions(expedition: str, rows: list[list[str]]) -> str:
    # ⛔ THE TEMPLATE STAMPS NO CLOCK. `registered_utc` gets the literal
    # REGISTER-BEFORE-RENDER, which `validate_music_atlas.py --preflight` refuses
    # as not-a-timestamp — so an unfilled template cannot pass. A planner that
    # helpfully wrote "now" would be a gate manufacturing its own alibi: the one
    # field whose whole job is to prove which of two events happened first.
    pred_rows = [[r[0], r[2], r[3], "—", REGISTER_BEFORE_RENDER] for r in rows]
    return "\n".join([
        "<!-- generated by scripts/skadi_plan.py; read by scripts/validate_music_atlas.py "
        "(--preflight, check_predictions) -->",
        "",
        f"# PREDICTIONS — {expedition}",
        "",
        "⛔ **Registered BEFORE the render, scored after.** Both cells below are `—` because "
        "the planner does not predict: it does arithmetic. A mind fills `prediction` with the "
        "verdict it expects and `registered_utc` with an ISO-8601 UTC timestamp, before "
        "anything is rendered. `skadi_bank.py` scores these against the actual verdicts into "
        "`PREDICTION_SCORE.md`, which is the falsifiable record of whether the rig earns itself.",
        "",
        table_md(PREDICTION_COLS, pred_rows),
        "",
    ])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Plan one SKAÐI expedition (a plate of probes).")
    ap.add_argument("--out", required=True, metavar="EXPEDITION_DIR",
                    help="expedition directory, normally under output/expeditions/")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing PLAN.md / PREDICTIONS.md")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; write nothing")
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass

    gates, warnings = load_gates()
    for w in warnings:
        print(f"[GATES] {w}", file=sys.stderr)

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    state = load_state(gates)
    rows = build_slots(state, gates)

    expedition = out.name
    plan_text = render_plan(expedition, rows, gates)
    pred_text = render_predictions(expedition, rows)

    if args.dry_run:
        print(plan_text)
        print(pred_text)
        return 0

    existing = [p for p in (out / "PLAN.md", out / "PREDICTIONS.md") if p.exists()]
    if existing and not args.force:
        for p in existing:
            print(f"REFUSED   {p} — already exists; --force to overwrite. A mind may already "
                  "have filled the non-obvious slot or registered predictions here.",
                  file=sys.stderr)
        return 1

    out.mkdir(parents=True, exist_ok=True)
    (out / "PLAN.md").write_text(plan_text, encoding="utf-8", newline="\n")
    (out / "PREDICTIONS.md").write_text(pred_text, encoding="utf-8", newline="\n")
    print(f"WROTE     {(out / 'PLAN.md')}")
    print(f"WROTE     {(out / 'PREDICTIONS.md')}")
    print(f"\n{len(rows)} slot(s). Next: fill the NON-OBVIOUS slot and register every "
          f"prediction, then\n  python3 scripts/validate_music_atlas.py --preflight "
          f"{out.relative_to(REPO).as_posix() if out.is_relative_to(REPO) else out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
