#!/usr/bin/env python3
"""video_cost_preflight.py — the machine reader for vault/VIDEO_MODEL_DOSSIER.yaml.

WHY THIS EXISTS
    An earlier gates file had zero machine readers, and every number in it could
    be violated silently. A gate with no reader is not a gate, it is a sentence.

    `lofn-video` enforced the spend line with the sentence "Do not call render
    tools." That control worked only because the capability was absent. There is
    now a funded fal account, so the control has to do real work: this script
    computes what a wave would cost BEFORE dispatch and exits non-zero if it
    breaches the ceiling. The bad state is unreachable, not discouraged.

USAGE
    python3 scripts/video_cost_preflight.py --list
    python3 scripts/video_cost_preflight.py --model ltx-2.3-fast --duration 8 --count 24
    python3 scripts/video_cost_preflight.py --model seedance-2.5 --tier 720p --duration 30 --count 24
    python3 scripts/video_cost_preflight.py --ladder \
        --draft-model ltx-2.3-fast --finish-model ltx-2.3 --duration 8 --count 24 --finish-count 6

EXIT CODES
    0  under ceiling — dispatch permitted (still needs The Scientist's approval to spend)
    2  OVER CEILING — refuse to dispatch
    3  bad input (unknown model / tier / duration outside the model's range)
    4  ladder families disagree without --bakeoff
    5  OFF-POLICY MODEL — render target outside policy.top_line_models without an override

⛔ A PASS HERE IS NOT PERMISSION TO SPEND. vault/AUTONOMY.md still owns the line:
   autonomous runs stop at drafts on disk. This script bounds an *approved* spend;
   it never authorises one.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("video_cost_preflight: PyYAML is required (pip install pyyaml)")

REPO = Path(__file__).resolve().parent.parent
DOSSIER = REPO / "vault" / "VIDEO_MODEL_DOSSIER.yaml"


def die(msg: str, code: int) -> "NoReturn":  # noqa: F821
    """Exit with a NAMED code.

    `sys.exit("message")` prints the string and exits 1 regardless of what the
    docstring promised — a documented-vs-actual mismatch caught by this script's
    own first test run. Exit codes are part of the contract, so set them here.
    """
    print(f"video_cost_preflight: {msg}  [exit {code}]", file=sys.stderr)
    raise SystemExit(code)


def load(path: Path) -> dict:
    if not path.exists():
        die(f"dossier not found at {path}", 3)
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def family(model_id: str) -> str:
    """Model family = the part before the version, e.g. ltx / veo / seedance.

    The draft rung exists to test OUR NOTATION, not the model. Drafting on one
    family and finishing on another measures nothing about the finish.
    """
    return model_id.split("-")[0]


def rate_for(model_id: str, spec: dict, tier: str | None, audio: str) -> tuple[float, str]:
    """Return (usd_per_second, resolved_tier). Exits 3 on anything unresolvable."""
    table = spec.get("usd_per_second") or {}
    if audio == "off" and spec.get("audio_toggle_changes_price"):
        table = spec.get("usd_per_second_audio_off") or table
    tier = tier or spec.get("default_tier")
    if tier not in table:
        die(f"model '{model_id}' has no rate for tier '{tier}'. "
            f"Available: {', '.join(sorted(table))}", 3)
    return float(table[tier]), tier


def check_duration(model_id: str, spec: dict, duration: float) -> list[str]:
    """Out-of-range duration is FATAL, not a warning.

    If the call would be clamped or rejected, the dollar figure this script just
    printed is not the figure that will be billed — and a wrong estimate that
    still prints a green tick is worse than no estimate.
    """
    hi = spec.get("max_duration_s")
    lo = spec.get("min_duration_s")
    if hi is not None and duration > hi:
        die(f"duration {duration:g}s exceeds {model_id} max of {hi}s — the call would fail or be "
            f"clamped, so this cost estimate is invalid", 3)
    if lo is not None and duration < lo:
        die(f"duration {duration:g}s is below {model_id} min of {lo}s — cost estimate is invalid", 3)
    return []


def banner(meta: dict) -> list[str]:
    out = []
    if meta.get("status") != "MEASURED" or not meta.get("measured_on"):
        out.append(
            "⚠️  PROVISIONAL — every rate in this dossier is `source: vendor-page`, "
            "i.e. a CLAIM, not an invoice."
        )
        out.append(
            "    THE EVIDENCE IS THE ARTIFACT: stamp meta.measured_on and flip models to "
            "`source: our-invoice` once a real fal receipt agrees."
        )
    return out


def quote(model_id: str, spec: dict, tier: str | None, duration: float, count: int,
          audio: str) -> tuple[float, float, str, list[str]]:
    rate, resolved = rate_for(model_id, spec, tier, audio)
    per_clip = rate * duration
    total = per_clip * count
    return per_clip, total, resolved, check_duration(model_id, spec, duration)


def main() -> int:
    p = argparse.ArgumentParser(description="Cost preflight for a Lofn video render wave.")
    p.add_argument("--dossier", default=str(DOSSIER))
    p.add_argument("--list", action="store_true", help="print the dossier and exit")

    p.add_argument("--model")
    p.add_argument("--tier", help="resolution key, e.g. 1080p / 720p / 4k / 2k")
    p.add_argument("--duration", type=float, help="seconds per clip")
    p.add_argument("--count", type=int, default=1,
                   help="RENDERS in this wave — not prompts. Prompts are free and the 24 mandate "
                        "is a divergence rule on prompts; renders are 1-3 selected prompts x takes.")
    p.add_argument("--takes", type=int,
                   help="takes per selected prompt (multiplies --count). The pipeline removes "
                        "prompt-caused regeneration, not seed-caused regeneration.")
    p.add_argument("--audio", choices=["on", "off"], default="on")
    p.add_argument("--wave", choices=["takes", "finish", "post", "draft", "bakeoff"], default="takes",
                   help="which ceiling applies (default: takes)")
    p.add_argument("--ceiling", type=float, help="override the dossier ceiling (USD)")

    p.add_argument("--ladder", action="store_true", help="cost a full draft->finish ladder")
    p.add_argument("--draft-model")
    p.add_argument("--draft-tier")
    p.add_argument("--finish-model")
    p.add_argument("--finish-tier")
    p.add_argument("--finish-count", type=int, default=6)
    p.add_argument("--finish-duration", type=float, help="defaults to --duration")
    p.add_argument("--bakeoff", action="store_true",
                   help="allow draft and finish to be different model families (budget as a bake-off)")
    p.add_argument("--allow-non-top-line", action="store_true",
                   help="render at a model outside policy.top_line_models. A BAD RENDER IS WORSE "
                        "THAN NO RENDER — this needs a stated reason, not a flag reflex.")

    a = p.parse_args()
    doc = load(Path(a.dossier))
    models, meta, budget = doc["models"], doc["meta"], doc["budget"]
    ceilings = budget["wave_ceiling_usd"]

    if a.list:
        print(f"VIDEO MODEL DOSSIER — compiled {meta['compiled_on']} · status {meta['status']}")
        print(f"account: {budget['account']} · funded ${budget['funded_usd']:.2f}\n")
        print(f"{'model':<20} {'role':<15} {'tier':<8} {'$/s':>8} {'max s':>6}  audio  shots")
        print("-" * 78)
        for mid, spec in models.items():
            tier = spec.get("default_tier")
            rate = (spec.get("usd_per_second") or {}).get(tier, 0.0)
            print(f"{mid:<20} {str(spec.get('role','')):<15} {str(tier):<8} "
                  f"{rate:>8.4f} {str(spec.get('max_duration_s','')):>6}  "
                  f"{'yes' if spec.get('native_audio') else 'no ':<5}  "
                  f"{str(spec.get('timed_shot_list',''))}")
        print()
        for line in banner(meta):
            print(line)
        return 0

    if a.ladder:
        dm, fm = a.draft_model, a.finish_model
        if not (dm and fm and a.duration):
            die("--ladder needs --draft-model, --finish-model and --duration", 3)
        for mid in (dm, fm):
            if mid not in models:
                die(f"unknown model '{mid}'. Known: {', '.join(sorted(models))}", 3)

        rows, breaches, warns = [], [], []
        if family(dm) != family(fm) and not a.bakeoff:
            print(f"⛔ LADDER FAMILY MISMATCH: draft '{dm}' and finish '{fm}' are different model families.")
            print("   The draft rung exists to test OUR NOTATION, not the model. Drafting on one family")
            print("   and finishing on another measures nothing about the finish.")
            print("   If the comparison IS the point, re-run with --bakeoff and budget it as a bake-off.")
            return 4

        d_clip, d_total, d_tier, w = quote(dm, models[dm], a.draft_tier, a.duration, a.count, a.audio)
        warns += w
        rows.append(("DRAFT", dm, d_tier, a.duration, a.count, d_clip, d_total, ceilings["draft"]))

        f_dur = a.finish_duration or a.duration
        f_clip, f_total, f_tier, w = quote(fm, models[fm], a.finish_tier, f_dur, a.finish_count, a.audio)
        warns += w
        rows.append(("FINISH", fm, f_tier, f_dur, a.finish_count, f_clip, f_total, ceilings["finish"]))

        print(f"RENDER LADDER · audio {a.audio}" + ("  [BAKE-OFF]" if a.bakeoff else ""))
        print(f"{'rung':<8} {'model':<20} {'tier':<7} {'sec':>5} {'n':>4} {'$/clip':>9} {'$ total':>9} {'ceiling':>9}  ")
        print("-" * 78)
        grand = 0.0
        for rung, mid, tier, dur, n, per, tot, ceil in rows:
            ceil = a.ceiling if a.ceiling is not None else ceil
            flag = "OVER" if tot > ceil else "ok"
            if tot > ceil:
                breaches.append((rung, mid, tot, ceil))
            grand += tot
            print(f"{rung:<8} {mid:<20} {tier:<7} {dur:>5.0f} {n:>4} {per:>9.4f} {tot:>9.2f} {ceil:>9.2f}  {flag}")
        print("-" * 78)
        sess = budget["session_ceiling_usd"]
        print(f"{'TOTAL':<8} {'':<20} {'':<7} {'':>5} {'':>4} {'':>9} {grand:>9.2f} {sess:>9.2f}  "
              f"{'OVER' if grand > sess else 'ok'}   (session ceiling)")
        print(f"\nremaining of ${budget['funded_usd']:.2f} account after this ladder: "
              f"${budget['funded_usd'] - grand:.2f}")
        for w in warns:
            print(f"⚠️  {w}")
        for line in banner(meta):
            print(line)
        if grand > sess:
            breaches.append(("SESSION", "-", grand, sess))
        if breaches:
            print("\n⛔ REFUSED — ceiling breached:")
            for rung, mid, tot, ceil in breaches:
                print(f"   {rung} {mid}: ${tot:.2f} > ${ceil:.2f}")
            return 2
        print("\n✅ under ceiling. NOT permission to spend — vault/AUTONOMY.md still requires The Scientist.")
        return 0

    if not (a.model and a.duration):
        die("need --model and --duration (or --list / --ladder)", 3)
    if a.model not in models:
        die(f"unknown model '{a.model}'. Known: {', '.join(sorted(models))}", 3)

    spec = models[a.model]
    takes = a.takes or 1
    renders = a.count * takes
    per_clip, total, tier, warns = quote(a.model, spec, a.tier, a.duration, renders, a.audio)
    ceiling = a.ceiling if a.ceiling is not None else ceilings[a.wave]

    policy = doc.get("policy", {})
    top_line = policy.get("top_line_models", [])
    off_policy = bool(top_line) and a.model not in top_line and a.wave in ("takes", "finish")

    print(f"model      {a.model}  ({spec.get('vendor','?')}, role {spec.get('role','?')})")
    print(f"tier       {tier}   audio {a.audio}")
    if takes > 1:
        print(f"renders    {a.count} prompt(s) x {takes} take(s) = {renders} renders")
    print(f"clip       {a.duration:g}s x {renders}  ->  ${per_clip:.4f}/clip")
    print(f"WAVE TOTAL ${total:.2f}   ceiling(${a.wave}) ${ceiling:.2f}   "
          f"account ${budget['funded_usd']:.2f}")

    if off_policy and not a.allow_non_top_line:
        print(f"\n⛔ OFF-POLICY MODEL — '{a.model}' is not in policy.top_line_models "
              f"({', '.join(top_line)}).")
        print("   ⭐ A BAD RENDER IS WORSE THAN NO RENDER. Cheap models are not a cheaper route to")
        print("   the same piece; they are a different, worse piece. The saving comes from getting")
        print("   the PROMPT right — the pipeline's job — not from buying weaker inference.")
        print("   The three-pass gate detects incoherence, NOT averageness, which is exactly how a")
        print("   cheap tier fails — so it would pass this gate on the way to shipping the mean.")
        print("   Override with --allow-non-top-line and a stated reason.")
        return 5
    if off_policy:
        print(f"⚠️  OFF-POLICY model '{a.model}' allowed by explicit override.")
    if spec.get("warning"):
        print(f"⚠️  {spec['warning'].strip()}")
    for w in warns:
        print(f"⚠️  {w}")
    for line in banner(meta):
        print(line)

    if total > ceiling:
        pct = total / budget["funded_usd"] * 100
        print(f"\n⛔ REFUSED — ${total:.2f} exceeds the {a.wave} ceiling of ${ceiling:.2f} "
              f"({pct:.0f}% of the whole account).")
        print("   Cut RENDERS, not quality: fewer selected prompts, fewer takes, shorter duration.")
        print("   ⭐ Prompts are free — diverge in text, then render the few that earned it.")
        print("   Dropping to a cheaper model is NOT the fix (policy.top_line_only).")
        return 2

    print("\n✅ under ceiling. NOT permission to spend — vault/AUTONOMY.md still requires The Scientist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
