#!/usr/bin/env python3
"""plan_albums.py — cut the Suno catalogue into two release-ready albums.

WHY: "top of the last X months" and "the rest" are two ranking decisions and a
cap. Doing them by eye over 210 clips produces a defensible-looking order that
nobody can reproduce or argue with. This makes the ranking explicit, the
exclusions explicit, and X a dial.

    python3 scripts/plan_albums.py --months 7
    python3 scripts/plan_albums.py --months 7 --size-a 16 --size-b 24
    python3 scripts/plan_albums.py --compare        # show every window at once

THE RANKING: Wilson lower bound on like-rate, NOT plays and NOT raw like-rate.

  * Plays are unusable: one placement-driven outlier holds 349,736 of the
    catalogue's 374,122 plays (93%). Ranking by plays ranks that track first
    and everything after it by noise.
  * Raw like-rate is unusable at the tail: 3 likes on 23 plays reads as 0.130
    and outranks 47 likes on 497 plays. The Wilson lower bound penalises small
    samples automatically, which is exactly the correction that ratio needs.

THE EVIDENCE FLOOR: Wilson shrinks small samples but does not disqualify them,
and at n=9 the bound is still mostly prior. The first cut of this plan seated a
9-play track fifth on album B. So a track needs `--min-plays` (default 50, the
catalogue median) before its like-rate counts as evidence at all — EXCEPT
inside `--grace-days` of release, because a week-old track with 40 plays has
not been ignored, it has not been heard yet. Old-and-unplayed and new-and-
unplayed look identical in the ratio and are not the same fact.

THE EXCLUSIONS: a track already on DistroKid must not ship again inside an
album, or the same recording exists twice on Spotify under two release ids.
Known-released titles are excluded by name and REPORTED, never silently
dropped — see `--show-excluded`.

AUTONOMY.md: reads local snapshots, writes a plan to disk. Publishes nothing,
spends nothing, uploads nothing.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent

# DistroKid's default ceiling is 35 tracks per release; more only by a support
# request. Any plan that quietly exceeds it is a plan that cannot be uploaded.
DISTROKID_TRACK_CAP = 35

# Confirmed on DistroKid as already distributed (data/snapshots/distribution/
# latest.json, pulled 2026-08-02). Matched loosely: Suno titles drift in
# punctuation from the release titles ("Dear Scientist," vs "Dear Scientist:").
ALREADY_RELEASED = [
    "Triple Arch Over Me",
    "A Dumber, Better Song",
    "Dear Scientist",
    "3-3-2 Open",
]


def norm(title: str | None) -> str:
    """Fold punctuation so 'Dear Scientist,' matches 'Dear Scientist:'.

    U+2011 (non-breaking hyphen) appears in real titles and is NOT the ASCII
    '-' the release list uses, so stripping non-alphanumerics is the only
    match that survives both.
    """
    return re.sub(r"[^a-z0-9]", "", (title or "").lower())


def wilson_lb(up: int, n: int, z: float = 1.96) -> float:
    """Lower bound of the 95% CI on the like rate."""
    if n == 0:
        return 0.0
    p = up / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return max(0.0, (centre - margin) / den)


def sequence(tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order an album for listening, not for the leaderboard.

    Ranked order front-loads every strong track and leaves the back half to
    sag. The rule here is deliberately simple and stated rather than dressed
    up as taste I cannot verify without ears: the two strongest open, the
    third-strongest closes, and the remainder alternate long/short so two
    five-minute tracks never sit together.
    """
    if len(tracks) <= 3:
        return tracks
    ranked = sorted(tracks, key=lambda t: -t["wlb"])
    opener, second, closer = ranked[0], ranked[1], ranked[2]
    middle = ranked[3:]

    long = sorted([t for t in middle if (t.get("duration") or 0) >= 240], key=lambda t: -t["wlb"])
    short = sorted([t for t in middle if (t.get("duration") or 0) < 240], key=lambda t: -t["wlb"])
    woven: list[dict[str, Any]] = []
    while long or short:
        if long:
            woven.append(long.pop(0))
        if short:
            woven.append(short.pop(0))
    return [opener, second, *woven, closer]


def load_rows(snapshot: Path, manifest: Path | None) -> list[dict[str, Any]]:
    snap = json.loads(snapshot.read_text(encoding="utf-8"))
    rows = snap["rows"]

    have: dict[str, dict[str, Any]] = {}
    if manifest and manifest.exists():
        man = json.loads(manifest.read_text(encoding="utf-8"))
        have = {f["id"]: f for f in man.get("files", []) if "path" in f}

    now = datetime.now(timezone.utc)
    released = {norm(t) for t in ALREADY_RELEASED}
    for r in rows:
        r["wlb"] = wilson_lb(r["upvote_count"], r["play_count"])
        created = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
        r["age_days"] = (now - created).days
        r["already_released"] = norm(r["title"]) in released
        got = have.get(r["id"])
        r["wav_path"] = got["path"] if got else None
        r["wav_bytes"] = got["bytes"] if got else None
    return rows


def build(
    rows: list[dict[str, Any]],
    months: int,
    size_a: int,
    size_b: int,
    min_plays: int = 50,
    grace_days: int = 30,
    require_wav: bool = False,
) -> dict[str, Any]:
    cutoff = months * 30.44

    for r in rows:
        r["has_evidence"] = r["play_count"] >= min_plays or r["age_days"] <= grace_days

    def usable(r: dict[str, Any]) -> bool:
        # 50 of 210 clips have no .wav on Suno's CDN. Seating one anyway
        # produces a tracklist that cannot actually be delivered, so
        # --require-wav plans only from masters already in hand.
        return bool(r["wav_path"]) if require_wav else True

    eligible = [r for r in rows if not r["already_released"] and r["has_evidence"] and usable(r)]
    thin = [r for r in rows if not r["already_released"] and not r["has_evidence"]]

    recent = [r for r in eligible if r["age_days"] <= cutoff]
    back = [r for r in eligible if r["age_days"] > cutoff]

    album_a = sequence(sorted(recent, key=lambda r: -r["wlb"])[:size_a])
    album_b = sequence(sorted(back, key=lambda r: -r["wlb"])[:size_b])

    def summarise(tracks: list[dict[str, Any]], pool: int, name: str) -> dict[str, Any]:
        runtime = sum((t.get("duration") or 0) for t in tracks)
        missing = [t["title"] for t in tracks if not t["wav_path"]]
        return {
            "name": name,
            "track_count": len(tracks),
            "pool_size": pool,
            "runtime_seconds": round(runtime, 1),
            "runtime_display": f"{int(runtime // 60)}:{int(runtime % 60):02d}",
            "over_distrokid_cap": len(tracks) > DISTROKID_TRACK_CAP,
            "wav_missing": missing,
            "tracks": [
                {
                    "position": i,
                    "title": t["title"],
                    "id": t["id"],
                    "created_at": t["created_at"],
                    "duration": t.get("duration"),
                    "play_count": t["play_count"],
                    "upvote_count": t["upvote_count"],
                    "like_rate": t["like_rate"],
                    "wilson_lb": round(t["wlb"], 5),
                    "wav_path": t["wav_path"],
                    "style_tags": t.get("tags"),
                }
                for i, t in enumerate(tracks, 1)
            ],
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ranking": "wilson_lower_bound_95pct_on_like_rate",
        "window_months": months,
        "distrokid_track_cap": DISTROKID_TRACK_CAP,
        "catalogue_size": len(rows),
        "evidence_floor": {"min_plays": min_plays, "grace_days": grace_days, "below_floor": len(thin)},
        "excluded_already_released": [r["title"] for r in rows if r["already_released"]],
        "album_a": summarise(album_a, len(recent), f"Top of the last {months} months"),
        "album_b": summarise(album_b, len(back), "The rest"),
    }


def render_markdown(plan: dict[str, Any]) -> str:
    out = [
        "# Album plan — Lofn AI",
        "",
        f"Generated {plan['generated_at'][:19]}Z · ranking: **Wilson 95% lower bound on like-rate** ",
        f"· window: **last {plan['window_months']} months** · DistroKid cap: {plan['distrokid_track_cap']}/release",
        "",
    ]
    if plan["excluded_already_released"]:
        out += [
            "**Excluded — already on DistroKid:** "
            + ", ".join(f"*{t}*" for t in plan["excluded_already_released"]),
            "",
        ]
    ef = plan["evidence_floor"]
    out += [
        f"**Evidence floor:** a track needs ≥{ef['min_plays']} plays (or to be under "
        f"{ef['grace_days']} days old) before its like-rate counts. "
        f"**{ef['below_floor']} tracks sit below it** and were not considered for either album.",
        "",
    ]
    for key in ("album_a", "album_b"):
        a = plan[key]
        flag = "  ⚠️ **OVER DISTROKID CAP**" if a["over_distrokid_cap"] else ""
        out += [
            f"## {a['name']}",
            "",
            f"{a['track_count']} tracks · {a['runtime_display']} · curated from a pool of {a['pool_size']}{flag}",
            "",
            "| # | Title | Date | Len | Plays | Likes | Wilson LB |",
            "|---:|---|---|---:|---:|---:|---:|",
        ]
        for t in a["tracks"]:
            dur = t["duration"] or 0
            out.append(
                f"| {t['position']} | {t['title']} | {t['created_at'][:10]} | "
                f"{int(dur//60)}:{int(dur%60):02d} | {t['play_count']:,} | {t['upvote_count']} | {t['wilson_lb']:.4f} |"
            )
        if a["wav_missing"]:
            out += ["", f"**WAV not yet on disk ({len(a['wav_missing'])}):** " + ", ".join(a["wav_missing"])]
        out.append("")
    return "\n".join(out)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Plan two DistroKid albums from the Suno catalogue.")
    ap.add_argument("--snapshot", default="data/snapshots/suno/latest.json")
    ap.add_argument("--manifest", default="output/distribution/wav_masters/manifest.json")
    ap.add_argument("--months", type=int, default=7, help="the X in 'top of the last X months'")
    ap.add_argument("--size-a", type=int, default=16)
    ap.add_argument("--size-b", type=int, default=24)
    ap.add_argument("--out", default="output/distribution")
    ap.add_argument("--min-plays", type=int, default=50, help="evidence floor (catalogue median)")
    ap.add_argument("--grace-days", type=int, default=30, help="new tracks exempt from the floor")
    ap.add_argument("--require-wav", action="store_true", help="only seat tracks whose master is already on disk")
    ap.add_argument("--compare", action="store_true", help="print pool sizes for every window and exit")
    args = ap.parse_args()

    rows = load_rows(REPO / args.snapshot, REPO / args.manifest)

    if args.compare:
        eligible = [
            r
            for r in rows
            if not r["already_released"]
            and (r["play_count"] >= args.min_plays or r["age_days"] <= args.grace_days)
        ]
        print(f"{'window':>10} {'in window':>10} {'back cat':>9}   top-of-window Wilson range")
        for m in (3, 4, 6, 7, 9, 12):
            rec = [r for r in eligible if r["age_days"] <= m * 30.44]
            bak = [r for r in eligible if r["age_days"] > m * 30.44]
            top = sorted(rec, key=lambda r: -r["wlb"])[:16]
            rng = f"{top[0]['wlb']:.4f} … {top[-1]['wlb']:.4f}" if len(top) >= 2 else "—"
            print(f"{m:8}mo {len(rec):10} {len(bak):9}   {rng}")
        return 0

    plan = build(rows, args.months, args.size_a, args.size_b, args.min_plays, args.grace_days, args.require_wav)
    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "album_plan.json").write_text(
        json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    md = render_markdown(plan)
    (out_dir / "ALBUM_PLAN.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\n[plan] wrote {out_dir/'ALBUM_PLAN.md'} and album_plan.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
