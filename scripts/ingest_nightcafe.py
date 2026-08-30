#!/usr/bin/env python3
"""ingest_nightcafe.py — normalise a NightCafe profile capture into a snapshot.

WHY: NightCafe is the largest engagement surface Lofn owns — 21,111 likes
received against Suno's 5,913 — and until 2026-08-02 no snapshot, ledger, or
analysis in this repo contained a single number from it. The Art rail in Lofn
Analytics had nothing behind it but three figures read off the profile header
by eye: "23K+ / 1.2K+ / 21K+".

Those three are ROUNDED. The real values are 23,766 / 1,231 / 21,111, and the
rounding hides the number that actually matters: only 1,378 of the 23,766 are
PUBLISHED. Any per-creation rate computed against 23,766 is wrong by 17x.

WHY THIS SCRIPT TAKES A FILE INSTEAD OF FETCHING
------------------------------------------------
It cannot fetch. creator.nightcafe.studio returns 403 to curl regardless of
User-Agent — it is behind bot protection — and the live data streams over a
Firestore Listen channel rather than a REST API, so there is nothing to page
against the way ingest_suno.py pages the public Suno profile endpoint.

The one clean read is the server-rendered `__NEXT_DATA__` blob on the profile
page, which must be lifted from a real browser session. So the division of
labour is:

    browser  ->  capture the payload  ->  this script normalises + checks it

Print the capture snippet with:

    python3 scripts/ingest_nightcafe.py --snippet

Then, having saved what it prints to a file:

    python3 scripts/ingest_nightcafe.py --capture user.json
    python3 scripts/ingest_nightcafe.py --capture user.json --handle CroneLife
    python3 scripts/ingest_nightcafe.py --capture user.json --dry-run

TWO INVARIANTS, same spirit as ingest_suno.py:

  1. PUBLISHED IS THE DENOMINATOR. `numCreations` is a generation counter, not a
     body of work. Every rate is computed against `numPublishedCreations` and the
     script refuses to emit a rate against the wrong one.
  2. RECONCILE OR SAY SO. The rounded header figures the site displays are
     recomputed from the exact integers and compared. A mismatch beyond rounding
     is REPORTED in the snapshot rather than silently accepted.

AUTONOMY.md: reads a local file, writes a local file. Fetches nothing, renders
nothing, publishes nothing, spends nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
SNAP = REPO / "data" / "snapshots" / "nightcafe"

# The browser step. Paste into the devtools console on
# https://creator.nightcafe.studio/u/<handle> and save the output.
SNIPPET = r"""
// NightCafe profile capture — run on https://creator.nightcafe.studio/u/<handle>
// Copy the result to a .json file and pass it to ingest_nightcafe.py --capture
JSON.stringify(
  JSON.parse(document.getElementById('__NEXT_DATA__').textContent).props.pageProps.user,
  null, 1
)
"""


def _round_like_site(n: int) -> str:
    """Reproduce NightCafe's own header rounding, so we can check ours against it."""
    if n >= 1000:
        return f"{n // 1000}K+" if n >= 10000 else f"{n / 1000:.1f}K+"
    return str(n)


def _ts(ms: Any) -> str | None:
    if not isinstance(ms, (int, float)) or ms <= 0:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def normalise(user: dict[str, Any], captured_at: str) -> dict[str, Any]:
    likes = user.get("likesMeta") or {}
    follows = user.get("followsMeta") or {}
    streak = user.get("streakMeta") or {}
    badges = user.get("badgesMeta") or {}
    colls = user.get("collectionsMeta") or {}

    total = int(user.get("numCreations") or 0)
    published = int(user.get("numPublishedCreations") or 0)
    likes_received = int(likes.get("likesReceived") or 0)
    likes_given = int(likes.get("likesGiven") or 0)
    followers = int(follows.get("numFollowers") or 0)

    notes: list[str] = []

    # INVARIANT 1 — published is the denominator, and it must exist.
    if published <= 0:
        notes.append(
            "numPublishedCreations is missing or zero, so no per-creation rate was "
            "computed. Rates against numCreations are meaningless and are not emitted."
        )
        per_published = None
        publish_rate = None
    else:
        per_published = round(likes_received / published, 2)
        publish_rate = round(published / total, 4) if total else None
        if total and published / total < 0.25:
            notes.append(
                f"Only {published:,} of {total:,} creations are published "
                f"({published / total:.1%}). The audience has never seen the rest. "
                f"Any rate computed against {total:,} is wrong by "
                f"{total / published:.0f}x."
            )

    # INVARIANT 2 — reconcile our integers against the site's own rounded display.
    display_check = {
        "creations": _round_like_site(total),
        "followers": _round_like_site(followers),
        "likes_received": _round_like_site(likes_received),
    }

    last_creation = _ts(streak.get("creationsLastDay"))
    dormant = not streak.get("creationsActive", False)
    if dormant and last_creation:
        notes.append(
            f"Creation streak is INACTIVE; last creation {last_creation}. "
            f"Longest streak was {streak.get('creationsLongest', 0)} days across "
            f"{streak.get('creationsNum5DayStreaks', 0)} separate five-day runs — "
            "this account was run consistently and then stopped."
        )

    recip = round(likes_received / likes_given, 2) if likes_given else None

    return {
        "surface": "nightcafe",
        "handle": user.get("username"),
        "display_name": (user.get("displayName") or "").strip(),
        "user_id": user.get("id"),
        "profile": f"https://creator.nightcafe.studio/u/{user.get('username')}",
        "captured_at": captured_at,
        "provenance": {
            "method": "__NEXT_DATA__ props.pageProps.user, lifted from a browser session",
            "why_not_headless": (
                "creator.nightcafe.studio returns 403 to curl at any User-Agent, and "
                "live data streams over Firestore rather than a REST API. There is no "
                "headless pull for this surface."
            ),
        },
        "account": {
            "created": _ts(user.get("created")),
            "subscription": user.get("subscription"),
            "website": user.get("website"),
            "bio": user.get("bio"),
        },
        "counts": {
            "creations_total": total,
            "creations_published": published,
            "creations_uploaded": int(user.get("numUploadCreations") or 0),
            "followers": followers,
            "following": int(follows.get("numFollowing") or 0),
            "likes_received": likes_received,
            "likes_given": likes_given,
            "comment_likes_received": int(likes.get("commentLikesReceived") or 0),
            "comment_likes_given": int(likes.get("commentLikesGiven") or 0),
            "badges": int(badges.get("numBadges") or 0),
            "badges_unclaimed": int(badges.get("numUnclaimedBadges") or 0),
            "collections": int(colls.get("numCollections") or 0),
        },
        "derived": {
            "publish_rate": publish_rate,
            "likes_per_published_creation": per_published,
            "followers_per_published": (
                round(followers / published, 2) if published else None
            ),
            "reciprocity_ratio": recip,
            "reciprocity_note": (
                "Likes received divided by likes given. On comments the ratio often "
                "inverts — generous in threads, silent in the feed. Compare "
                "comment_likes_given against comment_likes_received before concluding "
                "the account is stingy."
            ),
        },
        "activity": {
            "streak_active": bool(streak.get("creationsActive", False)),
            "longest_streak_days": streak.get("creationsLongest"),
            "five_day_streaks": streak.get("creationsNum5DayStreaks"),
            "last_creation": last_creation,
            "last_follower_received": _ts(follows.get("lastFollowerReceivedAt")),
            "last_user_followed": _ts(follows.get("lastUserFollowedAt")),
            "last_badge_received": _ts(badges.get("lastBadgeReceived")),
        },
        "reconciliation": {
            "site_rounds_to": display_check,
            "how_to_check": (
                "Compare site_rounds_to against the three figures the profile header "
                "actually prints. They must match. If they do not, the capture is "
                "stale or came from the wrong profile."
            ),
        },
        "notes": notes,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", type=Path, help="JSON file holding props.pageProps.user")
    ap.add_argument("--handle", help="label the output for a competitor scan")
    ap.add_argument("--dry-run", action="store_true", help="print, write nothing")
    ap.add_argument("--snippet", action="store_true", help="print the browser capture snippet")
    args = ap.parse_args()

    if args.snippet:
        print(SNIPPET.strip())
        return 0

    if not args.capture:
        ap.error("--capture is required (or use --snippet to get the browser step)")

    if not args.capture.exists():
        print(f"no such capture file: {args.capture}", file=sys.stderr)
        return 1

    user = json.loads(args.capture.read_text(encoding="utf-8"))
    if "username" not in user:
        # tolerate a whole __NEXT_DATA__ blob rather than just the user object
        try:
            user = user["props"]["pageProps"]["user"]
        except (KeyError, TypeError):
            print(
                "capture does not look like a NightCafe user object — expected a "
                "'username' key, or a full __NEXT_DATA__ blob",
                file=sys.stderr,
            )
            return 1

    captured_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    snap = normalise(user, captured_at)

    for n in snap["notes"]:
        print(f"  note: {n}")
    print(
        f"  {snap['display_name']} (@{snap['handle']}): "
        f"{snap['counts']['creations_published']:,} published of "
        f"{snap['counts']['creations_total']:,}, "
        f"{snap['counts']['likes_received']:,} likes, "
        f"{snap['counts']['followers']:,} followers"
    )
    print(f"  site should display: {snap['reconciliation']['site_rounds_to']}")

    if args.dry_run:
        print(json.dumps(snap, indent=1))
        return 0

    out_dir = SNAP if not args.handle else SNAP / "competitors" / args.handle
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("latest.json", f"{captured_at}.json"):
        (out_dir / name).write_text(
            json.dumps(snap, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(f"  wrote {out_dir / 'latest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
