#!/usr/bin/env python3
"""ingest_suno_comments.py — pull every comment on the Suno catalogue into a snapshot.

WHY: `vault/COMMENT_REPLY_STANDARD.md` is binding on every public reply, and
`memory/social-ledger.md` says it plainly — **no line, it didn't happen.** Until
this script existed the only way to see the inbox was to scroll it by hand in a
logged-in browser, which is why "roughly half the posted corpus survives only as
a count" and why two people went unanswered for weeks.

The endpoint is PUBLIC — no cookie, no key:

    GET https://studio-api.prod.suno.com/api/gen/{clip_id}/comments[?cursor=...]

    -> {"results": [...], "next_cursor": "<b64>|null", "total_count": N,
        "allow_comment": bool}

Discovered 2026-08-22 by reading the web app's own bundle (its route table lists
`/api/gen/{clip_id}/comments`); every older `/api/comment*` shape returns 404/405.

Replies are nested under a parent thread as `replies: [...]` with `num_replies`.
A thread counts as ANSWERED when OUR user id appears anywhere in it — that is the
only reliable "did we reply" signal, because reply text is indexed nowhere else.

    python3 scripts/ingest_suno_comments.py                 # full sweep
    python3 scripts/ingest_suno_comments.py --clip <id>     # one clip
    python3 scripts/ingest_suno_comments.py --dry-run       # print, write nothing

AUTONOMY.md: read-only over the public web, writes locally. Posts nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
API = "https://studio-api.prod.suno.com/api/gen/{clip_id}/comments"
REPLIES_API = "https://studio-api.prod.suno.com/api/comment/{comment_id}/replies"
UA = "Mozilla/5.0 (compatible; lofn-analytics-ingest/1.0; +https://www.lofn.ai)"

# Lofn AI's own Suno user id — the ANSWERED signal.
OUR_USER_ID = "7b4b0c35-2ae0-47c2-a779-0eb1b7b48ccc"
OUR_HANDLE = "localsymmetry"

OUT_DIR = REPO / "data" / "snapshots" / "suno_comments"


def fetch(url: str, timeout: int = 30, retries: int = 3) -> dict[str, Any]:
    req = urllib.request.Request(
        url, headers={"accept": "application/json", "user-agent": UA}
    )
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - report, do not hide
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("GET %s failed after %d tries: %s" % (url, retries, last))


def fetch_clip_comments(clip_id: str, max_pages: int = 80) -> dict[str, Any]:
    """Walk the cursor until it runs out. Returns the thread list plus meta."""
    url = API.format(clip_id=clip_id)
    threads: list[dict[str, Any]] = []
    seen: set[str] = set()
    cursor: str | None = None
    total_count: Any = None
    allow_comment: Any = None
    pages = 0
    truncated = False

    while pages < max_pages:
        u = url if cursor is None else url + "?cursor=" + urllib.parse.quote(cursor)
        data = fetch(u)
        pages += 1
        if total_count is None:
            total_count = data.get("total_count")
            allow_comment = data.get("allow_comment")
        batch = data.get("results") or []
        fresh = 0
        for row in batch:
            rid = row.get("id")
            if rid and rid not in seen:
                seen.add(rid)
                threads.append(row)
                fresh += 1
        cursor = data.get("next_cursor")
        if not cursor or not batch or fresh == 0:
            break
        time.sleep(0.25)
    else:
        truncated = True

    return {
        "clip_id": clip_id,
        "total_count": total_count,
        "allow_comment": allow_comment,
        "pages_walked": pages,
        "threads_pulled": len(threads),
        "truncated": truncated,
        "threads": threads,
    }


def fetch_replies(comment_id: str) -> list[dict[str, Any]]:
    """Fetch a thread's FULL reply list.

    ⛔ THE BUG THIS EXISTS FOR (2026-08-22, caught in production, publicly):
    the comments endpoint embeds only a PREVIEW of each thread's replies -- it
    returns `num_replies: 5` beside a `replies` array holding 1. Deciding
    ANSWERED from that array marks every thread whose first reply came from
    someone other than us as UNANSWERED.

    It reported `timzee65` as a 3.5-month-old unanswered debt. He had in fact
    been answered twice, on 2026-05-09 and 2026-08-02. A reply apologising for
    a silence that never happened was posted to his thread under The
    Scientist's name before the error was found, and had to be deleted.

    So: whenever the preview is short of `num_replies`, go and get the rest.
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    url = REPLIES_API.format(comment_id=comment_id)
    cursor: str | None = None
    for _ in range(20):
        u = url if cursor is None else url + "?cursor=" + urllib.parse.quote(cursor)
        data = fetch(u)
        batch = data.get("replies") or data.get("results") or []
        fresh = 0
        for r in batch:
            rid = r.get("id")
            if rid and rid not in seen:
                seen.add(rid)
                out.append(r)
                fresh += 1
        cursor = data.get("next_cursor")
        if not cursor or not batch or fresh == 0:
            break
        time.sleep(0.2)
    return out


def thread_state(thread: dict[str, Any]) -> dict[str, Any]:
    """ANSWERED iff our user id appears in the thread's replies.

    Reads the FULL reply list, never the embedded preview -- see fetch_replies.
    """
    replies = thread.get("replies") or []
    claimed = thread.get("num_replies") or 0
    if claimed > len(replies):
        try:
            replies = fetch_replies(thread["id"])
        except Exception:  # noqa: BLE001 - fall back to the preview, but say so
            thread["_replies_incomplete"] = True
    ours = [r for r in replies if r.get("user_id") == OUR_USER_ID]
    others = [r for r in replies if r.get("user_id") != OUR_USER_ID]
    return {
        "answered": bool(ours),
        "our_replies": [
            {"content": r.get("content"), "created_at": r.get("created_at")}
            for r in ours
        ],
        "other_replies": [
            {
                "handle": r.get("user_handle"),
                "display_name": r.get("user_display_name"),
                "content": r.get("content"),
                "created_at": r.get("created_at"),
            }
            for r in others
        ],
        "self_authored": thread.get("user_id") == OUR_USER_ID,
        "num_replies": thread.get("num_replies") or len(replies),
        "replies_fetched": len(replies),
        "replies_complete": len(replies) >= (thread.get("num_replies") or 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--clip",
        action="append",
        help="clip id (repeatable); default = every clip the catalogue snapshot says has comments",
    )
    ap.add_argument("--snapshot", default=str(REPO / "data/snapshots/suno/latest.json"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--all-clips",
        action="store_true",
        help="sweep every clip, not only those with comment_count>0",
    )
    args = ap.parse_args()

    titles: dict[str, dict[str, Any]] = {}
    if args.clip:
        clip_ids = list(dict.fromkeys(args.clip))
    else:
        snap = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
        rows = snap["rows"]
        for r in rows:
            titles[r["id"]] = {
                "title": r.get("title"),
                "created_at": r.get("created_at"),
                "play_count": r.get("play_count"),
                "upvote_count": r.get("upvote_count"),
                "comment_count": r.get("comment_count"),
            }
        clip_ids = [
            r["id"]
            for r in rows
            if args.all_clips or (r.get("comment_count") or 0) > 0
        ]

    print("sweeping %d clips" % len(clip_ids), file=sys.stderr)
    clips: list[dict[str, Any]] = []
    for i, cid in enumerate(clip_ids, 1):
        try:
            res = fetch_clip_comments(cid)
        except Exception as exc:  # noqa: BLE001
            print("  [%d/%d] %s FAILED: %s" % (i, len(clip_ids), cid, exc), file=sys.stderr)
            clips.append({"clip_id": cid, "error": str(exc), "threads": []})
            continue
        meta = titles.get(cid, {})
        res["title"] = meta.get("title")
        res["clip_created_at"] = meta.get("created_at")
        res["play_count"] = meta.get("play_count")
        res["upvote_count"] = meta.get("upvote_count")
        res["snapshot_comment_count"] = meta.get("comment_count")
        for t in res["threads"]:
            t["_state"] = thread_state(t)
        res["unanswered"] = sum(
            1
            for t in res["threads"]
            if not t["_state"]["answered"] and not t["_state"]["self_authored"]
        )
        clips.append(res)
        label = (res.get("title") or cid)[:44]
        print(
            "  [%d/%d] %-44s threads=%3d total=%s unanswered=%s"
            % (i, len(clip_ids), label, res["threads_pulled"], res["total_count"], res["unanswered"]),
            file=sys.stderr,
        )

    now = datetime.now(timezone.utc)
    out = {
        "feed": "suno_comments",
        "pulled_at": now.isoformat(),
        "source": API,
        "our_user_id": OUR_USER_ID,
        "our_handle": OUR_HANDLE,
        "clips_swept": len(clips),
        "threads_total": sum(len(c.get("threads", [])) for c in clips),
        "unanswered_total": sum(c.get("unanswered", 0) for c in clips),
        "reconciliation": {
            "clips_with_error": [c["clip_id"] for c in clips if c.get("error")],
            "clips_truncated": [c["clip_id"] for c in clips if c.get("truncated")],
            "count_mismatches": [
                {
                    "clip_id": c["clip_id"],
                    "api_total": c.get("total_count"),
                    "threads_pulled": c.get("threads_pulled"),
                }
                for c in clips
                if c.get("total_count") is not None
                and c.get("threads_pulled") is not None
                and c["total_count"] != c["threads_pulled"]
            ],
        },
        "clips": clips,
    }

    if args.dry_run:
        print(json.dumps({k: v for k, v in out.items() if k != "clips"}, indent=1))
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y-%m-%dT%H-%M-%S")
    path = OUT_DIR / (stamp + ".json")
    payload = json.dumps(out, ensure_ascii=False, indent=1)
    path.write_text(payload, encoding="utf-8")
    (OUT_DIR / "latest.json").write_text(payload, encoding="utf-8")
    print("\nwrote %s" % path, file=sys.stderr)
    print(
        "  %d clips - %d threads - %d UNANSWERED"
        % (out["clips_swept"], out["threads_total"], out["unanswered_total"]),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
