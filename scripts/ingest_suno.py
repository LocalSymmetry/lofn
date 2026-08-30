#!/usr/bin/env python3
"""ingest_suno.py — pull the live Suno catalogue into a machine-readable snapshot.

WHY: every engagement number Lofn Analytics shows about music comes from here.
Before this script the only record of it was prose
(`output/suno_catalog_analysis_2026-07-27.md`), which an app cannot read.

The profile endpoint is PUBLIC — no cookie, no key. That matters: the pull is
reproducible from any machine and does not depend on a logged-in browser.

    python3 scripts/ingest_suno.py                      # writes data/snapshots/suno/
    python3 scripts/ingest_suno.py --handle someone     # a competitor scan
    python3 scripts/ingest_suno.py --dry-run            # print, write nothing

TWO INVARIANTS, both learned the hard way on 2026-07-27:

  1. PAGES OVERLAP. A naive page-walk returned 242 rows for a 224-clip catalogue
     and double-counted plays to 731K. Rows are keyed by clip id in a dict.
  2. RECONCILE OR SAY SO. The summed per-clip plays are checked against the
     profile header's own total. A mismatch is REPORTED in the snapshot rather
     than silently accepted — the check is what caught the bug above. Small
     drift (a play landing mid-pull) is expected and recorded, not hidden.

AUTONOMY.md: read-only over the public web, writes locally. Renders nothing,
publishes nothing, spends nothing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
API = "https://studio-api.prod.suno.com/api/profiles/{handle}"
# Both sort params are required; `?page=N` alone returns HTTP 422.
QUERY = "?page={page}&playlists_sort_by=upvote_count&clips_sort_by=upvote_count"
UA = "Mozilla/5.0 (compatible; lofn-analytics-ingest/1.0; +https://www.lofn.ai)"

SECTION_RE = re.compile(r"^\[.*\]$")
SFX_RE = re.compile(r"^(\*.*\*|\(.*\))$")
EMO_RE = re.compile(r"EMO\s*:\s*([A-Za-z\- ]+)")
WORD_SPLIT = re.compile(r"\s+")
NORMALISE = re.compile(r"[^a-z0-9' ]")


def fetch_page(handle: str, page: int, timeout: int = 30) -> dict[str, Any]:
    url = API.format(handle=handle) + QUERY.format(page=page)
    req = urllib.request.Request(url, headers={"accept": "application/json", "user-agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def lyric_features(prompt: str | None) -> dict[str, Any]:
    """Derive the form measures the 2026-07-27 analysis found predictive.

    `words_per_line` was the strongest single correlate of like rate (r = -0.30)
    and `repeated_line_ratio` was monotonically positive, so both are computed
    here once rather than re-derived per view.

    A "sung line" excludes bracket section headers and *sfx* / (parenthetical)
    cues — counting those inflates line counts and deflates words-per-line.
    """
    if not prompt:
        return {}

    raw = [line.strip() for line in prompt.splitlines()]
    sung = [s for s in raw if s and not SECTION_RE.match(s) and not SFX_RE.match(s)]
    headers = [s for s in raw if SECTION_RE.match(s)]

    word_counts = [len([w for w in WORD_SPLIT.split(s) if w]) for s in sung]
    total_words = sum(word_counts)

    norm = [NORMALISE.sub("", s.lower()).strip() for s in sung]
    norm = [re.sub(r"\s+", " ", s) for s in norm if s]
    counts: dict[str, int] = {}
    for s in norm:
        counts[s] = counts.get(s, 0) + 1
    repeated = sum(1 for s in norm if counts[s] > 1)

    emo = [m.group(1).strip().lower() for m in EMO_RE.finditer(prompt)]
    section_names = {
        re.split(r"[–—\-]", h.strip("[]").lower())[0].strip() for h in headers
    }

    return {
        "sung_lines": len(sung),
        "words_per_line": round(total_words / len(sung), 3) if sung else None,
        "repeated_line_ratio": round(repeated / len(norm), 4) if norm else None,
        "max_line_repeats": max(counts.values()) if counts else 0,
        "section_count": len(headers),
        "unique_sections": len(section_names),
        "emo_tag_count": len(emo),
        "emo_tags": sorted(set(emo)),
        "lyric_chars": len(prompt),
        "has_chorus": bool(re.search(r"\[\s*chorus", prompt, re.I)),
    }


def normalise_clip(clip: dict[str, Any]) -> dict[str, Any]:
    meta = clip.get("metadata") or {}
    persona = clip.get("persona") or {}
    plays = clip.get("play_count") or 0
    upvotes = clip.get("upvote_count") or 0

    row = {
        "id": clip.get("id"),
        "title": clip.get("title"),
        "play_count": plays,
        "upvote_count": upvotes,
        # Plays measure distribution; likes-per-play measure response. On a
        # catalogue with one placement-driven outlier the second is the only
        # usable signal — so it is stored, not left to each view to rediscover.
        "like_rate": round(upvotes / plays, 5) if plays else None,
        "created_at": clip.get("created_at"),
        "model_name": clip.get("model_name"),
        "major_model_version": clip.get("major_model_version"),
        "is_public": clip.get("is_public"),
        "is_trashed": clip.get("is_trashed"),
        "is_pinned": clip.get("is_pinned"),
        "persona": persona.get("name") or persona.get("id") if persona else None,
        "duration": meta.get("duration"),
        # `tags` is the full Suno STYLE PROMPT, not a keyword list. It is the
        # only place our own style language is joined to engagement.
        "tags": meta.get("tags"),
        "negative_tags": meta.get("negative_tags"),
        "gpt_description_prompt": meta.get("gpt_description_prompt"),
        "type": meta.get("type"),
        "has_vocal": meta.get("has_vocal"),
        "is_remix": meta.get("is_remix"),
        # HOSTING FIELDS. Added 2026-08-02 after the Appendix C benchmark pull
        # found these were the ONLY public fields bearing on hosting behaviour
        # and that this function was throwing all three away — every
        # comment-surface number in vault/BENCHMARKS.md needed a bespoke raw
        # walk to recover them. Appendix C's six numbers are mostly about
        # hosting, so a pull that drops them cannot serve its own purpose.
        "comment_count": clip.get("comment_count"),
        "allow_comments": clip.get("allow_comments"),
        "can_remix": meta.get("can_remix"),
    }
    # Raw lyrics are deliberately NOT stored: they already live in
    # output/past_music/ and output/songs/, and duplicating ~500KB of them into
    # every snapshot makes the diff unreadable. The derived measures are kept.
    row.update(lyric_features(meta.get("prompt")))
    return row


CLIP_API = "https://studio-api.prod.suno.com/api/clip/{clip_id}"
# One clip id (or a full suno.com/song/<id> URL) per line; `#` comments allowed.
PRIVATE_MANIFEST = REPO / "data" / "snapshots" / "suno" / "private_clips.txt"
_ID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def read_private_manifest(path: Path = PRIVATE_MANIFEST) -> list[str]:
    """Clip ids the PROFILE ENDPOINT CANNOT SEE.

    WHY THIS EXISTS (2026-08-07): the profile feed returns only clips with
    `is_public: true`. "The Costume" (created 2026-08-02) was absent from four
    consecutive snapshots for that reason alone — the walk was correct, the
    catalogue was not. Private and unlisted work is still our work, and a
    snapshot that silently omits it is a partial archive that reads as a total
    one. The per-clip endpoint serves those clips by id without a cookie, so
    the gap is closeable as long as we record the ids.
    """
    if not path.exists():
        return []
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        m = _ID_RE.search(line)
        if m:
            ids.append(m.group(0))
        else:
            print(f"  [private] unparseable manifest line: {line!r}", file=sys.stderr)
    return ids


def fetch_clip(clip_id: str, timeout: int = 30) -> dict[str, Any]:
    req = urllib.request.Request(
        CLIP_API.format(clip_id=clip_id),
        headers={"accept": "application/json", "user-agent": UA},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def pull(handle: str, max_pages: int = 60, pause: float = 0.35) -> dict[str, Any]:
    by_id: dict[str, dict[str, Any]] = {}
    header: dict[str, Any] | None = None
    stats: dict[str, Any] | None = None
    # The endpoint is 1-indexed in effect: `page=0` and `page=1` BOTH return
    # `current_page: 1`. Starting at 0 makes the first two requests identical,
    # which reads as "no new rows" and ends the walk after 21 clips.
    page = 1
    pages_walked = 0
    barren = 0

    while page < max_pages:
        try:
            payload = fetch_page(handle, page)
        except urllib.error.HTTPError as err:
            if page == 0:
                raise
            print(f"  page {page}: HTTP {err.code} — stopping", file=sys.stderr)
            break

        if header is None:
            header = {
                "user_id": payload.get("user_id"),
                "display_name": payload.get("display_name"),
                "handle": payload.get("handle"),
                "profile_description": payload.get("profile_description"),
                "num_total_clips": payload.get("num_total_clips"),
                "is_verified": payload.get("is_verified"),
                "personas": len(payload.get("personas") or []),
                "playlists": len(payload.get("playlists") or []),
            }
            stats = payload.get("stats") or {}

        clips = payload.get("clips") or []
        if not clips:
            break

        before = len(by_id)
        for clip in clips:
            cid = clip.get("id")
            if cid:
                by_id[cid] = normalise_clip(clip)

        pages_walked += 1
        # Pages overlap, so a single page that adds nothing is normal. Only a
        # RUN of barren pages means the walk has converged — and if the header
        # says how many clips exist, that is the real stopping condition.
        barren = barren + 1 if len(by_id) == before else 0
        total = (header or {}).get("num_total_clips")
        if barren >= 3 or (total and len(by_id) >= total):
            break

        page += 1
        time.sleep(pause)

    # ── PUBLIC ROWS ONLY from here to the end of reconciliation ──────────────
    # The header's totals count public clips, so private rows must NEVER enter
    # the reconciliation sums or every future snapshot false-fails.
    rows = sorted(by_id.values(), key=lambda r: r.get("created_at") or "")
    for r in rows:
        r["source"] = "profile_feed"
    summed_plays = sum(r["play_count"] for r in rows)
    summed_upvotes = sum(r["upvote_count"] for r in rows)
    header_plays = (stats or {}).get("play_count__sum")
    header_upvotes = (stats or {}).get("upvote_count__sum")

    reconciliation = {
        "summed_plays": summed_plays,
        "header_plays": header_plays,
        "plays_delta": (summed_plays - header_plays) if header_plays is not None else None,
        "summed_upvotes": summed_upvotes,
        "header_upvotes": header_upvotes,
        "upvotes_delta": (summed_upvotes - header_upvotes) if header_upvotes is not None else None,
        "rows_pulled": len(rows),
        "header_clip_count": (header or {}).get("num_total_clips"),
        "pages_walked": pages_walked,
    }
    # A handful of plays can land mid-pull; that is drift, not corruption. A
    # large gap means the dedupe or the pagination is wrong and the snapshot
    # must not be trusted.
    delta = reconciliation["plays_delta"]
    plays_ok = delta is not None and abs(delta) <= max(5, summed_plays * 0.001)

    # CLIP COUNT IS A SECOND, INDEPENDENT CHECK — added 2026-08-02.
    # `rows_pulled` and `header_clip_count` were both computed here and then
    # never read, so `ok` rested on the play sum alone. That misses a real case:
    # a walk cut short by `max_pages` or a mid-walk error which drops only
    # ZERO-PLAY clips still sums its plays perfectly and was reported as good.
    # Those clips are exactly the ones a MEDIAN like rate needs — dropping them
    # biases the median upward, which is the number Appendix C benchmarks on.
    declared = reconciliation["header_clip_count"]
    rows_ok = declared is None or len(rows) >= declared
    truncated = pages_walked >= max_pages and not rows_ok

    reconciliation["plays_ok"] = plays_ok
    reconciliation["rows_ok"] = rows_ok
    reconciliation["truncated"] = truncated
    reconciliation["ok"] = bool(plays_ok and rows_ok and not truncated)
    if not rows_ok:
        reconciliation["why_not_ok"] = (
            f"pulled {len(rows)} rows against a declared {declared}. The play sum "
            "can still reconcile when the missing clips have zero plays, so this "
            "check is not redundant with plays_delta."
        )

    # ── PRIVATE / UNLISTED CLIPS — appended AFTER reconciliation, by design ──
    private_rows: list[dict[str, Any]] = []
    manifest_ids = read_private_manifest()
    for cid in manifest_ids:
        if cid in by_id:
            # It went public since it was listed. The feed's copy is canonical;
            # leave the manifest entry in place (harmless) rather than guessing.
            continue
        try:
            row = normalise_clip(fetch_clip(cid))
        except Exception as err:  # noqa: BLE001 - a dead id must not kill the pull
            print(f"  [private] {cid}: {type(err).__name__} {err} — skipped", file=sys.stderr)
            continue
        row["source"] = "private_manifest"
        private_rows.append(row)
        time.sleep(pause)

    all_rows = sorted(rows + private_rows, key=lambda r: r.get("created_at") or "")
    private_stats = {
        "manifest_ids": len(manifest_ids),
        "fetched": len(private_rows),
        "already_public": sum(1 for c in manifest_ids if c in by_id),
        "plays": sum(r["play_count"] for r in private_rows),
        "upvotes": sum(r["upvote_count"] for r in private_rows),
        "note": (
            "Rows with source='private_manifest' are EXCLUDED from reconciliation "
            "and from header stats, which count public clips only. Exclude them "
            "from public-catalogue engagement medians too — a private clip's play "
            "count is not a measure of public reach."
        ),
    }

    return {
        "feed": "suno",
        "pulled_at": datetime.now(timezone.utc).isoformat(),
        "source": API.format(handle=handle) + QUERY.format(page="N"),
        "handle": handle,
        "header": header,
        "stats": stats,
        "reconciliation": reconciliation,
        "private": private_stats,
        "row_count": len(all_rows),
        "public_row_count": len(rows),
        "private_row_count": len(private_rows),
        "rows": all_rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Pull a Suno profile into a snapshot.")
    ap.add_argument("--handle", default="localsymmetry")
    ap.add_argument("--out", default=None, help="snapshot directory (default data/snapshots/<feed>)")
    ap.add_argument("--feed", default=None, help="feed name; defaults to 'suno' for our own handle, 'competitors' otherwise")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    feed = args.feed or ("suno" if args.handle == "localsymmetry" else "competitors")
    print(f"[suno] pulling @{args.handle} …", file=sys.stderr)
    snapshot = pull(args.handle)
    snapshot["feed"] = feed

    rec = snapshot["reconciliation"]
    print(
        f"[suno] {rec['rows_pulled']} public clips over {rec['pages_walked']} pages | "
        f"plays {rec['summed_plays']:,} vs header {rec['header_plays']:,} "
        f"(delta {rec['plays_delta']:+,}) | reconciles={rec['ok']}",
        file=sys.stderr,
    )
    priv = snapshot.get("private") or {}
    if priv.get("manifest_ids"):
        print(
            f"[suno] + {priv['fetched']} private/unlisted from manifest "
            f"({priv['already_public']} since made public) | "
            f"total rows {snapshot['row_count']}",
            file=sys.stderr,
        )
    if not rec["ok"]:
        print("[suno] WARNING: reconciliation FAILED — do not trust this snapshot", file=sys.stderr)

    if args.dry_run:
        print(json.dumps({k: v for k, v in snapshot.items() if k != "rows"}, indent=1))
        return 0

    out_dir = Path(args.out) if args.out else REPO / "data" / "snapshots" / feed
    if feed == "competitors":
        out_dir = out_dir / args.handle
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    text = json.dumps(snapshot, indent=1, ensure_ascii=False)
    (out_dir / f"{stamp}.json").write_text(text, encoding="utf-8")
    (out_dir / "latest.json").write_text(text, encoding="utf-8")
    print(f"[suno] wrote {out_dir / (stamp + '.json')} ({len(text) / 1024:.0f} KB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
