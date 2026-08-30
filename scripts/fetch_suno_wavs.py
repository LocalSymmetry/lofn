#!/usr/bin/env python3
"""fetch_suno_wavs.py — pull the master WAVs for our Suno catalogue to local disk.

WHY: distribution needs the lossless source, not the profile page. Suno serves
`https://cdn1.suno.ai/{clip_id}.wav` PUBLICLY — no cookie, no key, same as the
profile endpoint `ingest_suno.py` already relies on. Verified 2026-08-02 by
reading the RIFF header of two pulled files: **48 kHz / 16-bit stereo**
(192,000 B/s), not the 44.1k the byte sizes first suggested. That matters
downstream — Mixea returns 24-bit/48k, so the sample rate already matches and
only the bit depth changes. This is Suno's own "Download > WAV" payload,
reachable without a browser session.

    python3 scripts/fetch_suno_wavs.py                 # all clips in latest.json
    python3 scripts/fetch_suno_wavs.py --ids a,b,c     # just these clip ids
    python3 scripts/fetch_suno_wavs.py --limit 3       # smoke test
    python3 scripts/fetch_suno_wavs.py --dry-run       # HEAD only, no bytes

THREE INVARIANTS:

  1. IDEMPOTENT. A file already on disk at the byte length the CDN reports is
     skipped. Re-running after an interruption resumes; it never re-pays for
     what is already down.
  2. VERIFY THE LENGTH. A truncated download is worse than a missing one
     because it looks finished. Every file is checked against Content-Length
     and written to `.part` first, so a killed process can never leave a
     half-file at the real name.
  3. THE MANIFEST IS THE RECORD. Ids, titles, byte counts and sha256 go to
     `manifest.json`. Titles collide and contain characters Windows forbids;
     the clip id is the only durable key, so the id is IN the filename.

AUTONOMY.md: reads the public web, writes locally. Renders nothing, publishes
nothing, spends nothing.
"""

from __future__ import annotations

import argparse
import hashlib
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
CDN = "https://cdn1.suno.ai/{cid}.wav"
UA = "Mozilla/5.0 (compatible; lofn-analytics-ingest/1.0; +https://www.lofn.ai)"

# Windows forbids <>:"/\|?* and trailing dots/spaces. Suno titles carry colons
# ("Dear Scientist:"), slashes, and quotes, so sanitising is not optional.
ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
DOTS = re.compile(r"[. ]+$")


def safe_stem(title: str | None, cid: str) -> str:
    base = ILLEGAL.sub("-", (title or "untitled").strip())
    base = DOTS.sub("", base).strip() or "untitled"
    return f"{base[:80]}__{cid[:8]}"


def head(cid: str, timeout: int = 30) -> tuple[int, int | None]:
    req = urllib.request.Request(CDN.format(cid=cid), headers={"user-agent": UA}, method="HEAD")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        length = resp.headers.get("content-length")
        return resp.status, int(length) if length else None


def download(cid: str, dest: Path, expected: int | None, timeout: int = 300) -> dict[str, Any]:
    """Stream to `.part`, hash on the way, rename only on a verified length."""
    part = dest.with_suffix(dest.suffix + ".part")
    sha = hashlib.sha256()
    written = 0
    req = urllib.request.Request(CDN.format(cid=cid), headers={"user-agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp, part.open("wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            sha.update(chunk)
            written += len(chunk)

    if expected is not None and written != expected:
        part.unlink(missing_ok=True)
        raise OSError(f"length mismatch: got {written:,}, expected {expected:,}")

    part.replace(dest)
    return {"bytes": written, "sha256": sha.hexdigest()}


def write_manifest(path: Path, entries: dict[str, dict[str, Any]], snapshot: str) -> None:
    path.write_text(
        json.dumps(
            {
                "source": CDN.format(cid="{clip_id}"),
                "snapshot": snapshot,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                # A 403 here means the CDN holds no .wav object for that clip.
                # The authenticated endpoint that mints one,
                # studio-api.prod.suno.com/api/gen/{id}/wav_file/, returns 401
                # without a session — so these are recoverable, but only by a
                # logged-in caller. See fetch_suno_wavs_auth.py.
                "files": sorted(entries.values(), key=lambda e: e.get("created_at") or ""),
            },
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> int:
    # Windows consoles default to cp1252. Titles carry U+2011 non-breaking
    # hyphens ("3-3-2 Open"), curly quotes and emoji, and a bare print() of one
    # raises UnicodeEncodeError and KILLS the run mid-catalogue — which is how
    # the first pass stopped at 22 of 210 while still reporting success through
    # a pipe. Progress output must never be able to end the download.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Fetch Suno master WAVs to local disk.")
    ap.add_argument("--snapshot", default="data/snapshots/suno/latest.json")
    ap.add_argument("--out", default="output/distribution/wav_masters")
    ap.add_argument("--ids", default=None, help="comma-separated clip ids (default: all in snapshot)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pause", type=float, default=0.5, help="seconds between clips")
    ap.add_argument("--dry-run", action="store_true", help="HEAD each clip, download nothing")
    args = ap.parse_args()

    snap = json.loads((REPO / args.snapshot).read_text(encoding="utf-8"))
    rows = snap["rows"]
    if args.ids:
        wanted = {c.strip() for c in args.ids.split(",") if c.strip()}
        rows = [r for r in rows if r["id"] in wanted]
    if args.limit:
        rows = rows[: args.limit]

    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    # Resume across runs: a prior manifest keeps the hashes we already paid for.
    entries: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = {e["id"]: e for e in prior.get("files", [])}

    ok = skipped = failed = 0
    total_bytes = 0

    for i, row in enumerate(rows, 1):
        cid, title = row["id"], row.get("title")
        dest = out_dir / f"{safe_stem(title, cid)}.wav"

        try:
            status, expected = head(cid)
        except urllib.error.HTTPError as err:
            print(f"[{i:3}/{len(rows)}] HTTP {err.code}  {title}", file=sys.stderr)
            # NEVER let a failed HEAD clobber a known-good entry. The HEAD runs
            # BEFORE the already-downloaded check, so a clip whose WAV is safely
            # on disk but which 403s on a later pass would otherwise have its
            # provenance replaced by an error. That silently cost 37 of 165
            # entries on 2026-08-07 — the audio survived, the record did not.
            if not entries.get(cid, {}).get("sha256"):
                entries[cid] = {"id": cid, "title": title, "error": f"HTTP {err.code}"}
            failed += 1
            continue
        except Exception as err:  # noqa: BLE001 — network shape varies; record and move on
            print(f"[{i:3}/{len(rows)}] ERR {type(err).__name__}  {title}", file=sys.stderr)
            if not entries.get(cid, {}).get("sha256"):
                entries[cid] = {"id": cid, "title": title, "error": f"{type(err).__name__}: {err}"}
            failed += 1
            continue

        if args.dry_run:
            print(f"[{i:3}/{len(rows)}] {status} {(expected or 0)/1e6:8.1f} MB  {title}")
            total_bytes += expected or 0
            continue

        if dest.exists() and expected is not None and dest.stat().st_size == expected:
            entry = entries.get(cid, {})
            if entry.get("sha256"):
                print(f"[{i:3}/{len(rows)}] skip (have)  {title}")
                skipped += 1
                total_bytes += expected
                continue

        try:
            result = download(cid, dest, expected)
        except Exception as err:  # noqa: BLE001
            print(f"[{i:3}/{len(rows)}] FAIL {type(err).__name__}: {err}  {title}", file=sys.stderr)
            entries[cid] = {"id": cid, "title": title, "error": f"{type(err).__name__}: {err}"}
            failed += 1
            continue

        entries[cid] = {
            "id": cid,
            "title": title,
            "path": str(dest.relative_to(REPO)).replace("\\", "/"),
            "created_at": row.get("created_at"),
            "duration": row.get("duration"),
            "play_count": row.get("play_count"),
            "upvote_count": row.get("upvote_count"),
            **result,
        }
        ok += 1
        total_bytes += result["bytes"]
        print(f"[{i:3}/{len(rows)}] ok  {result['bytes']/1e6:7.1f} MB  {title}")
        write_manifest(manifest_path, entries, args.snapshot)
        time.sleep(args.pause)

    # Flush once more at the end. Writing only inside the success branch loses
    # every error that happens after the last successful clip — a run whose
    # tail is all 403s would report no failures at all in the manifest.
    write_manifest(manifest_path, entries, args.snapshot)

    print(
        f"\n[wav] {ok} downloaded, {skipped} already had, {failed} failed | "
        f"{total_bytes/1e9:.2f} GB across {len(rows)} clips",
        file=sys.stderr,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
