#!/usr/bin/env python3
"""screen_explicit.py — per-track explicit-language screen for a distribution set.

WHY: DistroKid requires a per-track explicit flag, and getting it wrong can hold
a release. Guessing from memory is not a record. This pulls the ACTUAL lyrics
Suno holds for each clip and screens them, so the answer is evidence with a
timestamp rather than a recollection.

    python3 scripts/screen_explicit.py                  # screens the locked plan
    python3 scripts/screen_explicit.py --show-context   # print the matching line

THE ENDPOINT: `studio-api.prod.suno.com/api/clip/{id}` returns clip JSON —
including `metadata.prompt`, which is the lyric sheet — with NO auth, verified
2026-08-08. (The `/api/feed/` variants all 401; only `/api/clip/` is open.)

WHAT THIS IS AND IS NOT. It is a keyword screen over the lyric text, which is
the right first pass and catches every ordinary case. It is NOT a judgement
about tone, theme or intensity — DistroKid's flag is about profanity and
sexual content, not how dark a record is. Two failure modes stay open and are
reported rather than hidden:
  * SUNG vs WRITTEN — the lyric sheet is what was sent to the generator. A
    renderer can slur, drop or mangle a word. A clean sheet is strong evidence,
    not proof of a clean render.
  * EUPHEMISM — no word list catches what it does not contain. Sexual content
    with no profanity in it will screen clean and still be explicit.
A human ear is the final authority; this narrows what that ear has to check.
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

REPO = Path(__file__).resolve().parent.parent
CLIP = "https://studio-api.prod.suno.com/api/clip/{cid}"
UA = "Mozilla/5.0 (compatible; lofn-analytics-ingest/1.0; +https://www.lofn.ai)"

# Strong = DistroKid-explicit on its own.
STRONG = re.compile(
    r"\b(fuck\w*|motherfuck\w*|shit\w*|bullshit|cunt\w*|nigg\w+|faggot\w*|"
    r"whore\w*|slut\w*|cock|pussy|dick(?:head)?|bitch\w*|prick|twat|wank\w*)\b",
    re.I,
)
# Mild = surfaced for a human call, NOT auto-flagged. "hell" and "damn" are not
# explicit in any mainstream store's sense, and auto-flagging them would put a
# needless E on half a catalogue.
MILD = re.compile(r"\b(damn|goddamn|hell|ass|asshole|bastard|piss\w*|crap|screw(?:ed)?)\b", re.I)
# Structural markers and production directions are not sung; strip before screening.
SECTION = re.compile(r"^\s*[\[\(].*?[\]\)]\s*$", re.M)
INLINE_TAG = re.compile(r"\[[^\]]{0,80}\]")


def fetch(cid: str, timeout: int = 30) -> dict | None:
    req = urllib.request.Request(CLIP.format(cid=cid), headers={"user-agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        return {"__error": f"HTTP {e.code}"}
    except Exception as e:  # noqa: BLE001
        return {"__error": f"{type(e).__name__}: {e}"}


def lyrics_of(clip: dict) -> str:
    md = clip.get("metadata") or {}
    for k in ("prompt", "lyrics", "gpt_description_prompt"):
        v = md.get(k) or clip.get(k)
        if isinstance(v, str) and len(v) > 40:
            return v
    return ""


def sung_only(lyric: str) -> str:
    """Drop bracketed section headers and production cues — they are not sung."""
    body = SECTION.sub("", lyric)
    return INLINE_TAG.sub(" ", body)


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="output/distribution/ALBUM_PLAN_LOCKED.json")
    ap.add_argument("--out", default="output/distribution/EXPLICIT_SCREEN.json")
    ap.add_argument("--pause", type=float, default=0.4)
    ap.add_argument("--show-context", action="store_true")
    args = ap.parse_args()

    plan = json.loads((REPO / args.plan).read_text(encoding="utf-8"))
    tracks = [("A", t) for t in plan["side_a"]] + [("B", t) for t in plan["side_b"]]

    rows, explicit, unknown = [], [], []
    for side, t in tracks:
        clip = fetch(t["id"])
        time.sleep(args.pause)
        if not clip or "__error" in (clip or {}):
            rows.append({"side": side, "position": t["position"], "title": t["title"],
                         "verdict": "UNKNOWN", "error": (clip or {}).get("__error", "no response")})
            unknown.append(t["title"])
            print(f"  {'UNKNOWN':8} {side}{t['position']:02d} {t['title'][:44]:46} {(clip or {}).get('__error','')}")
            continue
        lyric = lyrics_of(clip)
        if not lyric:
            rows.append({"side": side, "position": t["position"], "title": t["title"],
                         "verdict": "NO_LYRICS", "note": "instrumental, or lyrics absent from the clip payload"})
            unknown.append(t["title"])
            print(f"  {'NO-LYRIC':8} {side}{t['position']:02d} {t['title'][:44]:46}")
            continue
        body = sung_only(lyric)
        strong = sorted({m.group(0).lower() for m in STRONG.finditer(body)})
        mild = sorted({m.group(0).lower() for m in MILD.finditer(body)})
        verdict = "EXPLICIT" if strong else ("REVIEW" if mild else "CLEAN")
        if strong:
            explicit.append(f"{side}{t['position']:02d} {t['title']}")
        ctx = []
        if args.show_context and strong:
            for ln in body.splitlines():
                if STRONG.search(ln):
                    ctx.append(ln.strip()[:110])
        rows.append({"side": side, "position": t["position"], "title": t["title"], "verdict": verdict,
                     "strong": strong, "mild": mild, "lyric_chars": len(lyric), "context": ctx})
        print(f"  {verdict:8} {side}{t['position']:02d} {t['title'][:44]:46} {strong or mild or ''}")
        for c in ctx:
            print(f"             > {c}")

    out = {
        "screened_at": datetime.now(timezone.utc).isoformat(),
        "source": CLIP.format(cid="{clip_id}"),
        "method": "keyword screen over Suno's stored lyric sheet, section headers and production tags removed",
        "limits": [
            "SUNG vs WRITTEN: the sheet is what was sent to the generator; a render can slur or drop a word.",
            "EUPHEMISM: sexual content without profanity screens CLEAN and may still warrant the flag.",
            "MILD hits are surfaced for a human call and are NOT auto-flagged — 'hell'/'damn' are not store-explicit.",
        ],
        "counts": {"total": len(rows), "explicit": len(explicit), "unknown": len(unknown),
                   "review": sum(1 for r in rows if r["verdict"] == "REVIEW"),
                   "clean": sum(1 for r in rows if r["verdict"] == "CLEAN")},
        "explicit_tracks": explicit,
        "tracks": rows,
    }
    (REPO / args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    c = out["counts"]
    print(f"\n  {c['clean']} clean · {c['review']} review · {c['explicit']} EXPLICIT · {c['unknown']} unknown")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
