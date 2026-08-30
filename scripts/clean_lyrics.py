#!/usr/bin/env python3
"""clean_lyrics.py — turn Suno lyric sheets into streaming-ready lyrics.

WHY: what Suno holds is a *generator script*, not a lyric sheet. It carries
section headers, EMO tags, production directions and SFX cues — none of which
are sung, and all of which look like nonsense on a lyrics screen. Musixmatch,
Apple and Spotify want only the words a listener hears, with the line and
stanza breaks intact so the lines can be time-synced later.

    python3 scripts/clean_lyrics.py                 # all tracks in the locked plan
    python3 scripts/clean_lyrics.py --keep-parens   # leave (backing vocals) in

WHAT IS STRIPPED, AND WHY EACH:
  * `[Verse 1 - EMO:Composure - Dry close alto - ...]`  section + emotion +
    performance direction. Never sung.
  * `*[a pen set down on wood]*`   SFX cue. Never sung.
  * `[Theme: ...]` / `[SONG FORM: ...]`  Lofn's own pipeline headers.
  * Trailing production sidecars and metadata blocks.

WHAT IS KEPT:
  * `(lonely...)` and other parenthesised lines — these are BACKING VOCALS and
    they ARE sung. Streaming lyrics conventionally keep them. `--strip-parens`
    if a particular track uses parentheses for directions instead.
  * Blank lines between stanzas — they carry the structure a sync editor needs.

⚠️ SUNG vs WRITTEN. This is the sheet sent to the generator, not a transcript of
the render. A model can slur, drop or repeat a line. These are publishable as
lyrics only after a human has heard the track against them; the file marks that
explicitly rather than implying the text is verified.
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

SFX = re.compile(r"\*\[[^\]]*\]\*")           # *[a pen set down on wood]*
BRACKET = re.compile(r"\[[^\]]*\]")           # [Verse 1 - EMO:... ] and friends
ASTERISK = re.compile(r"\*[^*\n]{0,80}\*")    # *stage direction*
ANGLED = re.compile(r"<[^>\n]{0,80}>")
LEADING_LABEL = re.compile(r"^\s*(?:Theme|SONG FORM|Disc_Channel|Production|Style|Exclude)\s*[:：].*$", re.I | re.M)

# Some sheets put PRODUCTION DIRECTIONS in parentheses instead of brackets —
# "(Bass enters — warm, sub-frequency)". Those are not sung, but an ordinary
# backing vocal "(lonely...)" is. Discovered on "I made a kite", which stripped
# only 4% because all of its markup was parenthetical. The tell is production
# vocabulary, not the punctuation.
PROD_WORDS = {
    # instruments / sources
    "bass", "drum", "drums", "synth", "synths", "piano", "guitar", "strings", "horns",
    "vocoder", "choir", "pad", "pads", "arp", "kick", "snare", "hat", "hats", "cymbal",
    # studio processes
    "reverb", "delay", "filter", "distortion", "sidechain", "compression", "eq",
    "granular", "glitch", "texture", "stem", "stems", "mix", "master",
    # arrangement events
    "enters", "enter", "fades", "fade", "drops", "drop", "builds", "build", "swell",
    "riser", "cut", "cuts", "loops", "loop", "sustain", "release",
    # structure / spec
    "verse", "chorus", "bridge", "intro", "outro", "instrumental", "acappella",
    "tempo", "bpm", "octave", "frequency", "sub-frequency", "sub-bass", "hz",
    # hyphenated compounds are ONE token to the word regex, so the head word
    # alone will not match them — "sub-bass" slipped through on that.
    "sub-frequencies", "half-time", "double-time", "lo-fi", "hi-hat",
}
WORD = re.compile(r"[a-z][a-z\-]+")


def is_production_paren(inner: str) -> bool:
    """A parenthetical is a DIRECTION, not a backing vocal, when it names studio
    machinery or an arrangement event.

    Set membership, not one long alternation. The alternation version of this
    silently matched NOTHING because an escaping accident left a literal
    backspace (\x08) at the head of the pattern — and a regex that fails OPEN
    looks exactly like a regex that works. A set cannot be half-broken, and its
    contents are inspectable.
    """
    if set(WORD.findall(inner.lower())) & PROD_WORDS:
        return True
    # descriptive sound-writing with a dash is a direction: "(warm - something broken)"
    return len(inner.split()) > 6 and ("—" in inner or " - " in inner)


def fetch(cid: str, timeout: int = 30) -> dict | None:
    try:
        req = urllib.request.Request(CLIP.format(cid=cid), headers={"user-agent": UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        return {"__error": f"HTTP {e.code}"}
    except Exception as e:  # noqa: BLE001
        return {"__error": f"{type(e).__name__}: {e}"}


def raw_lyrics(clip: dict) -> str:
    md = clip.get("metadata") or {}
    for k in ("prompt", "lyrics"):
        v = md.get(k) or clip.get(k)
        if isinstance(v, str) and len(v) > 40:
            return v
    return ""


def clean(raw: str, strip_parens: bool = False) -> str:
    t = raw.replace("\r\n", "\n").replace("\r", "\n")
    t = SFX.sub("", t)
    t = LEADING_LABEL.sub("", t)
    t = BRACKET.sub("", t)
    t = ANGLED.sub("", t)
    t = ASTERISK.sub("", t)
    if strip_parens:
        t = re.sub(r"\([^)\n]{0,160}\)", "", t)
    else:
        # Keep sung backing vocals; drop parenthetical PRODUCTION directions.
        t = re.sub(
            r"\(([^)\n]{0,160})\)",
            lambda m: "" if is_production_paren(m.group(1)) else m.group(0),
            t,
        )

    out, blank = [], 0
    for ln in t.split("\n"):
        s = ln.strip()
        # a line that was ONLY markup is now empty — drop it rather than leaving a hole
        if not s:
            blank += 1
            if blank == 1 and out:
                out.append("")
            continue
        blank = 0
        # drop orphaned punctuation left behind by a removed tag
        if re.fullmatch(r"[\-–—:;,.·|]+", s):
            continue
        out.append(s)
    while out and not out[-1]:
        out.pop()
    return "\n".join(out)


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="output/distribution/ALBUM_PLAN_LOCKED.json")
    ap.add_argument("--out", default="output/distribution/LYRICS")
    ap.add_argument("--strip-parens", action="store_true", help="also remove (backing vocals)")
    ap.add_argument("--pause", type=float, default=0.4)
    args = ap.parse_args()

    plan = json.loads((REPO / args.plan).read_text(encoding="utf-8"))
    tracks = [("A", t) for t in plan["side_a"]] + [("B", t) for t in plan["side_b"]]
    outdir = REPO / args.out
    for s in ("SIDE_A", "SIDE_B"):
        (outdir / s).mkdir(parents=True, exist_ok=True)

    ILL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
    rows, failed = [], []
    for side, t in tracks:
        clip = fetch(t["id"])
        time.sleep(args.pause)
        raw = raw_lyrics(clip) if clip and "__error" not in clip else ""
        if not raw:
            failed.append(f"{side}{t['position']:02d} {t['title']}")
            rows.append({"side": side, "position": t["position"], "title": t["title"],
                         "status": "NO_LYRICS", "detail": (clip or {}).get("__error", "instrumental or absent")})
            print(f"  {'--':>6}  {side}{t['position']:02d} {t['title'][:44]:46} no lyrics")
            continue
        body = clean(raw, args.strip_parens)
        lines = [l for l in body.split("\n") if l.strip()]
        p = outdir / f"SIDE_{side}" / f"{side}{t['position']:02d} - {ILL.sub('-', t['title'])[:70]}.txt"
        p.write_text(body + "\n", encoding="utf-8")
        rows.append({"side": side, "position": t["position"], "title": t["title"], "status": "OK",
                     "path": str(p.relative_to(REPO)).replace("\\", "/"),
                     "sung_lines": len(lines), "chars": len(body),
                     "raw_chars": len(raw), "stripped_pct": round(100 * (1 - len(body) / len(raw)), 1)})
        print(f"  {len(lines):>4}L  {side}{t['position']:02d} {t['title'][:44]:46} "
              f"{len(raw):>5} -> {len(body):>5} chars ({round(100*(1-len(body)/len(raw)))}% markup)")

    (outdir / "_index.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": CLIP.format(cid="{clip_id}"),
        "caveat": "This is the lyric sheet sent to the generator, NOT a transcript of the render. "
                  "A model can slur, drop or repeat lines. Verify against the audio before publishing.",
        "parentheses": "kept as backing vocals" if not args.strip_parens else "stripped",
        "tracks": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    ok = sum(1 for r in rows if r["status"] == "OK")
    print(f"\n  {ok}/{len(rows)} written to {args.out}/")
    if failed:
        print("  no lyrics: " + ", ".join(failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
