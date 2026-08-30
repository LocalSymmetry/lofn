#!/usr/bin/env python3
"""build_catalog.py — unify every creative work into one table the apps can read.

    python3 scripts/build_catalog.py            # -> data/catalog/works.json
    python3 scripts/build_catalog.py --stats    # coverage report, writes nothing

INPUT CORPORA (all local, no network)
    output/past_music/*.json                  468  music   prompt records
    output/music/*.json                        70  music   prompt records (3 shapes!)
    data/archive/music_index_cache_recovered   1487 music  recovered 2025-06..08
    output/songs/*.md                         132  music   packaged song files
    output/mp3/*.mp3                           66  music   rendered audio
    output/past_video/*.json                  210  video   shot/medium records
    output/images/**/*.json                   ~299 image   render receipts
    data/snapshots/suno/latest.json           210  ENGAGEMENT (the only source)

THE JOIN, MEASURED — do not assume better than this:
    mp3 title  -> Suno clip     90.9%   STRONG. Renders are the published songs.
    prompt     -> mp3           ~41%    WEAK. Most prompts were never rendered.
    prompt     -> Suno clip     1-5%    Thousands of drafts; 210 published.

    Consequence the apps must respect: "what did this SONG earn" is reliable;
    "what did this PROMPT earn" is only ~40% attributable. Never present a
    prompt-level engagement claim as if it covered the corpus.

NORMALIZATION RULES (each one is a bug that was actually hit)
    1. Basenames use U+F03A (private-use colon) where the timestamp has ':'.
    2. output/music has a field-swap on 38/70 records: when title is absurdly
       long the fields shifted by one.
    3. Three timestamp formats across corpora.
    4. `competition` is bool in one corpus and the string 'True' in others.
    5. EMO tags match bare `EMO:`, not `\\[EMO:` (4/468 vs 468/468).
    6. mp3 '(1)' duplicate suffixes.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "catalog" / "works.json"

# U+F03A is a private-use codepoint Windows substituted for ':' in filenames.
PUA_COLON = ""
EMO_RE = re.compile(r"EMO\s*:\s*([A-Za-z\- ]+)")
SECTION_RE = re.compile(r"^\[[^\]\n]{2,90}\]$", re.M)
BPM_RE = re.compile(r"(\d{2,3})\s*BPM", re.I)
KEY_RE = re.compile(r"\b([A-G](?:#|b)?)\s+(major|minor)\b", re.I)
VOCAL_RE = re.compile(
    r"\b(female|male|falsetto|mezzo|soprano|baritone|tenor|alto|whisper\w*|spoken|choir|vocal\w*)\b",
    re.I,
)
MP3_DUP_RE = re.compile(r"\s*\(\d+\)$")


def norm_title(s: str | None) -> str:
    """Aggressive title key for cross-corpus joins."""
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"\(feat\.?[^)]*\)", " ", s)
    s = MP3_DUP_RE.sub("", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def norm_ts(v: Any) -> str | None:
    """Three formats in the wild -> one ISO date string (UTC-naive)."""
    if not v:
        return None
    s = str(v).replace(PUA_COLON, ":")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y%m%d_%H%M%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(s.rstrip("Z"), fmt).isoformat()
        except ValueError:
            continue
    m = re.match(r"(\d{4}-\d{2}-\d{2})", s)
    return m.group(1) if m else None


def as_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1")
    return None


def lyric_measures(lyrics: str | None) -> dict[str, Any]:
    if not lyrics:
        return {}
    raw = [ln.strip() for ln in lyrics.splitlines()]
    sung = [
        s for s in raw
        if s and not re.fullmatch(r"\[.*\]", s) and not re.fullmatch(r"(\*.*\*|\(.*\))", s)
    ]
    if not sung:
        return {}
    words = [len([w for w in s.split() if w]) for s in sung]
    keys = [re.sub(r"[^a-z0-9' ]", "", s.lower()).strip() for s in sung]
    keys = [re.sub(r"\s+", " ", k) for k in keys if k]
    counts = Counter(keys)
    wpl = round(sum(words) / len(sung), 3)
    return {
        "sung_lines": len(sung),
        # Some records store the whole lyric on one physical line; the resulting
        # words_per_line (up to 972) is an artefact, not a measurement.
        "words_per_line": wpl if wpl <= 20 else None,
        "words_per_line_suspect": wpl > 20,
        "repeated_line_ratio": round(sum(1 for k in keys if counts[k] > 1) / len(keys), 4) if keys else None,
        "section_count": len(SECTION_RE.findall(lyrics)),
        "emo_tags": sorted({m.group(1).strip().lower() for m in EMO_RE.finditer(lyrics)}),
        "emo_tag_count": len(EMO_RE.findall(lyrics)),
        "lyric_chars": len(lyrics),
    }


def style_measures(style: str | None) -> dict[str, Any]:
    if not style:
        return {}
    bpm = BPM_RE.search(style)
    key = KEY_RE.search(style)
    return {
        "style_chars": len(style),
        "bpm": int(bpm.group(1)) if bpm else None,
        "musical_key": f"{key.group(1)} {key.group(2).lower()}" if key else None,
        "vocal_descriptors": sorted({m.group(1).lower() for m in VOCAL_RE.finditer(style)}),
    }


def make_row(**kw) -> dict[str, Any]:
    row = {
        "work_id": None, "modality": None, "source_corpus": None,
        "title": None, "created_at": None, "model": None,
        "style_text": None, "lyrics": None, "asset_path": None,
        "personality": None, "competition": None, "variant": None,
    }
    row.update(kw)
    row.update(lyric_measures(row.get("lyrics")))
    row.update(style_measures(row.get("style_text")))
    return row


def load_music_json(path: str, corpus: str) -> dict | None:
    try:
        d = json.load(open(path, encoding="utf-8-sig"))
    except Exception:
        return None
    if not isinstance(d, dict) or "title" not in d:
        return None  # the panel/axes and pair-list shapes are not works
    title, style, lyrics = d.get("title"), d.get("music_prompt"), d.get("lyrics_prompt")
    # Field-swap: 38/70 records in output/music have the columns shifted.
    if title and len(str(title)) > 120:
        title, style, lyrics = d.get("lyrics_prompt"), d.get("title"), d.get("music_prompt")
    return make_row(
        work_id=f"{corpus}:{os.path.basename(path)}",
        modality="music", source_corpus=corpus,
        title=(str(title)[:200] if title else None),
        created_at=norm_ts(d.get("timestamp")),
        model=d.get("model"), style_text=style, lyrics=lyrics,
        competition=as_bool(d.get("competition")),
        personality=(d.get("input_settings") or {}).get("selected_personality")
        if isinstance(d.get("input_settings"), dict) else None,
    )


def collect() -> list[dict]:
    rows: list[dict] = []

    for p in glob.glob(str(REPO / "output" / "past_music" / "*.json")):
        r = load_music_json(p, "past_music")
        if r:
            rows.append(r)
    for p in glob.glob(str(REPO / "output" / "music" / "*.json")):
        r = load_music_json(p, "music")
        if r:
            rows.append(r)

    rec = REPO / "data" / "archive" / "music_index_cache_recovered.json"
    if rec.exists():
        d = json.load(open(rec, encoding="utf-8"))
        for r in d["rows"]:
            rows.append(make_row(
                work_id=f"index_cache:{r.get('_source_key')}",
                modality="music", source_corpus="index_cache",
                title=r.get("title"), created_at=norm_ts(r.get("timestamp")),
                model=r.get("model"), style_text=r.get("music_prompt"),
                lyrics=r.get("lyrics_prompt"), competition=as_bool(r.get("competition")),
            ))

    for p in glob.glob(str(REPO / "output" / "songs" / "*.md")):
        txt = open(p, encoding="utf-8", errors="ignore").read()
        fm = re.match(r"^---\n(.*?)\n---", txt, re.S)
        meta = {}
        if fm:
            for line in fm.group(1).splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip().strip('"')
        title = meta.get("title")
        if not title:
            m = re.search(r"^#\s+(.+)$", txt, re.M)
            title = m.group(1).strip() if m else None
        style = re.search(r"##\s*MUSIC PROMPT\s*\n+(.+?)(?=\n##|\Z)", txt, re.S)
        lyr = re.search(r"##\s*LYRICS\s*\n+(.+?)(?=\n##|\Z)", txt, re.S)
        rows.append(make_row(
            work_id=f"songs:{os.path.basename(p)}", modality="music", source_corpus="songs",
            title=title, created_at=norm_ts(meta.get("created") or meta.get("date")),
            model=meta.get("renderer"), personality=meta.get("personality"),
            style_text=style.group(1).strip() if style else None,
            lyrics=lyr.group(1).strip() if lyr else None, asset_path=p,
        ))

    for p in glob.glob(str(REPO / "output" / "mp3" / "*.mp3")):
        stem = MP3_DUP_RE.sub("", Path(p).stem)
        rows.append(make_row(
            work_id=f"mp3:{os.path.basename(p)}", modality="music", source_corpus="mp3",
            title=stem, asset_path=p,
        ))

    # output/past_art holds 4,690 image files but only the slot-named ones carry
    # metadata:  YYYYMMDD_HHMMSS_model_concept_medium_variant_n_m.png
    # The other ~4,300 (Lofn-VideoFodder Gemini exports, NightCafe albumcovers)
    # are raw assets with opaque filenames. Admitting them as "works" would add
    # thousands of all-null rows and make every coverage percentage a lie, so
    # they are counted as assets instead — see `raw_assets` in the payload.
    slot_re = re.compile(
        r"^(\d{8})_(\d{6})_([^_]+)_(.+?)_(.+?)_(Revised|Synthesized)_(\d+)_(\d+)$"
    )
    raw_assets = 0
    for p in glob.glob(str(REPO / "output" / "past_art" / "**" / "*"), recursive=True):
        if not os.path.isfile(p) or Path(p).suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        m = slot_re.match(Path(p).stem)
        if not m:
            raw_assets += 1
            continue
        date, time, model, concept, medium, variant, _n, _m = m.groups()
        rows.append(make_row(
            work_id=f"past_art:{os.path.relpath(p, REPO)}", modality="image",
            source_corpus="past_art",
            title=concept, created_at=norm_ts(f"{date}_{time}"),
            model=model, style_text=medium, asset_path=p, variant=variant,
        ))
    globals()["_RAW_ASSETS"] = raw_assets

    for p in glob.glob(str(REPO / "output" / "past_video" / "*.json")):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        rows.append(make_row(
            work_id=f"past_video:{os.path.basename(p)}", modality="video",
            source_corpus="past_video",
            title=(d.get("concept") or "")[:120] or None,
            created_at=norm_ts(d.get("timestamp")), model=d.get("model"),
            style_text=d.get("medium"), competition=as_bool(d.get("competition")),
            personality=(d.get("input_settings") or {}).get("selected_personality")
            if isinstance(d.get("input_settings"), dict) else None,
        ))

    for p in glob.glob(str(REPO / "output" / "images" / "**" / "*.json"), recursive=True):
        try:
            d = json.load(open(p, encoding="utf-8-sig"))
        except Exception:
            continue
        items = d if isinstance(d, list) else [d]
        for i, it in enumerate(items):
            if not isinstance(it, dict):
                continue
            prompt = it.get("prompt") or it.get("image_prompt")
            if not prompt:
                continue
            rows.append(make_row(
                work_id=f"images:{os.path.relpath(p, REPO)}#{i}", modality="image",
                source_corpus="images", title=(it.get("title") or None),
                created_at=norm_ts(it.get("created_at") or it.get("timestamp")),
                model=it.get("model") or it.get("provider"),
                style_text=prompt, asset_path=it.get("image_url") or it.get("path"),
            ))

    return rows


def attach_engagement(rows: list[dict]) -> dict[str, Any]:
    """Join the Suno snapshot on normalized title. Report the rate honestly."""
    snap_path = REPO / "data" / "snapshots" / "suno" / "latest.json"
    if not snap_path.exists():
        for r in rows:
            r["engagement_status"] = "NO_SNAPSHOT"
        return {"status": "ABSENT", "hint": "python3 scripts/ingest_suno.py"}

    snap = json.load(open(snap_path, encoding="utf-8"))
    idx: dict[str, list[dict]] = defaultdict(list)
    for c in snap["rows"]:
        idx[norm_title(c.get("title"))].append(c)

    matched_clips: set[str] = set()
    per_corpus: Counter = Counter()
    totals: Counter = Counter()
    for r in rows:
        totals[r["source_corpus"]] += 1
        k = norm_title(r.get("title"))
        hits = idx.get(k) if k else None
        if not hits:
            r["engagement_status"] = "UNMATCHED"
            continue
        # Ambiguity is recorded, never silently resolved to the first hit.
        best = max(hits, key=lambda c: c.get("play_count") or 0)
        r["suno_clip_id"] = best["id"]
        r["play_count"] = best["play_count"]
        r["upvote_count"] = best["upvote_count"]
        r["like_rate"] = best["like_rate"]
        r["duration_s"] = best.get("duration")
        r["engagement_status"] = "MATCHED" if len(hits) == 1 else "AMBIGUOUS"
        r["engagement_candidates"] = len(hits)
        matched_clips.add(best["id"])
        per_corpus[r["source_corpus"]] += 1

    return {
        "status": "PRESENT",
        "snapshot_pulled_at": snap.get("pulled_at"),
        "clips_in_snapshot": len(snap["rows"]),
        "clips_matched": len(matched_clips),
        "clip_coverage_pct": round(len(matched_clips) / len(snap["rows"]) * 100, 1),
        "match_rate_by_corpus": {
            c: {"matched": per_corpus[c], "of": totals[c], "pct": round(per_corpus[c] / totals[c] * 100, 1)}
            for c in sorted(totals)
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true", help="print coverage, write nothing")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rows = collect()
    engagement = attach_engagement(rows)
    rows.sort(key=lambda r: (r.get("created_at") or "", r["work_id"]))

    by_modality = Counter(r["modality"] for r in rows)
    by_corpus = Counter(r["source_corpus"] for r in rows)
    fields = ["title", "created_at", "model", "style_text", "lyrics", "asset_path",
              "bpm", "musical_key", "words_per_line", "repeated_line_ratio",
              "sung_lines", "emo_tag_count", "play_count"]
    coverage = {
        f: round(sum(1 for r in rows if r.get(f) not in (None, "", [], {})) / len(rows) * 100, 1)
        for f in fields
    }

    print(f"works: {len(rows)}")
    print(f"  by modality: {dict(by_modality)}")
    print(f"  by corpus:   {dict(by_corpus)}")
    print("  coverage:")
    for f, pct in coverage.items():
        print(f"    {f:22s} {pct:5.1f}%")
    print(f"  engagement: {engagement.get('status')}", end="")
    if engagement.get("status") == "PRESENT":
        print(
            f" — {engagement['clips_matched']}/{engagement['clips_in_snapshot']} "
            f"clips ({engagement['clip_coverage_pct']}%) tied to a work"
        )
        for c, v in engagement["match_rate_by_corpus"].items():
            print(f"    {c:14s} {v['matched']:5d}/{v['of']:5d} = {v['pct']:5.1f}%")
    else:
        print()

    if args.stats:
        return 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "row_count": len(rows),
        "by_modality": dict(by_modality),
        "by_corpus": dict(by_corpus),
        "field_coverage_pct": coverage,
        "raw_assets_excluded": globals().get("_RAW_ASSETS", 0),
        "engagement": engagement,
        "caveats": [
            "prompt->render attribution is ~41%; prompt-level engagement claims do not cover the corpus.",
            "mp3->Suno title join is ~91%; song-level engagement is reliable.",
            "engagement_status is one of MATCHED / AMBIGUOUS / UNMATCHED / NO_SNAPSHOT. Never treat UNMATCHED as zero engagement — it means unknown.",
        ],
        "rows": rows,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
