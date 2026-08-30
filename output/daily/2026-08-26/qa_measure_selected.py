from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(ROOT))

from scripts.validate_step import _non_chorus_lyric_lines, load_gates
from skills.music.scripts.validate_suno_packages import (
    ARTIST_BLOCKLIST,
    EXCLUDE_RE,
    LYRICS_RE,
    PROMPT_RE,
    PROMPT_WITH_EXCLUDE_RE,
    TITLE_RE,
    strip_md,
)


ICB = (RUN_DIR / "CREATIVE_CONTEXT.md").read_text(encoding="utf-8")
GATES = load_gates()


def fenced_field(body: str) -> str:
    match = re.search(r"```[a-zA-Z]*\n(.*?)```", body, re.S)
    return (match.group(1) if match else body).strip()


def section(text: str, heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\s*\n(?P<body>.*?)(?=^## |\Z)",
        text,
        re.M | re.S,
    )
    return match.group("body").strip() if match else ""


def sung_lines(lyrics: str) -> list[str]:
    return [
        line.strip()
        for line in lyrics.splitlines()
        if line.strip()
        and not line.strip().startswith(("[", "*", "#", "```"))
    ]


rows = []
for path in sorted((RUN_DIR / "selected").glob("*.md")):
    text = path.read_text(encoding="utf-8")
    prompt_match = PROMPT_WITH_EXCLUDE_RE.search(text) or PROMPT_RE.search(text)
    exclude_match = EXCLUDE_RE.search(text)
    lyrics_match = LYRICS_RE.search(text)
    title_match = TITLE_RE.search(text)
    if not all((prompt_match, exclude_match, lyrics_match, title_match)):
        raise SystemExit(f"canonical extraction failed: {path.name}")

    prompt = strip_md(prompt_match.group("body"))
    exclude = strip_md(exclude_match.group("body"))
    lyrics = fenced_field(lyrics_match.group("body"))
    title = strip_md(title_match.group("body")).splitlines()[0].strip("* ")
    sung = sung_lines(lyrics)
    non_chorus = _non_chorus_lyric_lines(lyrics)
    emo_headers = re.findall(r"^\[[^\]\n]*EMO:([^\]\n]+)\]", lyrics, re.M | re.I)
    emotions = []
    for header in emo_headers:
        token = re.split(r"\s+[–—-]\s+", header, maxsplit=1)[0].strip()
        emotions.append(token)
    fact_lines = [
        line for line in sung
        if re.search(
            r"\d|\b(?:eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
            r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|"
            r"ninety|hundred|thousand|million)\s+(?:percent|degrees?|kilomet(?:er|re)s?|"
            r"miles?|met(?:er|re)s?|hours?|minutes?|seconds?|years?|days?|bars?)\b",
            line,
            re.I,
        )
    ]
    dedication = section(text, "DEDICATION")
    row = {
        "file": path.name,
        "title": title,
        "file_bytes": path.stat().st_size,
        "music_prompt_chars": len(prompt),
        "music_prompt_terminal_punctuation": prompt[-1:] in ".!?…",
        "exclude_chars": len(exclude),
        "lyrics_field_chars": len(lyrics),
        "sung_lines": len(sung),
        "emo_headers": len(emo_headers),
        "distinct_emotions": len(set(emotions)),
        "sfx_cues": len(re.findall(r"^\*[^*\n]{1,40}\*\s*$", lyrics, re.M)),
        "unique_line_ratio_chorus_exempt": round(
            len(set(line.lower() for line in non_chorus)) / len(non_chorus), 3
        ) if non_chorus else None,
        "house_lexicon_hits": sorted(
            phrase for phrase in GATES.get("house_lexicon", [])
            if str(phrase).lower() in (prompt + "\n" + lyrics).lower()
        ),
        "numeric_fact_lines": fact_lines,
        "artist_blocklist_hits_in_prompt": sorted(
            artist for artist in ARTIST_BLOCKLIST if artist in prompt.lower()
        ),
        "dedication_chars": len(dedication),
        "hard_pass": (
            850 <= len(prompt) <= 1000
            and len(exclude) <= 1000
            and len(lyrics) < 5000
            and 70 <= len(sung) <= 120
            and bool(emo_headers)
            and bool(re.search(r"^\*[^*\n]{1,40}\*\s*$", lyrics, re.M))
            and not any(artist in prompt.lower() for artist in ARTIST_BLOCKLIST)
            and len(dedication) <= 500
        ),
    }
    rows.append(row)

enhanced = sorted(RUN_DIR.glob("pair_*_step10_final_package_enhanced.md"))
integrity = {
    "icb_bytes": len(ICB.encode("utf-8")),
    "icb_sha256": hashlib.sha256(ICB.encode("utf-8")).hexdigest(),
    "speaker_tags_after": ICB.count("(after "),
    "enhanced_files": len(enhanced),
    "enhanced_prefix_matches": sum(
        p.read_text(encoding="utf-8").startswith(ICB) for p in enhanced
    ),
    "step06_files": len(list(RUN_DIR.glob("pair_*_step06_facets.md"))),
    "step07_files": len(list(RUN_DIR.glob("pair_*_step07_song_guides.md"))),
    "step08_files": len(list(RUN_DIR.glob("pair_*_step08_generation.md"))),
    "step09_files": len(list(RUN_DIR.glob("pair_*_step09_artist_refined.md"))),
    "step10_files": len(list(RUN_DIR.glob("pair_*_step10_revision_synthesis.md"))),
    "step11_files": len(enhanced),
    "final_packages": sum(
        len(re.findall(
            r"^## (?:1\. MUSIC PROMPT|SUNO STYLE PROMPT)\s*$",
            p.read_text(encoding="utf-8"),
            re.M,
        ))
        for p in enhanced
    ),
}

report = {"integrity": integrity, "selected": rows}
(RUN_DIR / "QA_MEASUREMENTS.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
    newline="\n",
)

print(json.dumps(integrity, indent=2))
print("title | prompt | exclude | lyrics | sung | EMO | SFX | uniq | dedication | hard")
for row in rows:
    print(
        f"{row['title']} | {row['music_prompt_chars']} | {row['exclude_chars']} | "
        f"{row['lyrics_field_chars']} | {row['sung_lines']} | {row['emo_headers']} | "
        f"{row['sfx_cues']} | {row['unique_line_ratio_chorus_exempt']} | "
        f"{row['dedication_chars']} | {row['hard_pass']}"
    )
