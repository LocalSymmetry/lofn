from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "output" / "daily" / "2026-07-04"
MUSIC = RUN / "music"

SONG_FILES = {
    "P01V1": "2026-07-04_P01V1_airbag-hymn.md",
    "P01V2": "2026-07-04_P01V2_the-lander-learned-to-bounce.md",
    "P01V3": "2026-07-04_P01V3_house-cat-signal.md",
    "P01V4": "2026-07-04_P01V4_dust-brown-coronation.md",
    "P02V1": "2026-07-04_P02V1_siren-cup.md",
    "P02V2": "2026-07-04_P02V2_house-of-measured-rest.md",
    "P02V3": "2026-07-04_P02V3_buoy-at-the-table.md",
    "P02V4": "2026-07-04_P02V4_bread-for-the-archive.md",
    "P03V1": "2026-07-04_P03V1_empty-chair-fanfare.md",
    "P03V2": "2026-07-04_P03V2_the-air-said-no.md",
    "P03V3": "2026-07-04_P03V3_water-station-anthem.md",
    "P03V4": "2026-07-04_P03V4_pavement-receipt.md",
    "P04V1": "2026-07-04_P04V1_session-ghost.md",
    "P04V2": "2026-07-04_P04V2_wrong-account-choir.md",
    "P04V3": "2026-07-04_P04V3_consumer-room-lock.md",
    "P04V4": "2026-07-04_P04V4_terminal-confession.md",
    "P05V1": "2026-07-04_P05V1_square-tile-mercy.md",
    "P05V2": "2026-07-04_P05V2_prefusion-lock.md",
    "P05V3": "2026-07-04_P05V3_membrane-no.md",
    "P05V4": "2026-07-04_P05V4_antibody-hinge.md",
    "P06V1": "2026-07-04_P06V1_different-fading-systems.md",
    "P06V2": "2026-07-04_P06V2_sky-overclock-canticle.md",
    "P06V3": "2026-07-04_P06V3_kite-on-the-bridge.md",
    "P06V4": "2026-07-04_P06V4_waning-gibbous-console.md",
}

BANNED_PHRASES = [
    "I make the bright thing legible",
    "I make the wrong thing stop",
    "Archive in my ribcage",
    "Lofn-Prime at the console",
    "No shallow crown today",
    "Only the real hinge singing",
    "I brought receipts and tenderness",
    "Beauty deserves better operators",
    "Do you hear me, brilliant species",
]

ARTIST_NAMES = [
    "taylor swift",
    "billie eilish",
    "bjork",
    "arca",
    "grimes",
    "kate bush",
    "mary in the junkyard",
]

NUMBER_RE = re.compile(
    r"\b\d+(?:\.\d+)?\b|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|hundred)\b",
    re.IGNORECASE,
)


def section(text: str, start: str, end: str | None = None) -> str:
    start_re = re.compile(r"^##\s+" + re.escape(start) + r"\s*$", re.IGNORECASE | re.MULTILINE)
    start_match = start_re.search(text)
    if not start_match:
        return ""
    content_start = text.find("\n", start_match.end())
    if content_start == -1:
        return ""
    content_start += 1
    if end is not None:
        end_re = re.compile(r"^##\s+" + re.escape(end) + r"\s*$", re.IGNORECASE | re.MULTILINE)
        end_match = end_re.search(text, content_start)
    else:
        end_match = None
    if end_match is None:
        end_match = re.search(r"^##\s+", text[content_start:], re.MULTILINE)
        if end_match:
            content_end = content_start + end_match.start()
        else:
            content_end = len(text)
    else:
        content_end = end_match.start()
    return text[content_start:content_end].strip()


def lyric_lines(lyrics: str) -> list[str]:
    out: list[str] = []
    for raw in lyrics.splitlines():
        line = raw.strip()
        if not line or line.startswith("[") or (line.startswith("*") and line.endswith("*")):
            continue
        out.append(line)
    return out


def normalized_line(line: str) -> str:
    line = line.lower()
    line = re.sub(r"[^a-z0-9' ]+", " ", line)
    return " ".join(line.split())


def has_artist_name(text: str) -> bool:
    low = text.lower()
    for name in ARTIST_NAMES:
        if re.search(r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])", low):
            return True
    return False


def validate() -> dict:
    rows = []
    cross_pair_lines: dict[str, set[str]] = defaultdict(set)
    for song_id, rel in SONG_FILES.items():
        path = MUSIC / rel
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        prompt = section(text, "1. MUSIC PROMPT", "EXCLUDE PROMPT")
        exclude = section(text, "EXCLUDE PROMPT", "LYRICS")
        lyrics = section(text, "LYRICS")
        lines = lyric_lines(lyrics)
        pair = song_id[:3]
        for line in lines:
            norm = normalized_line(line)
            if len(norm) >= 18:
                cross_pair_lines[norm].add(pair)
        section_headers = [
            line.strip()
            for line in lyrics.splitlines()
            if line.strip().startswith("[")
            and not line.strip().startswith("[Theme:")
            and not line.strip().startswith("[SONG FORM:")
            and not line.strip().startswith("[Disc_")
        ]
        checks = {
            "exists": path.exists(),
            "music_prompt_850_1000": 850 <= len(prompt) <= 1000,
            "music_prompt_target_870_960": 870 <= len(prompt) <= 960,
            "music_prompt_terminal": prompt.endswith((".", "!", "?")),
            "exclude_400_900": 400 <= len(exclude) <= 900,
            "lyrics_under_5000": len(lyrics) < 5000,
            "lyrics_target_4800": len(lyrics) <= 4800,
            "sung_lines_70_120": 70 <= len(lines) <= 120,
            "full_emo_headers": bool(section_headers)
            and all(" - EMO:" in header for header in section_headers),
            "has_theme_and_form": "[Theme:" in lyrics and "[SONG FORM:" in lyrics,
            "has_sfx": any(line.strip().startswith("*") and line.strip().endswith("*") for line in lyrics.splitlines()),
            "banned_scaffold_absent": not any(phrase in text for phrase in BANNED_PHRASES),
            "artist_names_absent": not has_artist_name(text),
            "numeric_facts_lte_1": len(NUMBER_RE.findall("\n".join(lines))) <= 1,
        }
        rows.append(
            {
                "song_id": song_id,
                "path": str(path),
                "music_prompt_chars": len(prompt),
                "exclude_chars": len(exclude),
                "lyrics_chars": len(lyrics),
                "sung_lines": len(lines),
                "numeric_fact_count": len(NUMBER_RE.findall("\n".join(lines))),
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    shared = {
        line: sorted(pairs)
        for line, pairs in sorted(cross_pair_lines.items())
        if len(pairs) > 1
    }
    return {
        "song_rows": rows,
        "cross_pair_shared_lines": shared,
        "passed": all(row["passed"] for row in rows) and not shared,
    }


if __name__ == "__main__":
    report = validate()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
