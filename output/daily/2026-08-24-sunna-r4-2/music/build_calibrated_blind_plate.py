"""Build the final blind QA plate from stable, auditable source artifacts.

Each set contains one current Step-11 finalist, one official judge-side golden,
and one nonselected Step-10 song from the same pair. Roles are written only to
the separate key directory; package files contain normalized song payloads.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
MUSIC = Path(__file__).resolve().parent
GOLDEN_SOURCE = ROOT / "skills/qa/references/golden_song_examples.md"
PLATE_DIR = MUSIC / "qa_blind_calibrated_v4_repair"
KEY_DIR = MUSIC / "qa_keys_calibrated_v4_repair"
DECOY_SOURCE = ROOT / "output/daily/2026-07-01/qa_blind_set/package_B.md"
TRIPLE_ARCH_SOURCE = ROOT / "skills/music/references/suno_format_example_triple_arch.md"


FINALISTS = [
    (4, "Shake It Into Mine"),
    (5, "Pretty Busy Tomorrow"),
]

GOLDENS = ["Dial Tone of God", "Radiant Doom", "Apology.DAT"]


def clean_field(value: str) -> str:
    value = value.strip()
    if value.startswith("```") and value.endswith("```"):
        value = re.sub(r"^```[^\n]*\n?", "", value)
        value = re.sub(r"\n?```$", "", value)
    value = re.sub(r"\n+---\s*$", "", value)
    return value.strip()


def field(text: str, heading: str, following: tuple[str, ...]) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\s*\n(.*?)(?=^## |\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if not match:
        raise ValueError(f"Missing field {heading!r}")
    return clean_field(match.group(1))


def normalize(title: str, music_prompt: str, lyrics: str, exclude: str = "") -> str:
    parts = [
        "# Blind Song Package",
        "",
        "## Title",
        title.strip(),
        "",
        "## Music Prompt",
        music_prompt.strip(),
    ]
    if exclude.strip():
        parts.extend(["", "## Exclude Prompt", exclude.strip()])
    parts.extend(["", "## Lyrics", lyrics.strip(), ""])
    return "\n".join(parts)


def current_finalist(pair: int, expected_title: str) -> str:
    text = (MUSIC / f"pair_{pair:02d}_step11_enhancement.md").read_text(encoding="utf-8-sig")
    title = field(text, "3. TITLE", ("4.",))
    if title != expected_title:
        raise ValueError(f"Pair {pair:02d} finalist title drift: {title!r}")
    return normalize(
        title,
        field(text, "1. MUSIC PROMPT", ("1B.",)),
        field(text, "2. LYRICS", ("3.",)),
        field(text, "1B. SUNO EXCLUDE PROMPT", ("2.",)),
    )


def archived_decoy() -> str:
    text = DECOY_SOURCE.read_text(encoding="utf-8-sig")
    title_match = re.search(r"^# (.+)$", text, flags=re.MULTILINE)
    if not title_match:
        raise ValueError("Archived decoy has no title")
    title = title_match.group(1).strip()
    if title != "Look Again, The World Is Lit":
        raise ValueError(f"Archived decoy title drift: {title!r}")
    return normalize(
        title,
        field(text, "1. MUSIC PROMPT", ("1B.",)),
        field(text, "2. LYRICS", ("3.",)),
        field(text, "1B. SUNO EXCLUDE PROMPT", ("2.",)),
    )


def triple_arch_golden() -> str:
    text = TRIPLE_ARCH_SOURCE.read_text(encoding="utf-8-sig")
    title = "Triple Arch Over Me"
    return normalize(
        title,
        field(text, "Music Prompt", ("Lyrics",)),
        field(text, "Lyrics", ("3.",)),
    )


def official_goldens() -> dict[str, str]:
    text = GOLDEN_SOURCE.read_text(encoding="utf-8-sig")
    matches = list(re.finditer(r"^Song Title: (.+)$", text, flags=re.MULTILINE))
    result: dict[str, str] = {}
    for index, match in enumerate(matches):
        title = match.group(1).strip()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.end() : end]
        music_match = re.search(
            r"^Music Prompt:\s*\n```\s*\n(.*?)\n```", block, flags=re.MULTILINE | re.DOTALL
        )
        lyrics_match = re.search(
            r"^Lyrics Prompt:\s*\n```\s*\n(.*?)\n```", block, flags=re.MULTILINE | re.DOTALL
        )
        if not music_match or not lyrics_match:
            raise ValueError(f"Malformed official golden: {title}")
        result[title] = normalize(title, music_match.group(1), lyrics_match.group(1))
    missing = set(GOLDENS) - set(result)
    if missing:
        raise ValueError(f"Missing official goldens: {sorted(missing)}")
    return result


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    for path in (PLATE_DIR, KEY_DIR):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)

    golden = triple_arch_golden()
    rng = random.Random("sunna-r4.2-calibrated-blind-v4-repair")
    manifest: list[dict[str, object]] = []

    decoy = archived_decoy()
    decoy_title = "Look Again, The World Is Lit"
    for set_index, (pair, finalist_title) in enumerate(FINALISTS, start=1):
        golden_title = "Triple Arch Over Me"
        roles = {
            "CANDIDATE": current_finalist(pair, finalist_title),
            "GOLDEN": golden,
            "DECOY": decoy,
        }
        labels = ["A", "B", "C"]
        rng.shuffle(labels)
        mapping = dict(zip(labels, roles))
        set_dir = PLATE_DIR / f"set_{set_index:02d}"
        set_dir.mkdir()
        key_lines: list[str] = []
        file_rows: list[dict[str, str]] = []
        for label in labels:
            role = mapping[label]
            payload = roles[role]
            package_path = set_dir / f"package_{label}.md"
            package_path.write_text(payload, encoding="utf-8", newline="\n")
            key_lines.append(f"{label} = {role}")
            file_rows.append(
                {
                    "label": label,
                    "role": role,
                    "sha256": sha256(payload),
                    "title": {
                        "CANDIDATE": finalist_title,
                        "GOLDEN": golden_title,
                        "DECOY": decoy_title,
                    }[role],
                }
            )
        (KEY_DIR / f"set_{set_index:02d}_key.txt").write_text(
            "\n".join(key_lines) + "\n", encoding="utf-8", newline="\n"
        )
        manifest.append(
            {
                "set": set_index,
                "pair": pair,
                "golden_source": str(TRIPLE_ARCH_SOURCE.relative_to(ROOT)).replace("\\", "/"),
                "decoy_source": str(DECOY_SOURCE.relative_to(ROOT)).replace("\\", "/"),
                "candidate_source": f"output/daily/2026-08-24-sunna-r4-2/music/pair_{pair:02d}_step11_enhancement.md",
                "files": file_rows,
            }
        )

    (KEY_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
