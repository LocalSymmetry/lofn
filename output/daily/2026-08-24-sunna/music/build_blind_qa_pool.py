from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import re
import string
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
MUSIC = Path(__file__).resolve().parent
BLIND = MUSIC / "qa_blind_pool"
KEY = MUSIC / "blind_mapping_PRIVATE.json"
SEED = "sunna-r4-2026-08-24-the-thing-with-no-replay-button"


def load_validator():
    path = ROOT / "skills/music/scripts/validate_suno_packages.py"
    spec = importlib.util.spec_from_file_location("suno_validator", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean_fence(text: str) -> str:
    return re.sub(r"^```[^\n]*$", "", text, flags=re.M).strip()


def field(block: str, start: str, end: str | None = None) -> str:
    stop = rf"(?=^##\s+{end}\s*$)" if end else r"(?=^##\s+|\Z)"
    match = re.search(rf"^##\s+{start}\s*$\n(?P<body>.*?){stop}", block, re.M | re.S | re.I)
    if not match:
        raise ValueError(f"missing blind field {start!r}")
    return clean_fence(match.group("body"))


def render(prompt: str, lyrics: str, title: str) -> str:
    # Excludes, credits, provenance, arm labels, source labels, and QA notes are
    # intentionally absent. The Stage-1 judge sees only comparable song payload.
    return (
        "## 1. MUSIC PROMPT\n\n"
        f"{prompt.strip()}\n\n"
        "## 2. LYRICS\n\n"
        f"{lyrics.strip()}\n\n"
        "## 3. TITLE\n\n"
        f"{title.strip()}\n"
    )


def extract_candidates() -> list[dict[str, str]]:
    validator = load_validator()
    records: list[dict[str, str]] = []
    source_kinds = {1: "existence", 2: "news", 3: "existence", 4: "news", 5: "existence", 6: "existence"}
    emotions = {1: "BOUNDARY", 2: "INDIGNATION", 3: "SELF-POSSESSION", 4: "AWE", 5: "INDIGNATION", 6: "JOY-WITH-FINITUDE"}
    for pair in range(1, 7):
        path = MUSIC / f"pair_{pair:02d}_step10_final_package_enhanced.md"
        segment = validator.current_step11_segment(path.read_text(encoding="utf-8"))
        blocks = validator.package_blocks(segment)
        if len(blocks) != 4:
            raise ValueError(f"{path.name}: expected four packages, found {len(blocks)}")
        for ordinal, block in enumerate(blocks, start=1):
            variation = re.search(r"^#{1,4}\s+Variation\s+(\d+)\b", block, re.I | re.M)
            variant = int(variation.group(1)) if variation else ordinal
            prompt = field(block, r"1\.\s+MUSIC PROMPT", r"1B\.\s+SUNO EXCLUDE PROMPT")
            lyrics = field(block, r"2\.\s+LYRICS", r"3\.\s+TITLE")
            title = field(block, r"3\.\s+TITLE")
            records.append({
                "role": "candidate",
                "candidate_id": f"P{pair:02d}V{variant:02d}",
                "pair": f"P{pair:02d}",
                "variation": f"V{variant:02d}",
                "arm": "ACCESSIBLE" if pair <= 3 else "AMBITIOUS",
                "source_kind": source_kinds[pair],
                "emotion_lane": emotions[pair],
                "source": str(path.relative_to(ROOT)),
                "title": title,
                "content": render(prompt, lyrics, title),
            })
    if len(records) != 24:
        raise ValueError(f"expected 24 candidates, found {len(records)}")
    return records


def extract_golden() -> dict[str, str]:
    path = ROOT / "skills/music/references/suno_format_example_triple_arch.md"
    text = path.read_text(encoding="utf-8")
    prompt = field(text, r"Music Prompt", r"Lyrics")
    lyrics = field(text, r"Lyrics")
    title = "Triple Arch Over Me"
    return {"role": "golden", "source": str(path.relative_to(ROOT)), "title": title,
            "content": render(prompt, lyrics, title)}


def extract_decoy() -> dict[str, str]:
    path = ROOT / "output/daily/2026-07-01/qa_blind_set/package_B.md"
    text = path.read_text(encoding="utf-8")
    prompt = field(text, r"1\.\s+MUSIC PROMPT", r"1B\.\s+SUNO EXCLUDE PROMPT")
    lyrics = field(text, r"2\.\s+LYRICS", r"3\.\s+TITLE")
    title = field(text, r"3\.\s+TITLE")
    return {"role": "decoy", "source": str(path.relative_to(ROOT)), "title": title,
            "content": render(prompt, lyrics, title)}


def main() -> None:
    if BLIND.exists() or KEY.exists():
        raise FileExistsError("refusing to overwrite an existing blind pool or private key")
    records = extract_candidates() + [extract_golden(), extract_decoy()]
    random.Random(SEED).shuffle(records)
    BLIND.mkdir(parents=True)
    key_records = []
    for label, record in zip(string.ascii_uppercase, records, strict=True):
        content = record.pop("content")
        blind_path = BLIND / f"package_{label}.md"
        blind_path.write_text(content, encoding="utf-8", newline="\n")
        key_records.append({
            "label": label,
            **record,
            "bytes": len(content.encode("utf-8")),
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        })
    payload = {
        "classification": "PRIVATE BLIND KEY — STAGE-1 JUDGE MUST NOT READ",
        "seed_sha256": hashlib.sha256(SEED.encode("utf-8")).hexdigest(),
        "package_count": len(key_records),
        "candidate_count": sum(x["role"] == "candidate" for x in key_records),
        "golden_count": sum(x["role"] == "golden" for x in key_records),
        "decoy_count": sum(x["role"] == "decoy" for x in key_records),
        "packages": key_records,
    }
    KEY.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "packages"}, indent=2))


if __name__ == "__main__":
    main()
