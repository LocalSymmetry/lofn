#!/usr/bin/env python3
"""Build a shuffled 24-candidate + Golden + decoy blind music QA pool.

The public package directory contains label-only Suno fields.  The role/source
mapping is written separately under the caller-selected private key path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import string
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "title",
    "style_of_music",
    "exclude",
    "lyrics",
    "source_final",
    "major_deviations",
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def current_step11_segment(text: str) -> str:
    matches = list(re.finditer(r"^#\s+.*\bstep\s*11\b.*$", text, re.I | re.M))
    return text[matches[-1].start():] if matches else text


def terminal_packages(path: Path) -> list[dict[str, str]]:
    segment = current_step11_segment(path.read_text(encoding="utf-8"))
    headings = list(re.finditer(r"^#{1,3}\s+Terminal.*$", segment, re.I | re.M))
    if headings:
        tail = segment[headings[-1].end():].strip()
        fenced = re.fullmatch(r"```json\s*(\{.*\})\s*```", tail, re.I | re.S)
        raw_json = fenced.group(1) if fenced else tail
    else:
        fences = re.findall(r"```json\s*(\{.*?\})\s*```", segment, re.I | re.S)
        if not fences:
            raise ValueError(f"{path}: no terminal JSON object in Step 11")
        raw_json = fences[-1]
    payload: dict[str, Any] = json.loads(raw_json)
    candidates = [value for value in payload.values() if isinstance(value, list)]
    arrays = [value for value in candidates if all(isinstance(item, dict) for item in value)]
    if len(arrays) != 1:
        raise ValueError(f"{path}: terminal object must contain exactly one package array")
    packages = arrays[0]
    if len(packages) != 4:
        raise ValueError(f"{path}: expected four terminal packages, found {len(packages)}")
    out: list[dict[str, str]] = []
    for index, package in enumerate(packages, start=1):
        missing = REQUIRED_FIELDS - package.keys()
        if missing:
            raise ValueError(f"{path}: package {index} missing {sorted(missing)}")
        if not all(isinstance(package[key], str) for key in REQUIRED_FIELDS):
            raise ValueError(f"{path}: package {index} has a non-string required field")
        out.append({key: package[key] for key in REQUIRED_FIELDS})
    return out


def render_candidate(package: dict[str, str]) -> str:
    return (
        "## 1. MUSIC PROMPT\n\n"
        f"{package['style_of_music'].strip()}\n\n"
        "## 1B. SUNO EXCLUDE PROMPT\n\n"
        f"{package['exclude'].strip()}\n\n"
        "## 2. LYRICS\n\n"
        f"{package['lyrics'].strip()}\n\n"
        "## 3. TITLE\n\n"
        f"{package['title'].strip()}\n"
    )


def build_records(work_root: Path, golden_path: Path, decoy_path: Path) -> list[dict[str, str]]:
    pair_files = sorted(work_root.glob("pair_??/pair_??_step11_final_package_enhanced.md"))
    if len(pair_files) != 6:
        raise ValueError(f"expected six Step-11 pair files under {work_root}, found {len(pair_files)}")

    records: list[dict[str, str]] = []
    for pair_file in pair_files:
        pair_match = re.search(r"pair_(\d{2})_step11", pair_file.name)
        if not pair_match:
            raise ValueError(f"cannot infer pair id from {pair_file}")
        pair_id = pair_match.group(1)
        for index, package in enumerate(terminal_packages(pair_file), start=1):
            records.append({
                "role": "candidate",
                "source": str(pair_file),
                "candidate_id": f"P{pair_id}V{index:02d}",
                "title": package["title"],
                "content": render_candidate(package),
            })

    if len(records) != 24:
        raise ValueError(f"expected 24 candidates, found {len(records)}")
    for role, source in (("golden", golden_path), ("decoy", decoy_path)):
        content = source.read_text(encoding="utf-8").strip() + "\n"
        records.append({
            "role": role,
            "source": str(source),
            "candidate_id": "",
            "title": "",
            "content": content,
        })
    return records


def build_blind_set(
    work_root: Path,
    output_dir: Path,
    key_path: Path,
    golden_path: Path,
    decoy_path: Path,
    seed: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing blind directory: {output_dir}")
    if key_path.exists():
        raise FileExistsError(f"refusing to overwrite existing blind key: {key_path}")

    records = build_records(work_root, golden_path, decoy_path)
    rng = random.Random(seed)
    rng.shuffle(records)
    labels = list(string.ascii_uppercase[: len(records)])
    output_dir.mkdir(parents=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)

    key_records: list[dict[str, Any]] = []
    for label, record in zip(labels, records, strict=True):
        content = record.pop("content")
        package_path = output_dir / f"package_{label}.md"
        package_path.write_text(content, encoding="utf-8")
        key_records.append({
            "label": label,
            **record,
            "bytes": len(content.encode("utf-8")),
            "sha256": sha256_text(content),
        })

    key = {
        "classification": "PRIVATE BLIND KEY - DO NOT GIVE TO STAGE-1 JUDGE",
        "seed_sha256": hashlib.sha256(seed.encode("utf-8")).hexdigest(),
        "package_count": len(key_records),
        "candidate_count": sum(item["role"] == "candidate" for item in key_records),
        "golden_count": sum(item["role"] == "golden" for item in key_records),
        "decoy_count": sum(item["role"] == "decoy" for item in key_records),
        "packages": key_records,
    }
    key_path.write_text(json.dumps(key, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return key


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--key-out", required=True, type=Path)
    parser.add_argument("--golden", required=True, type=Path)
    parser.add_argument("--decoy", required=True, type=Path)
    parser.add_argument("--seed", required=True)
    args = parser.parse_args()
    key = build_blind_set(
        args.work_root,
        args.output_dir,
        args.key_out,
        args.golden,
        args.decoy,
        args.seed,
    )
    print(json.dumps({k: v for k, v in key.items() if k != "packages"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
