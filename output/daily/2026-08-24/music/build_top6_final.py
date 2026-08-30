from __future__ import annotations

import hashlib
import importlib.util
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
MUSIC = Path(__file__).resolve().parent
SELECTED = [
    ("01", 1, "Still in Orbit"),
    ("02", 1, "No Significant Events"),
    ("03", 1, "Leave the Line Open"),
    ("04", 1, "The Last Verb Fails"),
    ("05", 1, "Short of the Groove"),
    ("06", 3, "Backwash Before Wave"),
]


def load_validator():
    path = ROOT / "skills/music/scripts/validate_suno_packages.py"
    spec = importlib.util.spec_from_file_location("suno_validator_top6", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def field(block: str, heading: str, following: str | None = None) -> str:
    if following:
        pattern = rf"^## {re.escape(heading)}\s*\n(.*?)(?=^## {re.escape(following)}\s*$)"
    else:
        pattern = rf"^## {re.escape(heading)}\s*\n(.*?)(?=^## |^# V|\Z)"
    match = re.search(pattern, block, re.M | re.S)
    return match.group(1).strip() if match else ""


def main() -> None:
    validator = load_validator()
    parts = [
        "# Lofn Prime Daily — Final Top 6",
        "",
        "**Date:** 2026-08-24  ",
        "**Theme:** UNCOUNTED, STILL HERE  ",
        "**QA:** SHIP — text packages only  ",
        "**Autonomy:** No audio was rendered, uploaded, published, or purchased. The Scientist's ear remains the render/publication gate.",
        "",
        "This file contains the six selected copy-paste packages: one per pair, three ACCESSIBLE and three AMBITIOUS.",
    ]
    dedication_parts = [
        "# Selected-Six Dedications",
        "",
        "All character counts include the dedication text only and are measured against the 500-character cap.",
    ]
    credit_parts = [
        "# Selected-Six Lineage & Credits",
        "",
        "These are public-surface companion credits. They preserve source, scene, player/recordist, and no-sample distinctions from each selected package.",
    ]
    manifest = []

    for order, (pair_id, package_index, expected_title) in enumerate(SELECTED, start=1):
        source = MUSIC / f"pair_{pair_id}_step10_final_package_enhanced.md"
        text = source.read_text(encoding="utf-8", errors="replace")
        blocks = validator.package_blocks(validator.current_step11_segment(text))
        block = blocks[package_index - 1].strip()
        if expected_title not in block.splitlines()[0]:
            raise RuntimeError(f"P{pair_id}: selected block does not contain {expected_title!r}")
        parts.extend(["", "---", "", f"## {order}. P{pair_id} — {expected_title}", "", block])
        dedication_match = re.search(r"^## Dedication[^\n]*\n\s*(.*?)(?=^## |^# V|\Z)", block, re.M | re.S)
        if not dedication_match:
            raise RuntimeError(f"P{pair_id}: selected package lacks dedication")
        dedication = dedication_match.group(1).strip()
        fenced = re.search(r"```(?:text)?\s*\n(.*?)\n```", dedication, re.S)
        if fenced:
            dedication = fenced.group(1).strip()
        dedication_parts.extend([
            "",
            f"## P{pair_id} — {expected_title}",
            "",
            dedication,
            "",
            f"**Characters:** {len(dedication)}",
        ])
        credit_match = re.search(
            r"^## (?:\d+\. )?Lineage & Credit[^\n]*\n(.*?)(?=^## |^# V|\Z)",
            block,
            re.M | re.S,
        )
        if not credit_match:
            raise RuntimeError(f"P{pair_id}: selected package lacks Lineage & Credit")
        credit_parts.extend([
            "",
            f"## P{pair_id} — {expected_title}",
            "",
            credit_match.group(1).strip(),
        ])
        manifest.append(
            f"P{pair_id} V{package_index} {expected_title}: "
            f"{hashlib.sha256(block.encode('utf-8')).hexdigest()}"
        )

    parts.extend(["", "---", "", "## Selected-package SHA-256 manifest", ""])
    parts.extend(f"- `{row}`" for row in manifest)
    (MUSIC / "TOP_6_FINAL.md").write_text("\n".join(parts) + "\n", encoding="utf-8", newline="\n")
    (MUSIC / "DEDICATIONS.md").write_text(
        "\n".join(dedication_parts) + "\n", encoding="utf-8", newline="\n"
    )
    (MUSIC / "CREDITS.md").write_text(
        "\n".join(credit_parts) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"TOP_6_FINAL.md\t{(MUSIC / 'TOP_6_FINAL.md').stat().st_size}")
    print(f"DEDICATIONS.md\t{(MUSIC / 'DEDICATIONS.md').stat().st_size}")
    print(f"CREDITS.md\t{(MUSIC / 'CREDITS.md').stat().st_size}")


if __name__ == "__main__":
    main()
