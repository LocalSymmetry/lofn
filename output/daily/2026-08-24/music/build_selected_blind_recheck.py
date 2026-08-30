from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
MUSIC = Path(__file__).resolve().parent
OUT = MUSIC / "qa_blind_recheck"
SELECTED_PACKAGE_INDEX = {"01": 1, "02": 1, "03": 1, "04": 1, "05": 1, "06": 3}


def module_from(path: Path):
    spec = importlib.util.spec_from_file_location("suno_validator_recheck", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def after(path: Path, marker: str) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    at = text.find(marker)
    return text[at:] if at >= 0 else text


def main() -> None:
    OUT.mkdir(exist_ok=True)
    validator = module_from(ROOT / "skills/music/scripts/validate_suno_packages.py")
    golden_paths = [
        ROOT / "skills/music/references/suno_format_example_triple_arch.md",
        ROOT / "skills/music/references/suno_format_example_blue_screen.md",
        ROOT / "skills/music/references/suno_format_example_five_wrong_colors.md",
    ]
    goldens = [after(path, "## Music Prompt") for path in golden_paths]
    decoy_path = ROOT / "output/daily/2026-05-23/daily-run/pair_01/step10_final_suno_package.md"
    decoy = after(decoy_path, "## Final Suno-Ready Lyrics")
    mapping: dict[str, object] = {
        "warning": "PRIVATE COORDINATOR KEY — RECHECK JUDGE MUST NOT READ",
        "sets": {},
    }

    for ordinal, pair_id in enumerate(SELECTED_PACKAGE_INDEX, start=1):
        path = MUSIC / f"pair_{pair_id}_step10_final_package_enhanced.md"
        segment = validator.current_step11_segment(path.read_text(encoding="utf-8", errors="replace"))
        packages = validator.package_blocks(segment)
        package_index = SELECTED_PACKAGE_INDEX[pair_id]
        candidate = packages[package_index - 1]
        golden_index = (ordinal - 1) % len(goldens)
        entries = [("candidate", candidate), ("golden", goldens[golden_index]), ("decoy", decoy)]
        random.Random(26082400 + ordinal).shuffle(entries)
        labels = ["A", "B", "C"]
        set_id = f"selected_{ordinal:02d}"
        lines = [
            f"# Blind selected-set recheck {set_id}",
            "",
            "Rank A, B, and C by publish-level song force. Ignore formatting differences and judge the song package itself.",
        ]
        key: dict[str, str] = {}
        for label, (role, content) in zip(labels, entries):
            lines.extend(["", f"## Package {label}", "", content.strip(), ""])
            key[label] = role
        (OUT / f"{set_id}.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
        mapping["sets"][set_id] = {
            "candidate_source": f"{path.name}#package{package_index}",
            "golden_source": str(golden_paths[golden_index].relative_to(ROOT)),
            "key": key,
        }

    (MUSIC / "blind_recheck_mapping_PRIVATE.json").write_text(
        json.dumps(mapping, indent=2), encoding="utf-8", newline="\n"
    )
    print(f"built {len(SELECTED_PACKAGE_INDEX)} selected blind sets")


if __name__ == "__main__":
    main()
