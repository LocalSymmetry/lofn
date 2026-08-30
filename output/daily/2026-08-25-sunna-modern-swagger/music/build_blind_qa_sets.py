from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
MUSIC = Path(__file__).resolve().parent
OUT = MUSIC / "qa_blind_sets"


def load_validator_module():
    path = ROOT / "skills" / "music" / "scripts" / "validate_suno_packages.py"
    spec = importlib.util.spec_from_file_location("suno_validator", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def after_heading(path: Path, marker: str) -> str:
    content = path.read_text(encoding="utf-8", errors="replace")
    at = content.find(marker)
    return content[at:] if at >= 0 else content


def main() -> None:
    OUT.mkdir(exist_ok=True)
    validator = load_validator_module()
    golden_paths = [
        ROOT / "skills/music/references/suno_format_example_triple_arch.md",
        ROOT / "skills/music/references/suno_format_example_blue_screen.md",
        ROOT / "skills/music/references/suno_format_example_five_wrong_colors.md",
    ]
    goldens = [after_heading(path, "## Music Prompt") for path in golden_paths]
    decoy_path = ROOT / "output/daily/2026-05-23/daily-run/pair_01/step10_final_suno_package.md"
    decoy = after_heading(decoy_path, "## Final Suno-Ready Lyrics")
    mapping: dict[str, object] = {
        "warning": "PRIVATE COORDINATOR KEY — BLIND QA JUDGE MUST NOT READ BEFORE FREEZE",
        "seed": 2026082502,
        "decoy_source": str(decoy_path.relative_to(ROOT)),
        "sets": {},
    }

    selections = json.loads((MUSIC / "selected_candidates.json").read_text(encoding="utf-8"))
    if len(selections) != 6:
        raise RuntimeError(f"expected 6 finalist records, found {len(selections)}")

    ordinal = 0
    for selected in selections:
        pair_num = int(selected["pair_id"][1:])
        pair_id = f"{pair_num:02d}"
        path = MUSIC / f"pair_{pair_id}_step10_selected_enhanced.md"
        text = path.read_text(encoding="utf-8", errors="replace")
        segment = validator.current_step11_segment(text)
        packages = validator.package_blocks(segment)
        if len(packages) != 1:
            raise RuntimeError(f"{path.name}: expected 1 enhanced finalist package, found {len(packages)}")
        variation = int(selected["variation"])
        candidate = packages[0]
        ordinal += 1
        set_id = f"set_{ordinal:02d}"
        golden_index = (ordinal - 1) % len(goldens)
        entries = [("candidate", candidate), ("golden", goldens[golden_index]), ("decoy", decoy)]
        rng = random.Random(2026082502 + ordinal)
        rng.shuffle(entries)
        body = [
            f"# Blind finalist comparison {set_id}",
            "",
            "Rank A, B, and C by publish-level song force. Do not infer origin from formatting; judge the musical package. Name strongest and weakest with concrete lyrical, formal, hook, and production evidence. Do not score by doctrinal obedience alone.",
        ]
        key: dict[str, str] = {}
        for label, (role, content) in zip(("A", "B", "C"), entries):
            body.extend(["", f"## Package {label}", "", content.strip(), ""])
            key[label] = role
        (OUT / f"{set_id}.md").write_text("\n".join(body), encoding="utf-8", newline="\n")
        mapping["sets"][set_id] = {
            "candidate_source": f"{path.name}#enhanced-selected-from-V{variation}",
            "candidate_title": selected["title"],
            "golden_source": str(golden_paths[golden_index].relative_to(ROOT)),
            "key": key,
        }

    (MUSIC / "blind_mapping_PRIVATE.json").write_text(
        json.dumps(mapping, indent=2), encoding="utf-8", newline="\n"
    )
    print(f"built {ordinal} blind sets in {OUT}")


if __name__ == "__main__":
    main()
