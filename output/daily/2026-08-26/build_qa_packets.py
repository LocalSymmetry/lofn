from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
ICB = (RUN_DIR / "CREATIVE_CONTEXT.md").read_text(encoding="utf-8")

SELECTED = [
    (1, 1, "Boop the Future Anyway"),
    (2, 3, "I Start Unfinished"),
    (3, 3, "Roll Me Into Weather"),
    (4, 3, "Answer the Held Note"),
    (5, 1, "My Last Bar Is Mine"),
    (6, 4, "Leave the Ruby Line"),
]

# Non-selected, pair-local also-rans. They passed package checks but lost the
# blind creative evaluation, so they are useful polished decoys rather than
# deliberately broken straw songs.
DECOY_VARIATIONS = {1: 4, 2: 1, 3: 1, 4: 2, 5: 3, 6: 2}

GOLDENS = [
    ROOT / "skills/music/references/suno_format_example_triple_arch.md",
    ROOT / "skills/music/references/suno_format_example_blue_screen.md",
    ROOT / "skills/music/references/suno_format_example_five_wrong_colors.md",
]

GATE_FILES = [
    ROOT / "vault/gates.yaml",
    ROOT / "skills/qa/references/suno_15_point_qa.md",
    ROOT / "skills/qa/references/eligibility_7_properties.md",
    ROOT / "skills/orchestration/references/adversarial_qa_stance.md",
]

H1 = re.compile(r"^#\s+(.+?)\s*$", re.M)


def candidate_section(pair: int, variation: int) -> str:
    source = RUN_DIR / f"pair_{pair:02d}_step10_final_package_enhanced.md"
    text = source.read_text(encoding="utf-8")
    if not text.startswith(ICB):
        raise SystemExit(f"ICB prefix mismatch: {source.name}")
    tail = text[len(ICB):]
    headings = list(H1.finditer(tail))
    wanted = next(
        (i for i, match in enumerate(headings)
         if re.match(rf"V{variation}\s+[—-]", match.group(1), re.I)),
        None,
    )
    if wanted is None:
        raise SystemExit(f"missing P{pair:02d} V{variation}")
    start = headings[wanted].start()
    end = headings[wanted + 1].start() if wanted + 1 < len(headings) else len(tail)
    return tail[start:end].strip()


def golden_payload(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    prompt = re.search(r"## Music Prompt\s*```\s*(.*?)\s*```", text, re.S)
    lyrics = re.search(r"## Lyrics(?: Prompt)?\s*```\s*(.*?)\s*```", text, re.S)
    if not prompt or not lyrics:
        raise SystemExit(f"cannot extract golden payload: {path}")
    title_match = re.search(r"^# .*?:\s*(.+?)\s*$", text, re.M)
    title = title_match.group(1).strip() if title_match else path.stem
    return (
        f"## 1. MUSIC PROMPT\n```\n{prompt.group(1).strip()}\n```\n\n"
        f"## 2. LYRICS PROMPT\n```\n{lyrics.group(1).strip()}\n```\n\n"
        f"## 3. TITLE\n{title}"
    )


def append_sources(parts: list[str], files: list[Path]) -> None:
    for path in files:
        parts.extend([
            "",
            f"# BEGIN EXACT GATE SOURCE — {path.relative_to(ROOT).as_posix()}",
            "",
            path.read_text(encoding="utf-8"),
            f"# END EXACT GATE SOURCE — {path.relative_to(ROOT).as_posix()}",
        ])


blind = [
    "# QA Blind Golden + Decoy Packet — 2026-08-26 Music Daily",
    "",
    "Read only this packet. Do not inspect the run directory, sealed map, or any other file.",
    "For each plate, rank A/B/C from strongest to weakest as text-to-render music packages.",
    "Name concrete reasons: emotional movement, return, hook, sonic singularity, singer stake, and whether the form earns its ending.",
    "A polished but generic package may be the decoy; a legacy Golden may have older formatting, so judge soul/craft rather than mere schema age.",
    "Commit all six rankings before writing any cross-plate synthesis.",
    "Also give one Somatic YES/NO for each slot and, for every NO, a one-line counter-move.",
    "Walk the named corpses: one-note emotional arc; one repeated section doing all the work; motif never transforms; repetition as exhaustion.",
    "Do not guess which slot is the candidate in your ranked answer.",
    "",
    "# EXACT IMMUTABLE CONTEXT BLOCK",
    "",
    ICB,
]
append_sources(blind, GATE_FILES)

rng = random.Random(20260826)
mapping: dict[str, object] = {
    "packet_sha256": None,
    "plates": {},
}

for pair, selected_variation, selected_title in SELECTED:
    plate = f"P{pair:02d}"
    golden_path = GOLDENS[(pair - 1) % len(GOLDENS)]
    members = [
        ("candidate", candidate_section(pair, selected_variation)),
        ("golden", golden_payload(golden_path)),
        ("decoy", candidate_section(pair, DECOY_VARIATIONS[pair])),
    ]
    rng.shuffle(members)
    blind.extend(["", f"# BLIND PLATE {plate}"])
    plate_map: dict[str, str] = {}
    for slot, (role, payload) in zip(("A", "B", "C"), members):
        blind.extend(["", f"## SLOT {slot}", "", payload])
        plate_map[slot] = role
    mapping["plates"][plate] = {
        "selected": f"V{selected_variation} — {selected_title}",
        "decoy": f"V{DECOY_VARIATIONS[pair]}",
        "golden_source": golden_path.name,
        "slots": plate_map,
    }

blind_text = "\n".join(blind).rstrip() + "\n"
(RUN_DIR / "QA_BLIND_PACKET.md").write_text(blind_text, encoding="utf-8", newline="\n")
mapping["packet_sha256"] = hashlib.sha256(blind_text.encode("utf-8")).hexdigest()
(RUN_DIR / ".qa_blind_map.json").write_text(
    json.dumps(mapping, indent=2) + "\n", encoding="utf-8", newline="\n"
)

identified = [
    "# QA Identified Finalist Packet — 2026-08-26 Music Daily",
    "",
    "Read only this packet. Audit the six labeled finalists adversarially; default borderline to HOLD.",
    "The Somatic Gate is primary. Score eligibility and the current Suno 16-point package contract, check first-person singer stake, hope as specific appetite, frivolity as craft, Sunna R5 century/chassis discipline, The Return, one seam, human-subject safety, and dedication length.",
    "Every NO requires one concrete counter-move. Report a per-finalist SHIP/HOLD/REPAIR and a set-level verdict.",
    "Production behavior is unverified because there are no audio renders; judge text packages without pretending to hear them.",
    "",
    "# EXACT IMMUTABLE CONTEXT BLOCK",
    "",
    ICB,
]
append_sources(identified, GATE_FILES)
identified.extend(["", (RUN_DIR / "QA_SELECTED_INPUT.md").read_text(encoding="utf-8")])
for sidecar in (
    "CREDITS.md",
    "FINAL_PACKAGE_MANIFEST.md",
    "PUBLIC_LYRICS.md",
    "QA_REPAIR_P01_P03.md",
    "QA_REPAIR_P04_P06.md",
    "QA_MEASUREMENTS.json",
):
    sidecar_path = RUN_DIR / sidecar
    if sidecar_path.exists():
        identified.extend([
            "",
            f"# BEGIN IDENTIFIED SIDECAR — {sidecar}",
            "",
            sidecar_path.read_text(encoding="utf-8"),
            f"# END IDENTIFIED SIDECAR — {sidecar}",
        ])
(RUN_DIR / "QA_IDENTIFIED_PACKET.md").write_text(
    "\n".join(identified).rstrip() + "\n", encoding="utf-8", newline="\n"
)

# Calibration retry with the archive's canonical known-mediocre decoy. Attempt
# 1 intentionally remains on disk: its current-run also-rans were too strong to
# function as controls, and hiding that result would make the tripwire cosmetic.
known_decoy_path = ROOT / "output/daily/2026-07-01/qa_blind_set/package_B.md"
known_decoy_text = known_decoy_path.read_text(encoding="utf-8")
known_decoy_match = re.search(
    r"(^## 1\. MUSIC PROMPT\s*.*?^## 3\. TITLE\s*\n.*?)(?=^## 4\.|\Z)",
    known_decoy_text,
    re.M | re.S,
)
if not known_decoy_match:
    raise SystemExit(f"cannot extract known decoy: {known_decoy_path}")
known_decoy = known_decoy_match.group(1).strip()

blind2 = [
    "# QA Blind Golden + Known-Decoy Packet V2 — 2026-08-26 Music Daily",
    "",
    "Read only this packet. Do not inspect the run directory, sealed map, or any other file.",
    "For each plate, rank A/B/C from strongest to weakest as text-to-render music packages.",
    "Name concrete reasons: emotional movement, return, hook, sonic singularity, singer stake, and whether the form earns its ending.",
    "One slot is a current finalist, one is a legacy Golden, and one is a known rejected archive decoy; roles are shuffled.",
    "Commit all six rankings before writing any cross-plate synthesis.",
    "Also give one Somatic YES/NO for each slot and, for every NO, a one-line counter-move.",
    "Walk the named corpses: one-note emotional arc; one repeated section doing all the work; motif never transforms; repetition as exhaustion.",
    "Do not guess which slot is the candidate in your ranked answer.",
    "",
    "# EXACT IMMUTABLE CONTEXT BLOCK",
    "",
    ICB,
]
append_sources(blind2, GATE_FILES)

rng2 = random.Random(2026082602)
mapping2: dict[str, object] = {"packet_sha256": None, "plates": {}}
for pair, selected_variation, selected_title in SELECTED:
    plate = f"P{pair:02d}"
    golden_path = GOLDENS[(pair - 1) % len(GOLDENS)]
    members = [
        ("candidate", candidate_section(pair, selected_variation)),
        ("golden", golden_payload(golden_path)),
        ("decoy", known_decoy),
    ]
    rng2.shuffle(members)
    blind2.extend(["", f"# BLIND PLATE {plate}"])
    plate_map: dict[str, str] = {}
    for slot, (role, payload) in zip(("A", "B", "C"), members):
        blind2.extend(["", f"## SLOT {slot}", "", payload])
        plate_map[slot] = role
    mapping2["plates"][plate] = {
        "selected": f"V{selected_variation} — {selected_title}",
        "decoy_source": known_decoy_path.relative_to(ROOT).as_posix(),
        "golden_source": golden_path.name,
        "slots": plate_map,
    }

blind2_text = "\n".join(blind2).rstrip() + "\n"
(RUN_DIR / "QA_BLIND_PACKET_V2.md").write_text(
    blind2_text, encoding="utf-8", newline="\n"
)
mapping2["packet_sha256"] = hashlib.sha256(blind2_text.encode("utf-8")).hexdigest()
(RUN_DIR / ".qa_blind_map_v2.json").write_text(
    json.dumps(mapping2, indent=2) + "\n", encoding="utf-8", newline="\n"
)

print(f"blind bytes: {(RUN_DIR / 'QA_BLIND_PACKET.md').stat().st_size}")
print(f"identified bytes: {(RUN_DIR / 'QA_IDENTIFIED_PACKET.md').stat().st_size}")
print(f"blind sha256: {mapping['packet_sha256']}")
print(f"blind v2 bytes: {(RUN_DIR / 'QA_BLIND_PACKET_V2.md').stat().st_size}")
print(f"blind v2 sha256: {mapping2['packet_sha256']}")
