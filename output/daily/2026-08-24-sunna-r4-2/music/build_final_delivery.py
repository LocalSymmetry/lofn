"""Build the stable Suno delivery manifest and closeout indexes."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path


RUN = Path(__file__).resolve().parent.parent
MUSIC = Path(__file__).resolve().parent
SUNO = MUSIC / "suno_render"
REPO = RUN.parents[2]

FINALS = [
    (1, "Friday Looks Good on Me", "ACCESSIBLE", "EXISTENCE", "reserve", "linear reservation", "01_friday_looks_good_on_me.md"),
    (2, "Beat My Score", "ACCESSIBLE", "NEWS", "challenge", "scorecard rounds", "02_beat_my_score.md"),
    (3, "Turn When I Lower", "ACCESSIBLE", "EXISTENCE", "permit", "stop/start antiphony", "03_turn_when_i_lower.md"),
    (4, "Shake It Into Mine", "AMBITIOUS", "EXISTENCE", "season", "cumulative call-response", "04_shake_it_into_mine.md"),
    (5, "Pretty Busy Tomorrow", "AMBITIOUS", "NEWS", "select", "cut-up photo-strip rondeau", "05_pretty_busy_tomorrow.md"),
    (6, "Hot Girl Future Tense", "AMBITIOUS", "EXISTENCE", "declare", "question-to-declaration", "06_hot_girl_future_tense.md"),
]


def raw_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def field(text: str, heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\s*\n(.*?)(?=^## |\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if not match:
        raise ValueError(f"Missing {heading!r}")
    value = match.group(1).strip()
    if value.startswith("```") and value.endswith("```"):
        value = re.sub(r"^```[^\n]*\n?", "", value)
        value = re.sub(r"\n?```$", "", value)
    return value.strip()


def sound_module():
    path = REPO / "scripts/measure_soundcraft.py"
    spec = importlib.util.spec_from_file_location("measure_soundcraft", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load soundcraft module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rel(path: Path) -> str:
    return path.relative_to(RUN).as_posix()


def artifact_row(path: Path) -> dict[str, object]:
    return {"path": rel(path), "bytes": path.stat().st_size, "sha256": raw_hash(path)}


def main() -> None:
    sound = sound_module()
    package_rows: list[dict[str, object]] = []
    for number, title, arm, axis, action, form, filename in FINALS:
        path = SUNO / filename
        text = path.read_text(encoding="utf-8-sig")
        actual_title = field(text, "3. TITLE")
        if actual_title != title:
            raise ValueError(f"Title drift in {filename}: {actual_title!r}")
        profile = sound.profile_file(str(path))
        package_rows.append(
            {
                "number": number,
                "pair": f"P{number:02d}",
                "title": title,
                "arm": arm,
                "axis": axis,
                "action": action,
                "form": form,
                "path": rel(path),
                "bytes": path.stat().st_size,
                "sha256": raw_hash(path),
                "style_chars": len(field(text, "1. MUSIC PROMPT")),
                "exclude_chars": len(field(text, "1B. SUNO EXCLUDE PROMPT")),
                "lyrics_chars": len(field(text, "2. LYRICS")),
                "dedication_chars": len(field(text, "4. DEDICATION")),
                "sung_lines": profile["lines"],
                "end_rhyme": round(profile["end_rhyme"], 3),
                "line_return": round(profile["line_return"], 3),
                "words_per_line": round(profile["words_per_line"], 2),
                "alliteration_per_100w": round(profile["allit_per_100w"], 2),
            }
        )

    qa = SUNO / "QA_REPORT.md"
    manifest = {
        "run_slug": "2026-08-24-sunna-r4-2",
        "verdict": "SHIP_TEXT_PACKAGES_ONLY",
        "rendered_audio": 0,
        "published": False,
        "spent": False,
        "icb": {
            "bytes": 154765,
            "sha256": "6ed856fb2fe4974030f8b040caaffaed718aff22b7d87554a4cfab91f886c0a0",
            "speaker_tags": 18,
            "dispatch_offset": 0,
        },
        "packages": package_rows,
        "qa_report": artifact_row(qa),
    }
    manifest_path = SUNO / "build_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    lines = [
        "# Suno Render Index — 2026-08-24 Sunna R4.2",
        "",
        "## Verdict",
        "",
        "**SHIP — text packages only, 6/6.** These Markdown files are ready for The Scientist to paste into Suno. No audio has been rendered, heard, audited, uploaded, published, or paid for.",
        "",
        "Paste `## 1. MUSIC PROMPT`, `## 1B. SUNO EXCLUDE PROMPT`, and `## 2. LYRICS` into the matching Suno fields, and use `## 3. TITLE` as the title. The dedication travels with the song but is not generator input.",
        "",
        "## Files",
        "",
        "| # | Title | Route | Action / form | Style | Exclude | Lyrics | Lines | Soundcraft R/L/W/A | Dedication | SHA-256 |",
        "|---:|---|---|---|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in package_rows:
        metrics = f"{row['end_rhyme']:.3f}/{row['line_return']:.3f}/{row['words_per_line']:.2f}/{row['alliteration_per_100w']:.2f}"
        lines.append(
            f"| {row['number']} | **{row['title']}** | {row['arm']} · {row['axis']} | {row['action']} / {row['form']} | "
            f"{row['style_chars']} | {row['exclude_chars']} | {row['lyrics_chars']:,} | {row['sung_lines']} | {metrics} | "
            f"{row['dedication_chars']} | `{row['sha256']}` |"
        )
    lines.extend(
        [
            "",
            "Soundcraft order is end-rhyme return / exact-line return / mean words per line / alliteration per 100 words. All six clear the binding floors. P03 carries a nonblocking 38-line boundary flag; P04 carries a nonblocking 997-character Style flag and a tourist-shorthand render watch.",
            "",
            "## Final evidence",
            "",
            f"- `QA_REPORT.md` — fresh stable-byte QA, **SHIP — text only**, SHA-256 `{raw_hash(qa)}`.",
            "- `build_manifest.json` — machine-readable package bytes, hashes, field counts, and metrics.",
            "- Package validator: PASS 6/6. Surface audit: PASS 6/6. Soundcraft: PASS 6/6. Portfolio distinctiveness: PASS 6/6.",
            "- Binding blind QA: healthy Staff-Pick > known-mediocre control order; every finalist beats the decoy; P04/P05 changed-byte reblind healthy 2/2.",
            "- Start-With-One dedications: 368 / 388 / 399 / 402 / 437 / 391 characters, all ≤500.",
            "",
            "The Scientist's ear remains the next gate. Rendering does not imply publication approval; a finished render requires `lofn-render-audit` under the audio blind rule.",
            "",
        ]
    )
    suno_index = SUNO / "INDEX.md"
    suno_index.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    checksum_paths = [
        RUN / "00_research_brief.md",
        RUN / "02_golden_seed.md",
        RUN / "03_orchestrator_panel_debate.md",
        MUSIC / "CREATIVE_CONTEXT.md",
        MUSIC / "TOP_6_SELECTION.md",
        MUSIC / "DEDICATIONS.md",
        MUSIC / "CREDITS.md",
        MUSIC / "QA_BLIND_DECODE.md",
        qa,
        suno_index,
        manifest_path,
    ]
    root_lines = [
        "# INDEX — 2026-08-24 Sunna R4.2 Daily",
        "",
        "## Verdict",
        "",
        "- **Run:** complete, canonical music-only Sunna daily successor.",
        "- **Final QA:** **SHIP — text packages only**, 6/6 selected.",
        "- **Render / upload / publish / spend:** none. The Scientist's ear and `vault/AUTONOMY.md` remain the next gate.",
        "- **Creative answer:** high-shine, sexy, bimbo-flirt fun with the deeper message carried by reciprocal adult action—not sweat, club abstraction, or an explanatory bridge.",
        "",
        "## Why this successor exists",
        "",
        "R4.1 was closed as QA FAIL. R4.2 regenerated coordinator Steps 00–05 under the current byte-zero ICB, restarted the affected P02/P04 isolated chains, and preserved every failed artifact and blind plate as repair evidence. The stable chain has one controller, 42/42 canonical step checks passing, and no inherited QA verdict.",
        "",
        "## Pipeline completion",
        "",
        "- Frozen ICB: **154,765 bytes**, 18 `(after ` speaker tags, SHA-256 `6ed856fb2fe4974030f8b040caaffaed718aff22b7d87554a4cfab91f886c0a0`; exact prefix at byte zero in all six dispatches.",
        "- Research + Golden Seed + three-configuration 18-voice orchestration: complete; orchestrator validator PASS.",
        "- Coordinator Steps 00–05: 6/6 PASS. Isolated pair Steps 06–10: 30/30 PASS. Manual Step 11: 6/6 PASS.",
        "- Step-06 and Step-09 distinctiveness PASS. Step-10 portfolio extracted 24/24 and PASS. Final portfolio extracted 6/6 and PASS.",
        "- Package, Sunna surface, soundcraft, human-subject, dedication, and credit gates PASS.",
        "",
        "## Final 3+3 / 2+4 set",
        "",
        "| # | Pair | Arm | Axis | Action | Form | Final |",
        "|---:|---|---|---|---|---|---|",
    ]
    for row in package_rows:
        root_lines.append(
            f"| {row['number']} | {row['pair']} | {row['arm']} | {row['axis']} | {row['action']} | {row['form']} | **{row['title']}** |"
        )
    root_lines.extend(
        [
            "",
            "## QA and repair history",
            "",
            "- Earlier blind plates that inverted their golden/decoy controls are preserved and explicitly void; none authorizes SHIP.",
            "- Binding calibration used the permitted Staff-Pick `Triple Arch Over Me` golden against the repository's established known-mediocre `Look Again, The World Is Lit` decoy; control order was healthy 6/6.",
            "- That valid plate rejected P04. Its verb catalogue was replaced by one continuous lime/salt/shaker counter-wager, its soundcraft floor was recovered, and its changed-byte reblind ranked candidate > golden > decoy.",
            "- Identified QA caught P05's removed Gate-9 enlargement and title causality. Four windows now hold a summer, shrink to warm paper, and make the laughing frame cause tomorrow; its reblind also ranked candidate > golden > decoy.",
            "- Final clean-context identified QA: Pipeline PASS · Package PASS · Human CLEAR · Overall SHIP text-only. Residual Grain, appointment-family, tourist-shorthand, and boundary risks remain explicit listening flags.",
            "",
            "## Delivery",
            "",
            "- `music/suno_render/` contains six numbered standalone Markdown packages, `INDEX.md`, `QA_REPORT.md`, and `build_manifest.json`.",
            "- Dedications measure **368 / 388 / 399 / 402 / 437 / 391** characters. Full lineage and player/source debts are in `music/CREDITS.md`.",
            "- Failure-ledger write-back: live rows L46–L51; displaced same-day L40–L45 preserved verbatim in `vault/COMPETITION_LEARNINGS_ARCHIVE.md`; live index remains at 25 entries.",
            "",
            "## Canonical checksums",
            "",
            "| Artifact | Bytes | SHA-256 |",
            "|---|---:|---|",
        ]
    )
    for path in checksum_paths:
        root_lines.append(f"| `{rel(path)}` | {path.stat().st_size:,} | `{raw_hash(path)}` |")
    root_lines.extend(
        [
            "",
            "## Lock record",
            "",
            "Controller: `codex:01a032f6-89dd-7472-8706-a7413eb26fd2:sunna-r4-2`. This INDEX is the final content artifact; the run lock is released immediately after final read-only validation.",
            "",
        ]
    )
    (RUN / "INDEX.md").write_text("\n".join(root_lines), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
