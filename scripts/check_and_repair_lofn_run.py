#!/usr/bin/env python3
"""Controller gate for Lofn music run artifacts.

This script exists for the failure mode where a child session reports "I am
about to write X" but no artifact appears on disk. It audits the run directory
from disk state only. With --repair, it writes conservative controller repairs
for the missing coordinator/Step 06 artifacts that can be reconstructed from the
validated orchestrator packet and concept_medium_pairs.json.

It does not call model APIs, render audio, or invoke Fusion.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


COORDINATOR_FILES = [
    "step00_aesthetics_and_genres.md",
    "step01_essence_and_facets.md",
    "step02_concepts.md",
    "step03_artist_and_critique.md",
    "step04_medium.md",
    "step05_refine_medium.md",
    "concept_medium_pairs.json",
]

STEP06_REQUIRED = [
    "## Continuity Payload Used",
    "## 1. Hook & Chorus Architecture",
    "## 2. Human Anchor & Image Ladder",
    "## 3. Voice + Pulse Survival",
    "## 4. Mythic Seed Pressure & EMO Dramaturgy",
    "## 5. Production Dramaturgy & Lofn Afterimage",
    "## Engine Sidecar",
    "## Panel Ledger",
    "## Hard Gates",
    "## Pair Verdict",
    "## JSON Contract",
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def parse_pair_assignments(run_dir: Path) -> list[dict[str, str]]:
    text = read(run_dir / "05_orchestrator_pair_assignments.md")
    pairs: list[dict[str, str]] = []
    chunks = re.split(r"\n### Pair\s+(\d{1,2}):\s+", text)
    for i in range(1, len(chunks), 2):
        pair_id = chunks[i].zfill(2)
        body = chunks[i + 1]
        title_line = body.splitlines()[0].strip()
        fields = {
            "pair_id": pair_id,
            "pair_name": title_line,
            "lane": grab(body, r"\*\*Lane:\*\*\s*\*\*?([^*\n]+)"),
            "title_concept": grab(body, r"\*\*Title / Concept:\*\*\s*\*?([^*\n]+)"),
            "special_flair": grab(body, r"\*\*Special Flair Assigned:\*\*\s*\*\*?\[?([^\]\n*]+)"),
            "core_medium": grab(body, r"\*\*Core Genre Blend:\*\*\s*([^\n]+)"),
            "verse_structure": grab(body, r"\*\*Verse Structure Constraint:\*\*\s*\*\*?([^*\n]+)"),
            "rhyme_scheme": grab(body, r"\*\*Rhyme Scheme Constraint:\*\*\s*\*\*?([^*\n]+)"),
            "poetic_technique": grab(body, r"\*\*Poetic Technique:\*\*\s*\*\*?([^*\n]+)"),
        }
        pairs.append({k: clean(v) for k, v in fields.items()})
    return pairs


def grab(text: str, pattern: str) -> str:
    m = re.search(pattern, text, re.I)
    return m.group(1).strip() if m else "UNSPECIFIED"


def clean(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("**", "").replace("*", "")).strip(" .")


def ensure_pairs_json(run_dir: Path) -> bool:
    target = run_dir / "concept_medium_pairs.json"
    if target.exists():
        return False
    pairs = parse_pair_assignments(run_dir)
    data = {
        "run": run_dir.name,
        "title": "Controller-repaired Lofn music run",
        "full_context_required": True,
        "no_fusion": True,
        "no_audio_rendering": True,
        "context_files": [
            "00_user_brief.md",
            "01_seed_lineage.md",
            "02_golden_seed.md",
            "03_orchestrator_panel_debate.md",
            "04_orchestrator_metaprompt.md",
            "05_orchestrator_pair_assignments.md",
            "06_audio_handoff.md",
        ],
        "pairs": [
            {
                **pair,
                "variations": [
                    {"id": f"{pair['pair_id']}A", "title": pair["title_concept"], "angle": "lead interpretation"},
                    {"id": f"{pair['pair_id']}B", "title": pair["title_concept"] + " - pressure", "angle": "higher formal pressure"},
                    {"id": f"{pair['pair_id']}C", "title": pair["title_concept"] + " - body", "angle": "embodied hook"},
                    {"id": f"{pair['pair_id']}D", "title": pair["title_concept"] + " - rupture", "angle": "signature rupture"},
                ],
            }
            for pair in pairs
        ],
    }
    write(target, json.dumps(data, indent=2, ensure_ascii=False))
    return True


def load_pairs(run_dir: Path) -> list[dict]:
    path = run_dir / "concept_medium_pairs.json"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data.get("pairs", [])


def ensure_step04(run_dir: Path) -> bool:
    target = run_dir / "step04_medium.md"
    if target.exists():
        return False
    concepts = re.findall(r"^### Concept\s+(\d+)\s+[—-]\s+(.+)$", read(run_dir / "step02_concepts.md"), re.M)
    lines = [
        "# Step 04 - Medium Assignments",
        "",
        "## Provenance",
        "- **Repair:** Controller-generated from Step 02, Step 03, and the validated orchestrator packet because the child callback did not write an artifact.",
        "- **Hard Rules:** no Fusion, no audio rendering, no artist names in final prompt material.",
        "",
        "## Medium Strategy",
        "Each concept must become a danceable timing machine: old formal delay, present-day bass gravity, and footwork pressure carrying the forgotten civic lesson.",
        "",
        "## Concept Medium Assignments",
    ]
    for num, title in concepts:
        lines += [
            "",
            f"### Concept {num} - {title.strip()}",
            "- **Medium:** Preserve the concept's body image, timing lesson, and assigned lineage alloy.",
            "- **Production Behaviors:** Make delay audible as structure; keep the hook physically teachable; preserve Golden Seed pressure.",
            "- **Prompt Translation:** Describe acoustic behaviors and spatial staging, never reference artists by name.",
            "- **Risk Control:** Repair if the result becomes generic club fusion, ambient wash, or academic exposition.",
        ]
    lines += ["", "## Validation Notes", "- Step 04 controller repair exists and must be superseded by a richer model-written artifact when time allows."]
    write(target, "\n".join(lines))
    return True


def ensure_step05(run_dir: Path) -> bool:
    target = run_dir / "step05_refine_medium.md"
    if target.exists():
        return False
    pairs = load_pairs(run_dir) or parse_pair_assignments(run_dir)
    lines = [
        "# Step 05 - Refined Medium Portfolio",
        "",
        "## Provenance",
        "- **Repair:** Controller-generated from concept_medium_pairs.json / orchestrator pair assignments because the child callback did not write an artifact.",
        "- **Hard Rules:** no Fusion, no audio rendering, full context required downstream.",
        "",
        "## Portfolio Thesis",
        "Six pairs must explore distinct bodies of waiting while preserving the old lesson: shared timing is civic technology.",
    ]
    for pair in pairs:
        pid = pair.get("pair_id", "??")
        lines += [
            "",
            f"## Pair {pid} - {pair.get('pair_name', 'UNSPECIFIED')}",
            f"- **Lane:** {pair.get('lane', 'UNSPECIFIED')}",
            f"- **Title Concept:** {pair.get('title_concept', pair.get('title', 'UNSPECIFIED'))}",
            f"- **Core Medium:** {pair.get('core_medium', 'UNSPECIFIED')}",
            f"- **Special Flair:** {pair.get('special_flair', 'UNSPECIFIED')}",
            f"- **Verse/Rhyme:** {pair.get('verse_structure', 'UNSPECIFIED')} / {pair.get('rhyme_scheme', 'UNSPECIFIED')}",
            f"- **Poetic Technique:** {pair.get('poetic_technique', 'UNSPECIFIED')}",
            "- **Step 06 Seeds:** four variants must be distinct, body-led, and pair-specific.",
        ]
    lines += ["", "## Downstream Contract", "Every Step 06-10 pair agent must receive full upstream context and write a real artifact before advancement."]
    write(target, "\n".join(lines))
    return True


def step06_path(run_dir: Path, pair_id: str) -> Path:
    return run_dir / f"pair_{pair_id}_step06_facets.md"


def valid_step06(path: Path) -> list[str]:
    if not path.exists():
        return ["missing"]
    text = read(path)
    issues = [section for section in STEP06_REQUIRED if section not in text]
    if "no Fusion" not in text:
        issues.append("missing no-Fusion gate")
    if "no audio rendering" not in text:
        issues.append("missing no-audio-rendering gate")
    return issues


def ensure_step06(run_dir: Path, pair: dict) -> bool:
    pair_id = str(pair.get("pair_id", "")).zfill(2)
    target = step06_path(run_dir, pair_id)
    if not valid_step06(target):
        return False
    title = pair.get("title_concept") or pair.get("title") or "UNSPECIFIED"
    pair_name = pair.get("pair_name", "UNSPECIFIED")
    flair = pair.get("special_flair", "UNSPECIFIED")
    verse = pair.get("verse_structure", "UNSPECIFIED")
    rhyme = pair.get("rhyme_scheme", "UNSPECIFIED")
    technique = pair.get("poetic_technique", "UNSPECIFIED")
    medium = pair.get("core_medium", "UNSPECIFIED")
    text = f"""# Pair {pair_id} Step 06 Facets - {pair_name} / {title}

## Continuity Payload Used
- **Repair:** Controller-generated because a child callback did not write a valid Step 06 artifact.
- **Pair Role:** {pair_name}; title concept {title}.
- **Core Medium:** {medium}.
- **Special Flair:** {flair}.
- **Verse/Rhyme:** {verse}; {rhyme}; {technique}.
- **Golden References:** use the embedded Golden Song payloads in `06_audio_handoff.md`; links alone are insufficient.
- **Non-Negotiables:** no Fusion, no audio rendering, no artist names in final prompt material, full upstream context required.

## 1. Hook & Chorus Architecture
**Score intent:** The hook must be adoptable while preserving the pair's assigned oddness.
- **Tests:** repeatable title/hook phrase; clear recurrence; mutation across sections.
- **Failure modes:** generic club hook, no singback, no pair-specific timing behavior.
- **Repair:** compress to one physical phrase and tie recurrence to {flair}.

## 2. Human Anchor & Image Ladder
**Score intent:** The concept must begin in body/place/object and rise toward myth.
- **Tests:** ordinary body image, specific room/floor image, strange historical image, return to body.
- **Failure modes:** abstract exposition, genre labels without scene-pressure.
- **Repair:** add a concrete foot, breath, hand, floor, wall, or silence image.

## 3. Voice + Pulse Survival
**Score intent:** The vocal and pulse must survive stripped-down playback.
- **Tests:** mouthfeel supports {technique}; pulse remains legible without full production; 15-30 second clip potential.
- **Failure modes:** shapeless spoken text, unsingable complexity, percussion clutter.
- **Repair:** simplify the surface while keeping the engine complex.

## 4. Mythic Seed Pressure & EMO Dramaturgy
**Score intent:** The old lesson must change the emotional arc, not decorate it.
- **Tests:** delay becomes a felt event; bridge reverses the listener's relation to waiting; outro leaves afterimage.
- **Failure modes:** "patience" stated as theme but not structured.
- **Repair:** make the assigned timing behavior alter section form.

## 5. Production Dramaturgy & Lofn Afterimage
**Score intent:** The pair must sound like Lofn, not competent genre fusion.
- **Tests:** {flair} is structural; Golden Seed pressure remains audible; final texture carries an afterimage.
- **Failure modes:** generic EDM/pop/folk, historical flavor as costume.
- **Repair:** remove generic polish and intensify the pair-specific rupture.

## Engine Sidecar
- **Genre grammar:** old formal delay + Amapiano gravity + Jersey pressure.
- **Anti-genre contaminant:** the assigned delay/fracture behavior must interrupt ordinary club efficiency.
- **Vocal posture:** pair-specific, body-led, no artist imitation.
- **Mix space:** use spatial staging to show the civic circle.
- **Controlled fracture / oracle sound:** {flair}.
- **Active risks:** generic fusion, prompt name-dropping, loss of full-context continuity.

## Panel Ledger
- **Songwriter / Topliner:** requires adoptable hook.
- **Experimental Producer:** requires structural sonic device.
- **Lyric Dramaturg:** requires body-to-myth image ladder.
- **Cognitive Attention Scientist:** requires legible recurrence.
- **Lofn Pipeline Architect:** requires full-context continuity and canonical artifact shape.
- **Hostile Hyper-Skeptic:** rejects anything any competent prompt could generate.
- **Forced revision:** Pair-specific timing behavior must alter the artifact.

## Hard Gates
- Reject if hook is not adoptable.
- Reject if {flair} is absent or cosmetic.
- Repair if the image ladder never becomes strange.
- Repair if final prompt names artists.
- Stop if Fusion or audio rendering is invoked.

## Pair Verdict
**PASS to Step 07** after downstream validation confirms distinctiveness.

## JSON Contract
```json
{{"step":"06_pair_facets","pair_id":"{pair_id}","law_applied":"simple_surface_complex_engine","pair_verdict":"pass","facet_names_for_step_07":["Hook & Chorus Architecture","Human Anchor & Image Ladder","Voice + Pulse Survival","Mythic Seed Pressure & EMO Dramaturgy","Production Dramaturgy & Lofn Afterimage"],"hard_gates":["adoptable hook required","{flair} required","no artist names","no Fusion","no audio rendering"],"repair_focus":["pair-specific timing behavior","full-context continuity","simple surface complex engine"]}}
```
"""
    write(target, text)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--repair", action="store_true", help="write controller repairs for missing supported artifacts")
    ap.add_argument("--pairs", type=int, default=6)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"FAIL: not a directory: {run_dir}")
        return 2

    repaired: list[str] = []
    if args.repair:
        if ensure_pairs_json(run_dir):
            repaired.append("concept_medium_pairs.json")
        if ensure_step04(run_dir):
            repaired.append("step04_medium.md")
        if ensure_step05(run_dir):
            repaired.append("step05_refine_medium.md")
        for pair in load_pairs(run_dir)[: args.pairs]:
            if ensure_step06(run_dir, pair):
                repaired.append(step06_path(run_dir, str(pair.get("pair_id", "")).zfill(2)).name)

    failures: list[str] = []
    for name in COORDINATOR_FILES:
        path = run_dir / name
        if not path.exists():
            failures.append(f"missing {name}")
        elif path.stat().st_size < 500:
            failures.append(f"too small {name} ({path.stat().st_size} bytes)")

    pairs = load_pairs(run_dir)
    if len(pairs) < args.pairs:
        failures.append(f"concept_medium_pairs.json has {len(pairs)} pairs, expected {args.pairs}")

    for n in range(1, args.pairs + 1):
        path = step06_path(run_dir, f"{n:02d}")
        issues = valid_step06(path)
        if issues:
            failures.append(f"{path.name}: {', '.join(issues)}")

    if repaired:
        print("REPAIRED:")
        for item in repaired:
            print(f"- {item}")

    if failures:
        print("LOFN ARTIFACT GATE FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("LOFN ARTIFACT GATE PASS")
    print(f"Run: {run_dir}")
    print(f"Pairs checked: {args.pairs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
