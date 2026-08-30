from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


RUN = Path(__file__).resolve().parent
MUSIC = RUN / "music"
ICB_PATH = MUSIC / "CREATIVE_CONTEXT.md"
ICB = ICB_PATH.read_bytes() if ICB_PATH.exists() else b""


@dataclass(frozen=True)
class Artifact:
    step: str
    pair: str
    modality: str
    path: Path
    icb_prefix: bool = False


def artifact(step: str, pair: str, modality: str, rel: str, icb_prefix: bool = False) -> Artifact:
    return Artifact(step, pair, modality, RUN / rel, icb_prefix)


def expected_artifacts() -> list[Artifact]:
    rows = [
        artifact("R0", "-", "research", "00_research_brief.md"),
        artifact("P1", "-", "phase1", "01_seed_lineage.md"),
        artifact("P2", "-", "phase1", "02_golden_seed.md"),
        artifact("P3", "-", "phase1", "03_orchestrator_panel_debate.md"),
        artifact("P4", "-", "phase1", "04_orchestrator_metaprompt.md"),
        artifact("P5", "-", "phase1", "05_orchestrator_pair_assignments.md"),
        artifact("P6", "-", "phase1", "06_audio_handoff.md"),
        artifact("ICB", "-", "shared", "music/CREATIVE_CONTEXT.md"),
        artifact("PRE", "-", "shared", "preflight.md"),
    ]
    for step in range(6):
        names = [
            "step00_aesthetics_and_genres.md",
            "step01_essence_and_facets.md",
            "step02_concepts.md",
            "step03_artist_and_critique.md",
            "step04_medium.md",
            "step05_refine_medium.md",
        ]
        rows.append(artifact(f"C{step:02d}", "-", "music", f"music/{names[step]}", True))
    for pair in range(1, 7):
        pid = f"P{pair:02d}"
        rows.append(artifact("B05", pid, "music", f"music/p{pair:02d}_step05_isolated_brief.md", True))
        suffixes = {
            1: ["step06_facets", "step07_song_guides", "step08_drafts", "step09_refinement", "step10_final_package"],
            2: ["step06_facets", "step07_song_guides", "step08_generation", "step09_artist_refined", "step10_final_package"],
            3: ["step06_facets", "step07_song_guides", "step08_generation", "step09_artist_refined", "step10_final_package"],
            4: ["step06_facets", "step07_song_guides", "step08_drafts", "step09_refinement", "step10_final_package"],
            5: ["step06_facets", "step07_song_guides", "step08_generation", "step09_artist_refined", "step10_final_package"],
            6: ["step06_facets", "step07_song_guides", "step08_generation", "step09_artist_refined", "step10_final_package"],
        }[pair]
        for offset, suffix in enumerate(suffixes, start=6):
            rows.append(artifact(f"{offset:02d}", pid, "music", f"music/pair_{pair:02d}_{suffix}.md", True))
        rows.append(artifact("S11P", pid, "music", f"music/pair_{pair:02d}_step11_dispatch_prompt.md", True))
        rows.append(artifact("S11", pid, "music", f"music/pair_{pair:02d}_step10_selected_enhanced.md", True))
    for path in sorted(MUSIC.glob("*.QUARANTINED_ATTEMPT_*.md")):
        match = re.search(r"pair_(\d{2})_step(\d{2})", path.name)
        rows.append(Artifact(f"Q{match.group(2) if match else '?'}", f"P{match.group(1) if match else '?'}", "music", path, True))
    rows.extend(
        [
            artifact("SEL", "-", "music", "music/selected_candidates.json"),
            artifact("SEL", "-", "music", "music/PAIR_SELECTION_LEDGER.md"),
            artifact("DED", "-", "music", "music/DEDICATIONS.md"),
            artifact("CRD", "-", "music", "music/CREDITS.md"),
            artifact("TOP", "-", "music", "music/TOP_6_FINAL.md"),
            artifact("BKEY", "-", "qa", "music/blind_mapping_PRIVATE.json"),
        ]
    )
    for n in range(1, 7):
        rows.append(artifact("BQA", "-", "qa", f"music/qa_blind_sets/set_{n:02d}.md"))
    rows.extend(
        [
            artifact("QAB", "-", "qa", "music/BLIND_FIRST_READ.md"),
            artifact("QA", "-", "qa", "music/QA_REPORT.md"),
            artifact("IDX", "-", "shared", "INDEX.md"),
        ]
    )
    if (MUSIC / "selected_candidates.json").exists():
        try:
            selected = json.loads((MUSIC / "selected_candidates.json").read_text(encoding="utf-8"))
            for item in selected:
                pid = item["pair_id"]
                pair_num = int(pid[1:])
                matches = sorted((MUSIC / "selected").glob(f"*_{pid}.md"))
                path = matches[0] if len(matches) == 1 else MUSIC / "selected" / f"MISSING_UNIQUE_{pid}.md"
                rows.append(Artifact("DEL", f"P{pair_num:02d}", "music", path))
        except (OSError, KeyError, TypeError, ValueError):
            pass
    return rows


def repair_attempts(path: Path) -> int:
    if not path.exists():
        return 0
    if ".QUARANTINED_ATTEMPT_" in path.name:
        return 3
    attempts = 1
    attempts += len(list(path.parent.glob(path.name + ".repair_attempt_*.md")))
    if path.name == "QA_REPORT.md":
        attempts += len(list(path.parent.glob("QA_REPORT_ATTEMPT_*.md")))
    elif (MUSIC / "QA_REPORT_ATTEMPT_1.md").exists() and (
        path == ICB_PATH
        or path.name.startswith("pair_")
        or path.name.startswith("p0")
        or path.name.startswith("step0")
        or path.name.startswith(("03_orchestrator", "04_orchestrator", "05_orchestrator", "06_audio"))
    ):
        attempts += 1
    return attempts


def qa_verdict(text: str) -> str:
    match = re.search(r"\*\*Overall:\*\*\s*\*\*(SHIP|HOLD|REPAIR|FAIL)(?: REQUIRED)?\*\*", text, re.I)
    if not match:
        match = re.search(r"^##?\s*(?:Overall\s+)?Verdict\s*[:—-]\s*\*\*(SHIP|HOLD|REPAIR|FAIL)", text, re.I | re.M)
    return match.group(1).upper() if match else "PRESENT"


def inspect(item: Artifact) -> tuple[bool, int, str, str, str]:
    if not item.path.is_file():
        return False, 0, "-", "NOT RUN", "pending"
    raw = item.path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if ".QUARANTINED_ATTEMPT_" in item.path.name:
        gate = "FAIL:QUARANTINED"
    elif item.path.name == "QA_REPORT.md":
        gate = qa_verdict(raw.decode("utf-8", errors="replace"))
    elif len(raw) < 128:
        gate = "FAIL:TRIVIAL"
    elif item.icb_prefix and (not ICB or not raw.startswith(ICB)):
        gate = "FAIL:ICB_PREFIX"
    else:
        gate = "PASS"
    if gate == "FAIL:QUARANTINED":
        status = "quarantined"
    else:
        status = "done" if gate in {"PASS", "SHIP", "HOLD", "REPAIR", "FAIL", "PRESENT"} else "pending"
    return True, len(raw), sha, gate, status


def main() -> None:
    rows: list[str] = []
    done = pending = quarantined = 0
    for item in expected_artifacts():
        exists, size, sha, gate, status = inspect(item)
        if status == "done":
            done += 1
        elif status == "quarantined":
            quarantined += 1
        else:
            pending += 1
        rel = item.path.relative_to(RUN).as_posix()
        rows.append(
            f"| {item.step} | {item.pair} | {item.modality} | `{rel}` | "
            f"{'yes' if exists else 'NO'} | {size} | `{sha}` | {gate} | "
            f"{repair_attempts(item.path)} | {status} |"
        )
    icb_sha = hashlib.sha256(ICB).hexdigest() if ICB else "-"
    out = [
        "# RUN_STATE — 2026-08-25 Sunna Modern Swagger",
        "",
        "> Metadata only. Rebuilt by stat-ing disk; disk wins if this cache disagrees.",
        "> ICB-prefix PASS proves byte continuity, not creative quality.",
        "",
        f"**ICB sha:** `{icb_sha}` ({len(ICB)} bytes)",
        f"**Artifacts:** {done} done · {pending} pending · {quarantined} quarantined",
        "",
        "| step | pair | modality | canonical_path | exists | byte_size | sha | gate_verdict | attempt_count | status |",
        "|---|---|---|---|---|---:|---|---|---:|---|",
        *rows,
        "",
        "Resume rule: rebuild from disk, then resume only pending or explicitly repaired artifacts. No render, upload, publish, or spend authority is recorded here.",
        "",
    ]
    (RUN / "RUN_STATE.md").write_text("\n".join(out), encoding="utf-8", newline="\n")
    print(f"RUN_STATE rebuilt: {done} done, {pending} pending, {quarantined} quarantined; ICB {icb_sha}")


if __name__ == "__main__":
    main()
