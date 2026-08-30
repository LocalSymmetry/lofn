from __future__ import annotations

import json
from pathlib import Path


MUSIC_DIR = Path(__file__).resolve().parent
RUN_DIR = MUSIC_DIR.parent


def main() -> None:
    icb = (MUSIC_DIR / "CREATIVE_CONTEXT.md").read_bytes()
    pairs = json.loads((MUSIC_DIR / "concept_medium_pairs.json").read_text(encoding="utf-8"))
    step05 = (MUSIC_DIR / "step05_refine_medium.md").read_bytes()
    dispatch_dir = MUSIC_DIR / "dispatch"
    dispatch_dir.mkdir(exist_ok=True)

    for pair in pairs:
        pair_id = pair["pair_id"]
        arm = pair["phase1_slot"]["arm"]
        suffix = (
            "\n\n# ISOLATED PAIR DISPATCH — PAIR "
            + pair_id
            + "\n\n"
            + "This packet is the complete injected creative prompt for one independent pair chain. "
              "The full ICB above is immutable. Work only on the selected pair below. Do not read or imitate "
              "any sibling pair's downstream artifacts.\n\n"
            + "## Current run contract\n\n"
            + "- Modality: MUSIC ONLY; LOFN-PRIME only; autonomous draft on disk; no audio rendering, publishing, uploading, or spend.\n"
            + f"- Pair: {pair_id}; slot: {arm}.\n"
            + "- Execute Steps 06, 07, 08, 09, 10, and the non-Fusion Step 11 enhancement as separate, sequential creative sub-steps.\n"
            + "- Current authority is six pairs × four complete candidate variations = 24 songs. Any legacy instruction asking for six guides/prompts per pair is superseded. Preserve four complete variations through Step 11.\n"
            + "- Every canonical step artifact must start with the ICB above byte-for-byte and then use the canonical provenance sections 0–7. Validate each step before advancing.\n"
            + "- Final positive Style prompt: one producer-grade paragraph, 850–1000 characters (target 870–960), no real artist names, no negative wording.\n"
            + "- Final Exclude prompt: separate comma-separated negative field, 400–900 characters, hard max 1000.\n"
            + "- Final Suno lyrics field begins with [Theme:] and [SONG FORM:], uses full EMO/performance headers, has at least 60 sung lines with 70–120 as the normal target, and is strictly under 5000 characters measured exactly.\n"
            + "- Preserve exact duration dramaturgy: opening, accumulation, brief flare, afterimage, with arithmetic accounting to the assigned runtime.\n"
            + "- The Return, Grain Law, One Seam, human-subject standard, Golden Move, and current producer-grade guidance are binding.\n"
            + "- Step 11 has veto authority: reject a structurally broken package rather than polishing it. Do not invoke Fusion or any paid model/tool.\n"
            + "- Golden-song payloads in the ICB are calibration evidence inherited from the orchestrator handoff; do not quote, copy, or reproduce their titles, lyrics, arrangements, keys, tempos, or vocal fingerprints in generated work. Use the Golden Move, not the mold.\n\n"
            + "## Selected pair — full structured object\n\n```json\n"
            + json.dumps(pair, ensure_ascii=False, indent=2)
            + "\n```\n\n"
            + "## Step 05 coordinator record\n\n"
            + "\n"
        )
        target = dispatch_dir / f"pair_{pair_id}_dispatch.md"
        target.write_bytes(icb + suffix.encode("utf-8") + step05 + b"\n")
        print(f"{target.name}\t{target.stat().st_size}")


if __name__ == "__main__":
    main()
