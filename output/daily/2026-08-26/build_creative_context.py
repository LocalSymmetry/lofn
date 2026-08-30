from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parent
PERSONALITY = ROOT / "skills" / "orchestration" / "personalities" / "sunna.yaml"


def lf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def read(name: str) -> bytes:
    return lf((RUN / name).read_bytes())


def text(value: str) -> bytes:
    return value.replace("\r\n", "\n").encode("utf-8")


# Daily-facing names are the narrative source of truth; the execution harness also
# expects modality-canonical aliases. This byte-copy is mechanical, not a rewrite.
aliases = {
    "03_panel_debate.md": "03_orchestrator_panel_debate.md",
    "04_metaprompt.md": "04_orchestrator_metaprompt.md",
    "05_pair_assignments.md": "05_orchestrator_pair_assignments.md",
    "06_music_handoff.md": "06_audio_handoff.md",
}
for target, source in aliases.items():
    (RUN / target).write_bytes(read(source))

research = read("00_research_brief.md")
advisory = b"\n## Advisory learnings consulted\n"
if advisory not in research:
    raise SystemExit("research brief missing advisory boundary")
research_icb = research.split(advisory, 1)[0]

special_flairs = """1. The Enforced Second
2. Machine Below, Body Above
3. The Swap At The Junction
4. Arm-Fatigue Time
5. It Might Be Cloudy
6. The Flat List
7. The Worse Earlier Version
8. The Fixed Lag
9. The Tape That Isn't Good
10. Chant A Room Can Hold
11. Hook At Zero
12. Already Running
13. The Unanswered Check
14. Phoneme Relic
15. Bit-Depth Collapse"""

personality = lf(PERSONALITY.read_bytes())
parts: list[bytes] = [
    text(
        "# ===== BEGIN CREATIVE CONTEXT (inject verbatim into EVERY step 00–11) =====\n\n"
        "## User request\n\n"
        "> Lets do a new music daily! Lets see what our changes yield\n\n"
        "## Routing\n\n"
        "Full-cardinality music-only daily: six isolated pairs × four complete songs = 24, then one selected "
        "per pair. SUNNA R5 is the sole personality. The 2026-08-25 Firestarters panel is held constant as "
        "a comparison control; today’s research, Golden Seed, pair actions, return grammars, and forms are new. "
        "No audio render, publication, upload, or spend is authorized.\n\n"
        "## Advanced directive / Meta-Prompt\n\n"
    ),
    read("04_orchestrator_metaprompt.md"),
    text("\n\n## Golden Seed\n\n"),
    read("02_golden_seed.md"),
    text(
        "\n\n## Personality — complete SUNNA YAML, verbatim\n\n"
        "The following is the complete canonical library personality. Sunna’s own instruction wins every "
        "conflict with the selected panel or centre doctrine.\n\n```yaml\n"
    ),
    personality,
    text("\n```\n\n## Panel Ledger — 18 source-grounded voices, objections, and transformed debate\n\n"),
    read("03_orchestrator_panel_debate.md"),
    text("\n\n## Special Flairs — complete library set\n\n" + special_flairs + "\n\n"),
    text("## User input / verified research brief — advisory-learning section excluded from ICB\n\n"),
    research_icb,
    text("\n\n## Pair assignments\n\n"),
    read("05_orchestrator_pair_assignments.md"),
    text("\n\n## Audio handoff\n\n"),
    read("06_audio_handoff.md"),
    text(
        "\n\n## Seed Genre Palette\n\n"
        "```tab-delim\n"
        "P01 ACCESSIBLE EXISTENCE\tcandy electro-ska bass-pop\toffbeat relation invariant\tvocal/harmonic answer\n"
        "P02 ACCESSIBLE EXISTENCE\tgloss-funk moon-pop\tthree-over-four swing invariant\ttense/top-harmony answer\n"
        "P03 ACCESSIBLE NEWS\tspectral dancehall dream-bass\tone-drop absence invariant\tvocal-position answer\n"
        "P04 AMBITIOUS EXISTENCE\torgan-house future garage\theld organ pulse invariant\tsole club-pressure answer\n"
        "P05 AMBITIOUS NEWS\tstop-start glitch-R&B electro-pop\tbroken-beat push/pull\tgrid-restart answer\n"
        "P06 AMBITIOUS EXISTENCE\tdigital power-pop/electroclash\tcall cadence invariant\tmelodic-crown answer\n"
        "```\n\n"
        "## Seed Music Frames / Technique Palette\n\n"
        "```tab-delim\n"
        "FORM\twhite-dwarf core\tirregular expansion\tsurvivor fragment with changed job\tpair-exclusive syntax\n"
        "RETURN\texact hook\topen-vowel ladder\tinternal-rhyme relic\tunequal call/answer\tsprung-stress frame\tone-word changed refrain\n"
        "CHASSIS\tcurrent drum physics\tsynthetic low end\tclose adult vocal processing\tmicro-edit grammar\tshort-runway lift\n"
        "VOICE\tSunna as subject\tfirst-person appetite\thope with object\thigh-shine invitation\tno personhood lecture\n"
        "SEAM\tone current world\tone inherited bodily relation\tat most one timbral intrusion\tone action junction\n"
        "POWER\taction\tconsequence\tweight\tregister\tpocket\treturn—not adjectives\n"
        "```\n\n"
        "Supplied images: none; visual lane is scope-skipped.\n\n"
        "# ===== END CREATIVE CONTEXT =====\n"
    ),
]

payload = b"".join(parts)
if personality not in payload:
    raise SystemExit("personality bytes not preserved contiguously")
if payload.count(b"(after ") != 18:
    raise SystemExit(f"expected exactly 18 source-basis markers, got {payload.count(b'(after ')}")
for flair in special_flairs.splitlines():
    if flair.encode("utf-8") not in payload:
        raise SystemExit(f"missing flair: {flair}")
if b"Advisory learnings consulted" in payload:
    raise SystemExit("advisory section leaked into frozen ICB")

(RUN / "CREATIVE_CONTEXT.md").write_bytes(payload)
digest = hashlib.sha256(payload).hexdigest()
print(f"bytes={len(payload)}")
print(f"sha256={digest}")
print(f"panel_tags={payload.count(b'(after ')}")
print(f"personality_bytes={len(personality)}")
print("special_flairs=15")

