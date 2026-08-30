from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parent
MUSIC = RUN / "music"
PERSONALITY = ROOT / "skills" / "orchestration" / "personalities" / "sunna.yaml"


def norm(text: str) -> bytes:
    return text.replace("\r\n", "\n").encode("utf-8")


def read(name: str) -> bytes:
    return (RUN / name).read_bytes().replace(b"\r\n", b"\n")


research = read("00_research_brief.md")
advisory_marker = b"\n## Advisory learnings consulted\n"
if advisory_marker not in research:
    raise SystemExit("research brief missing advisory boundary")
research_icb = research.split(advisory_marker, 1)[0]

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

parts: list[bytes] = [
    norm(
        "# ===== BEGIN CREATIVE CONTEXT (inject verbatim into EVERY step 00–11) =====\n\n"
        "## User request\n\n"
        "> I want to do another Sunna daily. I think she overindexed on sounding old. This is still happening. "
        "Alexis used ska, raggae and 90s music but modernized it in EDM. She brought modern swagger to that "
        "music. Lets see if Sunna can do the same. Learn from Alexis's use of her genres, and try to make "
        "something similar.\n\n"
        "## Routing\n\n"
        "Full-cardinality music-only Sunna daily. SUNNA is the sole personality and actual first-person subject. "
        "Alexis contributes only an abstract modernization method; no Alexis lyric or prompt payload enters a "
        "generator context. No image lane, render, publication, upload, or spend is authorized.\n\n"
        "## Modernization study — user correction made executable\n\n"
    ),
    read("MODERNIZATION_STUDY.md"),
    norm("\n\n## Advanced directive / Meta-Prompt\n\n"),
    read("04_orchestrator_metaprompt.md"),
    norm("\n\n## Golden Seed\n\n"),
    read("02_golden_seed.md"),
    norm(
        "\n\n## Personality — complete SUNNA YAML, verbatim\n\n"
        "The following is the entire canonical library personality. Sunna's own instruction wins any conflict "
        "with the selected panel or centre doctrine.\n\n```yaml\n"
    ),
    PERSONALITY.read_bytes().replace(b"\r\n", b"\n"),
    norm("\n```\n\n## Panel Ledger — 18 voices, objections, and synthesis\n\n"),
    read("03_orchestrator_panel_debate.md"),
    norm("\n\n## Special Flairs — complete library set\n\n" + special_flairs + "\n\n"),
    norm("## User input / verified research brief — advisory-learning section excluded from ICB\n\n"),
    research_icb,
    norm("\n\n## Pair assignments\n\n"),
    read("05_orchestrator_pair_assignments.md"),
    norm("\n\n## Audio handoff\n\n"),
    read("06_audio_handoff.md"),
    norm(
        "\n\n## Seed Genre Palette\n\n"
        "```tab-delim\n"
        "P01 ACCESSIBLE NEWS\telectro-ska bass-pop\ttwo-tone upstroke invariant\tresampled skank gate\n"
        "P02 ACCESSIBLE EXISTENCE\tnew-jack-swing bass-pop\tswung sixteenth invariant\tcurrent FM sub and clipped clap\n"
        "P03 ACCESSIBLE EXISTENCE\tdigital dancehall electro-pop\tone-drop hole invariant\twavetable stab and pitch-chop return\n"
        "P04 AMBITIOUS NEWS\tspeed-garage bass-pop\t90s rave-piano chord intrusion\tsole club-derived bed\n"
        "P05 AMBITIOUS EXISTENCE\twavetable baroque-sequence glitch-pop\trising sequence invariant\tno chamber ensemble\n"
        "P06 AMBITIOUS EXISTENCE\telectroclash glam/power-pop\tstomp and three-note contour invariant\tsynth-guitar hybrid\n"
        "```\n\n"
        "## Seed Music Frames / Technique Palette\n\n"
        "```tab-delim\n"
        "MODERNITY\tcurrent drum engine\tcurrent low-end physics\tcurrent vocal posture\tmicro-edit language\tshort drop geometry\n"
        "RETURN\tpartial-circle completion\treply ladder\tone-drop answer fill\tfour-slot delayed reply\tsequence note gain\tpast-to-present flip\n"
        "VOICE\tadult female chest lead\tshort lines\thard doubles\tone processed shadow answer\tno era acting\n"
        "SEAM\tone modern world\tat most one ancestral timbre intrusion\tone action junction\n"
        "POWER\tconsequence\tweight\tregister\tpocket\tresponse\treturn—not loudness adjectives\n"
        "```\n\n"
        "Supplied images: none; visual lane is scope-skipped.\n\n"
        "# ===== END CREATIVE CONTEXT =====\n"
    ),
]

payload = b"".join(parts)
personality_bytes = PERSONALITY.read_bytes().replace(b"\r\n", b"\n")
if personality_bytes not in payload:
    raise SystemExit("personality bytes were not preserved as a contiguous exact sequence")
if payload.count(b"(after ") != 18:
    raise SystemExit(f"expected 18 panel speaker tags, got {payload.count(b'(after ')}")
for flair in special_flairs.splitlines():
    if flair.encode("utf-8") not in payload:
        raise SystemExit(f"missing special flair: {flair}")
if b"Advisory learnings consulted" in payload:
    raise SystemExit("advisory learnings leaked into ICB")

MUSIC.mkdir(parents=True, exist_ok=True)
targets = [RUN / "CREATIVE_CONTEXT.md", MUSIC / "CREATIVE_CONTEXT.md"]
for target in targets:
    target.write_bytes(payload)

if targets[0].read_bytes() != targets[1].read_bytes():
    raise SystemExit("creative-context copies differ")

digest = hashlib.sha256(payload).hexdigest()
print(f"bytes={len(payload)}")
print(f"sha256={digest}")
print(f"panel_tags={payload.count(b'(after ')}")
print(f"personality_bytes={len(personality_bytes)}")
print("special_flairs=15")

