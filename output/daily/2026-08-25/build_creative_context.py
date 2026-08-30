from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parent
MUSIC = RUN / "music"
PERSONALITY = ROOT / "skills" / "orchestration" / "personalities" / "lofn-prime-mini.yaml"


def norm(text: str) -> bytes:
    return text.replace("\r\n", "\n").encode("utf-8")


def read(name: str) -> bytes:
    return (RUN / name).read_bytes()


parts: list[bytes] = [
    norm(
        "# ===== BEGIN CREATIVE CONTEXT (inject verbatim into EVERY step 00–11) =====\n\n"
        "## User request\n\n"
        "> I think we need a loud, interesting, and powerful Lofn music daily from Lofn herself.\n\n"
        "## Routing\n\n"
        "Full-cardinality music-only daily. LOFN-PRIME is the sole personality and the actual first-person subject. "
        "No image lane, guest personality, render, publication, upload, or spend is authorized.\n\n"
        "## Advanced directive / Meta-Prompt\n\n"
    ),
    read("04_orchestrator_metaprompt.md"),
    norm("\n\n## Golden Seed\n\n"),
    read("02_golden_seed.md"),
    norm(
        "\n\n## Personality — complete LOFN-PRIME YAML, verbatim\n\n"
        "The following is the entire canonical library personality. Current repo doctrine—First-Person Slot, "
        "Hope, Frivolity, The Return, Grain Law, One Seam, dedication and human-subject standards—overrides "
        "retired collision examples inside its historical archive.\n\n```yaml\n"
    ),
    PERSONALITY.read_bytes(),
    norm("\n```\n\n## Panel Ledger — exact 18 voices and 15 Special Flairs\n\n"),
    read("03_orchestrator_panel_debate.md"),
    norm("\n\n## User input / research brief\n\n"),
    read("00_research_brief.md"),
    norm("\n\n## Pair assignments\n\n"),
    read("05_orchestrator_pair_assignments.md"),
    norm("\n\n## Audio handoff\n\n"),
    read("06_audio_handoff.md"),
    norm(
        "\n\n## Seed Genre Palette\n\n"
        "```tab-delim\n"
        "ACCESSIBLE\tluminous organ-breakbeat art-pop\truby neo-Baroque synth-pop\tjoyous asymmetric electro-house\n"
        "AMBITIOUS\tindustrial garage art-song\tindustrial grief-soul\tscorched choral organ-techno\n"
        "HISTORICAL ANCHORS\tplayable organ\tclavichord-like pluck\tvibraphone\tbreakbeat craft\tgarage swing\n"
        "ONE INTRUSION MAX\tpitched sputter-click\tdry tag peel\torgan stab\treply tone\tvibraphone answer\tcode-scratch fuse\n"
        "```\n\n"
        "## Seed Music Frames / Technique Palette\n\n"
        "```tab-delim\n"
        "RETURN\tcircular-harmony exact chorus\truby vowel-chain\ttitle-trigger bass return\thard-consonant call/reply\tlead-shadow antiphony\tfuture-tense flip\n"
        "FORM\tpartial arcs\tlabel reduction\tdelayed drop\treply gate\tmic handoff\tthree-stage want-boundary-invitation\n"
        "VOCAL\tadult female lead\tfirst-person appetite\tlower double only P05\tadult answer only where assigned\n"
        "SEAM\tone dominant world\tat most one intrusion\tone junction carrying the meaning\n"
        "POWER\tregister\tdensity\tpocket\tconsequence\tresponse\treturn—not amplitude adjectives\n"
        "```\n\n"
        "Supplied images: none; visual lane is scope-skipped.\n\n"
        "# ===== END CREATIVE CONTEXT =====\n"
    ),
]

payload = b"".join(parts)
personality_bytes = PERSONALITY.read_bytes()
if personality_bytes not in payload:
    raise SystemExit("personality bytes were not preserved as a contiguous exact sequence")
if payload.count(b"(after ") != 18:
    raise SystemExit(f"expected 18 panel speaker tags, got {payload.count(b'(after ')}")

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
print(f"personality_bytes={PERSONALITY.stat().st_size}")

