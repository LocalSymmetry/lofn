from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parent
MUSIC = RUN / "music"
PERSONALITY = ROOT / "skills" / "orchestration" / "personalities" / "lofn-prime-mini.yaml"


def b(text: str) -> bytes:
    return text.replace("\r\n", "\n").encode("utf-8")


def read(name: str) -> bytes:
    return (RUN / name).read_bytes()


parts: list[bytes] = [
    b(
        "# ===== BEGIN CREATIVE CONTEXT (inject verbatim into EVERY step 00–11) =====\n\n"
        "## User request\n\n"
        "> Please update the Lofn skills in Codex to match with Claude then do a Lofn Prime only daily run\n\n"
        "## Routing\n\n"
        "Music-only daily by established local Prime-only precedent. LOFN-PRIME is the sole personality. "
        "No image lane, guest personality, render, publication, upload, or spend is authorized.\n\n"
        "## Advanced directive / Meta-Prompt\n\n"
    ),
    read("04_orchestrator_metaprompt.md"),
    b("\n\n## Golden Seed\n\n"),
    read("02_golden_seed.md"),
    b(
        "\n\n## Personality — complete LOFN-PRIME YAML, verbatim\n\n"
        "The following byte sequence is the entire canonical personality file. Current repo doctrine "
        "(One Seam, The Return, Grain Law, credit and human-subject standards) overrides retired "
        "collision examples inside its historical archive.\n\n```yaml\n"
    ),
    PERSONALITY.read_bytes(),
    b("\n```\n\n## Panel Ledger — exact 18 voices and 15 Special Flairs\n\n"),
    read("03_orchestrator_panel_debate.md"),
    b("\n\n## User input / research brief\n\n"),
    read("00_research_brief.md"),
    b("\n\n## Pair assignments\n\n"),
    read("05_orchestrator_pair_assignments.md"),
    b("\n\n## Audio handoff and Golden Song payloads\n\n"),
    read("06_audio_handoff.md"),
    b(
        "\n\n## Seed Genre Palette\n\n"
        "```tab-delim\n"
        "ACCESSIBLE AWE\tluminous chamber art-pop\ttense breakbeat pop\twarm abstract hip-hop\n"
        "AMBITIOUS INDIGNATION\tUK-garage pressure\tmonodic industrial art-song\tdub-techno undertow\n"
        "HISTORICAL-MEDIUM ANCHORS\tchamber ensemble\tbreakbeat craft\thip-hop pocket\tgarage swing\tmonody\tdub spatial practice\n"
        "INTRUSIONS — ONE PER PAIR\tdry announcement\tshelf impact\talarm harmonic\trelease click\tchisel pulse\tocean-field texture\n"
        "```\n\n"
        "## Seed Music Frames / Technique Palette\n\n"
        "```tab-delim\n"
        "RETURN\tidentical line/change harmony\tconsonant hammer\tledger rondo\tword-deletion debt\ttwo-script antiphony\treversed tidal refrain\n"
        "DURATION\texact target\textended accumulation\tbrief recognition flare\tmaterially shorter afterimage\n"
        "BODY RHYTHM\twipe\tbrace\ttally\treach\tchisel\tservice\n"
        "SEAM\tone dominant material\tat most one intrusion\tone junction carrying the meaning\n"
        "VOCAL\tadult female voice\tphysical proximity\tbreath and consonants as evidence\tregister rotates by pair\n"
        "```\n\n"
        "Supplied images: none; visual lane is scope-skipped.\n\n"
        "# ===== END CREATIVE CONTEXT =====\n"
    ),
]

payload = b"".join(parts)

if PERSONALITY.read_bytes() not in payload:
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

