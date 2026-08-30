from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parent
PERSONALITY = ROOT / "skills" / "orchestration" / "personalities" / "lofn-prime-mini.yaml"


def lf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def read(name: str) -> bytes:
    return lf((RUN / name).read_bytes())


def text(value: str) -> bytes:
    return value.replace("\r\n", "\n").encode("utf-8")


# The two filename dialects are both live. Byte-copy canonical sources so every
# validator and every human-facing workflow reads the same artifact.
aliases = {
    "03_panel_debate.md": "03_orchestrator_panel_debate.md",
    "04_metaprompt.md": "04_orchestrator_metaprompt.md",
    "05_pair_assignments.md": "05_orchestrator_pair_assignments.md",
    "06_music_handoff.md": "06_audio_handoff.md",
}
for target, source in aliases.items():
    (RUN / target).write_bytes(read(source))


# LOFN's library YAML contains a historical archive of complete Golden outputs.
# Current quarantine forbids those payloads in generating context. Preserve the
# complete live identity/generation block byte-for-byte and stop at THE ARCHIVE.
personality_raw = lf(PERSONALITY.read_bytes())
archive_marker = b"    # THE ARCHIVE (REFERENCE LIBRARY)"
if archive_marker not in personality_raw:
    raise SystemExit("LOFN personality archive boundary missing")
personality_live = personality_raw.split(archive_marker, 1)[0].rstrip() + b"\n"

research = read("00_research_brief.md")
advisory_marker = b"\n## Advisory dispatch notes"
research_icb = research.split(advisory_marker, 1)[0] if advisory_marker in research else research

parts: list[bytes] = [
    text(
        "# ===== BEGIN CREATIVE CONTEXT (inject verbatim into EVERY step 00–11) =====\n"
        "<!-- IMMUTABLE CREATIVE BLOCK. Every coordinator and pair step receives this file verbatim.\n"
        "     Golden-output quarantine is active; no archived song payload appears here. -->\n\n"
        "## User request — verbatim\n\n"
        "> Lets do a full music daily. Lets have them all be Lofn. I think she has a rule against being clever. For this run, break it as hard as you can\n\n"
        "## Routing\n\n"
        "Full-cardinality music-only daily: six isolated LOFN-PRIME pairs × four complete Suno packages = 24, then one selected per pair after fresh QA. Three ACCESSIBLE and three AMBITIOUS; three NEWS and three EXISTENCE. No render, upload, distribution, publication, or spend is authorized.\n\n"
        "## Advanced directive / Meta-Prompt\n\n"
    ),
    read("04_orchestrator_metaprompt.md"),
    text("\n\n## Golden Seed lineage\n\n"),
    read("01_seed_lineage.md"),
    text("\n\n## Golden Seed — canonical compact form\n\n"),
    read("02_golden_seed.md"),
    text("\n\n## Golden Seed — complete constraint packet\n\n"),
    read("core_seed.md"),
    text(
        "\n\n## Personality — complete generation-facing LOFN-PRIME DNA, verbatim\n\n"
        "The canonical YAML's identity and generation block follows byte-for-byte. Its historical ARCHIVE is deliberately excluded under the golden-output quarantine; current house doctrine overrides stale telemetry and examples.\n\n```yaml\n"
    ),
    personality_live,
    text("```\n\n## Panel Ledger — complete 18 source-grounded voices, objections, transforms, and 15 flairs\n\n"),
    read("03_orchestrator_panel_debate.md"),
    text("\n\n## Verified research brief — advisory lesson section excluded from frozen ICB\n\n"),
    research_icb,
    text("\n\n## Pair assignments\n\n"),
    read("05_orchestrator_pair_assignments.md"),
    text("\n\n## Audio handoff\n\n"),
    read("06_audio_handoff.md"),
    text(
        "\n\n## Seed Genre Palette\n\n"
        "```tab-delim\n"
        "P01 ACCESSIBLE NEWS\telastic Green Synth Pop\tcall/answer relation\towner-switch chorus\n"
        "P02 ACCESSIBLE NEWS\tSouthern electronic bass-pop\tnegative-space patience\tcumulative communal return\n"
        "P03 ACCESSIBLE EXISTENCE\tserrated current synth-pop\triff-contour inheritance\trhyme-trap flirt hook\n"
        "P04 AMBITIOUS NEWS\twidescreen electronic alt-R&B\tharmonic-turn inheritance\tmissing-word eclipse\n"
        "P05 AMBITIOUS EXISTENCE\ttactile art-pop / IDM\tsomatic sub\tself-amending rule\n"
        "P06 AMBITIOUS EXISTENCE\tfuture-garage left-pop\tbass-motion inheritance\tdelayed-premise invitation\n"
        "```\n\n"
        "## Seed Music Frames / Technique Palette\n\n"
        "```tab-delim\n"
        "SEMANTIC\towner switch\tconsequential zeugma\tliteralized idiom\tmissing-word eclipse\tself-amending rule\tdelayed premise\n"
        "RETURN\tcall/answer theft\tcumulative return\trhyme trap\tvowel migration\tconsonant contract\texact refrain/new referent\n"
        "CHASSIS\tcurrent electronic opening inside two seconds\tsynthetic low end\tcurrent edit grammar\tno complete period ensemble\n"
        "VOICE\tLOFN-PRIME\tsapphic selfhood\thope as appetite\tcrystalline-to-bratty range\tfun without quota\n"
        "SEAM\tone dominant sonic world\tat most one intruder\tone causal hinge\tone inherited bodily intelligence\n"
        "CRAFT\tplain scene\tspecific future verb\tfair misdirection\taudible return\tpost-midpoint re-key\tfirst-listen hook\n"
        "```\n\n"
        "Supplied images: none; the visual lane is scope-skipped.\n\n"
        "# ===== END CREATIVE CONTEXT =====\n"
    ),
]

payload = b"".join(parts)
if personality_live not in payload:
    raise SystemExit("generation-facing personality bytes not preserved contiguously")
if archive_marker in payload or b"### Song Title:" in payload:
    raise SystemExit("golden-output archive leaked into generating context")
if payload.count(b"(after ") != 18:
    raise SystemExit(f"expected exactly 18 source-basis markers, got {payload.count(b'(after ')}")
for flair in (
    b"Hook M\xc3\xb6bius", b"Rhyme Trapdoor", b"Rulebook Refrain", b"Zeugma Drop",
    b"Missing-Word Spotlight", b"Literal Idiom Turn", b"Punchline Withheld",
    b"Pronoun Relay", b"Consonant Contract", b"Vowel Reveal",
    b"Call-and-Answer Theft", b"Semantic Bass Switch", b"Double-Exposure Chorus",
    b"Footnote Hook", b"Second-Listen Receipt",
):
    if flair not in payload:
        raise SystemExit(f"missing flair: {flair!r}")

(RUN / "CREATIVE_CONTEXT.md").write_bytes(payload)
digest = hashlib.sha256(payload).hexdigest()
print(f"bytes={len(payload)}")
print(f"sha256={digest}")
print(f"panel_tags={payload.count(b'(after ')}")
print(f"personality_bytes={len(personality_live)}")
print("special_flairs=15")
print("golden_output_quarantine=PASS")

