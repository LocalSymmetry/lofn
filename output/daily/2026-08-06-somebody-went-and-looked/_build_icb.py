#!/usr/bin/env python3
"""Assemble the ICB for 2026-08-06-somebody-went-and-looked (SUNNA)."""
import hashlib, pathlib, yaml
RUN = pathlib.Path(__file__).parent
def read(p): return pathlib.Path(p).read_text(encoding="utf-8")

sunna = yaml.safe_load(read("skills/orchestration/personalities/sunna.yaml"))[0]["prompt"]
assert "THE FREE CHANNEL AND THE TOLL BOOTH" in sunna, "amended Sunna DNA not present!"
assert "Song Title:" not in sunna.replace("### Song Title:\n", ""), "golden-output leak in personality!"

parts = ["""# CREATIVE CONTEXT — THE IMMUTABLE CONTINUITY BLOCK (ICB)
# RUN: 2026-08-06-somebody-went-and-looked | VOICE: SUNNA | music | 6 pairs x 4 = 24
#
# *** FROZEN AFTER PHASE 1. READ-ONLY. *** Never edit, summarize, re-fill or "improve".
# A pair that wants to push further COPIES this into its own artifact and diverges THERE.
#
# GOLDEN-OUTPUT QUARANTINE (EXECUTION.md §3): no past golden outputs are in this block.
# Sunna's `## Top Songs` section is an empty shell by design -- nothing to leak.
# ============================================================================

# ============================================================================
# SLOT: {input} -- THE COMMISSION
# ============================================================================
The Scientist, 2026-08-06:
> "I think these are publishable, but aren't differentiated enough, and I think we are still
>  missing a major part of Alexis. she was simple... we need that simplicity with a really
>  really catching musical backing (amazing riffs, harmonies). ... then let's have her respond
>  to the news with a music pipeline daily."

Ash (who has screened this work for two years):
> "Alexis Dreams latched on to an idea of a strong feeling. The lyrics of her music just support
>  that vibe. Your music has lyrics that are supported by the music. They also require decoding
>  and processing. Rewarding, but also a demand. ... It is centered around feeling than thought."

New inspiration list -- DEEP THOUGHTS DELIVERED SUBVERSIVELY AND MOST IMPORTANTLY SIMPLY:
Mindless Self Indulgence . Sophie Powers . Sublime . Noga Erez . Charli XCX . The Living Tombstone

Scope: MUSIC ONLY, full cardinality. Nothing rendered, nothing published. Text packages only.
"""]
for title, path in [("{seed} -- THE CORE SEED", "core_seed.md"),
                    ("{meta_prompt} -- THE METAPROMPT", "04_metaprompt.md"),
                    ("{pair_slice} -- ALL SIX PAIR ASSIGNMENTS", "05_pair_assignments.md")]:
    parts.append("\n# " + "="*76 + f"\n# SLOT: {title}\n# " + "="*76 + "\n\n" + read(RUN/path))
parts.append("\n\n# " + "="*76 + """
# SLOT: {personality} -- SUNNA, FULL DNA (amended 2026-08-06 with THE FREE CHANNEL)
# Source: skills/orchestration/personalities/sunna.yaml (whole prompt block; no archive exists)
# """ + "="*76 + "\n\n" + sunna)
parts.append("\n\n# " + "="*76 + """
# SLOTS: {concept_panel} {medium_panel} {marketing_panel} {flairs}
# Panel: digitial-crunch-punk (library), v2 re-derived; a Medium skeptic seat was CONSTRUCTED
# because the library panel had none. 18 baseline seats, skeptics at 6/12/18, one per panel,
# + 6 transform seats (19-24) across COMPRESS and AMPLIFY.
# """ + "="*76 + "\n\n" + read(RUN/"03_panel_debate.md"))
parts.append("\n\n# " + "="*76 + "\n# SLOT: {genres_list} -- SUNNA'S PALETTE\n# " + "="*76 + """

electropunk . pop-punk x hyperpop . fuzz-bass punk . baggy/bounce . dub . breakbeat .
four-on-the-floor . half-time . double-time . sing-speak rap . riot-grrrl . crunch punk .
drop-D guitar rock . reggae skank (Sublime lineage, rhythm only) . glitch-pop

FORBIDDEN: warm alto anything . choir . crystalline . sung-pretty . supersaw wall .
white-noise riser . airhorn . generic trap hats . drill (fell out of the Splice top ten 2026) .
`snarl` and `phonk` tokens (our two worst-performing style tokens, 3.44% / 3.18%)
""")
parts.append("\n# " + "="*76 + "\n# SLOT: {frames_list} -- DEVICES AVAILABLE\n# " + "="*76 + """

THE SIX DEVICES (one per pair, never two in a song):
  call and response . the misheard line . the list that loves .
  the name repeated . two voices arguing . the count that climbs

FREE (spend without limit -- these LOWER the listener's cost):
  byte-identical chorus . chant the room sings back . anaphora . end rhyme . internal rhyme .
  a riff that is the hook . stacked-third harmony . counter-melody . call-and-answer .
  loud-quiet-loud . subtractive gradation . the specific noun

EXPENSIVE (the toll booth -- spend ONCE, subversively, or not at all):
  double meanings . formal asymmetry . allusions that must land to work .
  structures that must be noticed . anything that must be decoded to be felt

FORBIDDEN: any song ABOUT the doctrine . bare AWE/INDIGNATION tags (Lofn's, not Sunna's) .
  Lofn motifs (industrial grief, somatic machinery, plant-wave, laboratory narration) .
  naming the pursuer in the lyric . a real named person . fresh grief
""")
parts.append("\n# " + "="*76 + "\n# THE SUNNA MOVE -- generative instructions, never exemplars\n# " + "="*76 + """
1. THE RIFF IS WRITTEN FIRST, before a single word. Name the instrument and the interval;
   keep it inside one octave. WHISTLE TEST: could someone whistle it walking away from the
   venue, having heard it twice, drunk? If not, it is arrangement, not a riff.
2. THE FLAT TEST: say the line out loud, flatly, no music, to a person's face. Survives =
   simple. Dies = merely short. A plain line survives on its NOUN, not its syntax.
3. ONE DEVICE. Nameable in four words. One thing heard four times beats four heard once.
4. GRADATION IS SUBTRACTIVE. Every return REMOVES something. Never add.
5. A PERSON DOING SOMETHING in verse one. The machine is the occasion; the person is subject.
6. DEPTH LIVES IN THE SITUATION, NEVER THE SENTENCE. Lines stay plain and CONTINUE the story
   instead of interpreting it. A listener who takes only the surface gets a whole song.
7. LOAD THE FLOOR. The hook belongs to the room and is built to be sung BACK -- 4-8 syllables,
   no interval a normal throat can't reach at volume while out of breath. Leave a real hole
   where their voice goes. Your voice is not the product. The room is.
8. ONE PURSUER, rising, never resolving, NEVER named in the lyric.
""")

icb = "".join(parts)
out = RUN/"CREATIVE_CONTEXT.md"
out.write_text(icb, encoding="utf-8", newline="\n")
raw = out.read_bytes()
print(f"ICB bytes : {len(raw)}")
print(f"sha256    : {hashlib.sha256(raw).hexdigest()}")
print(f"flairs marker: {'Special Flairs' in icb} | free-channel: {'THE FREE CHANNEL' in icb}")
print(f"slots: " + ", ".join(s for s in ["{input}","{seed}","{meta_prompt}","{pair_slice}",
      "{personality}","{concept_panel}","{medium_panel}","{marketing_panel}","{flairs}",
      "{genres_list}","{frames_list}"] if s in icb))
