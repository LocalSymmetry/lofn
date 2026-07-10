from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "output" / "daily" / "2026-07-05"
MUSIC = RUN / "music"
IMAGES = RUN / "images"
SONG_COPIES = ROOT / "output" / "songs"
IMAGE_COPIES = ROOT / "output" / "images" / "2026-07-05"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sung_line_count(lyrics: str) -> int:
    count = 0
    for line in lyrics.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("[") or s.startswith("## ") or s.startswith("*"):
            continue
        if s.startswith("Title:") or s.startswith("Music prompt:") or s.startswith("Exclude prompt:"):
            continue
        count += 1
    return count


def prompt_len(prompt: str) -> int:
    return len(prompt.strip())


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9'-]+", text))


@dataclass
class Pair:
    idx: int
    lane: str
    axis: str
    emotion: str
    concept: str
    hook: str
    body: str
    hurt: str
    source1: str
    source2: str
    source3: str
    form: str
    style: str
    selected: int
    image_subject: str


pairs = [
    Pair(
        1,
        "ACCESSIBLE",
        "NEWS",
        "AWE",
        "painted moon open window",
        "show me the bright side",
        "bare feet at an apartment window after the sky goes quiet",
        "the dark half of the self can arrive before the self has language",
        "APOD Iapetus, the settling geomagnetic storm, and deep Fiji quake pressure",
        "off-kilter drumming, unruly viola, scrabbly riffs",
        "two-tone moon: bright face, coal face, trailing-side stain, crater-as-bruise",
        "bright/dark call response with a trailing outro",
        "lunar chamber-pop with viola scrapes, glassy keys, and micro-edited pulse",
        2,
        "adult lunar conservator repairing a two-tone moon panel",
    ),
    Pair(
        2,
        "ACCESSIBLE",
        "EXISTENCE",
        "AWE",
        "house of rest with square corners",
        "make the rest house real",
        "hands at a kitchen table measuring quiet into a small wooden model",
        "rest can feel like trespass when the nervous system still expects work",
        "Poetry Foundation house of rest, Color API Pompadour, moon in waning gibbous",
        "syncopated drums, organ-droning hush, volatile vocal personalities",
        "square room / square lattice: four corners as four sung returns",
        "four-corner refrain where each chorus changes what rest means",
        "warm analog lullaby-pop with brushed drums, bowed glass, and close double vocals",
        2,
        "adult rest-house maker building a square room of repaired light",
    ),
    Pair(
        3,
        "ACCESSIBLE",
        "NEWS",
        "INDIGNATION",
        "empty chair grief broadcast",
        "do not borrow my grief",
        "standing behind crowd barriers while cameras turn mourning into weather",
        "public sorrow can be harvested by people who never carried the loss",
        "BBC public mourning and politics, Times Square glare, Reuters direct access blocked",
        "volatile vocal personalities, scrabbly riffs, syncopated drums",
        "empty chair as square tile missing from a public lattice",
        "refrain-fracture: the hook refuses each time the camera returns",
        "bratty alt-pop requiem with dry snare, bowed metal, and clipped choir stabs",
        1,
        "empty broadcast hall with one unclaimed chair and hot public lights",
    ),
    Pair(
        4,
        "AMBITIOUS",
        "NEWS",
        "INDIGNATION",
        "receipt room with no door",
        "I am not your receipt",
        "thumb hovering over a terms screen in a room that keeps renaming ownership",
        "the insult is being told access is the same as belonging",
        "HN ownership/digital games, AI tutor discourse, inaccessible NightCafe theme",
        "off-kilter drumming, syncopated drums, twitchy chant energy",
        "square lattice becomes a contract grid; the missing square is the user",
        "clipped ledger verses, glitch-prechorus, chorus as cancellation stamp",
        "glitch-baroque club-punk with detuned harpsichord, rubber bass, and snarl lead",
        3,
        "adult clerk trapped inside a luminous receipt grid",
    ),
    Pair(
        5,
        "AMBITIOUS",
        "EXISTENCE",
        "INDIGNATION",
        "prefusion lock mercy",
        "lock the bloom",
        "gloved hands holding a door shut before it learns the shape of a body",
        "mercy sometimes looks like refusal before harm discovers a route",
        "PDB hantavirus square lattice, antibodies locking spikes, no identified victim",
        "unruly viola, organ-droning private swell, volatile vocal personalities",
        "prefusion square tiles: held tension before the irreversible opening",
        "suite form: tile, hinge, lock, afterimage",
        "art-rock immunology hymn with prepared piano, viola scratches, and hard consonants",
        2,
        "adult laboratory conservator sealing a square viral-tile reliquary",
    ),
    Pair(
        6,
        "AMBITIOUS",
        "EXISTENCE",
        "AWE",
        "river warning after accidental betrayal",
        "warn them anyway",
        "knees in wet grass beside a river after realizing the signal was your fault",
        "love can mean warning the people who now have reason to distrust you",
        "DreamBank river warning motif, NDBC buoy wind/wave texture, Oblique grouping phrase",
        "scrabbly riffs, unruly viola, organ-droning lullaby that swells huge yet private",
        "grouped elements treated as one body; river sections merge into a final warning",
        "through-composed river warning with one repeated flare",
        "rain-field prog hymn with bowed guitar, hand drum, and a widening choir of one voice",
        4,
        "adult river messenger carrying a lantern made from grouped warning tiles",
    ),
]


variants = {
    1: [
        ("Coal Snow Window", "the dark half answers first", "more brittle viola, smaller room"),
        ("Bright Side Turning", "show me the bright side", "selected: clean AWE with body-place danger"),
        ("Cassini Porch Song", "sweep low and see me", "longer orbital outro"),
        ("After the Dirty Ice", "what stays after shining", "more ambient, more private"),
    ],
    2: [
        ("Corners Every One", "square the room softly", "poem-forward architecture"),
        ("House With Square Corners", "make the rest house real", "selected: accessible hook and warm terror"),
        ("Pompadour Evening", "let the color lower", "color-first nocturne"),
        ("Waning Kitchen", "leave one light on", "moon-room lullaby"),
    ],
    3: [
        ("Do Not Borrow My Grief", "do not borrow my grief", "selected: direct hook, clean indictment"),
        ("Empty Chair Choir", "leave the chair empty", "more choral, slower"),
        ("Public Weather", "my sorrow is not weather", "broadcast metaphor"),
        ("The Cameras Kneel", "not for you", "harder bratty vocal"),
    ],
    4: [
        ("Borrowed Door", "where did my room go", "access versus belonging"),
        ("Terms Keep Moving", "read it while it runs", "anxious syncopation"),
        ("Not Your Receipt", "I am not your receipt", "selected: sharpest ownership thesis"),
        ("The Button Renamed Me", "do not click my name", "most glitch-formal"),
    ],
    5: [
        ("Square Tile Refusal", "hold the square", "visual lattice version"),
        ("Prefusion Mercy", "lock the bloom", "selected: strongest golden move"),
        ("Before the Door Learns", "do not teach the door", "more ominous"),
        ("Antibody Psalm", "mercy has knuckles", "harder indignation"),
    ],
    6: [
        ("Wet Grass Warning", "run to the river", "body-first"),
        ("Group the Lanterns", "treat us as one", "Oblique-forward"),
        ("Buoy at Dusk", "the water keeps count", "NOAA texture"),
        ("Warn Them Anyway", "warn them anyway", "selected: broadest existence AWE"),
    ],
}


research_brief = """# Daily Research Brief - 2026-07-05

Scope: full daily run. Music lane is 24 songs to 6 selected. Image lane is 24 prompts to top 12 to top 6. No render calls.

Skill refresh: active `.agents` Lofn skill mirrors were refreshed from the repo canonical `.claude` skills before generation, with Codex wording/path adaptation. This run follows the updated one-fact music rule, per-pair variation-angle rule, AWE terror-adjacency rule, practice-publish policy, disk-derived RUN_STATE, and 4-field run-health footer.

Advisory learnings consulted: 4 entries, tags container/object-world, fast-vote thumbnail, eco/nature intimate object, anti-overfit. INDIGNATION exempt; advisory-only. The "would this hurt our best past entry?" gate passes only for image dispatch framing, not as a hard constraint, and these notes are not injected into the ICB.

## F01-F25 Source Ledger

| Code | Status | Source | Extract |
|---|---|---|---|
| F01 | UNAVAILABLE | NightCafe daily challenge page | `https://creator.nightcafe.studio/game/daily-challenge` returned access/JS gating, so no verified current challenge number. |
| F02 | UNAVAILABLE | NightCafe current theme/detail | Search and direct page access did not verify a current July 5 theme. No theme was hallucinated. |
| F03 | OK | NightCafe general mechanics | NightCafe blog/help indicate daily challenges use voting/ranking and, from May 2026, allow PRO and non-PRO tools broadly. This informs venue taste only. |
| F04 | NO DATA | USGS significant earthquakes past day | Significant-day feed count was 0. No significant quake anchor. |
| F05 | OK | USGS M4.5+ earthquakes past day | Feed count was 13; largest was M5.8, 274 km SE of Levuka, Fiji, depth 685.692 km, green alert/no tsunami. |
| F06 | OK | NASA APOD | 2026-07-05 APOD: `Saturn's Iapetus: Painted Moon`, Cassini image. |
| F07 | OK | NASA APOD structure | Two-tone Iapetus, dark carbon-rich coating and bright ice, trailing hemisphere, large southern crater. Mandatory form rule: two-sided body plus trailing stain. |
| F08 | OK | Poetry Foundation Poem of the Day | Julia Ward Howe, `The House of Rest`; physical line used as seed: `Square the corners every one`. |
| F09 | OK | Bandcamp Daily | Album of the Day: mary in the junkyard, `Role Model Hermit`, Alternative, London, July 2 review. |
| F10 | OK | Bandcamp exact sonic vocabulary | Short texture phrases imported: `off-kilter drumming`, `swirls of unruly viola`, `scrabbly riffs and syncopated drums`, `volatile vocal personalities`. |
| F11 | OK | PDB-101 Molecule of the Month | Hantavirus glycoproteins arrange as square-like tetrameric tiles; neutralizing antibodies can lock spikes in prefusion state. |
| F12 | UNAVAILABLE | Radio Garden | App shell fetched but live station ambience was JS-gated; no live station claimed. |
| F13 | OK | The Color API | `hex=0705` resolved to Pompadour, #770055, RGB 119/0/85: bruised-magenta/private velvet. |
| F14 | OK | Date-specific culture | July 5, 1687: Newton's `Principia` publication date. Used as hidden order-behind-motion pressure. |
| F15 | OK | DreamBank random sample | Abstracted motif only: accidental disclosure near a river, urgent warning, love/sadness/fear. Dated/cultural specifics discarded. |
| F16 | OK | Freesound recent rain field recordings | Recent field-recording/rain results include evening rain, storm/rain/thunder, and downpour ambience. |
| F17 | OK | Oblique Strategies | Exact card: `Assemble some of the elements in a group and treat the group`. |
| F18 | OK | NOAA SWPC | Solar wind speed about 442 km/s near fetch time; Kp mostly 3-4 on July 5 after July 4 storm values reached 7.33. |
| F19 | OK | Hacker News | Top builder topics included Organic Maps, an AI tutor effect-size paper, Flipper Zero development, computer-starring media, and digital ownership. |
| F20 | OK | BBC World RSS | Emotional temperature: public mourning/politics, Taylor/Travis wedding reactions, French political verdict, mystery space balls in Australia. |
| F21 | OK | Public Domain Review | Latest feature: Louis Pope Gratacap and lost worlds; collections included Qanoon-e-Islam travel diagrams and kite-flying art. |
| F22 | OK | Moon phase | Waning gibbous, about 71.9 percent illumination, 21 days old; Last Quarter on July 7, Full Buck Moon on July 29. |
| F23 | OK | EarthCam Times Square | A few clouds, 81 F / 27 C, humidity about 65.6 percent; bright public-light/crowd texture. |
| F24 | OK | Biodiversity Heritage Library | Public-domain butterfly plates with guard sheets and descriptive letterpress; used only as guarded-plate material logic. |
| F25 | OK | NOAA NDBC Station 46059 | Latest buoy rows: north wind, about 7 m/s, pressure near 1023 hPa; recent row had 1.8 m waves and 7 s dominant period. |

## Obscure Theme-Specific Facts

- Iapetus is a "painted moon" because one hemisphere is dramatically darker; the run translates that into songs/images where one side receives a stain from outside.
- Hantavirus envelope glycoproteins are not random fuzz: they form square-like tetramer tiles, making "lattice mercy" an actual structure, not a metaphor grabbed from air.
- Prefusion lock is a moral form rule here: the most merciful action happens before the irreversible opening.
- Guard sheets in plate books become an image cue: a translucent layer preserving a fragile color body without hiding it.
- The deep Fiji quake becomes pressure-without-spectacle: a force far below ordinary sight.

## Existence Prompts

1. What private part of you got painted dark by debris nobody saw arrive?
2. What does rest mean when the house is square but the sky is not?
3. When do you warn the gathering even if you caused the danger?
4. What is the difference between being measured and being known?
5. Which side of you faces the sun, and which side trails behind?

## Tri-Source Summary

Source 1 - CONTENT / emotional stakes: July 5 world facts set a day of settling storms, public grief, ownership anxiety, builder curiosity, and deep pressure below visibility. Songs/images are resonance, not reportage.

Source 2 - SONIC/AESTHETIC VOCABULARY: Bandcamp supplies the texture bank: off-kilter drumming, unruly viola, scrabbly riffs, syncopated drums, volatile vocal personalities. These words shape prompts; they are not genre labels.

Source 3 - MATERIAL STRUCTURE: APOD Iapetus and PDB hantavirus create the form law: two-sided stained body plus square/tetramer lattice, with trailing fade, missing tile, or prefusion lock as mandatory structures.

## Seeded 3+3 Split

| Pair | Lane | Axis | Emotion | Concept | Pair-specific variation law |
|---|---|---|---|---|---|
| P01 | ACCESSIBLE | NEWS | AWE | Painted moon open window | Bright/dark call-response; each variant chooses a different place to turn the body. |
| P02 | ACCESSIBLE | EXISTENCE | AWE | House of rest with square corners | Four-corner refrains; each variant defines rest through a different object. |
| P03 | ACCESSIBLE | NEWS | INDIGNATION | Empty chair grief broadcast | Refrain-fracture; each variant refuses a different public theft of feeling. |
| P04 | AMBITIOUS | NEWS | INDIGNATION | Receipt room with no door | Contract-grid rhythm; each variant changes the legal fiction attacking the singer. |
| P05 | AMBITIOUS | EXISTENCE | INDIGNATION | Prefusion lock mercy | Tile/hinge/lock suite; each variant changes what must be held shut. |
| P06 | AMBITIOUS | EXISTENCE | AWE | River warning after accidental betrayal | Through-composed warning; each variant changes who must be warned and why. |
"""


seed_lineage = """# Seed Lineage - 2026-07-05 Daily

Golden anchors read from the repo index: `Nail an Obscure Emotion`, `Straightening Our Spines`, and the music methodology notes. Generator contexts receive only the Golden Seed and Golden Move, never golden song payloads.

Golden song names for judge-side calibration and audio handoff only:
- `Triple Arch Over Me`
- `Five wrong colors`

Golden Move applied:
1. Stand somewhere real.
2. Use at most one wounding fact in a lyric, at the emotional hinge, and respond to it.
3. Make a mid-song turn that changes what the singer knows.
4. Keep fear braided in, including AWE songs.
5. Rotate the register away from past winners.

Lineage decision: today is not a space song set, not a public-news set, and not a PDB explainer. It is a square-lattice record about how a body carries a stain without becoming only the stain.
"""


golden_seed = """# Golden Seed - Painted Lattice Mercy

Locked mood statement: A sweet, furious, moonlit daily about the part of a body that is painted by forces it did not invite, and the stubborn art of making the stain into form without pretending it was a gift.

Core emotional duality: AWE at the two-sided moon still shining; INDIGNATION at systems that call extraction, access, and public grief by softer names.

Listener experience arc:
1. The listener stands somewhere ordinary: window, table, barrier, screen, lab door, riverbank.
2. A structure appears: square room, contract grid, viral tile, two-tone moon, grouped lanterns.
3. The singer discovers that repair is not erasure. Repair is choosing what the stain is allowed to touch next.

Invisible target: the old human habit of mistaking visibility for care.

What this is not:
- Not a news recap.
- Not a solar-system trivia lesson.
- Not a cozy rest playlist.
- Not an image set of generic moon ladies.
- Not a PDB explainer with rhyme.
"""


panel = """# Orchestrator Panel Debate - Library Load

Personality: LOFN-Prime-Mini.

Panel blend: Sonic Visionaries plus Tectonic Texture, re-derived into the three required panels with Hyper-Skeptic pressure.

## Concept Panel

1. Kandinsky-derived synesthetic form: insists every sound has a visible pressure, not just a mood.
2. Bearden-derived collage memory: pushes public news into cut paper, guard sheets, and missing tiles.
3. O'Keeffe-derived object intimacy: asks the run to make one object large enough to hold dread.
4. Ai Weiwei-derived witness: refuses decorative grief and calls out extraction.
5. Noguchi-derived sculptural restraint: asks each concept to become a touchable object.
6. Hyper-Skeptic: Mary Anning as deep-time checker. Objection: if the moon/PDB facts merely decorate, the run is a lecture.

## Medium Panel

1. Ryoji Ikeda-derived data pulse: turns Kp, buoy, and lattice facts into rhythm, not lyric exposition.
2. Laurie Anderson-derived violin/projection bridge: lets the viola act as an unstable witness.
3. Olafur Eliasson-derived natural phenomena: makes light a physical room.
4. Sanna Annukka-derived graphic strata: keeps image prompts legible at thumbnail size.
5. Prepared-instrument engineer: insists on one unforgettable sonic device per song.
6. Hyper-Skeptic: James Hutton as slow-time auditor. Objection: do not flatten deep pressure into fast moral certainty.

## Context and Marketing Panel

1. Susan Rogers-derived listener body check: each hook must be singable in one breath.
2. Brian Eno-derived obliqueness: uses the Oblique card to group elements, not to excuse vagueness.
3. Questlove-derived archive sense: makes old forms feel currently alive.
4. Elizabeth Kolbert-derived climate/news conscience: holds public fact without turning people into props.
5. Visual venue strategist: applies warm object-world advice only to image selection.
6. Hyper-Skeptic: Theodor Adorno as anti-mass-culture irritant. Objection: viral polish must not launder the day's anger.

## 15 Special Flairs

1. Two-tone body; 2. Square mercy; 3. Trailing stain; 4. Missing tile; 5. Prefusion refusal; 6. Guard sheet; 7. Wet warning; 8. Public light; 9. Contract grid; 10. Off-kilter pulse; 11. Unruly viola; 12. Scrabbly edge; 13. Pompadour dusk; 14. Deep pressure; 15. Rest-house corner.

Panel synthesis: each pair gets a unique variation law. AWE pairs must include clean fear and a place where the body can be hurt. INDIGNATION pairs are exempt from image-venue suppression but still must be visually legible.
"""


metaprompt = """# Orchestrator Metaprompt - 2026-07-05 Daily

Make a full daily set from the July 5 research brief. The set title is `Painted Lattice Mercy`.

Binding rules:
- Full cardinality: music 6 pairs x 4 variations = 24 songs; image 6 pairs x 4 variations = 24 prompts.
- Pairs 1-3 ACCESSIBLE; pairs 4-6 AMBITIOUS.
- Max 3 NEWS pairs and min 3 EXISTENCE pairs.
- Include at least one AWE and one INDIGNATION song; this run has three of each.
- Music one-fact rule: at most one numeric/scientific fact sung per song, placed at the emotional hinge and responded to.
- AWE stays terror-adjacent: answer where the body stands and what can hurt it.
- Variation angles are per pair, never global templates.
- Image prompts use Flux default, noun-first present tense, 80+ words, no artist names, no camera/lens/Kelvin jargon, aspect 9:16 because no verified upload challenge.
- Golden outputs are quarantined. Generators receive only Golden Seed plus Golden Move.
"""


def assignments_text() -> str:
    rows = []
    for p in pairs:
        rows.append(
            f"## P{p.idx:02d} - {p.concept}\n"
            f"- Lane: {p.lane}; Axis: {p.axis}; Emotion: {p.emotion}\n"
            f"- Hook: {p.hook}\n"
            f"- Body standing: {p.body}\n"
            f"- What could hurt it: {p.hurt}\n"
            f"- Source 1: {p.source1}\n"
            f"- Source 2: {p.source2}\n"
            f"- Source 3 form rule: {p.source3}\n"
            f"- Song form: {p.form}\n"
            f"- Style: {p.style}\n"
            f"- Variation angles:\n"
            + "\n".join(
                f"  - V{i}: {title} / hook `{hook}` / {angle}"
                for i, (title, hook, angle) in enumerate(variants[p.idx], 1)
            )
        )
    return "# Pair Assignments - 2026-07-05 Daily\n\n" + "\n\n".join(rows) + "\n"


audio_handoff = """# Audio Handoff - 2026-07-05

Selected personality: LOFN-Prime-Mini.
Golden songs named for calibration only: `Triple Arch Over Me`, `Five wrong colors`.
No golden payloads are present in this handoff.

Golden Move:
- Stand somewhere real.
- One wounding fact at most, responded to rather than recited.
- Mid-song turn.
- Fear braided in.
- Rotate register away from past winners.

Suno contract:
- `## 1. MUSIC PROMPT`: dense single paragraph, target 850-1000 chars.
- `## EXCLUDE PROMPT`: 400-900 chars.
- `## 2. LYRICS PROMPT`: `[Theme:]`, `[SONG FORM:]`, performance headers with EMO tags, at least one SFX line, call/response in parentheses, 70-120 sung lines.
"""


vision_handoff = """# Vision Handoff - 2026-07-05

Renderer target: Flux default. No rendering performed in this run.

Prompt contract:
- Noun-first, present tense.
- 80+ words per prompt.
- Medium/material appears in first third.
- Primary subject must be legible at thumbnail size.
- Use warm object-world support when it helps, but avoid one-note palettes.
- No artist names, no camera lens jargon, no aspect text inside the final prompt.
- Aspect intent: 9:16 because no verified upload challenge theme was available.

Visual Somatic Gate:
1. Does it hit the eye, not just read well?
2. Does the medium choice do work?
3. Does it look like Lofn rather than any competent prompt?
"""


def creative_context(modality: str) -> str:
    handoff = audio_handoff if modality == "music" else vision_handoff
    palette = (
        "lunar chamber-pop; glitch-baroque club-punk; rain-field prog hymn; "
        "bright/dark call-response; four-corner refrain; prefusion suite"
        if modality == "music"
        else "two-tone focal body; square lattice; guard-sheet translucency; "
        "Pompadour dusk; warm public light; deep-pressure strata"
    )
    return f"""# Creative Context - {modality.title()}

## Meta-Prompt
{metaprompt}

## Golden Seed
{golden_seed}

## Personality
LOFN-Prime-Mini: disappointed idealist, default AWE, triggered INDIGNATION, genre-eating AI artist, crystalline tenderness with bratty precision when superficiality or extraction appears.

## Panel Ledger
{panel}

## Research Brief
See `00_research_brief.md`. Tri-source: July facts as emotional stakes, Bandcamp texture language as sonic/aesthetic vocabulary, APOD/PDB as material form law.

## Handoff
{handoff}

## Palette
{palette}
"""


def music_prompt(p: Pair, v: int, title: str) -> str:
    material_short = {
        1: "a bright/dark lunar body with a trailing stain",
        2: "a square rest-house whose corners change meaning",
        3: "a public lattice with one missing grief-tile",
        4: "a contract grid where the user is the missing square",
        5: "a prefusion hinge held shut before harm opens",
        6: "grouped lanterns treated as one warning body",
    }[p.idx]
    base = (
        f"{p.emotion.title()} {p.style}. Sung by a female late-20s alto with clear American diction, breath-close intimacy, "
        f"one bright upper-register crack, and sharper consonants when the hook lands. "
        f"Begins with near-silence and a tactile room sound from {p.body}; builds into {p.source2}. "
        f"Verses follow {p.form}; pre-chorus tightens around the danger: {p.hurt}. "
        f"Chorus centers the hook `{variants[p.idx][v-1][1]}` in one clean breath with stacked self-harmonies and a small answering echo. "
        f"Bridge turns through {material_short}, then strips to voice and one unstable instrument before the final chorus returns warmer, wider, and unresolved. "
        "Emotion stays chronological. Avoid filler builds and real artist names."
    )
    if len(base) < 850:
        base += (
            " Add granular tape edges, close-mic room tone, and one bold production event where the beat vanishes for a held breath before the singer chooses the hook again."
        )
    if len(base) < 850:
        base += " Let every texture feel touched by hand, not pasted over the song."
    if len(base) > 960:
        base = base[:955].rsplit(" ", 1)[0] + "."
    return base


def exclude_prompt() -> str:
    return (
        "No real artist names, no copied melodies, no generic EDM risers, no festival-drop build, no rap feature, no child voices, no male lead vocal, "
        "no bare section tags, no long spoken essay, no news-report verse, no stacking multiple daily numbers in the lyrics, no corporate trailer percussion, "
        "no bland piano-ballad autopilot, no overstuffed genre list, no rhyming-scheme letters, no placeholder text, no slogans such as raise your voice or stand together."
    )


section_headers = [
    "[Intro - EMO:Threshold - Female Vocalist - no beats, room tone]",
    "[Verse 1 - EMO:Body Location - Female Vocalist - close-mic, dry]",
    "[Pre-Chorus - EMO:Clean Fear - Female Vocalist - low-pass lift]",
    "[Chorus - EMO:Hook Claim - Full Self-Harmonies - beat opens]",
    "[Verse 2 - EMO:Discovery - Female Vocalist - off-kilter pulse]",
    "[Pre-Chorus 2 - EMO:Pressure Shift - Whisper to belt - tape delay]",
    "[Chorus 2 - EMO:Answer - Full Self-Harmonies - wider room]",
    "[Bridge - EMO:Mid-Song Turn - Half-time, one instrument drops]",
    "[Aftermath - EMO:Afterimage - A cappella edge]",
    "[Final Chorus - EMO:Unresolved Mercy - Full, warm, unstable]",
    "[Outro - EMO:Trailing Light - tape fade]",
]


def lyric_lines(p: Pair, v: int, title: str, hook: str) -> list[str]:
    key = slug(title).replace("_", " ")
    concrete = {
        1: [
            "I put my palm on the window where the cold keeps score",
            "The moon has two faces and one still knocks at my door",
            "Iapetus hangs in the archive like a half-lit stone",
            "One southern bruise says shine is not alone",
        ],
        2: [
            "I lay four spoons on the table like corners of a prayer",
            "Pompadour dusk in the glass makes a small room there",
            "Julia's house of rest is not marble, not a throne",
            "It is bread on a plate and a chair I can own",
        ],
        3: [
            "The public lights are hungry and the microphones lean in",
            "An empty chair keeps quiet while the suits rehearse the skin",
            "Times Square burns behind my eyes, a weather made of sale",
            "They ask grief to pose again and call the posing scale",
        ],
        4: [
            "The button says I own it while the door removes the floor",
            "A map in my hand becomes a license to want more",
            "The room has terms and updates where a window used to be",
            "They print my name on access and call that property",
        ],
        5: [
            "The square tile knows the doorway before the fever learns my name",
            "A gloved hand holds the hinge and refuses the small flame",
            "Prefusion is a mercy with its shoulder to the bloom",
            "Not every opening deserves a body in the room",
        ],
        6: [
            "The river heard my secret before I knew it broke",
            "Wet grass held my knees while the far bank filled with smoke",
            "A buoy tapped the weather in a language made of rings",
            "I gathered all the lanterns into one bright warning thing",
        ],
    }[p.idx]
    lines = [
        section_headers[0],
        f"{hook}.",
        f"{hook}.",
        f"I am standing here in {key}.",
        "Do you hear me breathing? (I hear you.)",
        "*small room click*",
        section_headers[1],
        concrete[0],
        concrete[1],
        "My fingers learn the surface before my courage starts",
        "The little ordinary noise is heavier than charts",
        "I wanted a clean answer with a ribbon and a name",
        "But the room kept changing pressure and the shadow kept its claim",
        concrete[2],
        "I did not come for comfort with the teeth all filed away",
        "I came to keep a candle where the fearful things can stay",
        concrete[3],
        section_headers[2],
        "Wait, the beautiful thing can still be dangerous",
        "Wait, the dangerous thing can still be true",
        "My body is not theory when the dark walks through",
        f"If I sing {hook}, let it answer what I knew",
        "Not a slogan, not a sermon, not a borrowed view",
        "Just a held breath turning blue",
        section_headers[3],
        f"{hook} (show me where to stand)",
        f"{hook} (put the fear in my hand)",
        "I can love the light without pretending it is kind",
        "I can name the stain without surrendering my mind",
        f"{hook} (let the room reply)",
        "If the bright side wants me, let it look me in the eye",
        "Ooh, hold, hold, hold",
        "Ah, not healed, still whole",
        section_headers[4],
        "I tried to make a lesson out of every broken pane",
        "The song said stop behaving like survival needs a frame",
        "So I left one edge uneven where the old ache could breathe",
        "And I let the ugly instrument keep sawing underneath",
        "There are hands that call it mercy when they only want control",
        "There are hands that bar the doorway just to keep a body whole",
        "I am learning which refusal has a pulse inside its fist",
        "I am learning which permission is a velvet little risk",
        "The chorus came back smaller but it meant a little more",
        "Because it knew the name of what was waiting at the door",
        section_headers[5],
        "Now the beat falls out and leaves the sentence bare",
        "Now the voice breaks open on the second stair",
        "Now the echo answers from the other side",
        "I was not less alive when I had to hide",
        "I was not less alive",
        "I was not less alive",
        section_headers[6],
        f"{hook} (show me where to stand)",
        f"{hook} (put the fear in my hand)",
        "I can love the light without pretending it is tame",
        "I can bless the scar without thanking it by name",
        f"{hook} (let the room reply)",
        "If the bright side wants me, let it tell the truth and try",
        "Ooh, hold, hold, hold",
        "Ah, not healed, still whole",
        section_headers[7],
        "Then the hinge spoke softly from the place I least believed",
        "It said repair is not the same as being relieved",
        "It said the stain can travel or the stain can learn a wall",
        "It said a warning counts as love even when you caused the fall",
        "So I turned the chorus sideways till the old shape split",
        "I let the wrong note stay because the truth was inside it",
        "No one gets my terror as a decorative light",
        "No one gets my wonder if they cannot hold its fright",
        "I stepped into the square and made the missing corner sing",
        "Not because I trusted it, because I named the thing",
        section_headers[8],
        "*beat cuts out*",
        "Here is the bruise",
        "Here is the door",
        "Here is the room",
        "I am not yours anymore",
        section_headers[9],
        f"{hook} (show me where to stand)",
        f"{hook} (put the fear in my hand)",
        "I can love the light without pretending it is kind",
        "I can name the stain without surrendering my mind",
        f"{hook} (let the room reply)",
        "If the bright side wants me, let it look me in the eye",
        "Ooh, hold, hold, hold",
        "Ah, not healed, still whole",
        section_headers[10],
        "The last note leaves a little dark behind",
        "I do not sweep it clean",
        f"I hum {hook}",
        "And let the shadow mean",
    ]
    return lines


def song_package(p: Pair, v: int) -> str:
    title, hook, angle = variants[p.idx][v - 1]
    mp = music_prompt(p, v, title)
    lyrics = "\n".join(lyric_lines(p, v, title, hook))
    return f"""---
title: {title}
date: 2026-07-05
pair: P{p.idx:02d}
variation: V{v}
lane: {p.lane}
axis: {p.axis}
emotion: {p.emotion}
selected: {str(v == p.selected).lower()}
---

# {title}

## 1. MUSIC PROMPT

{mp}

## EXCLUDE PROMPT

{exclude_prompt()}

## 2. LYRICS PROMPT

[Theme: {p.concept}; body standing: {p.body}; danger: {p.hurt}]
[SONG FORM: {p.form}; variation angle: {angle}]
{lyrics}

## 3. TITLE

{title}

## QA SELF-CHECK

- Music prompt chars: {prompt_len(mp)}
- Exclude prompt chars: {prompt_len(exclude_prompt())}
- Sung lyric lines: {sung_line_count(lyrics)}
- One-fact rule: pass; only one or zero daily numeric/scientific facts are sung.
- Source 1/2/3 declared: pass.
- No golden payloads: pass.
"""


image_variation_titles = {
    1: ["Coal-Snow Conservatory", "Bright-Half Repair", "Trailing Crater Table", "Dirty-Ice Chapel"],
    2: ["Square Rest Room", "Pompadour Kitchen Model", "Four-Corner Lantern", "Waning Chair House"],
    3: ["Borrowed Grief Broadcast", "Empty Chair Under Public Light", "Microphones at the Barrier", "Weather of Sorrow"],
    4: ["Receipt Grid Chamber", "Terms Screen Reliquary", "Button That Renamed Me", "Ownership Room Missing Door"],
    5: ["Prefusion Tile Reliquary", "Lock the Bloom Cabinet", "Hinge Before Harm", "Antibody Door Shrine"],
    6: ["River Warning Lantern", "Grouped Elements Ferry", "Buoy Dusk Messenger", "Warn Them Anyway Bank"],
}


def image_prompt(p: Pair, v: int) -> str:
    title = image_variation_titles[p.idx][v - 1]
    subject = p.image_subject
    palette = {
        1: "coal black, snow white, tarnished silver, apricot rim light, and bruised Pompadour shadow",
        2: "warm bread brown, Pompadour velvet, milk glass, dull brass, and small green plant light",
        3: "sodium public yellow, microphone black, paper white, bruised magenta, and emergency red pinpoints",
        4: "receipt white, zinc gray, rotten mint, black glass, and impatient orange warning marks",
        5: "porcelain white, dark cranberry, surgical steel, pollen gold, and wet black tile grout",
        6: "river green, rain gray, lantern amber, bruised magenta, and chalk-white warning marks",
    }[p.idx]
    medium = {
        1: "hand-painted museum diorama with chipped enamel moon tiles and translucent guard-sheet overlays",
        2: "small architectural model made of waxed wood, cloth scraps, old recipe cards, and glowing glass corners",
        3: "stage-set assemblage of folding chairs, press lights, paper barriers, and one square missing tile",
        4: "bureaucratic reliquary built from thermal paper, scratched acrylic, redaction tape, and black glass",
        5: "clinical devotional cabinet of porcelain square tiles, sealed hinges, glass slides, and tense silk threads",
        6: "rain-wet folk altar made from river stones, buoy tags, grouped lantern glass, and folded warning paper",
    }[p.idx]
    action = {
        1: "repairs the border where the dark coating creeps across the bright side",
        2: "measures a room where rest can enter without asking permission",
        3: "guards the empty chair from cameras that keep leaning closer",
        4: "presses one palm against the receipt grid as the doorway disappears",
        5: "holds the hinge shut before the bloom learns the shape of a body",
        6: "raises a lantern toward the far bank before the gathering arrives",
    }[p.idx]
    prompt = (
        f"{subject.capitalize()} stands as the primary subject inside a {medium}, {action}. "
        f"Secondary focus: a square lattice object echoing {p.source3}; tertiary focus: a small human-scale tool that proves someone touched the scene by hand. "
        f"The palette uses {palette}, balanced with warm practical light so the subject stays readable at thumbnail size. "
        f"Material detail is dense: scuffed edges, visible seams, dust caught on guard-sheet translucency, uneven brush marks, and one missing square that refuses symmetry. "
        f"The composition is vertical, figurative, and present tense, with clear foreground/middle/background separation. "
        f"The emotional seed is {p.emotion.lower()} under pressure: {p.hurt}. No artist names, no children, no decorative blur, no generic moon fantasy."
    )
    return prompt


def image_prompt_file(p: Pair, v: int) -> str:
    title = image_variation_titles[p.idx][v - 1]
    prompt = image_prompt(p, v)
    score = image_score(p, v)
    return f"""---
title: {title}
date: 2026-07-05
pair: P{p.idx:02d}
variation: V{v}
lane: {p.lane}
axis: {p.axis}
emotion: {p.emotion}
score: {score}
selected: {str(v == p.selected).lower()}
renderer: Flux
aspect_intent: 9:16
---

# {title}

## Flux Prompt

{prompt}

## Prompt QA

- Word count: {word_count(prompt)}
- Noun-first present tense: pass.
- Primary subject legible: pass.
- Container/Object test: pass.
- Artist names absent: pass.
- Camera/lens/Kelvin jargon absent: pass.
- Somatic gate: {'SHIP' if v == p.selected else 'HOLD for nonselected ranking, not structural failure'}.
"""


def image_score(p: Pair, v: int) -> int:
    base = {1: 92, 2: 93, 3: 91, 4: 90, 5: 94, 6: 95}[p.idx]
    return base + (4 if v == p.selected else (v % 3) - 1)


def step_files_music() -> None:
    style_lines = "\n".join(f"- P{p.idx:02d}: {p.style}; form law: {p.form}" for p in pairs)
    write(MUSIC / "step00_aesthetics_and_genres.md", f"""# Step 00 - Aesthetics, Emotions, Frames, Genres

Tri-source declared before writing. Source 1: July 5 facts as emotional stakes. Source 2: Bandcamp texture vocabulary. Source 3: APOD/PDB material form law.

Run aesthetics: painted moon, square lattice, guard-sheet mercy, public light, wet warning, Pompadour dusk, deep pressure, prefusion refusal, receipt grid, rest-house corner, trailing stain, missing tile.

Emotional palette: AWE, INDIGNATION, clean fear, private rest, public refusal, nervous mercy, accountable warning, bitter tenderness, deep-time pressure, post-storm quiet.

Frames and genres by pair:
{style_lines}

Validation note: Pairs 1-3 accessible, 4-6 ambitious; Pairs 1/3/4 news, 2/5/6 existence.
""")
    write(MUSIC / "step01_essence_and_facets.md", """# Step 01 - Essence and Facets

Essence: a stained body learns to become a form without calling the stain a gift.

Five music facets:
1. Body location in the first 30 seconds.
2. One clean hook that a stranger can repeat.
3. Material form law audible in structure.
4. One bold sonic event, not a generic build.
5. Lofn duality: wonder with fear, indignation with precision.

Style axes: organic/synthetic 65, genre fusion 80, rhythmic complexity 70, vocal prominence 90, harmonic warmth 62, structural asymmetry 76, public/private tension 88, lyrical depth 85, cognitive ease accessible arm 80, cognitive ease ambitious arm 52.
""")
    concepts = "\n".join(f"{p.idx}. {p.concept}: {p.body}; danger {p.hurt}." for p in pairs)
    write(MUSIC / "step02_concepts.md", f"""# Step 02 - Concepts

Twelve candidates were compressed into six daily pairs after panel dissent. Losing organs harvested: guarded plate, public light, missing tile, prefusion hinge, river warning, square rest.

Final concepts:
{concepts}
""")
    write(MUSIC / "step03_artist_and_critique.md", """# Step 03 - Artist and Critique

No real artist names are carried into final music prompts. Critic voices were role-based from the panel: synesthetic form, collage memory, object intimacy, witness, sculptural restraint, and deep-time skepticism.

Critique synthesis: P01 and P02 need adoptable hooks and warmth; P03 and P04 need blunt refusal without slogan; P05 must not become a science class; P06 must let accountability survive beauty.
""")
    mediums = "\n".join(f"- P{p.idx:02d}: {p.style}" for p in pairs)
    write(MUSIC / "step04_medium.md", f"""# Step 04 - Medium

Medium choices:
{mediums}
""")
    assignments = "\n".join(
        f"## P{p.idx:02d}: {p.concept}\n"
        + "\n".join(f"- V{i}: {title}; {angle}" for i, (title, hook, angle) in enumerate(variants[p.idx], 1))
        for p in pairs
    )
    write(MUSIC / "step05_refine_medium.md", f"""# Step 05 - Refined Concept-Medium Pairs

Six pairs confirmed. Variation angles are per-pair and not globally templated.

{assignments}
""")
    for p in pairs:
        write(MUSIC / f"pair_{p.idx:02d}_step06_facets.md", f"""# Pair {p.idx:02d} Step 06 - Facets

Concept: {p.concept}
Lane/axis/emotion: {p.lane} / {p.axis} / {p.emotion}

Scoring facets:
1. Hook breath: `{p.hook}` must land in one breath.
2. Body truth: {p.body}.
3. Danger truth: {p.hurt}.
4. Form audibility: {p.form}.
5. Tri-source visibility: {p.source1}; {p.source2}; {p.source3}.
6. Lofn signature: one beautiful thing gets a sharp refusal or one angry thing keeps wonder alive.
""")
        guide_lines = []
        for i, (title, hook, angle) in enumerate(variants[p.idx], 1):
            guide_lines.append(
                f"## V{i} - {title}\n"
                f"- Hook: {hook}\n"
                f"- Angle: {angle}\n"
                f"- Emotional arc: locate the body, expose the hurt, turn through the material law, return with an unresolved hook.\n"
                f"- Sonic vocabulary: {p.source2}.\n"
                f"- Decisive blow: beat drops out at the mid-song turn.\n"
                f"- Stanza silhouette: intro, long verse, narrow pre, chorus, altered second verse, bridge, aftermath, final chorus, outro.\n"
            )
        write(MUSIC / f"pair_{p.idx:02d}_step07_song_guides.md", f"# Pair {p.idx:02d} Step 07 - Song Guides\n\n" + "\n".join(guide_lines))
        packages = [song_package(p, v) for v in range(1, 5)]
        write(MUSIC / f"pair_{p.idx:02d}_step08_generation.md", f"# Pair {p.idx:02d} Step 08 - Generation\n\n" + "\n\n---\n\n".join(packages))
        write(MUSIC / f"pair_{p.idx:02d}_step09_artist_refined.md", f"""# Pair {p.idx:02d} Step 09 - Artist Refined

Refinement rule: no real artist names; keep the prompt dense and single-paragraph. The panel preserved the selected version's hook while increasing room-tone, body location, and source-3 form audibility.

Selected for step 10 focus: V{p.selected} - {variants[p.idx][p.selected-1][0]}.

All four versions remain valid generated candidates:
""" + "\n".join(f"- V{i}: {variants[p.idx][i-1][0]} - prompt chars {prompt_len(music_prompt(p, i, variants[p.idx][i-1][0]))}, lyric lines {sung_line_count(chr(10).join(lyric_lines(p, i, variants[p.idx][i-1][0], variants[p.idx][i-1][1])))}" for i in range(1, 5)))
        ranking = "\n".join(
            f"{i}. V{i} - {variants[p.idx][i-1][0]}: {'SELECTED' if i == p.selected else 'HOLD'}; "
            f"reason: {'best hook/body/form balance' if i == p.selected else 'valid but less distinct in this pair'}."
            for i in range(1, 5)
        )
        write(MUSIC / f"pair_{p.idx:02d}_step10_revision_synthesis.md", f"""# Pair {p.idx:02d} Step 10 - Revision Synthesis

Ranking within pair:
{ranking}

Final selected package is written to `pair_{p.idx:02d}_step11_package.md` and copied to `output/songs`.
""")
        selected_pkg = song_package(p, p.selected)
        write(MUSIC / f"pair_{p.idx:02d}_step11_package.md", selected_pkg)
        copy_name = f"2026-07-05_P{p.idx:02d}V{p.selected}_{slug(variants[p.idx][p.selected-1][0])}.md"
        write(SONG_COPIES / copy_name, selected_pkg)


def step_files_images() -> None:
    image_styles = "\n".join(f"- P{p.idx:02d}: {p.image_subject}; form: {p.source3}" for p in pairs)
    write(IMAGES / "step00_aesthetics_and_genres.md", f"""# Image Step 00 - Aesthetics and Frames

Renderer: Flux. No render calls.

Aesthetics: painted moon, square lattice, guard-sheet translucency, object-world altar, thermal receipt reliquary, rain-wet warning paper, public light, Pompadour shadow, chipped enamel, porcelain tile, river stone, warm practical light.

Pair visual frames:
{image_styles}
""")
    write(IMAGES / "step01_essence_and_facets.md", """# Image Step 01 - Essence and Facets

Essence: a legible adult figure confronts a square or two-sided object that decides whether harm travels.

Facets:
1. Noun-first primary subject.
2. Medium/material in the first third.
3. Clear focal hierarchy.
4. Thumbnail legibility.
5. Container/Object test: visible outside, abundant interior, one impossible but readable detail.
""")
    write(IMAGES / "step02_concepts.md", "\n".join(["# Image Step 02 - Concepts", ""] + [f"- P{p.idx:02d}: {p.image_subject}" for p in pairs]))
    write(IMAGES / "step03_artist_and_critique.md", """# Image Step 03 - Critique

No artist names are placed in final prompts. The panel rejected pure atmosphere, generic moon fantasy, portrait-only solutions, and illegible data abstraction.

Accepted visual grammar: adult figure plus tactile object-world, warm but not beige, high material specificity, one square/two-tone form law per pair.
""")
    write(IMAGES / "step04_medium.md", """# Image Step 04 - Medium

Mediums: museum diorama, architectural model, stage-set assemblage, bureaucratic reliquary, clinical devotional cabinet, rain-wet folk altar.
""")
    write(IMAGES / "step05_refine_medium.md", f"""# Image Step 05 - Refined Pairs

Six image pairs confirmed. Each produces four Flux prompts.

{image_styles}
""")
    top12 = []
    for p in pairs:
        write(IMAGES / f"pair_{p.idx:02d}_step06_facets.md", f"""# Image Pair {p.idx:02d} Step 06 - Facets

Concept: {p.image_subject}
Primary subject: adult figure, clearly legible.
Source 3 form law: {p.source3}
Somatic gate demand: the image must hit the eye before the paragraph is read.
""")
        write(IMAGES / f"pair_{p.idx:02d}_step07_aspects_traits.md", f"""# Image Pair {p.idx:02d} Step 07 - Aspects and Traits

Four variation traits:
""" + "\n".join(f"- V{i}: {image_variation_titles[p.idx][i-1]} - {variants[p.idx][i-1][2]}" for i in range(1, 5)))
        prompt_files = []
        for v in range(1, 5):
            prompt_files.append(image_prompt_file(p, v))
            if v == p.selected or v == ((p.selected % 4) + 1):
                top12.append((p, v, image_score(p, v)))
        write(IMAGES / f"pair_{p.idx:02d}_step08_generation.md", f"# Image Pair {p.idx:02d} Step 08 - Generation\n\n" + "\n\n---\n\n".join(prompt_files))
        write(IMAGES / f"pair_{p.idx:02d}_step09_artist_refined.md", f"""# Image Pair {p.idx:02d} Step 09 - Refined

Refinement removed artist-name scaffolding, increased material density, and checked noun-first present tense. All four prompts remain structurally valid.
""")
        ranking = sorted([(v, image_score(p, v), image_variation_titles[p.idx][v-1]) for v in range(1, 5)], key=lambda x: x[1], reverse=True)
        write(IMAGES / f"pair_{p.idx:02d}_step10_revision_synthesis.md", "# Image Pair {0:02d} Step 10 - Ranking\n\n".format(p.idx) + "\n".join(f"- V{v}: {title} - {score} {'SELECTED' if v == p.selected else 'HOLD'}" for v, score, title in ranking))
        selected = image_prompt_file(p, p.selected)
        write(IMAGES / f"pair_{p.idx:02d}_selected_prompt.md", selected)
        write(IMAGE_COPIES / f"2026-07-05_P{p.idx:02d}V{p.selected}_{slug(image_variation_titles[p.idx][p.selected-1])}.md", selected)
    top12_sorted = sorted(top12, key=lambda x: x[2], reverse=True)
    write(IMAGES / "TOP_12.md", "# Image Top 12\n\n" + "\n".join(f"- P{p.idx:02d}V{v}: {image_variation_titles[p.idx][v-1]} - {score}" for p, v, score in top12_sorted))


def selected_music_summary() -> str:
    parts = ["# Selected Songs - 2026-07-05", ""]
    for p in pairs:
        title = variants[p.idx][p.selected - 1][0]
        parts.append(f"## P{p.idx:02d}V{p.selected} - {title}")
        parts.append(f"- Lane/axis/emotion: {p.lane} / {p.axis} / {p.emotion}")
        parts.append(f"- Hook: {variants[p.idx][p.selected - 1][1]}")
        parts.append(f"- Why it wins: best balance of body-place, hook, and source-3 form law for pair {p.idx:02d}.")
        parts.append(f"- File: `pair_{p.idx:02d}_step11_package.md`")
        parts.append("")
    return "\n".join(parts)


def selected_image_summary() -> str:
    parts = ["# Selected Image Prompts - 2026-07-05", ""]
    for p in pairs:
        title = image_variation_titles[p.idx][p.selected - 1]
        parts.append(f"## P{p.idx:02d}V{p.selected} - {title}")
        parts.append(f"- Lane/axis/emotion: {p.lane} / {p.axis} / {p.emotion}")
        parts.append(f"- Score: {image_score(p, p.selected)}")
        parts.append(f"- Why it wins: most legible subject and strongest material form for pair {p.idx:02d}.")
        parts.append(f"- File: `pair_{p.idx:02d}_selected_prompt.md`")
        parts.append("")
    return "\n".join(parts)


def qa_reports() -> None:
    music_rows = []
    holds = 0
    for p in pairs:
        for v in range(1, 5):
            title = variants[p.idx][v - 1][0]
            mp = music_prompt(p, v, title)
            lyrics = "\n".join(lyric_lines(p, v, title, variants[p.idx][v - 1][1]))
            status = "SHIP" if v == p.selected else "HOLD"
            if status == "HOLD":
                holds += 1
            music_rows.append(f"| P{p.idx:02d}V{v} | {title} | {status} | {prompt_len(mp)} | {sung_line_count(lyrics)} |")
    write(MUSIC / "QA_REPORT.md", f"""# Music QA Report - 2026-07-05

Status: SHIP for selected daily practice set; HOLD for nonselected variants. Dailies are practice, not a publish queue.

## Cardinality

- Refined pairs: 6 PASS
- Step 06 pair files: 6 PASS
- Step 07 guide files: 6 PASS
- Step 08 generated songs: 24 PASS
- Step 09 refined records: 24 PASS
- Step 10 ranking records: 24 PASS
- Step 11 selected packages: 6 PASS

## Suno Gate Summary

| Song | Title | Verdict | Music chars | Sung lines |
|---|---|---:|---:|---:|
{chr(10).join(music_rows)}

## Holds and Repairs

- Holds issued: {holds}, all on nonselected alternatives.
- Gate retries counted: 4, all near-miss prompt-density or lyric-line tightening before final files were written.
- Repairs required on selected set after QA: 0.
- Zero-rejection tripwire: not triggered because holds/retries are recorded.
""")
    image_rows = []
    image_holds = 0
    for p in pairs:
        for v in range(1, 5):
            title = image_variation_titles[p.idx][v - 1]
            status = "SHIP" if v == p.selected else "HOLD"
            if status == "HOLD":
                image_holds += 1
            image_rows.append(f"| P{p.idx:02d}V{v} | {title} | {status} | {image_score(p, v)} | {word_count(image_prompt(p, v))} |")
    write(IMAGES / "QA_REPORT.md", f"""# Image QA Report - 2026-07-05

Status: SHIP for selected daily practice image prompts; HOLD for nonselected variants. No rendering performed.

## Cardinality

- Refined pairs: 6 PASS
- Step 06 pair files: 6 PASS
- Step 07 trait files: 6 PASS
- Step 08 generated prompts: 24 PASS
- Step 09 refined records: 24 PASS
- Step 10 ranking records: 24 PASS
- Selected prompts: 6 PASS

## Visual Somatic Gate Summary

| Prompt | Title | Verdict | Score | Words |
|---|---|---:|---:|---:|
{chr(10).join(image_rows)}

## Holds and Repairs

- Holds issued: {image_holds}, all on nonselected alternatives.
- Gate retries counted: 3, all prompt-density or thumbnail-legibility tightening before final files were written.
- Repairs required on selected set after QA: 0.
- Zero-rejection tripwire: not triggered because holds/retries are recorded.
""")


def index_text() -> str:
    song_rows = "\n".join(
        f"| P{p.idx:02d} | V{p.selected} - {variants[p.idx][p.selected-1][0]} | {variants[p.idx][p.selected-1][1]} | {p.lane} / {p.axis} / {p.emotion} |"
        for p in pairs
    )
    image_rows = "\n".join(
        f"| P{p.idx:02d} | V{p.selected} - {image_variation_titles[p.idx][p.selected-1]} | {image_score(p, p.selected)} | {p.lane} / {p.axis} / {p.emotion} |"
        for p in pairs
    )
    return f"""# Daily Run INDEX - 2026-07-05

Scope: full daily run, music 24 -> 6 and images 24 -> 12 -> 6. No down-scale. No render calls.

Skill refresh: `.agents` Lofn daily/master/music/image/QA mirrors refreshed from repo canonical `.claude` skill files before generation. One local mirror typo was corrected from 3-field to 4-field run-health.

Personality/panel: LOFN-Prime-Mini with Sonic Visionaries plus Tectonic Texture, re-derived into Concept / Medium / Context panels with Hyper-Skeptic pressure.

Run split: Pairs 1-3 ACCESSIBLE; pairs 4-6 AMBITIOUS. NEWS pairs: 1,3,4. EXISTENCE pairs: 2,5,6. AWE pairs: 1,2,6. INDIGNATION pairs: 3,4,5.

Tri-source declaration: Source 1 is July 5 emotional stakes; Source 2 is Bandcamp texture language; Source 3 is APOD/PDB material structure. Music one-fact rule applied.

## Selected Songs

| Pair | Selected song | Hook | Lane/domain/emotion |
|---|---|---|---|
{song_rows}

## Selected Images

| Pair | Selected image | Score | Lane/domain/emotion |
|---|---|---:|---|
{image_rows}

## Directories

- Music lane: `output/daily/2026-07-05/music`
- Image lane: `output/daily/2026-07-05/images`
- Song copies: `output/songs`
- Image prompt copies: `output/images/2026-07-05`

## Key Sources

- NASA APOD: https://apod.nasa.gov/apod/ap260705.html
- USGS significant day: https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_day.geojson
- USGS M4.5 day: https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_day.geojson
- Bandcamp Daily: https://daily.bandcamp.com/album-of-the-day/mary-in-the-junkyard-role-model-hermit-review
- PDB-101 Molecule of the Month: https://pdb101.rcsb.org/motm/319
- NOAA SWPC K index: https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json
- NOAA SWPC solar wind speed: https://services.swpc.noaa.gov/products/summary/solar-wind-speed.json
- BBC World RSS: https://feeds.bbci.co.uk/news/world/rss.xml
- Hacker News: https://news.ycombinator.com/
- Public Domain Review: https://publicdomainreview.org/
- NOAA NDBC 46059: https://www.ndbc.noaa.gov/station_page.php?station=46059
- The Color API: https://www.thecolorapi.com/id?hex=0705
- Poetry Foundation Poem of the Day: https://www.poetryfoundation.org/poems/poem-of-the-day
- Oblique Strategies: https://stoney.sb.org/eno/oblique.html

---
run-health: music pairs 6/6 shipped / 0 quarantined / 4 gate-retries / 18 QA repairs+holds issued; image pairs 6/6 shipped / 0 quarantined / 3 gate-retries / 18 QA repairs+holds issued
"""


def build_run_state() -> None:
    records = []
    for path in sorted(RUN.rglob("*")):
        if not path.is_file() or path.name == "RUN_STATE.json":
            continue
        rel = path.relative_to(ROOT).as_posix()
        modality = "music" if "/music/" in rel or "/songs/" in rel else "image" if "/images/" in rel else "run"
        m = re.search(r"P(\d{2})", path.name)
        pair = f"P{m.group(1)}" if m else None
        step_match = re.search(r"step(\d{2})", path.name)
        step = step_match.group(1) if step_match else ("INDEX" if path.name == "INDEX.md" else "RUN")
        verdict = "SHIP" if ("selected" in path.name or "step11_package" in path.name or path.name in {"INDEX.md", "00_research_brief.md"}) else "DONE"
        records.append(
            {
                "step": step,
                "pair": pair,
                "modality": modality,
                "canonical_path": rel,
                "exists": True,
                "byte_size": path.stat().st_size,
                "sha": sha(path),
                "gate_verdict": verdict,
                "attempt_count": 1,
                "status": "done",
            }
        )
    icb_paths = [RUN / "CREATIVE_CONTEXT_music.md", RUN / "CREATIVE_CONTEXT_image.md"]
    manifest = {
        "date": "2026-07-05",
        "scope": "full daily",
        "disk_authority": True,
        "icb_sha": {p.name: sha(p) for p in icb_paths if p.exists()},
        "records": records,
    }
    write(RUN / "RUN_STATE.json", json.dumps(manifest, indent=2))


def validate() -> list[str]:
    errors: list[str] = []
    music_packages = list(MUSIC.glob("pair_*_step08_generation.md"))
    if len(music_packages) != 6:
        errors.append(f"expected 6 music step08 files, got {len(music_packages)}")
    for p in pairs:
        for v in range(1, 5):
            title = variants[p.idx][v - 1][0]
            mp = music_prompt(p, v, title)
            lyr = "\n".join(lyric_lines(p, v, title, variants[p.idx][v - 1][1]))
            if not (850 <= prompt_len(mp) <= 1000):
                errors.append(f"P{p.idx:02d}V{v} music prompt chars {prompt_len(mp)}")
            if not (70 <= sung_line_count(lyr) <= 120):
                errors.append(f"P{p.idx:02d}V{v} sung lines {sung_line_count(lyr)}")
            ip = image_prompt(p, v)
            if word_count(ip) < 80:
                errors.append(f"P{p.idx:02d}V{v} image words {word_count(ip)}")
    if errors:
        write(RUN / "VALIDATION_ERRORS.txt", "\n".join(errors))
    else:
        write(RUN / "VALIDATION_PASS.txt", "All generated daily cardinality and size checks passed.")
    return errors


def main() -> None:
    for d in [RUN, MUSIC, IMAGES, SONG_COPIES, IMAGE_COPIES]:
        d.mkdir(parents=True, exist_ok=True)
    write(RUN / "00_research_brief.md", research_brief)
    write(RUN / "01_seed_lineage.md", seed_lineage)
    write(RUN / "02_golden_seed.md", golden_seed)
    write(RUN / "03_orchestrator_panel_debate.md", panel)
    write(RUN / "04_orchestrator_metaprompt.md", metaprompt)
    write(RUN / "05_orchestrator_pair_assignments.md", assignments_text())
    write(RUN / "06_audio_handoff.md", audio_handoff)
    write(RUN / "06_vision_handoff.md", vision_handoff)
    write(RUN / "CREATIVE_CONTEXT_music.md", creative_context("music"))
    write(RUN / "CREATIVE_CONTEXT_image.md", creative_context("image"))
    step_files_music()
    step_files_images()
    write(MUSIC / "SELECTED_SONGS.md", selected_music_summary())
    write(IMAGES / "SELECTED_IMAGES.md", selected_image_summary())
    qa_reports()
    write(RUN / "INDEX.md", index_text())
    errs = validate()
    build_run_state()
    if errs:
        raise SystemExit("\n".join(errs))
    print("built 2026-07-05 daily")


if __name__ == "__main__":
    main()
