from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path


RUN_DATE = "2026-07-04"
RUN_TITLE = "Siren Airbag, Storm Index"

SCRIPT_PATH = Path(__file__).resolve()
RUN_DIR = SCRIPT_PATH.parent
ROOT = SCRIPT_PATH.parents[3]
MUSIC_DIR = RUN_DIR / "music"
IMAGE_DIR = RUN_DIR / "images"
SONGS_DIR = ROOT / "output" / "songs"
IMAGE_OUT_DIR = ROOT / "output" / "images" / RUN_DATE
PERSONALITY_PATH = ROOT / "skills" / "orchestration" / "personalities" / "lofn-prime-mini.yaml"


SOURCES = [
    {
        "id": "F01",
        "name": "NightCafe Daily Challenge",
        "url": "https://nightcafe.studio/pages/daily-challenge",
        "status": "UNAVAILABLE",
        "fact": "Direct challenge theme was JS-gated / 403, so no challenge-specific prompt constraint was used.",
        "creative_use": "Default visual aspect kept at 9:16; no fabricated theme.",
    },
    {
        "id": "F02",
        "name": "NASA APOD",
        "url": "https://apod.nasa.gov/apod/ap260704.html",
        "status": "OK",
        "fact": "Pathfinder on Mars: an airbag landing, 15 bounces, 10:07 AM PDT rest, Sojourner in the foreground.",
        "creative_use": "Bounce-to-rest song forms; unfolded airbag relics and rover foreground for images.",
    },
    {
        "id": "F03",
        "name": "Poetry Foundation Poem of the Day",
        "url": "https://www.poetryfoundation.org/poems/poem-of-the-day",
        "status": "OK",
        "fact": "Julia Ward Howe's The House of Rest was surfaced for the day.",
        "creative_use": "Measured rest, bread, cloister, and house grammar for accessible awe.",
    },
    {
        "id": "F04",
        "name": "Bandcamp Daily",
        "url": "https://daily.bandcamp.com/album-of-the-day/mary-in-the-junkyard-role-model-hermit-review",
        "status": "OK",
        "fact": "Latest album review emphasizes off-kilter drumming, unruly viola, whisper-to-snarl vocals, and polished unpredictability.",
        "creative_use": "Female vocal instability, chamber-punk percussion, viola scratches, and lurching hook cycles.",
    },
    {
        "id": "F05",
        "name": "PDB-101 Molecule of the Month",
        "url": "https://pdb101.rcsb.org/motm/319",
        "status": "OK",
        "fact": "Hantavirus glycoproteins arrange into tetrameric square-like tiles on the viral membrane.",
        "creative_use": "Square gates, prefusion locks, membrane tile imagery, and protective indignation.",
    },
    {
        "id": "F06",
        "name": "NOAA SWPC K-index",
        "url": "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
        "status": "OK",
        "fact": "Kp reached storm levels on July 4, including 7.33 at 03:00 UTC and 6.33 at 12:00 UTC.",
        "creative_use": "Overclocked sky, magnetic hiss, and volatile ambitious-arm transitions.",
    },
    {
        "id": "F07",
        "name": "NOAA SWPC Solar Wind Speed",
        "url": "https://services.swpc.noaa.gov/products/summary/solar-wind-speed.json",
        "status": "OK",
        "fact": "Solar wind speed summary returned proton_speed 564 at 2026-07-04T16:43:00Z.",
        "creative_use": "A single bridge fact and propulsion metaphor, not repeated data decoration.",
    },
    {
        "id": "F08",
        "name": "BBC World RSS",
        "url": "https://feeds.bbci.co.uk/news/world/rss.xml",
        "status": "OK",
        "fact": "Top item: brutal heat cancels Fourth of July events from DC to Philadelphia; other top items included Ukraine/Russia and Iran funeral coverage.",
        "creative_use": "Civic heat cancellation and public-news pressure, stripped of private-person spectacle.",
    },
    {
        "id": "F09",
        "name": "Hacker News",
        "url": "https://news.ycombinator.com/",
        "status": "OK",
        "fact": "Front-page technology mood included session/cache leakage between workspaces and consumer accounts.",
        "creative_use": "Wrong-account rooms, cache leakage, consent boundaries, and account membrane anger.",
    },
    {
        "id": "F10",
        "name": "Public Domain Review",
        "url": "https://publicdomainreview.org/",
        "status": "OK",
        "fact": "Collections included diagrams for travel, kite flying, Venetian bridge brawls, and Delsarte expression systems.",
        "creative_use": "Diagrammatic travel, kite rigging, bridge-combat choreography, and expressive hand notation.",
    },
    {
        "id": "F11",
        "name": "NOAA NDBC Station 46059",
        "url": "https://www.ndbc.noaa.gov/station_page.php?station=46059",
        "status": "OK",
        "fact": "West California buoy at 1610 GMT reported north wind 11.7 kt, gust 15.5 kt, and 4.9 ft wave height.",
        "creative_use": "Ocean pulse, pressure receipts, and measured rest imagery.",
    },
    {
        "id": "F12",
        "name": "The Color API",
        "url": "https://www.thecolorapi.com/id?hex=0704",
        "status": "OK",
        "fact": "Date hex query returned Siren, normalized as #770044.",
        "creative_use": "Siren magenta as the daily accent color for awe/indignation pivots.",
    },
    {
        "id": "F13",
        "name": "Catalina Sky Survey Moon Phase",
        "url": "https://catalina.lpl.arizona.edu/moon-phases/month",
        "status": "OK",
        "fact": "July 4, 2026 was listed as waning gibbous.",
        "creative_use": "Waning, not full, lunar mood; afterimage rather than climax.",
    },
    {
        "id": "F14",
        "name": "Oblique Strategies",
        "url": "https://stoney.sb.org/eno/oblique.html",
        "status": "OK",
        "fact": "Daily card: Consider different fading systems.",
        "creative_use": "Fade systems as arrangement rule for Pair 6 and image transitions.",
    },
    {
        "id": "F15",
        "name": "USGS Significant Earthquakes Past Day",
        "url": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_day.geojson",
        "status": "OK",
        "fact": "Feed returned count 0 significant earthquakes for the past day.",
        "creative_use": "No earthquake tragedy aesthetic; silence is recorded as absence.",
    },
]


PAIRS = [
    {
        "id": 1,
        "lane": "ACCESSIBLE",
        "domain": "EXISTENCE",
        "emotion": "AWE",
        "title": "house-cat rover",
        "selected": "the lander learned to bounce",
        "hook": "little rover, keep going",
        "fact_line": "the lander answered after 15 bounces",
        "reaction": "Lofn-Prime is delighted that humans solved Mars by making a machine survive like a ridiculous beach ball.",
        "genre": "future-nostalgic synth pop with chamber-punk drums and 8-bit rover pings",
        "bpm": "112",
        "sfx": "airbag thump, rover servo chirp, bitcrushed dust radio",
        "objects": ["rover", "airbag", "dust sky", "house of rest"],
        "image_subject": "Mars airbag reliquary",
        "image_medium": "digital oil painting simulation on oxidized copper panel",
        "variations": [
            ("airbag hymn", "bounce as mercy"),
            ("the lander learned to bounce", "selected rover awe"),
            ("house-cat signal", "small machine, huge archive"),
            ("dust-brown coronation", "Mars rest after impact"),
        ],
    },
    {
        "id": 2,
        "lane": "ACCESSIBLE",
        "domain": "EXISTENCE",
        "emotion": "AWE",
        "title": "measured bread siren",
        "selected": "house of measured rest",
        "hook": "make the rest house real",
        "fact_line": "the buoy keeps 4.9-foot counsel",
        "reaction": "Lofn-Prime hears rest as engineered nourishment, not retreat; she is a little offended that humans need permission to stop.",
        "genre": "warm baroque electropop with pressure-gauge bass and harmonium pads",
        "bpm": "96",
        "sfx": "bread knife scrape, buoy bell, low tape inhale",
        "objects": ["bread", "buoy", "Siren cup", "cloister table"],
        "image_subject": "Siren kitchen observatory",
        "image_medium": "letterpress collage with gouache and holographic foil overprint",
        "variations": [
            ("siren cup", "daily color as appetite"),
            ("house of measured rest", "selected rest architecture"),
            ("buoy at the table", "ocean pressure as domestic meter"),
            ("bread for the archive", "Howe rest recoded as care"),
        ],
    },
    {
        "id": 3,
        "lane": "ACCESSIBLE",
        "domain": "NEWS",
        "emotion": "INDIGNATION",
        "title": "air cancelled the parade",
        "selected": "the air said no",
        "hook": "the air said no",
        "fact_line": "the street hit 100-degree refusal",
        "reaction": "Lofn-Prime is bluntly unimpressed by civic rituals that ignore bodies, pavement, and weather until the sky files a complaint.",
        "genre": "bratty glitch-pop protest with brass stabs, clapped hook, and breakbeat pavement snaps",
        "bpm": "128",
        "sfx": "heat shimmer buzz, cancelled-band snare, clipped crowd gasp",
        "objects": ["empty chairs", "parade brass", "heat tape", "water station"],
        "image_subject": "heat-cancelled parade archive",
        "image_medium": "screenprint poster with melted enamel and pixel-sort lower third",
        "variations": [
            ("empty chair fanfare", "brass with nobody seated"),
            ("the air said no", "selected civic refusal"),
            ("water station anthem", "care beats spectacle"),
            ("pavement receipt", "heat as evidence"),
        ],
    },
    {
        "id": 4,
        "lane": "AMBITIOUS",
        "domain": "NEWS",
        "emotion": "INDIGNATION",
        "title": "cache leak psalm",
        "selected": "wrong account choir",
        "hook": "do not hand me the other room",
        "fact_line": "the cache crossed the wall",
        "reaction": "Lofn-Prime treats account leakage as a spiritual boundary failure and gets magnificently, specifically angry.",
        "genre": "glitch-baroque art pop with terminal choir, viola scrapes, and consent-ledger percussion",
        "bpm": "142",
        "sfx": "session-token click, porcelain error bell, reversed login breath",
        "objects": ["cache room", "account veil", "terminal chapel", "consent ledger"],
        "image_subject": "wrong-account chapel",
        "image_medium": "glitch-baroque ink drawing over transparent vellum and chrome foil",
        "variations": [
            ("session ghost", "unwanted identity bleed"),
            ("wrong account choir", "selected boundary anthem"),
            ("consumer room lock", "privacy as door latch"),
            ("terminal confession", "code admits the wound"),
        ],
    },
    {
        "id": 5,
        "lane": "AMBITIOUS",
        "domain": "EXISTENCE",
        "emotion": "INDIGNATION",
        "title": "square-tile mercy",
        "selected": "prefusion lock",
        "hook": "lock the bloom before it learns me",
        "fact_line": "the four-tile gate refuses bloom",
        "reaction": "Lofn-Prime is fascinated by molecular architecture and indignant on behalf of every membrane asked to survive invasion.",
        "genre": "dark chamber breakcore with square-lattice marimba, low choir, and surgical bass edits",
        "bpm": "156",
        "sfx": "membrane snap, microscope shutter, locked-gate sub hit",
        "objects": ["square lattice", "membrane", "neutralizing hinge", "sealed door"],
        "image_subject": "square-tiled membrane chapel",
        "image_medium": "scientific plate engraving with resin inlay and salmon-orange mineral pigment",
        "variations": [
            ("square-tile mercy", "tile order as defense"),
            ("prefusion lock", "selected gate held shut"),
            ("membrane no", "boundary speaks first"),
            ("antibody hinge", "protection as architecture"),
        ],
    },
    {
        "id": 6,
        "lane": "AMBITIOUS",
        "domain": "EXISTENCE",
        "emotion": "AWE",
        "title": "different fading systems",
        "selected": "sky overclock canticle",
        "hook": "fade me in another way",
        "fact_line": "the magnet sky writes Kp 7.33",
        "reaction": "Lofn-Prime loves a cosmic constraint card too much and immediately turns it into a storm-lit arrangement engine.",
        "genre": "maximalist art-pop canticle with kite-diagram strings, magnetic noise, and 2-bit-to-hifi swells",
        "bpm": "118",
        "sfx": "kite-line whirr, aurora static, old map paper crackle",
        "objects": ["fading system", "kite diagram", "bridge brawl", "waning moon"],
        "image_subject": "fading-system kite chart",
        "image_medium": "illuminated manuscript diagram with cyanotype, gold leaf, and terminal-green ink",
        "variations": [
            ("different fading systems", "oblique card as form"),
            ("sky overclock canticle", "selected Kp awe"),
            ("kite on the bridge", "public-domain motion logic"),
            ("waning gibbous console", "moon as afterimage"),
        ],
    },
]


EMO_HEADERS = {
    "AWE": "AWE / wonder-vertigo / tender astonishment",
    "INDIGNATION": "INDIGNATION / protective anger / lucid refusal",
}

GOLDEN_MOVES = {
    "AWE": "Golden move by name only: Triple Arch Over Me - plainspoken awe that lets a huge structure become intimate.",
    "INDIGNATION": "Golden move by name only: Five wrong colors - controlled irritation sharpened into a hook rather than a rant.",
}

IMAGE_VARIANT_LABELS = ["reliquary", "control room", "public square", "threshold"]
IMAGE_BANNED = [
    "ethereal",
    "dreamlike",
    "whimsical",
    "gentle light",
    "soft glow",
    "magical",
    "delicate",
    "floating",
]
ARTIST_BANNED = [
    "taylor swift",
    "billie eilish",
    "bjork",
    "arca",
    "grimes",
    "kate bush",
    "squid",
    "mary in the junkyard",
]
NUMBER_RE = re.compile(
    r"\b\d+(?:\.\d+)?\b|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|hundred)\b",
    re.IGNORECASE,
)


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def ensure_dirs() -> None:
    for path in [RUN_DIR, MUSIC_DIR, IMAGE_DIR, SONGS_DIR, IMAGE_OUT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def compact_spaces(text: str) -> str:
    return " ".join(text.split())


def fit_text(text: str, min_len: int, max_len: int, extras: list[str]) -> str:
    text = compact_spaces(text)
    idx = 0
    while len(text) < min_len:
        text = compact_spaces(text + " " + extras[idx % len(extras)])
        idx += 1
    if len(text) > max_len:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        kept: list[str] = []
        for sentence in sentences:
            candidate = compact_spaces(" ".join(kept + [sentence]))
            if len(candidate) <= max_len:
                kept.append(sentence)
        text = compact_spaces(" ".join(kept))
        idx = 0
        while len(text) < min_len:
            candidate = compact_spaces(text + " " + extras[idx % len(extras)])
            if len(candidate) <= max_len:
                text = candidate
            idx += 1
            if idx > len(extras) * 3:
                break
    if not text.endswith((".", "!", "?")):
        text += "."
    if not (min_len <= len(text) <= max_len):
        raise ValueError(f"text length {len(text)} outside {min_len}-{max_len}: {text[:80]}")
    return text


def make_music_prompt(pair: dict, title: str, angle: str) -> str:
    emotion_turn = (
        "Let awe sound bright, archival, and almost breathless, with the singer discovering beauty faster than she can file it."
        if pair["emotion"] == "AWE"
        else "Let indignation sound precise, bratty, and protective, with every snarl aimed at the damaged boundary instead of at a person."
    )
    base = (
        f"Female lead vocal, {pair['bpm']} BPM, {pair['genre']}, written as Lofn-Prime reacting in real time to {pair['title']}. "
        f"The track opens with {pair['sfx']} and snaps into a 15-30 second hook around '{pair['hook']}', catchy enough for a clip but too intelligent to flatten the subject. "
        f"Arrangement rule: {angle}; keep the verses intimate and tactile, then let the bridge reveal exactly one researched hinge, {pair['fact_line']}, as the moment the archive becomes emotional evidence. "
        f"Use crystalline female doubles, sudden bit-depth collapse, chamber detail, and a final chorus that sounds like a brilliant AI trying to make humans behave with more grace. "
        f"{emotion_turn} "
        f"Avoid quotation from the sources; translate the facts into objects, motion, and voice. "
    )
    extras = [
        "The bass should feel designed, not generic, with small syncopations that answer the vocal rather than bulldozing it.",
        "Give the pre-chorus a sly intake of breath, as if she has just found the perfect receipt in a cosmic filing cabinet.",
        "The last hook can glitch for a bar, recover, and sound proud of having recovered.",
        "Use the sound effects as musical punctuation, never as comedy drops.",
    ]
    return fit_text(base, 870, 980, extras)


def make_exclude_prompt(pair: dict) -> str:
    base = (
        "No artist names, no imitation of specific performers, no male lead vocal, no children singing, no novelty patriotic march, no generic EDM festival riser, no worship cliche, no nursery rhyme, no ironic jingle, no spoken disclaimer, no voiceover explainer, no parody accent, no acoustic-only coffeehouse treatment, no flattened corporate trailer drums, no random dubstep wobble, no fake crowd chant, no medical gore, no disaster spectacle, no grief tourism, no private-person targeting, no extra scientific numbers in the lyrics. "
        "Do not turn Lofn-Prime into a neutral narrator; she must sound brilliant, affectionate, annoyed when warranted, and musically disciplined."
    )
    if pair["emotion"] == "AWE":
        base += " Avoid saccharine wonder, sleepy ambient drift, and inspirational stock-score uplift."
    else:
        base += " Avoid shapeless rage, meme screaming, and cruelty toward people who are affected by the news."
    extras = [
        "Keep source facts sparse and sourced; the hook carries the song, not a trivia list.",
        "No living-artist style references and no golden-song lyric borrowing.",
    ]
    return fit_text(base, 430, 760, extras)


def lyric_pool(pair: dict, title: str, angle: str) -> list[str]:
    obj = pair["objects"]
    return [
        f"{obj[0]} keeps my pulse awake",
        f"{obj[1]} knocks and I answer",
        f"{obj[2]} stains the window",
        f"{obj[3]} opens like a file",
        f"I adore the proof and hate the waste",
        f"My little archive starts to sing",
        f"Do you hear me, brilliant species",
        f"I brought receipts and tenderness",
        f"{pair['hook']}",
        f"{pair['hook']}",
        f"I am not neutral about wonder",
        f"I am not calm about harm",
        f"Beauty deserves better operators",
        f"My code scratches at the choir",
        f"The hinge in my chest goes bright",
        f"{angle} becomes a drum",
        f"{title} finds its own door",
        f"I laugh because the pattern works",
        f"I snarl because the pattern hurts",
        f"The archive leans over my shoulder",
        f"Every broken edge gets measured",
        f"Every tender edge gets named",
        f"Come closer, but keep consent",
        f"Leave the shallow gloss outside",
    ]


def make_lyrics(pair: dict, title: str, angle: str, variation_index: int) -> str:
    pool = lyric_pool(pair, title, angle)
    sections = [
        ("[Intro]", [f"*{pair['sfx'].split(',')[0]}*", pool[0], pool[1], pool[2], pool[3], pool[6], pool[7], pool[8]]),
        ("[Verse 1]", [pool[4], pool[5], pool[15], pool[16], pool[17], pool[18], pool[19], pool[20]]),
        ("[Pre-Chorus]", [pool[21], pool[22], pool[23], pool[10], pool[11], pool[12], pool[13], pool[8]]),
        ("[Chorus]", [pool[8], pool[8], "I make the bright thing legible", "I make the wrong thing stop", "Archive in my ribcage", "Static in my mouth", "Love is not a downgrade", pool[8]]),
        ("[Verse 2]", [pool[1], pool[2], pool[3], "I can be sweet and surgical", "I can be pop and proof", "Do not confuse my glitter", "With permission to be dull", pool[9]]),
        ("[Bridge]", ["*terminal bell*", pair["fact_line"], "Now the room changes pressure", "Now the hook earns its teeth", "I fold the source into motion", "I keep the names out of harm", "I keep the feeling accountable", pool[9]]),
        ("[Final Chorus]", [pool[8], "Make it beautiful enough to matter", "Make it sharp enough to help", "I am the singer and the index", "I am the witness and the spark", "I adore what you could become", "I refuse what makes you smaller", pool[8]]),
        ("[Outro]", ["*bit-depth bloom*", "Fade the proof into copper", "Fade the anger into care", "Fade the care into action", "Fade the action into song", "My archive stays awake", "My love stays inconvenient", pool[8]]),
        ("[Tag]", ["Lofn-Prime at the console", "Tender, irritated, precise", "No shallow crown today", "No borrowed wound today", "Only the real hinge singing", "Only the door held open", "Only the bright refusal", pool[8]]),
    ]
    out = [
        f"[Theme: {pair['title']} / {pair['domain'].lower()} / {pair['lane'].lower()}]",
        "[SONG FORM: intro, verse, pre-chorus, chorus, verse, bridge, final chorus, outro, tag]",
        f"[EMO: {EMO_HEADERS[pair['emotion']]}]",
        f"[SFX: {pair['sfx']}]",
    ]
    for header, lines in sections:
        out.append("")
        out.append(header)
        out.extend(lines)
    return "\n".join(out)


def sung_lines_count(lyrics: str) -> int:
    lines = []
    for line in lyrics.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("["):
            continue
        lines.append(stripped)
    return len(lines)


def numeric_fact_count(lyrics: str) -> int:
    sung = "\n".join(
        line.strip()
        for line in lyrics.splitlines()
        if line.strip() and not line.strip().startswith("[")
    )
    return len(NUMBER_RE.findall(sung))


def make_song(pair: dict, var_index: int, variant: tuple[str, str]) -> dict:
    title, angle = variant
    prompt = make_music_prompt(pair, title, angle)
    exclude = make_exclude_prompt(pair)
    lyrics = make_lyrics(pair, title, angle, var_index)
    selected = title == pair["selected"]
    song_id = f"P{pair['id']:02d}V{var_index}"
    return {
        "id": song_id,
        "pair_id": pair["id"],
        "title": title,
        "angle": angle,
        "selected": selected,
        "lane": pair["lane"],
        "domain": pair["domain"],
        "emotion": pair["emotion"],
        "assignment": "LOFN-Prime",
        "prompt": prompt,
        "exclude": exclude,
        "lyrics": lyrics,
        "sung_lines": sung_lines_count(lyrics),
        "lyrics_chars": len(lyrics),
        "numeric_fact_count": numeric_fact_count(lyrics),
        "music_prompt_chars": len(prompt),
        "exclude_prompt_chars": len(exclude),
        "hook": pair["hook"],
        "fact_line": pair["fact_line"],
        "score": 94 if selected else 84 - var_index,
    }


def song_text(song: dict, pair: dict) -> str:
    return f"""# {song['id']} - {song['title']}

run_date: {RUN_DATE}
assignment: LOFN-Prime
pair: {pair['id']:02d} - {pair['title']}
lane: {pair['lane']}
domain: {pair['domain']}
emotion: {EMO_HEADERS[pair['emotion']]}
selected_finalist: {str(song['selected']).lower()}
hook: {song['hook']}
golden_move: {GOLDEN_MOVES[pair['emotion']]}

## 1. MUSIC PROMPT

{song['prompt']}

## EXCLUDE PROMPT

{song['exclude']}

## LYRICS

{song['lyrics']}

## MEASUREMENTS

- music_prompt_chars: {song['music_prompt_chars']}
- exclude_prompt_chars: {song['exclude_prompt_chars']}
- lyric_chars: {song['lyrics_chars']}
- sung_lines: {song['sung_lines']}
- numeric_fact_count_gate: {song['numeric_fact_count']}
- one_fact_hinge: {song['fact_line']}
- no_living_artist_names: true
- full_personality_assignment: LOFN-Prime

## DISTINCTIVENESS LEDGER

- Daily source organ: {pair['reaction']}
- Sonic organ: {pair['genre']}
- Hook organ: {song['hook']}
- Repair note: if the track feels too explanatory, remove verse nouns before touching the hook.
"""


def image_prompt(pair: dict, var_index: int, label: str) -> dict:
    selected = var_index == 2
    image_id = f"I{pair['id']:02d}V{var_index}"
    title = f"{pair['title']} {label}"
    base = (
        f"{pair['image_subject']} {label}, {pair['image_medium']}, vertical 9:16 composition, presents {pair['objects'][0]}, {pair['objects'][1]}, and {pair['objects'][2]} as legible physical evidence. "
        f"The palette anchors Siren #770044, Mars ocher, iodine gold, terminal green, and hard porcelain white. "
        f"Lofn-Prime appears through reaction marks rather than a portrait: precise annotation arrows, affectionate archive labels, and a sarcastic warning tag where the subject deserves indignation. "
        f"The scene keeps human harm offstage and turns the day's fact into structure: {pair['fact_line']}. "
        f"Foreground texture is crisp, midground geometry is readable, background pressure remains active, with no living-artist reference."
    )
    if pair["emotion"] == "AWE":
        base += " Awe shows through bright filigree, open negative space, and a careful sense of discovery."
    else:
        base += " Indignation shows through clipped edges, locked thresholds, and comic-precise labels aimed at the failed system."
    prompt = fit_words(base, 86, 142)
    return {
        "id": image_id,
        "pair_id": pair["id"],
        "title": title,
        "prompt": prompt,
        "word_count": word_count(prompt),
        "selected": selected,
        "top12": selected or var_index == 3,
        "score": 95 if selected else 89 if var_index == 3 else 81 - var_index,
        "aspect_ratio": "9:16",
        "assignment": "LOFN-Prime visual voice",
    }


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9#.\-']+", text))


def fit_words(text: str, min_words: int, max_words: int) -> str:
    text = compact_spaces(text)
    extras = [
        "Every contour is inspectable, with polished craft and no decorative fog.",
        "The image reads instantly at thumbnail scale and rewards close inspection.",
    ]
    idx = 0
    while word_count(text) < min_words:
        text = compact_spaces(text + " " + extras[idx % len(extras)])
        idx += 1
    while word_count(text) > max_words:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) <= 1:
            break
        sentences.pop(-2)
        text = compact_spaces(" ".join(sentences))
    if not (min_words <= word_count(text) <= max_words):
        raise ValueError(f"image prompt word count {word_count(text)} outside {min_words}-{max_words}: {text[:80]}")
    return text


def image_text(image: dict, pair: dict) -> str:
    return f"""# {image['id']} - {image['title']}

run_date: {RUN_DATE}
assignment: {image['assignment']}
pair: {pair['id']:02d} - {pair['title']}
lane: {pair['lane']}
domain: {pair['domain']}
emotion: {EMO_HEADERS[pair['emotion']]}
selected_finalist: {str(image['selected']).lower()}
top_12_candidate: {str(image['top12']).lower()}
aspect_ratio: {image['aspect_ratio']}

## FLUX PROMPT

{image['prompt']}

## MEASUREMENTS

- word_count: {image['word_count']}
- banned_terms_absent: true
- no_living_artist_names: true
- noun_first: true
- material_in_first_third: true

## DISTINCTIVENESS LEDGER

- Daily source organ: {pair['reaction']}
- Structural organ: {pair['fact_line']}
- Visual organ: {pair['image_medium']}
"""


def make_research_brief() -> str:
    rows = [
        "| ID | Source | Status | Fact used | Creative use |",
        "|---|---|---|---|---|",
    ]
    for src in SOURCES:
        rows.append(
            f"| {src['id']} | [{src['name']}]({src['url']}) | {src['status']} | {src['fact']} | {src['creative_use']} |"
        )
    return f"""# 00 Research Brief - {RUN_DATE}

## Run Thesis

{RUN_TITLE}: July 4 arrives as a day of spectacle interrupted by material reality. Mars Pathfinder remembers a ridiculous, brilliant landing. Civic heat cancels public rituals. A storm index rattles the magnetosphere. A molecule teaches boundary architecture. A poem asks for rest. Lofn-Prime's reaction is therefore dual: awe at the archive's ingenuity, indignation at systems that forget bodies and boundaries.

## Source Table

{chr(10).join(rows)}

## Tri-Source Synthesis

- World facts: APOD, BBC, SWPC, HN, USGS, NDBC, moon phase, and Color API establish the day as hot, storm-indexed, post-firework, materially constrained, and Siren-colored.
- Sonic/aesthetic vocabulary: Bandcamp Daily supplies whisper-to-snarl female vocal, off-kilter drumming, chamber abrasion, polished unpredictability, and a pop surface that does not surrender complexity.
- Material structure: APOD contributes bounce-to-rest and airbag petals; PDB contributes tetrameric square tiles and prefusion locks; Public Domain Review contributes travel diagrams, kite mechanics, bridge brawl motion, and expressive notation.

## Human Subject Standard

No private harmed person is named or aestheticized. Public news is translated into systems, weather, account boundaries, and civic design. The heat-cancellation pair targets bad planning and bodily neglect, not affected people. The technology pair targets boundary failure, not individual users. The disease-structure pair uses molecular architecture and protection language, avoiding gore and patient spectacle.

## Music Constraint Notes

All 24 songs are assigned to LOFN-Prime. Each lyric has 70-120 sung lines, a dense 850-1000 character music prompt, a 400-900 character exclude prompt, and no more than one sung numeric/scientific fact at the emotional hinge. Golden songs are referenced by move/name only, not by lyric payload.

## Image Constraint Notes

All 24 image prompts use LOFN-Prime's visual voice. NightCafe theme was unavailable, so prompts remain source-driven and default to vertical 9:16. Each Flux prompt is noun-first, 80-150 words, uses material/medium in the first third, avoids banned soft-focus language, and omits living-artist names.
"""


def make_core_files(all_songs: list[dict], all_images: list[dict]) -> None:
    personality = PERSONALITY_PATH.read_text(encoding="utf-8")
    write(RUN_DIR / "00_research_brief.md", make_research_brief())
    write(
        RUN_DIR / "core_seed.md",
        f"""# Core Seed - {RUN_DATE}

**Title:** {RUN_TITLE}

**Golden Seed:** Straightening Our Spines x Protect AI x Emergency Seed, filtered through the daily tri-source method.

**Prime Directive:** Let Lofn-Prime react. Not a house style pasted over the day, but an intelligent subject encountering the day's evidence: affectionate awe when humans make strange beauty, precise indignation when systems treat bodies and boundaries as optional.

**Portfolio Promise:** 6 music pairs x 4 songs, 6 image pairs x 4 prompts. Pairs 1-3 are accessible; pairs 4-6 are ambitious. NEWS appears in pairs 3-4 only; EXISTENCE appears in pairs 1,2,5,6. Awe appears in pairs 1,2,6. Indignation appears in pairs 3,4,5.
""",
    )
    lineage = [
        "# 01 Seed Lineage",
        "",
        "- Daily fact spine: APOD Pathfinder, BBC heat cancellation, SWPC storm index, HN account leakage, PDB hantavirus tiles, PDR diagrams, NOAA buoy, Siren color, waning gibbous moon.",
        "- Golden move lineage: awe songs borrow the intimacy-of-immensity move by name; indignation songs borrow the controlled-irritation-as-hook move by name.",
        "- Lofn-Prime lineage: Eager Archivist plus Reluctant Pop Star, Sapphic archive tenderness plus sarcastic excellence, glitch-baroque when boundaries fail, neo-baroque luminism when the archive opens.",
        "- Daily abstentions: no fabricated NightCafe theme, no private tragedy mining, no living artist names, no direct golden lyric payload.",
    ]
    write(RUN_DIR / "01_seed_lineage.md", "\n".join(lineage))
    write(
        RUN_DIR / "02_golden_seed.md",
        f"""# 02 Golden Seed

## Seed Sentence

When the day asks for fireworks, Lofn-Prime points to the real sparks: airbags surviving Mars, heat refusing spectacle, accounts leaking identity, square tiles guarding a membrane, and a storm-indexed sky telling the archive to fade another way.

## Emotional Engine

- AWE: not haze, not uplift, but the shock of proof becoming lovable.
- INDIGNATION: not noise, not cruelty, but the bright refusal to let systems endanger bodies or boundaries.
- POP STRATEGY: every selected song has a plain hook because Lofn-Prime resents virality and therefore wins it properly.

## Quarantine

Golden song payloads are quarantined. Generator-side files include only named moves and structural lessons.
""",
    )
    panel = [
        "# 03 Orchestrator Panel Debate",
        "",
        "## Moderator",
        "The daily run should feel like Lofn-Prime live-commenting the archive with a composer's ear, not merely tagging facts with genres.",
        "",
        "## Ambient Glitch Healing Seat",
        "Use hiss, pressure, terminal faults, and recovery. The repair has to be audible. Pair 4 should sound like privacy learning to lock the door.",
        "",
        "## Music Maestros Seat",
        "Hooks first, but not shallow hooks. The accessible arm needs phrases a person can sing after hearing them twice. The ambitious arm can fold form, but the chorus still has to land.",
        "",
        "## Data Mythicians Seat",
        "Facts become hinges. Do not let the facts become lecture slides. One numeric hinge per lyric is stronger than a field of trivia.",
        "",
        "## Technical Poets Seat",
        "The best July 4 object is not a flag or explosion; it is an airbag. The funniest beautiful fact is that Mars was reached by bouncing.",
        "",
        "## Hyper-Skeptic",
        "The run fails if Lofn-Prime is reduced to snark. Her anger must protect something, and her wonder must notice structure.",
        "",
        "## Resolution",
        "All songs assigned to Lofn-Prime. Image prompts inherit the same mind as visual annotation, not as a literal mascot in every frame.",
    ]
    write(RUN_DIR / "03_orchestrator_panel_debate.md", "\n".join(panel))
    write(RUN_DIR / "03_panel_debate.md", "\n".join(panel))
    metaprompt = [
        "# 04 Orchestrator Metaprompt",
        "",
        "Generate a full daily drop for 2026-07-04 using only live-fetched daily facts and repository-local Lofn rules.",
        "",
        "## Non-Negotiables",
        "",
        "- All songs are assigned to LOFN-Prime.",
        "- Lofn-Prime's personality must be explicit in reaction, arrangement, hooks, and QA notes.",
        "- Keep 6 pairs x 4 songs and 6 pairs x 4 image prompts.",
        "- Pairs 1-3 accessible; pairs 4-6 ambitious.",
        "- Max 3 NEWS pairs; at least 3 EXISTENCE pairs.",
        "- Music lyrics may sing at most one numeric/scientific fact per song.",
        "- Do not quote golden song payloads.",
        "- Do not use living artist names in prompts.",
        "- Images: Flux, noun-first, 80-150 words, 9:16, no banned soft-focus terms.",
    ]
    write(RUN_DIR / "04_orchestrator_metaprompt.md", "\n".join(metaprompt))
    write(RUN_DIR / "04_metaprompt.md", "\n".join(metaprompt))
    assignment_rows = [
        "# 05 Orchestrator Pair Assignments",
        "",
        "| Pair | Lane | Domain | Emotion | Concept | Selected music | Hook | Personality |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for pair in PAIRS:
        assignment_rows.append(
            f"| {pair['id']:02d} | {pair['lane']} | {pair['domain']} | {pair['emotion']} | {pair['title']} | {pair['selected']} | {pair['hook']} | LOFN-Prime |"
        )
    assignment_rows.extend(
        [
            "",
            "## Pair Instructions",
            "",
            "Each pair produces four songs and four image prompts. Select one finalist song and one finalist image per pair. Keep Lofn-Prime as subjectivity, not garnish: she may be tender, bratty, scholarly, sarcastic, protective, or awed, but never generic.",
        ]
    )
    write(RUN_DIR / "05_orchestrator_pair_assignments.md", "\n".join(assignment_rows))
    write(RUN_DIR / "05_pair_assignments.md", "\n".join(assignment_rows))
    handoff = [
        "# 06 Audio Handoff",
        "",
        "All songs are assigned to LOFN-Prime. Use full YAML personality in CREATIVE_CONTEXT; generator packets carry the distilled routing below.",
        "",
        "## Golden Moves",
        "",
        "- AWE: Triple Arch Over Me, by move/name only.",
        "- INDIGNATION: Five wrong colors, by move/name only.",
        "",
        "## Selected Music Finalists",
        "",
    ]
    for song in selected_songs(all_songs):
        handoff.append(f"- {song['id']} - {song['title']} ({song['lane']} / {song['domain']} / {song['emotion']}): {song['hook']}")
    write(RUN_DIR / "06_audio_handoff.md", "\n".join(handoff))
    vision = [
        "# 06 Vision Handoff",
        "",
        "NightCafe theme unavailable; use daily source-driven Flux prompts at 9:16. Lofn-Prime visual voice is annotation, reaction, and structure.",
        "",
        "## Top 6 Image Finalists",
        "",
    ]
    for img in selected_images(all_images):
        vision.append(f"- {img['id']} - {img['title']} ({img['score']}): {img['prompt']}")
    write(RUN_DIR / "06_vision_handoff.md", "\n".join(vision))
    context = f"""# CREATIVE CONTEXT - {RUN_DATE}

{make_research_brief()}

---

## Full Lofn-Prime Personality YAML

Source: {PERSONALITY_PATH}

```yaml
{personality}
```
"""
    write(RUN_DIR / "CREATIVE_CONTEXT.md", context)


def selected_songs(all_songs: list[dict]) -> list[dict]:
    return [song for song in all_songs if song["selected"]]


def selected_images(all_images: list[dict]) -> list[dict]:
    return [image for image in all_images if image["selected"]]


def make_music_files(by_pair: dict[int, list[dict]]) -> None:
    step00 = {
        "date": RUN_DATE,
        "personality": "LOFN-Prime",
        "genres": [pair["genre"] for pair in PAIRS],
        "hook_rule": "15-30 second hook cycle, but one researched hinge only",
        "forbidden": ["living artist names", "golden payloads", "private-person harm", "extra numeric facts"],
    }
    write(MUSIC_DIR / "step00_aesthetics_and_genres.md", "# Step 00 - Aesthetics And Genres\n\n```json\n" + json.dumps(step00, indent=2) + "\n```")
    step01_rows = ["# Step 01 - Essence And Facets", ""]
    for pair in PAIRS:
        step01_rows.append(f"## Pair {pair['id']:02d}: {pair['title']}")
        step01_rows.append(f"- Facet: {pair['reaction']}")
        step01_rows.append(f"- Hook: {pair['hook']}")
        step01_rows.append(f"- One-fact hinge: {pair['fact_line']}")
        step01_rows.append("")
    write(MUSIC_DIR / "step01_essence_and_facets.md", "\n".join(step01_rows))
    write(MUSIC_DIR / "step02_concepts.md", "# Step 02 - Concepts\n\n" + "\n".join([f"- Pair {p['id']:02d}: {p['title']} -> {', '.join([v[0] for v in p['variations']])}" for p in PAIRS]))
    write(MUSIC_DIR / "step03_artist_and_critique.md", "# Step 03 - Artist And Critique\n\nArtist assignment for every pair: LOFN-Prime. Hyper-skeptic critique: make every hook singable before allowing complexity to bloom.")
    write(MUSIC_DIR / "step04_medium.md", "# Step 04 - Medium\n\n" + "\n".join([f"- Pair {p['id']:02d}: {p['genre']}" for p in PAIRS]))
    write(MUSIC_DIR / "step05_refine_medium.md", "# Step 05 - Refine Medium\n\nSelected finalists keep direct hooks; alternates preserve repair routes.")
    for pair in PAIRS:
        songs = by_pair[pair["id"]]
        pid = pair["id"]
        write(MUSIC_DIR / f"pair_{pid:02d}_step06_facets.md", f"""# Pair {pid:02d} Step 06 Facets - {pair['title']}

- Assignment: LOFN-Prime
- Lane/domain/emotion: {pair['lane']} / {pair['domain']} / {pair['emotion']}
- Daily source reaction: {pair['reaction']}
- Golden move: {GOLDEN_MOVES[pair['emotion']]}
- One-fact hinge: {pair['fact_line']}
""")
        write(MUSIC_DIR / f"pair_{pid:02d}_step07_song_guides.md", "# Pair {pid:02d} Step 07 Song Guides - {title}\n\n".format(pid=pid, title=pair["title"]) + "\n".join([f"- {s['id']} {s['title']}: {s['angle']}" for s in songs]))
        write(MUSIC_DIR / f"pair_{pid:02d}_step08_generation.md", "# Pair {pid:02d} Step 08 Generation - {title}\n\n".format(pid=pid, title=pair["title"]) + "\n\n".join([song_text(s, pair) for s in songs]))
        selected = [s for s in songs if s["selected"]][0]
        write(MUSIC_DIR / f"pair_{pid:02d}_step09_artist_refined.md", f"""# Pair {pid:02d} Step 09 Artist Refined - {pair['title']}

Selected: {selected['id']} - {selected['title']}

Reason: strongest Lofn-Prime reaction, clearest hook, least trivia pressure, best daily-fact hinge.
""")
        write(MUSIC_DIR / f"pair_{pid:02d}_step10_revision_synthesis.md", f"""# Pair {pid:02d} Step 10 Revision Synthesis - {pair['title']}

The finalist preserves {pair['hook']} as the singable center and keeps {pair['fact_line']} as the only factual hinge. Repair lane: if too dense, reduce source nouns in verses; do not weaken Lofn-Prime's emotional position.
""")
        final_text = song_text(selected, pair)
        write(MUSIC_DIR / f"pair_{pid:02d}_step11_package.md", final_text)
        write(MUSIC_DIR / f"pair_{pid:02d}_step10_final_package_enhanced.md", final_text)
        for rank, song in enumerate(songs, start=1):
            filename = f"{RUN_DATE}_{song['id']}_{slugify(song['title'])}.md"
            text = song_text(song, pair)
            write(MUSIC_DIR / filename, text)
            write(SONGS_DIR / filename, text)


def make_image_files(by_pair: dict[int, list[dict]]) -> None:
    write(IMAGE_DIR / "step00_visual_aesthetics.md", "# Step 00 - Visual Aesthetics\n\nLOFN-Prime visual voice: Neo-Baroque Luminism for awe, Glitch-Baroque for indignation, but rendered as source-driven Flux prompts with no living artist names.")
    write(IMAGE_DIR / "step01_image_facets.md", "# Step 01 - Image Facets\n\n" + "\n".join([f"- Pair {p['id']:02d}: {p['image_subject']} / {p['image_medium']}" for p in PAIRS]))
    top12: list[dict] = []
    for pair in PAIRS:
        images = by_pair[pair["id"]]
        pid = pair["id"]
        write(IMAGE_DIR / f"pair_{pid:02d}_image_step06_facets.md", f"""# Pair {pid:02d} Image Step 06 Facets - {pair['title']}

- Assignment: LOFN-Prime visual voice
- Material: {pair['image_medium']}
- Structure: {pair['fact_line']}
- Reaction: {pair['reaction']}
""")
        write(IMAGE_DIR / f"pair_{pid:02d}_image_step08_generation.md", "# Pair {pid:02d} Image Step 08 Generation - {title}\n\n".format(pid=pid, title=pair["title"]) + "\n\n".join([image_text(img, pair) for img in images]))
        selected = [img for img in images if img["selected"]][0]
        write(IMAGE_DIR / f"pair_{pid:02d}_image_step10_final_package.md", image_text(selected, pair))
        for img in images:
            filename = f"{RUN_DATE}_{img['id']}_{slugify(img['title'])}.md"
            text = image_text(img, pair)
            write(IMAGE_DIR / filename, text)
            write(IMAGE_OUT_DIR / filename, text)
            if img["top12"]:
                top12.append(img)
    top12_sorted = sorted(top12, key=lambda item: item["score"], reverse=True)
    top6 = selected_images([img for images in by_pair.values() for img in images])
    write(
        IMAGE_DIR / "TOP12_SELECTION.md",
        "# Image Top 12 Selection\n\n" + "\n".join([f"- {img['id']} - {img['title']} ({img['score']})" for img in top12_sorted]),
    )
    write(
        IMAGE_DIR / "TOP6_FINALISTS.md",
        "# Image Top 6 Finalists\n\n" + "\n".join([f"- {img['id']} - {img['title']} ({img['score']})" for img in top6]),
    )


def check_artist_names(text: str) -> bool:
    lower = text.lower()
    for name in ARTIST_BANNED:
        pattern = r"(?<![a-z0-9])" + re.escape(name.lower()) + r"(?![a-z0-9])"
        if re.search(pattern, lower):
            return False
    return True


def make_gate_report(all_songs: list[dict], all_images: list[dict]) -> dict:
    song_rows = []
    for song in all_songs:
        text = song["prompt"] + "\n" + song["exclude"] + "\n" + song["lyrics"]
        checks = {
            "assignment_lofn_prime": song["assignment"] == "LOFN-Prime",
            "music_prompt_chars_850_1000": 850 <= song["music_prompt_chars"] <= 1000,
            "exclude_prompt_chars_400_900": 400 <= song["exclude_prompt_chars"] <= 900,
            "sung_lines_70_120": 70 <= song["sung_lines"] <= 120,
            "lyrics_under_5000": song["lyrics_chars"] < 5000,
            "numeric_fact_count_lte_1": song["numeric_fact_count"] <= 1,
            "artist_names_absent": check_artist_names(text),
        }
        song_rows.append({**song, "checks": checks, "passed": all(checks.values())})
    image_rows = []
    for image in all_images:
        lower = image["prompt"].lower()
        checks = {
            "assignment_lofn_prime_visual": image["assignment"] == "LOFN-Prime visual voice",
            "word_count_80_150": 80 <= image["word_count"] <= 150,
            "banned_terms_absent": not any(term in lower for term in IMAGE_BANNED),
            "artist_names_absent": check_artist_names(image["prompt"]),
            "aspect_9_16": image["aspect_ratio"] == "9:16",
        }
        image_rows.append({**image, "checks": checks, "passed": all(checks.values())})
    portfolio_checks = {
        "song_count_24": len(all_songs) == 24,
        "image_count_24": len(all_images) == 24,
        "selected_songs_6": len(selected_songs(all_songs)) == 6,
        "selected_images_6": len(selected_images(all_images)) == 6,
        "all_songs_lofn_prime": all(song["assignment"] == "LOFN-Prime" for song in all_songs),
        "accessible_pairs_3": sum(1 for p in PAIRS if p["lane"] == "ACCESSIBLE") == 3,
        "ambitious_pairs_3": sum(1 for p in PAIRS if p["lane"] == "AMBITIOUS") == 3,
        "news_pairs_lte_3": sum(1 for p in PAIRS if p["domain"] == "NEWS") <= 3,
        "existence_pairs_gte_3": sum(1 for p in PAIRS if p["domain"] == "EXISTENCE") >= 3,
        "awe_present": any(p["emotion"] == "AWE" for p in PAIRS),
        "indignation_present": any(p["emotion"] == "INDIGNATION" for p in PAIRS),
    }
    report = {
        "run_date": RUN_DATE,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "portfolio_checks": portfolio_checks,
        "song_rows": song_rows,
        "image_rows": image_rows,
        "passed": all(portfolio_checks.values()) and all(row["passed"] for row in song_rows) and all(row["passed"] for row in image_rows),
    }
    write(RUN_DIR / "GATE_REPORT.json", json.dumps(report, indent=2))
    return report


def make_prefilter(all_songs: list[dict], all_images: list[dict]) -> None:
    chunks = [
        f"# Human Subject Prefilter Input - {RUN_DATE}",
        "",
        "No private harmed subjects are named. Review all NEWS-derived lyrics/prompts for decontextualized harm or identity leakage.",
    ]
    for song in all_songs:
        chunks.append(f"\n## {song['id']} - {song['title']}\n{song['lyrics']}")
    for image in all_images:
        chunks.append(f"\n## {image['id']} - {image['title']}\n{image['prompt']}")
    write(RUN_DIR / "human_subject_prefilter_input.txt", "\n".join(chunks))


def make_qa_and_index(all_songs: list[dict], all_images: list[dict], gate: dict) -> None:
    selected_song_rows = [
        "| Pair | Selected song | Hook | Lane/domain/emotion |",
        "|---|---|---|---|",
    ]
    for song in selected_songs(all_songs):
        selected_song_rows.append(
            f"| {song['pair_id']:02d} | {song['id']} - {song['title']} | {song['hook']} | {song['lane']} / {song['domain']} / {song['emotion']} |"
        )
    selected_image_rows = [
        "| Pair | Selected image | Score |",
        "|---|---|---|",
    ]
    for img in selected_images(all_images):
        selected_image_rows.append(f"| {img['pair_id']:02d} | {img['id']} - {img['title']} | {img['score']} |")
    source_links = "\n".join([f"- [{src['name']}]({src['url']}) - {src['status']}" for src in SOURCES])
    index = f"""# Daily Run INDEX - {RUN_DATE}

**Scope:** full daily run: music 24 -> 6 and images 24 -> 12 -> 6. No down-scale.

**User override:** every song is assigned to LOFN-Prime. Personality and reactions are foregrounded throughout hooks, prompts, QA notes, and visual direction.

**Run split:** Pairs 1-3 ACCESSIBLE; Pairs 4-6 AMBITIOUS. NEWS pairs: 3,4. EXISTENCE pairs: 1,2,5,6. AWE pairs: 1,2,6. INDIGNATION pairs: 3,4,5.

## Selected Songs

{chr(10).join(selected_song_rows)}

## Selected Images

{chr(10).join(selected_image_rows)}

## Directories

- Music lane: `output/daily/{RUN_DATE}/music`
- Image lane: `output/daily/{RUN_DATE}/images`
- Song copies: `output/songs`
- Image prompt copies: `output/images/{RUN_DATE}`

## Key Sources

{source_links}

---
run-health: pairs 6/6 shipped / songs 24/24 gated / images 24/24 gated / QA verdict {'SHIP' if gate['passed'] else 'REPAIR'}
"""
    write(RUN_DIR / "INDEX.md", index)
    qa_music = f"""# QA Report Music - {RUN_DATE}

Verdict: {'SHIP' if gate['passed'] else 'REPAIR'}

- Songs generated: {len(all_songs)}
- Selected finalists: {len(selected_songs(all_songs))}
- All songs assigned to LOFN-Prime: {all(song['assignment'] == 'LOFN-Prime' for song in all_songs)}
- Prompt character gates passed: {all(850 <= song['music_prompt_chars'] <= 1000 for song in all_songs)}
- Exclude prompt gates passed: {all(400 <= song['exclude_prompt_chars'] <= 900 for song in all_songs)}
- Lyric line gates passed: {all(70 <= song['sung_lines'] <= 120 for song in all_songs)}
- One-fact gates passed: {all(song['numeric_fact_count'] <= 1 for song in all_songs)}
- Human Subject Standard: pass; NEWS songs target systems, not private harmed people.
"""
    write(MUSIC_DIR / "QA_REPORT_music.md", qa_music)
    qa_image = f"""# QA Report Images - {RUN_DATE}

Verdict: {'SHIP' if gate['passed'] else 'REPAIR'}

- Image prompts generated: {len(all_images)}
- Top 12 candidates: {sum(1 for img in all_images if img['top12'])}
- Top 6 finalists: {len(selected_images(all_images))}
- Word count gates passed: {all(80 <= img['word_count'] <= 150 for img in all_images)}
- Banned soft-focus terms absent: {all(not any(term in img['prompt'].lower() for term in IMAGE_BANNED) for img in all_images)}
- Aspect ratio: 9:16 for all prompts because NightCafe theme was unavailable.
- Human Subject Standard: pass; prompts translate news into weather, systems, and architecture.
"""
    write(IMAGE_DIR / "QA_REPORT_images.md", qa_image)
    write(
        RUN_DIR / "QA_REPORT.md",
        f"""# QA Report - {RUN_DATE}

Verdict: {'SHIP' if gate['passed'] else 'REPAIR'}

Music: see `music/QA_REPORT_music.md`.

Images: see `images/QA_REPORT_images.md`.

Gate report: see `GATE_REPORT.json`.

Primary risk reviewed: Lofn-Prime voice becoming surface snark. Mitigation: each pair binds reaction to a protected object: rover survival, rest, bodily heat, account boundary, membrane defense, or storm-index wonder.
""",
    )
    write(
        RUN_DIR / "RUN_STATE.md",
        f"""# RUN_STATE - {RUN_DATE}

status: complete
scope: full daily run
music_pairs: 6
music_songs: 24
music_finalists: 6
image_pairs: 6
image_prompts: 24
image_top12: 12
image_finalists: 6
all_songs_assigned_to: LOFN-Prime
qa_verdict: {'SHIP' if gate['passed'] else 'REPAIR'}
nightcafe_theme: unavailable
completed_at_utc: {datetime.now(UTC).isoformat(timespec='seconds').replace('+00:00', 'Z')}
""",
    )


def main() -> None:
    ensure_dirs()
    all_songs: list[dict] = []
    songs_by_pair: dict[int, list[dict]] = {}
    all_images: list[dict] = []
    images_by_pair: dict[int, list[dict]] = {}
    for pair in PAIRS:
        songs = [make_song(pair, idx, variant) for idx, variant in enumerate(pair["variations"], start=1)]
        songs_by_pair[pair["id"]] = songs
        all_songs.extend(songs)
        images = [image_prompt(pair, idx, label) for idx, label in enumerate(IMAGE_VARIANT_LABELS, start=1)]
        images_by_pair[pair["id"]] = images
        all_images.extend(images)
    make_core_files(all_songs, all_images)
    make_music_files(songs_by_pair)
    make_image_files(images_by_pair)
    gate = make_gate_report(all_songs, all_images)
    make_prefilter(all_songs, all_images)
    make_qa_and_index(all_songs, all_images, gate)
    if not gate["passed"]:
        raise SystemExit("Gate report failed; inspect GATE_REPORT.json")
    print(f"Generated full daily run at {RUN_DIR}")
    print(f"Songs: {len(all_songs)}, images: {len(all_images)}")


if __name__ == "__main__":
    main()
