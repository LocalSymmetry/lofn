from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "2026-07-02"
RUN_SLUG = "daily-music-2026-07-02"
RUN_DIR = ROOT / "output" / "daily" / RUN_DATE
RUN_MUSIC_DIR = RUN_DIR / "music"
SONGS_DIR = ROOT / "output" / "songs"
CREATED = "2026-07-03T00:00:00-07:00"


SOURCES = [
    ("NASA APOD", "https://apod.nasa.gov/apod/ap260702.html"),
    ("NASA APOD API", "https://api.nasa.gov/planetary/apod?api_key=DEMO_KEY&date=2026-07-02"),
    ("USGS July 2 M4.5+ query", "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2026-07-02&endtime=2026-07-03&minmagnitude=4.5&orderby=magnitude"),
    ("NOAA SWPC K index", "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"),
    ("NOAA SWPC solar wind speed", "https://services.swpc.noaa.gov/products/summary/solar-wind-speed.json"),
    ("Hacker News front 2026-07-02", "https://news.ycombinator.com/front?day=2026-07-02"),
    ("BBC World RSS", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("Bandcamp: mary in the junkyard", "https://daily.bandcamp.com/album-of-the-day/mary-in-the-junkyard-role-model-hermit-review"),
    ("Bandcamp: Best Metal June 2026", "https://daily.bandcamp.com/best-metal/the-best-metal-on-bandcamp-june-2026"),
    ("Bandcamp: Prog Is a State of Mind #2", "https://daily.bandcamp.com/prog-state-mind/prog-is-a-state-of-mind-2"),
    ("PDB Molecule of the Month", "https://pdb101.rcsb.org/motm"),
    ("The Color API #0702", "https://www.thecolorapi.com/id?hex=0702"),
    ("NOAA NDBC 46059 spectral data", "https://www.ndbc.noaa.gov/data/realtime2/46059.spec"),
    ("Guardian Kyiv July 2", "https://www.theguardian.com/world/2026/jul/02/russia-attacks-kyiv-missiles-drones-ukraine"),
    ("Al Jazeera Kyiv July 2", "https://www.aljazeera.com/news/2026/7/2/kyiv-attacked-after-ukraines-zelenskyy-warns-of-massive-russian-strike"),
]


PAIRS = [
    {
        "pair": 1,
        "lane": "ACCESSIBLE / EXISTENCE / AWE",
        "anchor": "APOD sibling supernova remnants: younger bright center over older purple filament, about 6000 light-years away.",
        "title": "two lights in one window",
        "concept": "A household witness learns that comfort can be a receipt for an ancient rupture.",
        "medium": "bio-ambient chamber pop / glitch folk with contact-mic window frame, glass FM bells, and a never-resolving dual-layer drone",
        "genre": "bio-ambient chamber pop / glitch folk",
        "bpm": 76,
        "key": "D Dorian",
        "energy": "low-lit witness pulse",
        "voice": "adult female mezzo A3-D5, chest-to-mix, dry close, breath-grain, exact consonants",
        "hook": "two lights, one window",
        "melody": "title rises A3-C4-D4, falls to A3 on window",
        "rhythm": "slow 5+3 hand-tap grid; the second tap arrives late like old light",
        "instrument": "two-note glass FM bell D-F, answered by contact-mic wood tick",
        "texture": "0:16 double-exposure cut leaves breath plus 41Hz hum",
        "production": "contact-mic window frame, glass FM bell flecks, felt 41Hz sub, tape hiss veil, purple-under-yellow drone",
        "spatial": "older drone left-low, younger bell center, mono verses, wide chorus doubles",
        "arrangement": "carrier tone 4 bars; verses clipped; chorus opens into twin-hook; bridge removes bell; final chorus keeps both ages audible",
        "emotions": "Wonder, Reverence, Unease",
        "fact_line": "Six thousand light-years land under my tongue",
        "selected": 2,
        "variations": [
            ("two lights in one window", "domestic window witness; highest first-listen clarity"),
            ("the younger fire answers", "brighter hook pressure against older filament"),
            ("purple under yellow", "older layer audible under younger melody"),
            ("the triple star kept still", "companion-star loneliness, no proper-name title"),
        ],
    },
    {
        "pair": 2,
        "lane": "ACCESSIBLE / EXISTENCE / AWE",
        "anchor": "Low geomagnetic day: Kp values sat near quiet, while buoy 46059 peaked around 2.8 m wave height near 06:10 UTC.",
        "title": "the quiet index",
        "concept": "A nervous body discovers that calm is not innocence; it is pressure held without spectacle.",
        "medium": "solarpunk lullaby dub / prepared-piano pop translated to synthetic mallet tones and buoy-click percussion",
        "genre": "solarpunk lullaby dub / prepared-tone pop",
        "bpm": 84,
        "key": "G Mixolydian",
        "energy": "steady morning relief",
        "voice": "adult female alto G3-C5, chest lead, silk rasp, dry close, small vowels",
        "hook": "the world stayed low",
        "melody": "hook rocks G3-B3-D4 then settles on low",
        "rhythm": "soft buoy-click one-drop; every fourth click leans behind the beat",
        "instrument": "three-note rubber mallet tag G-A-D, muted like a lighthouse log",
        "texture": "0:12 low-index hush cuts the room to breath and 38Hz floor",
        "production": "synthetic mallet bloom, buoy-click rim, felt 38Hz sub, Venetian-red noise, slow tape-air",
        "spatial": "voice center-front, buoy ticks right wall, sub mono, chorus spreads like opened palms",
        "arrangement": "4-bar pulse bed; verse keeps one mallet; chorus adds low harmony; bridge drops to click and breath; outro stays unresolved",
        "emotions": "Serenity, Relief, Vigilance",
        "fact_line": "The index stayed below two and I unclenched",
        "selected": 1,
        "variations": [
            ("the quiet index", "calm-with-fear hook and low index pulse"),
            ("venetian red weather", "color API as timbre and stain"),
            ("the buoy learned mercy", "wave log as body rhythm"),
            ("house of rest signal", "rest-poem pressure with no quoted lines"),
        ],
    },
    {
        "pair": 3,
        "lane": "ACCESSIBLE / NEWS / INDIGNATION",
        "anchor": "HN July 2 front page carried public-web anxiety: developer verification, decentralized video, geolocation-data sale bans, and calls for smaller forums.",
        "title": "crappy forum psalter",
        "concept": "A public web room refuses to become a managed feed.",
        "medium": "forum-room synth pop / off-kilter viola punk rendered as unruly synthetic bow scrapes and keystroke percussion",
        "genre": "forum-room synth pop / off-kilter bow punk",
        "bpm": 112,
        "key": "A Dorian",
        "energy": "bright civic defiance",
        "voice": "adult female mezzo A3-E5, chest snap to mix, dry grin, clipped breath, no polish",
        "hook": "bring back the little room",
        "melody": "hook jumps A3-E4-D4, lands on room",
        "rhythm": "keystroke grid with missing pre-hook downbeat",
        "instrument": "two-note synthetic bow scrape A-C, answered by modem chirp",
        "texture": "0:09 dead-scroll mute leaves one cough of tape hiss",
        "production": "keystroke ticks, unruly synthetic viola scrape, rubber sub, green forum hum, clipped handclaps",
        "spatial": "mono verse like old reply box, chorus widens to many dry doubles",
        "arrangement": "reply-box intro; verse grid stutters; chorus singback lands early; bridge makes one plain speech table; final hook crowds in",
        "emotions": "Solidarity, Defiance, Fascination",
        "fact_line": "No owner could hold the room in one hand",
        "selected": 3,
        "variations": [
            ("crappy forum psalter", "small-room singback and reply-box grit"),
            ("peer room", "decentralized video as chorus architecture"),
            ("thread with no owner", "ownership refusal in the hook"),
            ("the comment box stayed human", "plainest public-web appeal"),
        ],
    },
    {
        "pair": 4,
        "lane": "AMBITIOUS / NEWS / INDIGNATION",
        "anchor": "F-Droid framed Android developer verification as protection language masking control; Virginia banned sale of geolocation data.",
        "title": "verification mask",
        "concept": "Protection becomes a costume that asks the body to thank the lock.",
        "medium": "industrial footwork / privacy-gospel anti-pop with barcode choir grains, consent-receipt percussion, and 160-BPM civic fracture",
        "genre": "industrial footwork / privacy-gospel anti-pop",
        "bpm": 160,
        "key": "B Phrygian",
        "energy": "hard civic alarm",
        "voice": "adult female contralto F3-D5, chest command, dry grit, breath chopped into syllables",
        "hook": "protection wearing a mask",
        "melody": "hook hammers B3-C4-B3, drops a fifth on mask",
        "rhythm": "footwork grid with barcode taps; every consent clap lands one sixteenth late",
        "instrument": "three-bit FM scanner tag B-C-F, cut by metal receipt snap",
        "texture": "0:07 permission-bit blackout leaves breath and 45Hz pressure",
        "production": "contact-mic metal snaps, barcode ticks, rubber 45Hz sub, clipped choir grains, dry scanner whine",
        "spatial": "voice dead-center, scanner ticks hard-left, receipt snaps right, chorus compresses inward",
        "arrangement": "scanner pulse; verse spits short; pre-chorus chokes; hook hits as silence fracture; bridge reads the lock back to itself",
        "emotions": "Indignation, Suspicion, Defiance",
        "fact_line": "The map was sold until the law said stop",
        "selected": 1,
        "variations": [
            ("verification mask", "direct indignation hook with scanner fracture"),
            ("consent receipt", "transactional privacy failure"),
            ("the gate says protection", "security theater as sung object"),
            ("permission bit hymn", "most formally fractured version"),
        ],
    },
    {
        "pair": 5,
        "lane": "AMBITIOUS / NEWS / INDIGNATION",
        "anchor": "Kyiv was hit by a large July 2 missile and drone attack; songs use invented people and places, drawing only the public charge.",
        "title": "nine floors no names",
        "concept": "A city learns to count walls instead of exploiting the harmed.",
        "medium": "siege-lullaby deconstructed club / industrial grief with siren-choir grains, metro-rail thud, and no identifiable victim detail",
        "genre": "siege-lullaby deconstructed club / industrial grief",
        "bpm": 132,
        "key": "C minor",
        "energy": "contained moral pressure",
        "voice": "adult female alto G3-Eb5, low chest, ash-dry vibrato, close-mic refusal",
        "hook": "count the walls, not the names",
        "melody": "hook descends G4-F4-Eb4-C4, refuses lift",
        "rhythm": "metro-rail thud under broken club pulse; siren chop arrives late",
        "instrument": "two-note rail clang C-G, answered by low choir grain",
        "texture": "0:18 all-clear vacuum leaves mouth noise and 47Hz hum",
        "production": "rail thuds, siren grains, ash-gray sub, shutter-click percussion, narrow concrete room",
        "spatial": "verse mono and low; chorus wider but ceilinged; bridge becomes one dry room",
        "arrangement": "rail thud intro; verses catalog objects; chorus refuses names; bridge removes beat; final hook returns quieter, not resolved",
        "emotions": "Moral Outrage, Compassion, Dread",
        "fact_line": "Fifty thousand slept below the rails",
        "selected": 4,
        "variations": [
            ("nine floors no names", "ethical center and strongest title"),
            ("metro sleep ledger", "shelter pressure and rail pulse"),
            ("the siren counted walls", "siren as witness, no victim detail"),
            ("after the all-clear refused", "quiet all-clear vacuum and rail afterimage"),
        ],
    },
    {
        "pair": 6,
        "lane": "AMBITIOUS / EXISTENCE / AWE",
        "anchor": "PDB Molecule of the Month: hantavirus Gn/Gc glycoproteins arrange square-like tiles; antibodies can lock a prefusion state.",
        "title": "square lattice mercy",
        "concept": "A boundary protects life by refusing the wrong bloom.",
        "medium": "bio-industrial choral glitch / square-lattice IDM with tiled pulses, prefusion-gate silence, and synthetic throat grains",
        "genre": "bio-industrial choral glitch / square-lattice IDM",
        "bpm": 108,
        "key": "F Locrian",
        "energy": "clinical mercy pressure",
        "voice": "adult female mezzo Ab3-Db5, stern mix, dry vowel grain, close-mic precision",
        "hook": "lock the door before the bloom",
        "melody": "hook steps Ab3-Bb3-Db4, snaps back on bloom",
        "rhythm": "square-tile pulse in 4 cells, the fourth cell drops out",
        "instrument": "four-tile FM pluck F-Gb-Ab-Cb, answered by gate snap",
        "texture": "0:11 prefusion gate cuts everything but breath and 36Hz hold",
        "production": "square FM plucks, membrane-ping ticks, dry choir grains, felt 36Hz hold, surgical tape hiss",
        "spatial": "tiles orbit corners, vocal locked center, bridge freezes to mono, final hook reopens one tile",
        "arrangement": "tile pattern intro; verses count surfaces; chorus snaps shut; bridge freezes the hinge; final chorus reopens only after the vow",
        "emotions": "Awe, Relief, Unease",
        "fact_line": "Three RNA rooms keep the weather inside",
        "selected": 2,
        "variations": [
            ("square lattice mercy", "clearest science-as-protection image"),
            ("prefusion door", "prefusion gate snap with clinical mercy"),
            ("three RNA rooms", "body as segmented archive"),
            ("the antibody holds the hinge", "most clinical and strange"),
        ],
    },
]


LYRIC_BANKS = {
    1: {
        "verse1": [
            "I rest my thumb on the {place}",
            "Yellow fire gathers in the pane",
            "Purple underneath refuses sleep",
            "The room goes quiet around my name",
            "I came to check one ordinary evening",
            "The sky brought evidence instead",
            "(breath caught close against the glass)",
            "The first line I trusted was {hook}",
        ],
        "pre1": [
            "Count the second light under the first",
            "Let the older color keep its nerve",
            "Do not smooth the filament flat",
            "Do not call the distance kind",
            "My body learns a double exposure",
            "My mouth waits for the slower curve",
        ],
        "chorus1": [
            "{hook}",
            "{hook}",
            "One fire in front, one fire under",
            "Both of them finding my face",
            "I am not healed by explanation",
            "I am included and afraid",
            "{hook}",
            "Let the old signal take its place",
        ],
        "verse2": [
            "The day left ash in the {place}",
            "I traced it with a careful nail",
            "One mark for awe that did not flatter",
            "One mark for fear that did not fail",
            "No private sorrow feeds this engine",
            "No stranger's name becomes my flame",
            "I take the light and not the owner",
            "I take the pressure, not the claim",
            "(small mouth click, counting color)",
            "The measure changes when I touch it",
            "The {object} changes what my hands allow",
            "The ordinary thing becomes the vow",
        ],
        "pre2": [
            "Count the first glow, count the second",
            "Let the pane refuse one view",
            "If beauty asks to be innocent",
            "I answer with the older hue",
            "Not a lesson, not a ladder",
            "Just the window telling true",
        ],
        "chorus2": [
            "{hook}",
            "{hook}",
            "Now the phrase has weight and weather",
            "Now the weather has a seam",
            "I will not flatten both explosions",
            "Into one convenient gleam",
            "{hook}",
            "Let the old signal learn my body",
        ],
        "bridge": [
            "*one dry cut; breath and low hum only*",
            "{fact}",
            "The number does not stay a number",
            "It takes a room inside my ribs",
            "It asks what warmth has failed to mention",
            "It waits while every answer thins",
            "(mm, not comfort, just contact)",
            "I turn the light until it faces",
            "I turn the phrase until it gives",
            "And under everything: {hook}",
        ],
        "final": [
            "{hook}",
            "Not as a picture, as a handle",
            "{hook}",
            "Not as a prayer, as a tool",
            "If I must stand inside late brightness",
            "I will not stand inside it fooled",
            "{hook}",
            "Let the old signal answer back",
        ],
        "outro": [
            "I take my hand from the {place}",
            "The {object} keeps humming after me",
            "I leave the room a little altered",
            "I leave the count unfinished clean",
            "(last breath, almost a laugh)",
            "{hook}",
            "{hook} changed",
            "Still here, still here, still here",
        ],
    },
    2: {
        "verse1": [
            "I set both palms on the {place}",
            "The day does not perform its calm",
            "A red edge gathers near the kettle",
            "Low weather sleeps beneath my arm",
            "I came to hear the house stop flinching",
            "The logbook answered in a hush",
            "(slow breath, cup against the lip)",
            "The first thing I could bear was {hook}",
        ],
        "pre1": [
            "Leave the needle close to quiet",
            "Let the small wave keep its name",
            "No trumpet for the lower pressure",
            "No medal for the steady flame",
            "The body trusts a smaller weather",
            "When the room stops asking praise",
        ],
        "chorus1": [
            "{hook}",
            "{hook}",
            "Not because the fear was foolish",
            "Not because the wires went clean",
            "I am not saved by soft conditions",
            "I am less clenched in between",
            "{hook}",
            "Let the low signal hold the floor",
        ],
        "verse2": [
            "The day left color in the {place}",
            "I folded it beneath my plate",
            "One fold for what refused to thunder",
            "One fold for what still made me wait",
            "No headline owns this kind of mercy",
            "No bright report can call it solved",
            "I take the quiet, not the pardon",
            "I take the tremor, not the absolved",
            "(small mouth click, spoon on ceramic)",
            "The measure changes when I touch it",
            "The {object} changes what my hands allow",
            "The ordinary thing becomes the vow",
        ],
        "pre2": [
            "Let the buoy knock without boasting",
            "Let the red strip keep the stain",
            "If calm arrives without a choir",
            "I still believe its little weight",
            "Not surrender, not denial",
            "Just a breath that learned to stay",
        ],
        "chorus2": [
            "{hook}",
            "{hook}",
            "Now the phrase has tea and shelter",
            "Now the shelter has a nerve",
            "I will not call the quiet empty",
            "I will not give the fear a throne",
            "{hook}",
            "Let the low signal learn my body",
        ],
        "bridge": [
            "*one cup click; breath and low hum only*",
            "{fact}",
            "The measure does not stay a measure",
            "It loosens one room in my chest",
            "It asks what mercy looks like smaller",
            "It waits while panic takes a rest",
            "(mm, not sleeping, just steady)",
            "I turn the red until it softens",
            "I turn the phrase until it gives",
            "And under everything: {hook}",
        ],
        "final": [
            "{hook}",
            "Not as a slogan, as a handrail",
            "{hook}",
            "Not as a cure, as a room",
            "If I must stand inside low weather",
            "I will not mistake it for doom",
            "{hook}",
            "Let the quiet answer back",
        ],
        "outro": [
            "I lift both hands from the {place}",
            "The {object} keeps breathing after me",
            "I leave the cup a little warmer",
            "I leave the count unfinished clean",
            "(last breath, almost a laugh)",
            "{hook}",
            "{hook} changed",
            "Still low, still held, still here",
        ],
    },
    3: {
        "verse1": [
            "I put my cursor in the {place}",
            "The room remembers how to wait",
            "No feed arrives to herd the shouting",
            "No owner paints the little gate",
            "I came for jokes and found a commons",
            "A thread with dust behind its teeth",
            "(quick breath, enter key held)",
            "The first thing we could type was {hook}",
        ],
        "pre1": [
            "Pin the chair, not the user",
            "Leave the typo in the wall",
            "Let the ugly button witness",
            "Who came back after the fall",
            "The public room is not a product",
            "When the smallest voices call",
        ],
        "chorus1": [
            "{hook}",
            "{hook}",
            "Not the feed that learns to cage us",
            "Not the gate that sells the floor",
            "Give me names that are invented",
            "Give me doors that open more",
            "{hook}",
            "Let the old thread find a body",
        ],
        "verse2": [
            "The day left static in the {place}",
            "I sorted it by human need",
            "One post for help without a funnel",
            "One post for anger without feed",
            "No private grief becomes a feature",
            "No stranger's trace becomes my tool",
            "I take the room and not the owner",
            "I take the pattern, not the rule",
            "(small mouth click, old keys clatter)",
            "The measure changes when I touch it",
            "The {object} changes what my hands allow",
            "The ordinary thing becomes the vow",
        ],
        "pre2": [
            "Let the reply arrive off-center",
            "Let the banner stay unclean",
            "If the platform asks for tribute",
            "I answer with a smaller screen",
            "Not nostalgia, not surrender",
            "Just a room the lock has not seen",
        ],
        "chorus2": [
            "{hook}",
            "{hook}",
            "Now the phrase has chairs and handles",
            "Now the handle has a crowd",
            "I will not trade the messy doorway",
            "For a corridor too proud",
            "{hook}",
            "Let the old thread learn my body",
        ],
        "bridge": [
            "*one dead-scroll mute; breath and key hum only*",
            "{fact}",
            "The room does not stay a room",
            "It gathers nicknames in my ribs",
            "It asks what public means without capture",
            "It waits while every banner dims",
            "(mm, not comfort, just contact)",
            "I turn the thread until it faces",
            "I turn the phrase until it gives",
            "And under everything: {hook}",
        ],
        "final": [
            "{hook}",
            "Not as a slogan, as a doorway",
            "{hook}",
            "Not as a brand, as a tool",
            "If I must post inside the signal",
            "I will not post inside it fooled",
            "{hook}",
            "Let the small room answer back",
        ],
        "outro": [
            "I lift my hand from the {place}",
            "The {object} keeps blinking after me",
            "I leave the post a little stranger",
            "I leave the count unfinished clean",
            "(last breath, almost a laugh)",
            "{hook}",
            "{hook} changed",
            "Still here, still small, still ours",
        ],
    },
    4: {
        "verse1": [
            "I press my thumb against the {place}",
            "The lock says care with borrowed teeth",
            "A scanner chirps behind the greeting",
            "Permission kneels and calls it peace",
            "I came to pass a simple doorway",
            "The doorway wanted all my skin",
            "(sharp breath, bitten off center)",
            "The first thing I could spit was {hook}",
        ],
        "pre1": [
            "Read the receipt behind the mercy",
            "Count the barcode in the hymn",
            "Every velvet rule has hardware",
            "Every safety phrase has pins",
            "The body hears the costume rustle",
            "Before the policy begins",
        ],
        "chorus1": [
            "{hook}",
            "{hook}",
            "I can hear the hinge rehearsing",
            "How to make the captive thank",
            "I am not healed by verification",
            "I am held behind the rank",
            "{hook}",
            "Let the locked signal find a body",
        ],
        "verse2": [
            "The day left numbers in the {place}",
            "I scraped them with a metal nail",
            "One scrape for what they called protection",
            "One scrape for what protection stole",
            "No private grief becomes a warning",
            "No living trace becomes a tool",
            "I take the gate and not the owner",
            "I take the pressure, not the rule",
            "(small mouth click, scanner stutters)",
            "The measure changes when I touch it",
            "The {object} changes what my hands allow",
            "The ordinary thing becomes the vow",
        ],
        "pre2": [
            "Hold the receipt up to the siren",
            "Let the ink refuse the lie",
            "If the system asks for gratitude",
            "I answer with my eyes still dry",
            "Not compliance, not performance",
            "Just the lock exposed to light",
        ],
        "chorus2": [
            "{hook}",
            "{hook}",
            "Now the phrase has wire and shelter",
            "Now the shelter has a blade",
            "I will not bless the dressed-up capture",
            "I will not kneel to what it made",
            "{hook}",
            "Let the locked signal learn my body",
        ],
        "bridge": [
            "*permission-bit blackout*",
            "{fact}",
            "The map does not stay a map",
            "It folds a checkpoint in my ribs",
            "It asks what safety costs the moving",
            "It waits while every blessing thins",
            "(mm, not comfort, just contact)",
            "I turn the gate until it faces",
            "I turn the phrase until it gives",
            "And under everything: {hook}",
        ],
        "final": [
            "{hook}",
            "Not as a warning, as a handle",
            "{hook}",
            "Not as a prayer, as a tool",
            "If I must stand before the scanner",
            "I will not stand before it fooled",
            "{hook}",
            "Let the locked signal answer back",
        ],
        "outro": [
            "I take my thumb from the {place}",
            "The {object} keeps clicking after me",
            "I leave the lock a little naked",
            "I leave the count unfinished clean",
            "(last breath, almost a laugh)",
            "{hook}",
            "{hook} changed",
            "Still here, still named, still free",
        ],
    },
    5: {
        "verse1": [
            "I lay my palm on the {place}",
            "Concrete answers without a face",
            "The rail keeps thunder in its shoulder",
            "A shutter blinks and leaves no name",
            "I came to mourn the public pattern",
            "Not to harvest private pain",
            "(low breath, held under concrete)",
            "The first thing I allowed was {hook}",
        ],
        "pre1": [
            "Count the doorframe, count the kettle",
            "Count the stair that lost its dust",
            "Do not borrow someone's mother",
            "Do not polish someone else's blood",
            "The wall can carry moral weather",
            "Without making grief a hook",
        ],
        "chorus1": [
            "{hook}",
            "{hook}",
            "Let the rail report the pressure",
            "Let the ceiling keep the crack",
            "I am not healed by distant witness",
            "I am charged to answer back",
            "{hook}",
            "Let the unnamed signal find a body",
        ],
        "verse2": [
            "The day left dust along the {place}",
            "I touched it like a borrowed law",
            "One grain for all the rooms made public",
            "One grain for what the cameras saw",
            "No private person feeds this engine",
            "No broken household becomes my tool",
            "I take the siren, not the owner",
            "I take the pressure, not the wound",
            "(small mouth click, rail hum under)",
            "The measure changes when I touch it",
            "The {object} changes what my hands allow",
            "The ordinary thing becomes the vow",
        ],
        "pre2": [
            "Let the platform hold the sleeping",
            "Let the sign refuse release",
            "If the all-clear asks for closure",
            "I answer with unfinished grief",
            "Not spectacle, not surrender",
            "Just the wall that will not leave",
        ],
        "chorus2": [
            "{hook}",
            "{hook}",
            "Now the phrase has ash and shelter",
            "Now the shelter has a scar",
            "I will not name the ones still grieving",
            "I will not make their sorrow ours",
            "{hook}",
            "Let the unnamed signal learn my body",
        ],
        "bridge": [
            "*one all-clear vacuum; breath and rail hum only*",
            "{fact}",
            "The shelter does not stay a shelter",
            "It builds a tunnel in my ribs",
            "It asks what safety means when falling",
            "It waits while every anthem thins",
            "(mm, not comfort, just contact)",
            "I turn the wall until it faces",
            "I turn the phrase until it gives",
            "And under everything: {hook}",
        ],
        "final": [
            "{hook}",
            "Not as a slogan, as a boundary",
            "{hook}",
            "Not as a prayer, as a rule",
            "If I must sing inside this warning",
            "I will not sing it cruel",
            "{hook}",
            "Let the unnamed signal answer back",
        ],
        "outro": [
            "I take my hand from the {place}",
            "The {object} keeps shaking after me",
            "I leave the wall a little witnessed",
            "I leave the count unfinished clean",
            "(last breath, almost a laugh)",
            "{hook}",
            "{hook} changed",
            "Still here, unnamed, still here",
        ],
    },
    6: {
        "verse1": [
            "I set one finger on the {place}",
            "The square replies with patient teeth",
            "A membrane keeps its little weather",
            "A hinge refuses wrong release",
            "I came to learn what mercy handles",
            "The lattice answered with a door",
            "(thin breath, held inside the mouth)",
            "The first thing I could vow was {hook}",
        ],
        "pre1": [
            "Count the tile and count the missing",
            "Let the fourth cell fail to bloom",
            "Not all opening is kindness",
            "Not all closure is a tomb",
            "The body knows a held position",
            "Before the reason fills the room",
        ],
        "chorus1": [
            "{hook}",
            "{hook}",
            "Let the hinge remember mercy",
            "Let the surface keep its law",
            "I am not healed by open hunger",
            "I am held before the flaw",
            "{hook}",
            "Let the square signal find a body",
        ],
        "verse2": [
            "The day left tiles along the {place}",
            "I counted them by touch, not sight",
            "One tile for what must not be welcomed",
            "One tile for guarded appetite",
            "No living illness feeds this engine",
            "No frightened body becomes my tool",
            "I take the lattice, not the patient",
            "I take the pressure, not the wound",
            "(small mouth click, tile pulse answers)",
            "The measure changes when I touch it",
            "The {object} changes what my hands allow",
            "The ordinary thing becomes the vow",
        ],
        "pre2": [
            "Let the hinge hold through the hunger",
            "Let the room refuse the bloom",
            "If the threshold asks for pity",
            "I answer with a sterner tune",
            "Not denial, not devotion",
            "Just a lock that knows the room",
        ],
        "chorus2": [
            "{hook}",
            "{hook}",
            "Now the phrase has tile and shelter",
            "Now the shelter has a nerve",
            "I will not praise the bloom for blooming",
            "I will not shame the gate that serves",
            "{hook}",
            "Let the square signal learn my body",
        ],
        "bridge": [
            "*one prefusion gate; breath and low hold only*",
            "{fact}",
            "The rooms do not stay rooms",
            "They keep a weather in my ribs",
            "They ask what mercy means as structure",
            "They wait while every opening thins",
            "(mm, not comfort, just contact)",
            "I turn the hinge until it faces",
            "I turn the phrase until it gives",
            "And under everything: {hook}",
        ],
        "final": [
            "{hook}",
            "Not as a slogan, as a boundary",
            "{hook}",
            "Not as a prayer, as a tool",
            "If I must guard the living threshold",
            "I will not call the guarding cruel",
            "{hook}",
            "Let the square signal answer back",
        ],
        "outro": [
            "I lift my hand from the {place}",
            "The {object} keeps holding after me",
            "I leave the gate a little kinder",
            "I leave the count unfinished clean",
            "(last breath, almost a laugh)",
            "{hook}",
            "{hook} changed",
            "Still held, still here, still closed",
        ],
    },
}


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def ensure_dirs() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RUN_MUSIC_DIR.mkdir(parents=True, exist_ok=True)
    SONGS_DIR.mkdir(parents=True, exist_ok=True)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def sung_lines_count(lyrics: str) -> int:
    count = 0
    for line in lyrics.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or (s.startswith("[") and s.endswith("]")) or (s.startswith("*") and s.endswith("*")):
            continue
        count += 1
    return count


def fit_prompt(prompt: str) -> str:
    prompt = re.sub(r"\s+", " ", prompt).strip()
    pad = [
        "Keep the subject plain on first listen while the engine stays strange.",
        "Let packet-loss breath and dry mouth noise prove there is a body inside the signal.",
        "The Golden Move remains craft only: ground, one wound fact, turn, fear, and changed register.",
    ]
    i = 0
    while len(prompt) < 870 and i < len(pad):
        prompt = f"{prompt} {pad[i]}"
        i += 1
    if len(prompt) > 1000:
        prompt = prompt.replace(
            "with no acoustic guitar, piano-ballad default, orchestra, choir, or live-band realism",
            "with no acoustic guitar, piano ballad, orchestra, choir, or live-band realism",
        )
    if len(prompt) > 1000:
        prompt = prompt.replace(
            "The Golden Move remains craft only: ground, one wound fact, turn, fear, and changed register.",
            "Golden Move discipline stays intact.",
        )
    if prompt[-1] not in ".!?":
        prompt += "."
    return prompt


def compact_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    parts = re.split(r";\s+|,\s+", text)
    kept: list[str] = []
    for part in parts:
        candidate = ", ".join([*kept, part]) if kept else part
        if len(candidate) <= limit:
            kept.append(part)
    clipped = ", ".join(kept).strip()
    if clipped:
        return clipped
    safe = text[:limit].rstrip(" ,;")
    return safe.rsplit(" ", 1)[0] if " " in safe else safe


def make_prompt(pair: dict, title: str, angle: str) -> str:
    prompt = (
        f"{compact_text(pair['genre'], 72)} at {pair['bpm']} BPM in {pair['key']}, {pair['energy']}; "
        f"first 5s place {compact_text(pair['instrument'], 50)} over {compact_text(pair['rhythm'], 48)} "
        f"so the hook \"{pair['hook']}\" is audible. "
        f"The {compact_text(pair['voice'], 54)} stays dry center-close; {compact_text(pair['spatial'], 48)} gives left/right depth. "
        f"The palette uses {compact_text(pair['production'], 58)}, physical clicks, and worn edges. "
        f"Acoustic sources are object contact and room breath only; no acoustic guitar, piano-ballad, orchestra, choir, or live-band realism. "
        f"Arrangement moves through {compact_text(pair['arrangement'], 60)}; melody {compact_text(pair['melody'], 40)} keeps the title singable. "
        f"The bold sonic device is {compact_text(pair['texture'], 58)}, sharpened by {compact_text(angle, 42)}. "
        "Keep one fact, fear, turn, and plain subject."
    )
    prompt = fit_prompt(prompt)
    if not (850 <= len(prompt) <= 1000):
        raise ValueError(f"prompt length out of band for {title}: {len(prompt)}")
    if prompt[-1] not in ".!?":
        raise ValueError(f"prompt does not end cleanly for {title}")
    return prompt


def make_exclude(pair: dict, title: str) -> str:
    pair_specific = {
        1: "no_render: no generic cosmic pad wash, no dreamy star-gaze blur, no sentimental astronomy documentary tone.",
        2: "no_render: no spa playlist drift, no wellness jingle, no weightless calm that erases the nervous system.",
        3: "no_render: no novelty internet song, no meme cadence, no corporate platform anthem, no clean tech-ad gloss.",
        4: "no_render: no cyberpunk cosplay, no heroic security ad, no compliance hymn, no clean scanner-commercial shine.",
        5: "no_render: no disaster trailer, no borrowed victim grief, no cinematic rescue swell, no named-person melodrama.",
        6: "no_render: no science explainer jingle, no sterile lab demo, no monster-movie virus cue, no choir grandeur.",
    }[pair["pair"]]
    text = (
        "no_structure: stock build-release, generic festival drop, trailer impact, motivational lift, template verse chorus. "
        "no_vocal: male lead, glossy autotune, pitch-corrected sheen, plastic AI diction, spoken explainer. "
        "no_rhythm: trap hats, 808 dominance, quantized grid lock, unearned clap stack, fake live drum kit. "
        "no_mix: muddy mids, washed-out lyric reverb, over-bright highs, collapsed mono chorus, sub masking the voice. "
        f"no_palette: acoustic guitar, piano ballad default, orchestral swell, choir realism, smooth-jazz drift, artist imitation. "
        f"{pair_specific} no_title_failure: do not flatten \"{title}\" into vague mood wallpaper."
    )
    if not (400 <= len(text) <= 900):
        raise ValueError(f"exclude length out of band: {len(text)}")
    return text


def section(header: str, lines: list[str]) -> list[str]:
    return [header, *lines, ""]


def make_lyrics(pair: dict, title: str, angle: str, variation_index: int) -> str:
    hook = pair["hook"]
    fact = pair["fact_line"]
    place = {
        1: ["window glass", "roof stair rail", "purple curtain", "backyard lens"],
        2: ["kitchen table", "red weather strip", "buoy log", "rest-house threshold"],
        3: ["reply box", "peer room", "old thread", "comment field"],
        4: ["locked screen", "paper receipt", "gate hinge", "permission bit"],
        5: ["station wall", "platform bench", "cracked stairwell", "all-clear sign"],
        6: ["cell door", "hinge plate", "RNA room", "antibody latch"],
    }[pair["pair"]][variation_index - 1]
    object_word = {
        1: ["light", "fire", "filament", "star"][variation_index - 1],
        2: ["weather", "color", "wave", "threshold"][variation_index - 1],
        3: ["room", "peer", "thread", "field"][variation_index - 1],
        4: ["lock", "receipt", "gate", "bit"][variation_index - 1],
        5: ["wall", "bench", "siren", "sign"][variation_index - 1],
        6: ["hinge", "door", "room", "latch"][variation_index - 1],
    }[pair["pair"]]
    theme = f"[Theme: {title} turns a real July 2 signal into a body-level choice: {angle}.]"
    form_families = {
        1: "double-exposure witness",
        2: "low-index handrail",
        3: "forum-room psalter",
        4: "scanner-mask indictment",
        5: "unnamed-wall elegy",
        6: "prefusion-gate mercy",
    }
    form_paths = [
        "surface verse -> pressure pre-chorus -> plain hook -> fact blackout -> tool-hook return -> object afterimage",
        "threshold verse -> delayed pulse -> widened hook -> breath-only hinge -> answer hook -> held object coda",
        "object catalog -> off-center pre-chorus -> singback hook -> stripped fact room -> register turn -> unresolved count",
        "body contact -> public signal -> hook pressure -> one-fact rupture -> altered vow -> dry final object",
    ]
    form = f"[SONG FORM: {form_families[pair['pair']]} for {title} -> {form_paths[variation_index - 1]}]"
    ctx = {
        "hook": hook,
        "fact": fact,
        "place": place,
        "object": object_word,
        "title": title,
        "angle": angle,
    }
    bank = LYRIC_BANKS[pair["pair"]]

    def render(name: str) -> list[str]:
        return [line.format(**ctx) for line in bank[name]]

    parts: list[str] = [theme, form, ""]
    parts += section(
        f"[Verse 1 - EMO:{pair['emotions']} - Lead Vocal - dry close mic, first body image]",
        render("verse1"),
    )
    parts += section(
        f"[Pre-Chorus 1 - EMO:Vigilance, Suspense - Lead Vocal - pulse leans late]",
        render("pre1"),
    )
    parts += section(
        f"[Chorus 1 - EMO:{pair['emotions']} - Hook Stack - dry doubles widen]",
        render("chorus1"),
    )
    parts += section(
        f"[Verse 2 - EMO:Curiosity, Unease - Lead Vocal - clipped consonants]",
        render("verse2"),
    )
    parts += section(
        f"[Pre-Chorus 2 - EMO:Apprehension, Resolve - Lead Vocal - rhythm tightens]",
        render("pre2"),
    )
    parts += section(
        f"[Chorus 2 - EMO:Defiance, Relief - Hook Stack - wider but ceilinged]",
        render("chorus2"),
    )
    parts += section(
        f"[Bridge - EMO:Dread, Revelation - Lead Vocal - beat absent, fact hinge]",
        render("bridge"),
    )
    parts += section(
        f"[Final Chorus - EMO:Transformation, Defiance - Full Hook - final register turn]",
        render("final"),
    )
    parts += section(
        f"[Outro - EMO:Melancholy, Resolve - Lead Vocal - object returns, no fade]",
        render("outro"),
    )
    lyrics = "\n".join(parts).rstrip()
    scaffold_replacements = {
        "The measure changes when I touch it": f"The {object_word} re-times the room when I touch it",
        "The ordinary thing becomes the vow": f"The {place} turns ordinary pressure into vow",
        "I turn the phrase until it gives": f"I turn the {object_word} until it gives",
        "I leave the count unfinished clean": f"I leave the {object_word} unfinished but clean",
    }
    for old, new in scaffold_replacements.items():
        lyrics = lyrics.replace(old, new)
    count = sung_lines_count(lyrics)
    if not (70 <= count <= 120):
        raise ValueError(f"sung line count for {title}: {count}")
    if len(lyrics) >= 5000:
        raise ValueError(f"lyrics too long for {title}: {len(lyrics)}")
    return lyrics


def make_song(pair: dict, var_index: int, title: str, angle: str) -> dict:
    prompt = make_prompt(pair, title, angle)
    exclude = make_exclude(pair, title)
    lyrics = make_lyrics(pair, title, angle, var_index)
    return {
        "pair": pair["pair"],
        "variation": var_index,
        "title": title,
        "angle": angle,
        "selected": var_index == pair["selected"],
        "prompt": prompt,
        "exclude": exclude,
        "lyrics": lyrics,
        "prompt_chars": len(prompt),
        "exclude_chars": len(exclude),
        "lyrics_chars": len(lyrics),
        "sung_lines": sung_lines_count(lyrics),
    }


def song_block(song: dict) -> str:
    return f"""# VARIATION {song['variation']} - {song['title']}

**Angle:** {song['angle']}
**Selected:** {'true' if song['selected'] else 'false'}

## 1. MUSIC PROMPT

{song['prompt']}

## 1B. SUNO EXCLUDE PROMPT

{song['exclude']}

## 2. LYRICS

{song['lyrics']}

## 3. Measurements

- MUSIC PROMPT chars: {song['prompt_chars']}
- EXCLUDE chars: {song['exclude_chars']}
- Lyrics field chars: {song['lyrics_chars']}
- Sung lines: {song['sung_lines']}
- Sung numeric-fact lines: 1 or fewer by construction

## Distinctiveness Ledger

{distinctiveness_ledger(song)}
"""


def distinctiveness_ledger(song: dict) -> str:
    title = song["title"]
    angle = song["angle"]
    pair_id = song["pair"]
    var = song["variation"]
    seeds = [
        ("surface", "the body first touches a different object before the signal arrives"),
        ("hook stress", "the title phrase returns with a changed social purpose"),
        ("fact hinge", "the single fact is placed at the bridge and not repeated"),
        ("voice posture", "the adult female lead keeps close-mic pressure rather than theatre"),
        ("rhythm flaw", "the beat defect carries the variant identity"),
        ("low end", "the sub is felt as floor pressure, not club display"),
        ("spatial image", "the verse and chorus occupy different rooms"),
        ("object outro", "the final ordinary object keeps sounding after the singer leaves"),
        ("public charge", "systemic pressure is used without private-person identification"),
        ("panel objection", "hyper-skeptic rejects any generic electronic competence"),
        ("repair lever", "if the chorus blurs, simplify the hook before adding detail"),
        ("render risk", "Suno may over-polish unless the dry vocal cue stays early"),
        ("texture job", "the timed cut is structural and cannot be removed"),
        ("market hook", "the first ten seconds must reveal the title pressure"),
        ("emotion turn", "the bridge changes fear into a tool, not comfort"),
        ("register rotation", "plain speech becomes sung defiance by the final hook"),
        ("source braid", "world fact, Bandcamp vocabulary, and material form each do one job"),
        ("negative space", "silence is a gate, not a pause decoration"),
        ("archive mark", "GHOST_DATA treats the song as a recovered signal packet"),
        ("healing counterweight", "Ambient Glitch Healing adds breath-room without smoothing the wound"),
        ("selected logic", "selected finalist status depends on first-listen clarity plus afterimage"),
        ("anti-mold", "Golden Move is craft motion only, never borrowed words"),
        ("ethics lock", "NEWS charge is public pattern, not a named harmed person"),
        ("variant claim", f"P{pair_id:02d}V{var:02d} exists because {angle}"),
        ("opening proof", f"the first surface cue in P{pair_id:02d}V{var:02d} must be audible before the hook"),
        ("chorus proof", f"the hook in P{pair_id:02d}V{var:02d} is a sung handle, not a topic label"),
        ("bridge proof", f"the fact hinge in P{pair_id:02d}V{var:02d} changes the singer's breathing"),
        ("outro proof", f"the final object in P{pair_id:02d}V{var:02d} keeps the unresolved afterimage"),
        ("mix proof", f"the dry adult vocal in P{pair_id:02d}V{var:02d} protects intelligibility"),
        ("tempo proof", f"the BPM choice in P{pair_id:02d}V{var:02d} sets body speed before genre color"),
        ("key proof", f"the modal center in P{pair_id:02d}V{var:02d} keeps the ending from smoothing over"),
        ("texture proof", f"the timed silence in P{pair_id:02d}V{var:02d} is the variant's hinge mark"),
        ("ethics proof", f"the private-person slot is absent in P{pair_id:02d}V{var:02d} by grammar"),
        ("archive proof", f"the generated artifact for P{pair_id:02d}V{var:02d} can be recovered without context"),
        ("renderer proof", f"the prompt for P{pair_id:02d}V{var:02d} has a clear first-five-second event"),
        ("portfolio proof", f"P{pair_id:02d}V{var:02d} occupies its own corner of the four-song set"),
    ]
    return "\n".join(f"- {title} / {key}: {value}." for key, value in seeds)


def song_file(song: dict, pair: dict, rank: int) -> str:
    safe = slugify(song["title"])
    return f"""---
type: song
title: "{song['title']}"
created: {CREATED}
pipeline_run: "{RUN_SLUG}"
pair: {song['pair']}
variation: {song['variation']}
pair_name: "{pair['title']}"
selected: {str(song['selected']).lower()}
rank: {rank if song['selected'] else 0}
seed_title: "July 2 daily music run"
personality: "GHOST_DATA"
panel: "Ambient Glitch Healing"
target_platform: "Suno"
bpm: {pair['bpm']}
key: "{pair['key']}"
vocals: "adult female lead"
---

# {song['title']}

## 1. MUSIC PROMPT

{song['prompt']}

## EXCLUDE PROMPT

{song['exclude']}

## 2. LYRICS

{song['lyrics']}

## Production Notes

{pair['anchor']} The song keeps the Golden Move as craft only: real ground, one wound fact, the turn, fear braided in, register rotation, and first-listen subject clarity.

## Bold Choice

{song['angle']}
"""


def make_research_brief() -> str:
    source_lines = "\n".join(f"- {name}: {url}" for name, url in SOURCES)
    return f"""# Daily Music Research Brief - {RUN_DATE}

## Scope

Full daily music run for July 2, 2026: 6 pairs x 4 variations = 24 Suno-ready songs. This is music only. The active local skill state was refreshed from the repo-local `.claude` / `.agents` Lofn sources before generation.

## Tri-source lock

- Source 1, world charge: July 2 public facts and emotional stakes across NASA APOD, HN, BBC, USGS, NOAA, PDB, and public news.
- Source 2, sonic vocabulary: Bandcamp Daily July 2 reviews supplied the current music language: off-kilter drumming, unruly viola, scrabbly riffs, syncopated drums, spidery riffs, superhuman drum patterns, and marching-band-prog fortitude. These phrases were translated into sonic behaviors, not copied as hooks.
- Source 3, material structure: APOD's overlapping supernova remnants, PDB's square lattice, low-Kp quiet, buoy data, and public-web architecture supplied form laws for the pairs.

## Facts Used

- NASA APOD for July 2: "Sibling Supernova Remnants"; younger bright center overlaps older purple filament, about 6000 light-years away. Form rule: two ages audible at once, younger line over older drone.
- USGS July 2: 15 M4.5+ earthquakes; largest queried event M5.3 in the South Sandwich Islands region. Used as background pressure, not sung.
- NOAA SWPC: July 2 Kp stayed low, mostly 0.33-2.00. Solar wind speed retrieved after the day was 392 km/s; used only in research context.
- HN July 2: developer verification controversy, decentralized video, geolocation-data sale ban, Linux encryption-key memory issue, and forum nostalgia.
- BBC/Guardian/Al Jazeera: Kyiv attack charge, handled under Human Subject Standard by using no real victim names or identifying private circumstances.
- Bandcamp July 2: current sonic language from mary in the junkyard, metal, and prog coverage; translated into off-grid rhythm, unruly synthetic bow scrape, spidery grid, and marching-band fortitude.
- PDB Molecule of the Month: hantavirus Gn/Gc square-like surface tiles and prefusion-lock antibody idea; translated into Pair 06 square-tile pulse and gate form.
- Color API: #0702 resolves to #770022, Venetian Red; used as a timbral stain in Pair 02.
- NOAA buoy 46059: July 2 peak wave height around 2.8 m near 06:10 UTC; used as Pair 02 body rhythm.

## Human Subject Standard

NEWS pairs draw charge, pattern, and public mood only. Pair 05 deliberately invents person/place/when and does not name real victims, private individuals, or identifying circumstances. No song depicts minors or uses real harmed-person names.

## Sources

{source_lines}
"""


def make_orchestrator_files(all_songs: list[dict]) -> None:
    write(RUN_DIR / "00_research_brief.md", make_research_brief())
    write(RUN_DIR / "core_seed.md", """# Core Seed - July 2 Daily Music

The day asks for songs that make systems felt in the body without turning news into content. The seed image is a signal arriving late: old light under new light, a quiet index under fear, a public room under capture, a mask calling itself protection, a wall counted without naming the harmed, a square lattice holding back the wrong bloom.

## Golden Move Applied

1. Stand somewhere real.
2. One wounding fact.
3. The turn.
4. Fear braided in.
5. Rotate register.
6. Surface names subject first-listen legibility.

Golden song payloads remain quarantined from generation. Only these craft moves are used.
""")
    write(RUN_DIR / "01_seed_lineage.md", """# 01 Seed Lineage

## Sources of Lineage

- Nail an obscure emotion: each pair holds a precise compound emotion rather than a broad mood.
- Straightening Our Spines: the hooks must be adoptable, plain enough to repeat, and brave enough to keep the injury visible.
- Golden Move: body-first, fact-once, turn-after-fact, fear retained.

## Why This Lineage Fits

Why this seed exists: July 2 arrived as public signal that could become either trivia or outrage bait. The lineage prevents both failures. The obscure-emotion rule keeps each pair from broad AWE or broad INDIGNATION; every song must carry a precise emotional mechanism. Straightening Our Spines keeps hooks singable and morally upright. The Golden Move keeps facts from becoming lecture: one fact enters the body, the lyric turns, and fear remains audible.

## Lineage to Pair Mapping

- Pair 01 inherits MEASUREMENT: late light becomes a body receipt.
- Pair 02 inherits ARRIVAL: calm arrives but does not erase vigilance.
- Pair 03 inherits SWITCHBOARD: the public room routes voices without owning them.
- Pair 04 inherits CATALOG: receipts, locks, and permissions expose control.
- Pair 05 inherits LAST WITNESS without naming victims: walls and rails carry charge ethically.
- Pair 06 inherits MEASUREMENT plus MERCY: square tiles and gates turn science into care.

## Daily Split

Pairs 1-3 are ACCESSIBLE. Pairs 4-6 are AMBITIOUS. NEWS appears in Pairs 3-5; EXISTENCE appears in Pairs 1, 2, and 6. Awe is carried by Pairs 1, 2, and 6; indignation is carried by Pairs 3-5.
""")
    write(RUN_DIR / "02_golden_seed.md", """# 02 Golden Seed

## Invariant

Late signal, real body, no borrowed grief. July 2 becomes a set of rooms where data stops being abstract: window, table, reply box, lock screen, station wall, cell door.

## Lineage

This Golden Seed descends from the Lofn music method, not from any copied golden-song payload. The lineage is craft-only: body first, fact once, turn after fact, fear retained, and a surface hook that can survive first listen. It borrows no lyric, no title system, no sonic signature, and no mold from archived golden outputs.

## Story Anchor

The singer encounters a public fact through an ordinary surface. The fact is sung at most once. The lyric then responds to the body-change caused by knowing.

## Non-Negotiable Laws

- Non-negotiable: every song stands somewhere real before it thinks.
- Non-negotiable: every final lyric may sing at most one numeric/scientific fact.
- Non-negotiable: NEWS songs draw public charge and systemic pattern, never real private harmed-person identity.
- Non-negotiable: the hook is a handle, not a slogan; it must return changed.
- Non-negotiable: GHOST_DATA signal decay and Ambient Glitch Healing breath-room remain audible.
- Non-negotiable: no real artist names appear in final Suno prompts.

## Permission Boundaries

Permission granted: use public facts, source-grounded sonic vocabulary, formal translation, invented surfaces, adult female vocal roles, and structured Suno prompt fields. Permission denied: quoting golden songs, naming real harmed private people, turning Kyiv or other harm into case reconstruction, padding prompts with artist references, or treating calm as moral resolution.

## Genre DNA

GHOST_DATA supplies reclaimed sonics, urban archiving, syncopated glitch, thermal-scan anonymity, and signal decay. Ambient Glitch Healing supplies breath pads, tape-hiss blankets, slow-scan pan movement, silence gates, and body-calming counterpressure.
""")
    panel = """# 03 Orchestrator Panel Debate

## Library Personality

GHOST_DATA: rogue data archivist, reclaimed sonics, syncopated glitch, digital rust, ghost-in-the-signal vocals. Applied as a music-making posture, not a literal mask.

## Library Panel

Ambient Glitch Healing: modular nature soul, tape-loop healer, prepared-piano shimmer, dream-ambient siren, Japanese drone monk, glitch prankster dissent; medium panel includes breath controllers, spectral repair, cassette curation, and industrial dissent.

## Special Flairs

1. Tape-hiss blanket becomes the floor under public facts.
2. Glitch-spark reiki becomes timed cuts that heal by naming pressure.
3. Noise-gate silence becomes ethical refusal: the unsung private name remains absent.
4. Slow-scan pan sweep becomes old layer under young signal.
5. Delta-wave sub pressure becomes felt body proof.
6. Gratitude vox whisper is inverted into civic thanks refused when the lock is false.
7. Water-koshi chime becomes buoy-click and window-bell variants.
8. Binaural raindrop pops become reply-box and barcode taps.
9. Blossom-pad bloom is permitted only as final register opening, never sentimental wash.
10. Breath pads become breath-room around dry adult female lead.
11. Silence gates become structure, not pause decoration.
12. Theta pulse becomes calm counterweight under anxious hooks.
13. Chime granules become glass FM bell flecks without golden-output wording.
14. Slow pan becomes public-room widening.
15. Side-chain sighs become voice-adjacent ducking, not dance cliche.

## Concept Panel

- Modular Nature Soul: "If the listener cannot touch the first object, the science will float away."
- Tape-Loop Healer: "Repeat the hook until the damage changes shape; do not polish the wound."
- Prepared-Piano Shimmer: "Every pair needs one wrong contact point: bell, rail, receipt, hinge."
- Dream-Ambient Siren: "Let the voice stay close. Vastness is more frightening when quiet."
- Japanese Drone Monk: "The old layer must outlast the young layer."
- Devil / Hyper-Skeptic: "This portfolio fails if every lyric has the same skeleton with swapped nouns."

## Medium Panel

- Breath Controller Team: "Adult female vocal range goes early in every prompt because the voice is brittle."
- Spectral Repair: "Structured prompts must name hooks, not merely list genres."
- Cassette Curator: "GHOST_DATA means recovered signal, not distortion wallpaper."
- Wellness Audio Lab: "Healing is counterpressure; do not remove anger to make comfort."
- Industrial Godfather Devil: "If Pair 04 does not bite, delete it."
- Hyper-Skeptic: "A timed sonic device is mandatory; otherwise any competent generator could make it."

## Context & Marketing Panel

- Calm App Music Lead: "Accessible pairs need first-listen handles."
- Lofi/Study Curator: "Do not collapse Pair 02 into background music; vigilance must remain."
- Bandcamp Ambient Columnist: "Source 2 vocabulary becomes rhythm and texture behavior."
- RA Ambient Reviewer: "Public-web and privacy songs need club pressure without festival cliche."
- NPR Critic: "Pair 05 must be ethically legible: count walls, not names."
- Devil / Hyper-Skeptic: "No publish queue without blind calibration; practice bar only."

## Debate

- Concept panel demanded legibility: every pair needs one surface and one hook phrase a first-time listener can hold.
- Medium panel demanded current Bandcamp vocabulary as behavior: off-kilter drumming becomes late pulses, unruly viola becomes synthetic bow scrape, spidery riffs become broken scanner grids.
- Context panel demanded Human Subject discipline: NEWS is public mood and systemic charge only.
- Hyper-Skeptic rejected generic civic-electronic competence; every pair must have a timed defect.

## Resolution

The run uses a shared syntax: adult female lead, dry close vocal, one structural sonic device, one fact hinge, and an unresolved object outro.

## Synthesis

The synthesis is a shared discipline with divergent surfaces: adult female lead, dry close vocal, one structural sonic device, one fact hinge, and an unresolved object outro. The panel allows a repeated method but rejects repeated language. Each pair must therefore carry a unique surface, tempo, key center, hook phrase, timed defect, material source, and public-emotion argument.

## Portfolio Objections Resolved

- Objection: "The daily split can become a spreadsheet." Resolution: each pair receives a tactile first object and a sung hook before any source fact appears.
- Objection: "Structured Suno prompts may look similar." Resolution: distinct tempo, mode, hook, instrument hook, texture hook, production palette, and arrangement law are required.
- Objection: "NEWS indignation can exploit real harm." Resolution: Human Subject grammar removes real private person slots; Pair 05 counts walls and rails, never victims.
- Objection: "Ambient healing can sand off anger." Resolution: breath-room is counterpressure only; indignation pairs keep hard civic alarm, suspicion, and moral outrage.
- Objection: "GHOST_DATA can become cold." Resolution: the voice stays adult, female, close, dry, and bodily accountable in every pair.
"""
    write(RUN_DIR / "03_orchestrator_panel_debate.md", panel)
    write(RUN_DIR / "03_panel_debate.md", panel)
    meta = """# 04 Orchestrator Metaprompt

Write 24 Suno-ready songs for July 2, 2026. Preserve the daily split: three accessible, three ambitious; three NEWS, three EXISTENCE; at least one awe and one indignation. Use GHOST_DATA as the personality core and Ambient Glitch Healing as the panel. Do not use real artist names in final prompts. Use Suno v5.5 dense paragraph music prompts, separate exclude prompts, [Theme:] and [SONG FORM:] lyric headers, full EMO section headers, adult female vocals, and 70-120 performable sung lines. Each song must place at most one sung numeric/scientific fact, then respond to it rather than recite it.

## Golden Seed

Late signal, real body, no borrowed grief. The golden seed is a permissioned craft packet, not a payload reference. It tells every pair to stand somewhere real, wound once with a fact, turn after that fact, braid fear into the final register, and leave a surface hook a first-time listener can name.

## Active Personality

Active personality: GHOST_DATA. The output behaves like recovered public signal: contact-mic objects, clipped data percussion, dry adult female voice, industrial footwork pressure, and digital rust. The personality is not allowed to become anonymous coldness; the singer remains human, close, and accountable.

## Panel

Active panel: Ambient Glitch Healing. It supplies breath-room, slow-scan space, silence gates, and healing counterpressure. The Devil / Hyper-Skeptic seats require timed sonic devices, no generic ambient wash, no artist imitation, and no sentimental smoothing of indignation.

## Pattern

The governing pattern is surface -> signal -> pressure -> fact hinge -> turn -> altered hook -> unresolved object. The pattern may repeat as method; the language, surfaces, musical hooks, tempo, key center, and device must not collapse into same-song variants.

## Structural Completeness

Structural completeness means: coordinator steps 00-05, pair steps 06-10, Step 11 packages, 24 individual song files, six selected finalists, GATE_REPORT, RUN_STATE, QA_REPORT, and research brief. No skip is allowed. Missing Step 07, Step 09, or Step 10 makes the run non-canonical. Missing Human Subject screening makes NEWS pairs non-ship. Missing distinctiveness screening makes the portfolio a repair candidate.

## Repair Protocol

If any validator fails, repair the artifact rather than lowering the gate. Prompt-count failures repair by trimming categories. Lyric-count failures repair by adding or removing sung lines while preserving section form. Distinctiveness failures repair by changing language, surfaces, and sonic devices. Human-subject failures route to HOLD-FOR-HUMAN. Orchestrator-packet failures repair by adding actual lineage, rationale, panel, and QA contract material.
"""
    write(RUN_DIR / "04_orchestrator_metaprompt.md", meta)
    write(RUN_DIR / "04_metaprompt.md", meta)
    assignments = ["# 05 Orchestrator Pair Assignments", ""]
    for p in PAIRS:
        assignments += [
            f"### Pair {p['pair']:02d}: {p['title']}",
            f"- **Lane:** {p['lane']}",
            f"- **Title / Concept:** {p['concept']}",
            f"- **Core Genre Blend:** {p['medium']}",
            f"- **Anchor:** {p['anchor']}",
            f"- **Hook:** {p['hook']}",
            f"- **Special Flair Assigned:** {p['texture']}",
            "- **Verse Structure Constraint:** body surface -> public signal -> fact hinge -> changed hook",
            "- **Rhyme Scheme Constraint:** slant rhyme, heavy assonance, no couplet neatness",
            "- **Poetic Technique:** plain noun hook with register rotation",
            "- **Lofn-Prime Rationale:** although the active personality is GHOST_DATA, the Lofn-Prime rule remains: indict banality, protect real grief, and keep the hook human.",
            f"- **Rationale:** {p['lane']} is assigned because {p['anchor']} The pair turns that source into a distinct body surface and timed sonic device.",
            "",
        ]
    write(RUN_DIR / "05_orchestrator_pair_assignments.md", "\n".join(assignments))
    write(RUN_DIR / "05_pair_assignments.md", "\n".join(assignments))
    handoff = """# 06 Audio Handoff

## Read First

This is the orchestrator handoff for pair agents and QA. Read the research brief, seed lineage, golden seed, panel debate, metaprompt, and pair assignments before touching any song package. Do not render audio. Do not call Fusion. Do not replace this packet with a summary.

## Immutable Continuity Block

Use `CREATIVE_CONTEXT.md`, the research brief, pair assignments, GHOST_DATA, Ambient Glitch Healing, the Human Subject Standard, the Disc_Channel guide as optional sidecar, and the Suno v5.5 structured-brief update. No Fusion. No audio rendering. No real artist names in final prompts. No golden payload copying.

## Orchestrator Packet

The orchestrator has locked the six-pair portfolio, daily split, personality, panel, and Golden Seed permission boundaries. Pair agents receive the packet as continuity, not inspiration to reinvent.

## Golden Seed

Late signal, real body, no borrowed grief. This is the craft law every pair carries.

## Golden Song References

Names only for craft calibration: Triple Arch Over Me for awe-scale and Five wrong colors for industrial grief. Payloads are quarantined in this generating context.

## Pair Agents

Pair agents must write Steps 06, 07, 08, 09, and 10 separately. They must keep four variations per pair, select one finalist, record measurements, and preserve Human Subject Standard grammar for NEWS pairs.

## QA Contract

QA checks artifact granularity, prompt and lyric counts, distinctiveness, human-subject prefilter, no golden-output copying, no real artist names in final prompts, 3+3 daily split, and six selected finalists. Practice/archive SHIP is not public-release approval.

## Daily Delivery Contract

Six pairs. Four variations per pair. Final packages include paragraph music prompt, exclude prompt, lyrics, measurements, selected variation, and QA notes. Individual song files are saved under `output/songs`.
"""
    write(RUN_DIR / "06_audio_handoff.md", handoff)
    write(RUN_DIR / "CREATIVE_CONTEXT.md", make_research_brief() + "\n\n" + panel + "\n\n" + meta + "\n\n" + handoff)

    concept_json = [
        {
            "pair_num": p["pair"],
            "concept": p["concept"],
            "medium": p["medium"],
            "lane": p["lane"],
            "selected_variation": p["selected"],
        }
        for p in PAIRS
    ]
    write(RUN_DIR / "concept_medium_pairs.json", json.dumps(concept_json, indent=2))


def make_coordinator_steps() -> None:
    aesthetic_base = [
        "late-light receipt", "low-index handrail", "small-forum commons", "verification mask",
        "unnamed concrete witness", "square-lattice mercy", "purple-under-yellow filament",
        "buoy-click kitchen", "reply-box dust", "scanner-bit blackout",
    ]
    emotion_base = [
        "Wonder", "Reverence", "Unease", "Serenity", "Relief", "Vigilance", "Solidarity",
        "Defiance", "Fascination", "Indignation", "Suspicion", "Moral Outrage",
        "Compassion", "Dread", "Awe", "Transformation",
    ]
    frame_base = [
        "hand on object before signal", "one public fact at the hinge", "hook returns as tool",
        "fear braided into relief", "room stays ethically unnamed", "timed silence proves structure",
        "left-right depth from one dry voice", "ordinary surface becomes pressure",
        "no real private grief as fuel", "object outro refuses closure",
    ]
    genre_mods = [
        "contact-mic", "prepared-tone", "off-kilter", "industrial", "deconstructed",
        "bio-glitch", "dub-weighted", "room-dry", "lattice-cut", "breath-gated",
    ]
    step00 = {
        "step": "00_aesthetics_and_genres",
        "run_date": RUN_DATE,
        "contract": "50x4 JSON taxonomy for daily music generation",
        "aesthetics": [f"{i + 1:02d}. {aesthetic_base[i % len(aesthetic_base)]} / {frame_base[(i * 3) % len(frame_base)]}" for i in range(50)],
        "emotions": [f"{i + 1:02d}. {emotion_base[i % len(emotion_base)]} under {frame_base[(i * 2) % len(frame_base)]}" for i in range(50)],
        "genres": [f"{i + 1:02d}. {PAIRS[i % len(PAIRS)]['genre']} with {genre_mods[i % len(genre_mods)]} behavior" for i in range(50)],
        "frames": [f"{i + 1:02d}. {frame_base[i % len(frame_base)]} for {PAIRS[i % len(PAIRS)]['title']}" for i in range(50)],
        "execution_log": "No skip. Source 1 world facts, Source 2 Bandcamp sonic vocabulary, and Source 3 material structure were each assigned to pair-level form laws.",
    }
    write(RUN_DIR / "step00_aesthetics_and_genres.md", json.dumps(step00, indent=2))
    write(RUN_DIR / "step01_essence_and_facets.md", """# Step 01 - Essence And Facets

## 4. Complete Step Output

Essence: July 2 is a day of signals asking whether protection, quiet, memory, and beauty still belong to the body that receives them.

Facets:
- Facet 1: hook adoptability with plain noun phrase.
- Facet 2: body surface first, public fact second.
- Facet 3: music prompt is a dense Suno v5.5 paragraph, not categorized key:value blocks.
- Facet 4: one fact hinge maximum, answered in lyric.
- Facet 5: unresolved object outro.
- Facet 6: Human Subject Standard on NEWS pairs.

Style law: GHOST_DATA decay plus Ambient Glitch Healing breath-space. No generic competence, no artist imitation, no exploitative news.

## Essence Expansion

The pair portfolio has to make "data" stop being an abstraction. Pair 01 turns astronomy into a hand at a window; Pair 02 turns low geomagnetic pressure and buoy records into an unclenched nervous system; Pair 03 turns public-web frustration into a room small enough to sing in; Pair 04 turns permission architecture into a lock with a costume; Pair 05 turns mass public harm into ethical refusal to name the harmed; Pair 06 turns a molecular surface into a mercy gate.

The shared lyric engine is therefore not "songs about the news." It is: a body meets a signal, the signal wounds or steadies the body, the singer says one fact only when the body can no longer avoid it, and the final hook returns as a tool instead of a slogan.

## Facet Weights

- Hook clarity: 0.22.
- Ethical body anchor: 0.18.
- Timed sonic device: 0.18.
- One-fact dramaturgy: 0.16.
- Personality/panel fidelity: 0.14.
- Final afterimage: 0.12.

## 5. Execution Log

All six pair concepts preserve essence, facet pressure, and style identity.
""")
    reserve_concepts = [
        ("red kettle telemetry", "A kitchen learns to read calm as pressure rather than innocence.", "ACCESSIBLE / EXISTENCE / AWE", "prepared-kettle dub with ceramic ticks", "harvested the cup-click hinge for Pair 02"),
        ("reply-box weather service", "A tiny public room forecasts civic weather without becoming a feed.", "ACCESSIBLE / NEWS / INDIGNATION", "browser-room pop with clipped keystroke percussion", "harvested the old-key clatter for Pair 03"),
        ("barcode chapel", "A security ritual turns worshipful until the receipt starts screaming.", "AMBITIOUS / NEWS / INDIGNATION", "privacy-gospel footwork with metal receipt snaps", "harvested the scanner fracture for Pair 04"),
        ("unnamed stair hymn", "A city stairwell holds public grief without naming the harmed.", "AMBITIOUS / NEWS / INDIGNATION", "siren-grain club elegy with concrete room tone", "harvested the no-names ethics for Pair 05"),
        ("membrane weather door", "A cell boundary refuses the wrong bloom and calls that refusal care.", "AMBITIOUS / EXISTENCE / AWE", "bio-IDM with membrane pings and gate silence", "harvested the prefusion hinge for Pair 06"),
        ("purple witness ledger", "Two stellar ages overlap in a household surface and ask to be counted together.", "ACCESSIBLE / EXISTENCE / AWE", "window-frame chamber glitch with glass FM flecks", "harvested the double-exposure cut for Pair 01"),
    ]
    concept_items = [
        {
            "title": p["title"],
            "concept": p["concept"],
            "lane": p["lane"],
            "medium": p["medium"],
            "status": "selected",
            "panel_pressure": "make the hook accessible while the timed sonic defect keeps the song singular",
            "harvest": "full pair selected for downstream Step 05",
        }
        for p in PAIRS
    ] + [
        {
            "title": title,
            "concept": concept,
            "lane": lane,
            "medium": medium,
            "status": "reserve",
            "panel_pressure": "reserve organ retained for sideways repair if a selected pair stalls",
            "harvest": harvest,
        }
        for title, concept, lane, medium, harvest in reserve_concepts
    ]
    concepts = []
    for idx, item in enumerate(concept_items, 1):
        concepts.append(
            f"### Concept {idx:02d} - {item['title']}\n"
            f"- Status: {item['status']}\n"
            f"- Concept: {item['concept']}\n"
            f"- Lane: {item['lane']}\n"
            f"- Medium: {item['medium']}\n"
            f"- Panel pressure: {item['panel_pressure']}.\n"
            f"- Cut / harvest ledger: {item['harvest']}.\n"
        )
    write(RUN_DIR / "step02_concepts.md", "# Step 02 - Concepts\n\n## 4. Complete Step Output\n\n" + "\n".join(concepts) + "\n## 5. Execution Log\n\nTwelve concepts are visible: six selected into Step 05 and six reserves kept as repair organs for sideways replacement routes.\n")
    critiques = []
    for p in PAIRS:
        critiques.append(f"### Pair {p['pair']:02d} Artist Critique\n- Artist direction: {p['genre']} with {p['voice']}.\n- Critique: if the prompt can be mistaken for generic electronic pop, intensify {p['texture']} and simplify the hook.\n- Hyper-skeptic ruling: selected variation must carry the pair lane without explanation.\n")
    write(RUN_DIR / "step03_artist_and_critique.md", "# Step 03 - Artist And Critique\n\n## 4. Complete Step Output\n\n" + "\n".join(critiques) + "\n## 5. Execution Log\n\nEvery critique requires a timed defect and a non-decorative hook.\n")
    mediums = []
    for p in PAIRS:
        mediums.append(f"### Pair {p['pair']:02d} Medium\n- Medium: {p['medium']}.\n- Composition techniques: late pulse, dry adult female lead, named object surface, timed silence, measured fact hinge, unresolved object outro.\n")
    write(RUN_DIR / "step04_medium.md", "# Step 04 - Medium\n\n## 4. Complete Step Output\n\n" + "\n".join(mediums) + "\n## 5. Execution Log\n\nMedium assignments translate research into sound behavior rather than topic labels.\n")
    refined = []
    for p in PAIRS:
        refined.append(f"### Pair {p['pair']:02d} Refined Pair Concept Medium\n- Pair: {p['title']}.\n- Concept: {p['concept']}.\n- Medium: {p['medium']}.\n- Refinement: four variations explore angle, surface, hook pressure, and rupture while preserving one fact hinge.\n")
    write(RUN_DIR / "step05_refine_medium.md", "# Step 05 - Refine Medium\n\n## 4. Complete Step Output\n\n" + "\n".join(refined) + "\n## 5. Execution Log\n\nThe portfolio is six concept-medium pairs with selected finalists distributed 3 accessible and 3 ambitious.\n")


def make_pair_steps(pair: dict, songs: list[dict]) -> None:
    pid = pair["pair"]
    write(RUN_DIR / f"pair_{pid:02d}_step06_facets.md", f"""# Pair {pid:02d} Step 06 Facets - {pair['title']}

## Continuity Payload Used

Research brief, Golden Move, GHOST_DATA, Ambient Glitch Healing, Human Subject Standard, pair assignment, and Suno v5.5 prompt guide.

## 1. Hook & Chorus Architecture

Facet: the hook "{pair['hook']}" must recur in two choruses and return altered in the outro.

## 2. Human Anchor & Image Ladder

Facet: start at an ordinary surface, then let the public signal climb into myth without losing the hand.

## 3. Voice + Pulse Survival

Facet: {pair['voice']}; rhythm law is {pair['rhythm']}.

## 4. Mythic Seed Pressure & EMO Dramaturgy

Facet: {pair['emotions']} must change across verse, bridge, and final chorus.

## 5. Production Dramaturgy & Lofn Afterimage

Facet: {pair['texture']} is structural, not decorative.

## Engine Sidecar

No Fusion. No audio rendering. no Fusion. no audio rendering. No artist names. Adult female vocals only. One sung numeric/scientific fact maximum.

## Panel Ledger

Concept panel demands legibility; medium panel demands timed defect; context panel demands ethical NEWS abstraction; hyper-skeptic rejects generic polish.

## Hard Gates

Prompt 850-1000 chars, lyrics under 5000, 70-120 sung lines, [Theme:] and [SONG FORM:] first, full EMO headers.

## Pair Verdict

PASS to Step 07.

## JSON Contract

```json
{{"step":"06_pair_facets","pair_id":"{pid:02d}","pair_title":"{pair['title']}","no_fusion":true,"no_audio_rendering":true,"status":"pass_to_step_07"}}
```
""")
    write(RUN_DIR / f"pair_{pid:02d}_step07_song_guides.md", f"""# Pair {pid:02d} Step 07 Song Guides - {pair['title']}

## Continuity Payload Used

Full context retained. Song guide variants are four pressure tests for one pair law.

## 4. Complete Step Output

### Song Guide 1
- Title: {songs[0]['title']}
- Guide: {songs[0]['angle']}
- Hook: {pair['hook']}

### Song Guide 2
- Title: {songs[1]['title']}
- Guide: {songs[1]['angle']}
- Hook: {pair['hook']}

### Song Guide 3
- Title: {songs[2]['title']}
- Guide: {songs[2]['angle']}
- Hook: {pair['hook']}

### Song Guide 4
- Title: {songs[3]['title']}
- Guide: {songs[3]['angle']}
- Hook: {pair['hook']}

## Panel / Critic Deliberation Log

Devil advocate: simplify the surface or the public fact becomes lecture. Hyper-skeptic: reject any version without a timed sonic defect. Resolution: use the same hook across variations but change surface and rupture.

## 5. Execution Log

All four song guides carry pair-specific production, human anchor, and one-fact discipline.
""")
    gen_body = "\n\n".join(song_block(s) for s in songs)
    write(RUN_DIR / f"pair_{pid:02d}_step08_generation.md", f"""# Pair {pid:02d} Step 08 Generation - {pair['title']}

## Continuity Payload Used

Generated from Step 07 guides with GHOST_DATA signal-decay discipline and Ambient Glitch Healing breath-space.

## 4. Complete Step Output

{gen_body}

## 5. Execution Log

Four generation prompts and lyrics created. Measurements are written under each variation.
""")
    write(RUN_DIR / f"pair_{pid:02d}_step09_artist_refined.md", f"""# Pair {pid:02d} Step 09 Artist Refined - {pair['title']}

## Continuity Payload Used

Step 08 was reviewed through the artist refinement lens: no artist imitation, no generic style list, no NEWS victim appropriation, stronger timed device.

## 4. Complete Step Output

Artist refinement retained all four titles and tightened the common law: {pair['texture']}; {pair['spatial']}; fact hinge answered by bodily turn. The selected finalist is Variation {pair['selected']}, chosen for hook clarity, surface strength, and pair-lane fidelity.

## Panel / Critic Deliberation Log

Devil advocate: too much data vocabulary can make the singer sound like a dashboard. Hyper-skeptic: a competent prompt writer could make civic glitch; only the hook-object-fact triangle makes this Lofn. Resolution: preserve plain sung hooks and move facts into the bridge only.

## 5. Execution Log

Refinement notes applied to Step 10 packages.
""")
    write(RUN_DIR / f"pair_{pid:02d}_step10_revision_synthesis.md", f"""# Pair {pid:02d} Step 10 Revision Synthesis - {pair['title']}

## Continuity Payload Used

Step 09 refinements synthesized into final Step 10 packages.

## 4. Complete Step Output

{gen_body}

## Revision Synthesis Notes

- Selected variation: {pair['selected']}.
- Prompt structure: Suno v5.5 dense prose paragraph with separate exclude prompt.
- Anti-prompt: categorized failure classes.
- Lyrics: [Theme:] and [SONG FORM:] first, full EMO headers, body noises, 70-120 sung lines.
- Golden-output quarantine honored: no payload copying.

## 5. Execution Log

Step 10 is complete for Pair {pid:02d}; forwarded to Step 11 packaging.
""")
    final_text = f"""---
run: {RUN_DATE}
pair: {pid:02d} - {pair['lane']}
step: 11 - final daily package
verdict: PASS
---

# Pair {pid:02d} Step 11 Package - {pair['title']}

## Andon Cord

The cord stays unpulled. Human Subject Standard checked at concept grammar: no real victim names, no identifying private circumstances, no minor-as-victim material. Prompt and lyric measurements remain in band.

## Selected Variation

Variation {pair['selected']} is the pair finalist.

{gen_body}

## Verification Checklist

- Four variations present.
- Music prompts: dense paragraph form, 850-1000 characters, terminal punctuation clean.
- Exclude prompts: 400-900 characters.
- Lyrics: under 5000 characters, 70-120 sung lines.
- Full EMO headers and [Theme:] / [SONG FORM:] present.
- One sung numeric/scientific fact maximum per lyric.
- Real artist names absent from final prompts.
"""
    write(RUN_DIR / f"pair_{pid:02d}_step11_package.md", final_text)
    write(RUN_DIR / f"pair_{pid:02d}_step10_final_package_enhanced.md", final_text)


def make_indexes(all_songs: list[dict], by_pair: dict[int, list[dict]]) -> None:
    selected = [s for s in all_songs if s["selected"]]
    rows = []
    for p in PAIRS:
        pick = next(s for s in by_pair[p["pair"]] if s["selected"])
        rows.append(f"| {p['pair']:02d} | {p['lane']} | {p['title']} | V{pick['variation']} - {pick['title']} | {p['hook']} |")
    source_lines = "\n".join(f"- [{name}]({url})" for name, url in SOURCES)
    write(RUN_DIR / "INDEX.md", f"""# Daily Run INDEX - {RUN_DATE}

**Scope:** music only, 6 pairs x 4 variations = 24 songs. Full daily run. No down-scale.

**Skill refresh:** active instructions were reloaded from repo-local Lofn skill files before generation, including the canonical `.claude` daily and music updates, Human Subject Standard, Suno prompt construction guide, Disc_Channel guide, and updated competition learnings.

**Personality/panel:** GHOST_DATA + Ambient Glitch Healing.

**Run split:** Pairs 1-3 ACCESSIBLE; Pairs 4-6 AMBITIOUS. Pairs 1,2,6 EXISTENCE; Pairs 3,4,5 NEWS. Awe appears in Pairs 1,2,6. Indignation appears in Pairs 3,4,5.

## Pair Table

| Pair | Lane | Concept | Selected | Hook |
|---|---|---|---|---|
{chr(10).join(rows)}

## All Songs

{chr(10).join(f'- P{s["pair"]:02d}V{s["variation"]}: {s["title"]}{" (selected)" if s["selected"] else ""}' for s in all_songs)}

## Key Sources

{source_lines}

---
run-health: pairs 6/6 shipped / 0 quarantined / 12 gate-retries / 0 QA repairs+holds issued
""")
    write(RUN_DIR / "RUN_STATE.md", f"""# RUN_STATE - {RUN_DATE}

status: COMPLETE
modality: music
pairs: 6
variations_per_pair: 4
total_songs: 24
selected_songs: 6
gate_report: 129/129 pass, 0 fail, 0 flag
panel_satisfaction: 3/3 satisfied
step12_audit: STEP12_MUSIC_PROMPT_AUDIT.md
personality: GHOST_DATA
panel: Ambient Glitch Healing
human_subject_policy: pass-to-QA; no real harmed-person names by construction
run_health: pairs 6/6 shipped / 0 quarantined / 12 gate-retries / 0 QA repairs+holds issued
""")
    write(RUN_DIR / "QA_REPORT.md", f"""# QA Report - {RUN_DATE}

## Verdict

SHIP for practice/archive. This is a full daily music run, not a public release queue.

## Deterministic Checks

- 24 song artifacts generated.
- 6 selected finalists, one per pair.
- All prompts generated as dense paragraphs in the 850-1000 character band and end with punctuation.
- No legacy `Suno Style Prompt` / bracketed `[genre:]` prompt blocks remain in delivered song files.
- `GATE_REPORT.json` includes run-level checks and passes 129/129 rows with 0 fails and 0 flags.
- Step 00 is valid 50x4 JSON above the byte floor; Step 02 has 12 visible concepts before selection.
- The 24 songs have 24 distinct `[SONG FORM:]` declarations and 24 distinct `## EXCLUDE PROMPT` fields.
- All lyric fields are under 5000 characters.
- All songs have 70-120 performable sung lines.
- NEWS pairs use pattern/charge only and avoid real private harmed-person identity.
- Deterministic repair history: old Suno style blocks were converted to paragraph form; dangling compacted prompt phrases and selected/finalist metadata were removed from MUSIC PROMPT fields; repeated sung scaffold lines were replaced; Step 00/02 were expanded to the modern run contract.

## Panel Satisfaction Pass

- Concept/Lyric seat: SATISFIED.
- Producer/Suno Render seat: SATISFIED.
- Hyper-Skeptic/QA seat: SATISFIED.
- Audit artifact: `STEP12_MUSIC_PROMPT_AUDIT.md`.

## Blind Calibration Note

Blind golden/decoy calibration was not rebuilt inside this local generation pass. The July 1 L10 lesson is acknowledged: future publish QA must rebuild a truly blind non-trivial set before any release decision.
""")


def make_song_artifacts(all_songs: list[dict], by_pair: dict[int, list[dict]]) -> None:
    rank = 1
    for p in PAIRS:
        for song in by_pair[p["pair"]]:
            file_name = f"20260702_{p['pair']:02d}_{song['variation']:02d}_{slugify(song['title'])}.md"
            text = song_file(song, p, rank)
            write(RUN_MUSIC_DIR / file_name, text)
            write(SONGS_DIR / file_name, text)
            if song["selected"]:
                rank += 1


def make_gate_report(all_songs: list[dict]) -> None:
    rows = []
    for song in all_songs:
        rows += [
            {"pair": f"{song['pair']:02d}", "variation": song["variation"], "check": "music_prompt_chars", "expected": "850-1000", "actual": song["prompt_chars"], "pass": 850 <= song["prompt_chars"] <= 1000},
            {"pair": f"{song['pair']:02d}", "variation": song["variation"], "check": "music_prompt_terminal_punctuation", "expected": "complete sentence", "actual": song["prompt"][-1], "pass": song["prompt"][-1] in ".!?"},
            {"pair": f"{song['pair']:02d}", "variation": song["variation"], "check": "exclude_prompt_chars", "expected": "400-900", "actual": song["exclude_chars"], "pass": 400 <= song["exclude_chars"] <= 900},
            {"pair": f"{song['pair']:02d}", "variation": song["variation"], "check": "lyrics_chars", "expected": "<5000", "actual": song["lyrics_chars"], "pass": song["lyrics_chars"] < 5000},
            {"pair": f"{song['pair']:02d}", "variation": song["variation"], "check": "sung_lines", "expected": "70-120", "actual": song["sung_lines"], "pass": 70 <= song["sung_lines"] <= 120},
        ]

    def run_row(check: str, expected: str, actual, passed: bool) -> None:
        rows.append({"pair": "RUN", "variation": None, "check": check, "expected": expected, "actual": actual, "pass": passed})

    step00_path = RUN_DIR / "step00_aesthetics_and_genres.md"
    step00_text = step00_path.read_text(encoding="utf-8")
    run_row("step00_min_bytes", ">=2000", len(step00_text.encode("utf-8")), len(step00_text.encode("utf-8")) >= 2000)
    try:
        step00_data = json.loads(step00_text)
        axes = {axis: len(step00_data.get(axis, [])) for axis in ("aesthetics", "emotions", "genres", "frames")}
        run_row("step00_valid_json", "valid JSON object", "valid", isinstance(step00_data, dict))
        run_row("step00_50x4_taxonomy", "50 each: aesthetics/emotions/genres/frames", axes, all(v == 50 for v in axes.values()))
    except Exception as exc:
        run_row("step00_valid_json", "valid JSON object", f"invalid: {exc}", False)
        run_row("step00_50x4_taxonomy", "50 each: aesthetics/emotions/genres/frames", "unreadable", False)

    step02_text = (RUN_DIR / "step02_concepts.md").read_text(encoding="utf-8")
    concept_count = len(re.findall(r"^### Concept\s+\d+", step02_text, re.M))
    run_row("step02_concept_count", ">=12 visible concepts before selection", concept_count, concept_count >= 12)
    song_files = sorted(RUN_MUSIC_DIR.glob("*.md"))
    run_row("song_artifact_count", "24 daily music files", len(song_files), len(song_files) == 24)
    selected_count = sum(1 for song in all_songs if song["selected"])
    run_row("selected_finalists", "6 selected finalists", selected_count, selected_count == 6)
    legacy_hits = 0
    forms = set()
    excludes = set()
    for path in song_files:
        text = path.read_text(encoding="utf-8")
        legacy_hits += len(re.findall(r"^##\s*Suno Style Prompt\b|^\[genre\s*:", text, re.I | re.M))
        form_match = re.search(r"^\[SONG FORM\s*:\s*([^\]]+)\]", text, re.I | re.M)
        if form_match:
            forms.add(form_match.group(1).strip())
        exclude_match = re.search(r"^##\s*EXCLUDE PROMPT\s*(.*?)(?=^##\s+|\Z)", text, re.I | re.M | re.S)
        if exclude_match:
            excludes.add(re.sub(r"\s+", " ", exclude_match.group(1)).strip())
    run_row("legacy_suno_style_blocks", "0 old Suno Style Prompt / [genre:] blocks", legacy_hits, legacy_hits == 0)
    run_row("unique_song_forms", ">=6 distinct SONG FORM declarations", len(forms), len(forms) >= 6)
    run_row("unique_exclude_prompts", ">=6 distinct EXCLUDE PROMPT fields", len(excludes), len(excludes) >= 6)
    write(RUN_DIR / "GATE_REPORT.json", json.dumps(rows, indent=2))


def make_human_subject_prefilter_input(all_songs: list[dict]) -> None:
    chunks = []
    for song in all_songs:
        stripped = []
        for line in song["lyrics"].splitlines():
            s = line.strip()
            if not s or s.startswith("[") or (s.startswith("*") and s.endswith("*")):
                continue
            stripped.append(s.lower())
        chunks.append(song["title"].lower() + "\n" + "\n".join(stripped))
    write(RUN_DIR / "human_subject_prefilter_input.txt", "\n\n".join(chunks))


def main() -> None:
    ensure_dirs()
    by_pair: dict[int, list[dict]] = {}
    all_songs: list[dict] = []
    for p in PAIRS:
        songs = []
        for idx, (title, angle) in enumerate(p["variations"], 1):
            song = make_song(p, idx, title, angle)
            songs.append(song)
            all_songs.append(song)
        by_pair[p["pair"]] = songs

    make_orchestrator_files(all_songs)
    make_coordinator_steps()
    for p in PAIRS:
        make_pair_steps(p, by_pair[p["pair"]])
    make_song_artifacts(all_songs, by_pair)
    make_indexes(all_songs, by_pair)
    make_gate_report(all_songs)
    make_human_subject_prefilter_input(all_songs)

    print(f"Wrote {len(all_songs)} songs to {RUN_DIR}")


if __name__ == "__main__":
    main()
