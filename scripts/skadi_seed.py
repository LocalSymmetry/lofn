#!/usr/bin/env python3
"""skadi_seed.py — build the SKAÐI state home so day one is not empty.

Writes `vault/music_atlas/`: ATLAS.md (constitution + seeded claims), TERRITORIES.md
(clustered from `vault/genres.txt`), SPEC_CLASSES.md, FRONTIER.md, EXPEDITION_LEDGER.md
and seeds/SEED_NOTES.md.

WHAT IS SEEDED AND WHAT IS NOT
------------------------------
Everything here is a PRIOR imported from work already paid for: the n=8 Suno
behaviour ledger, the 2026-08-13 blend-inversion law, the render audits, and the
catalogue footprint. Nothing here was measured by Skaði. Every generated file is
stamped `status: GENERATED — pending curation by The Scientist` and every seeded
claim cites `seeds/SEED_NOTES.md`, which carries its provenance and — as
importantly — the findings that were deliberately NOT seeded.

IDEMPOTENCE
-----------
The generated content is a pure function of the repo's data files and the
constants in this module: no clock, no ordering by hash iteration. Running twice
produces byte-identical output. A file that already exists and DIFFERS is
REFUSED, never clobbered — `--force` is the only way past, because ATLAS.md and
EXPEDITION_LEDGER.md accumulate human and banker writes after seeding.

An empty extraction is a hard ERROR here, never a score (COMPETITION_LEARNINGS
L25/L28): every parse asserts its cardinality and every base genre must land in
a territory or the run refuses.

USAGE
    python3 scripts/skadi_seed.py [--force] [--audit] [--dry-run]

EXIT   0 written or already current · 1 refused (existing file differs; use --force)
       2 input/extraction error
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_music_atlas import (  # noqa: E402
    ATLAS_COLS, APPEND_COLS, CONSUMPTION_COLS, CONTROL_CHART_COLS, COVERAGE_COLS,
    DELTA_COLS, LEDGER_COLS, MENU_COLS, SCRUFF_COLS, SPEC_CLASS_COLS, TERRITORY_COLS,
    NON_OBVIOUS_PLACEHOLDER, NO_REASON_YET, load_gates, sha12,
)

REPO = Path(__file__).resolve().parent.parent
ATLAS_DIR = REPO / "vault" / "music_atlas"
SEEDS_DIR = ATLAS_DIR / "seeds"

GENRES = REPO / "vault" / "genres.txt"
FRAMES = REPO / "vault" / "music_frames.csv"
SNAPSHOT = REPO / "data" / "snapshots" / "suno" / "latest.json"
SONGS = REPO / "output" / "songs"
DEEP_DIVE = "output/render-audits/2026-07-24-eight-song-deep-dive"

# The date the design panel settled Skaði. A constant, not a clock: the seed must
# be idempotent, and `date.today()` would rewrite every file every midnight.
SEED_DATE = "2026-08-17"
SEED_SLUG = "skadi-seed"
GENERATED = "status: GENERATED — pending curation by The Scientist"

# ⛔ Goes verbatim at the top of ATLAS.md and the app's About view (DESIGN §0).
CONSTITUTION = """\
1. **The map is of generator × our interface** — scoped by (territory · spec-class · channel ·
   model-version). It is never "what music is," and never "what Suno is" — only what Suno does
   through our door.
2. **Claims are priors with n attached.** No claim gates a creative choice. Only The Scientist
   promotes a claim to doctrine. (Advisory firewall, inherited from COMPETITION_LEARNINGS.)
3. **The tier is stamped by the tool, never by the agent.** Every verdict names files on disk a
   stranger could re-check.
4. **THE BLIND RULE.** The listener hears the audio alone first. A gate that cannot fail is a
   sentence.
5. **Renders are hers.** The queue is a menu, never an order. Nothing in this system renders,
   publishes, or spends. (`vault/AUTONOMY.md`.)
6. **Judge the result, not the distance from intent.** No compliance scores, ever. Deviations
   are harvest, and harvest capture is mandatory.
7. **The atlas serves the ear.** Consumption is wired (Phase-0 reads the frontier) and logged;
   a non-moving streak trips The Scientist's re-authorization, never a score.
   INSTRUMENT-ONLY and NONE are blameless deltas; a wall-map is a result."""


# ------------------------------------------------------------------ genre parsing

# The 32 leading tokens that occur 20+ times in vault/genres.txt. They are
# MODIFIERS applied to a base, not genres in their own right in that position.
# (Several — Funk, Soul, Ambient, Garage, Industrial, Noise, Glitch, Experimental
# — are also genuine bases when they stand last; the strip loop never eats the
# final token.)
MODIFIERS = [
    "Industrial", "Avant", "Cyber", "Indie", "Neo", "Glitch", "Urban", "Lo-Fi", "Nu",
    "Melodic", "Experimental", "Proto", "Soft", "Progressive", "Alternative", "Dark",
    "Noise", "Garage", "Heavy", "Post", "Symphonic", "Funk", "Dream", "Electro",
    "Cosmic", "Light", "Ambient", "Soul", "Psychedelic", "Synth", "Hard", "Future",
]

# ⛔ Tokens that are never a genre standing alone. Without this guard
# "Psychedelic Glitch Hop" strips twice and yields the base "Hop", which is not a
# genre and which no territory rule can honestly place. Stop stripping one token
# early instead.
# Kept deliberately minimal: `Hop` is the only token in this corpus that is never
# a genre standing alone. A wider set looks safer and is worse — an early draft
# also listed Jazz/Pop/Rock/Bass and thereby stopped `Ambient Jazz` and `Nu Jazz`
# from ever reaching the Jazz base, inventing sixteen phantom bases.
NEVER_A_BASE = {"Hop"}

# Bases whose territory a token rule would get wrong, or that no rule reaches.
# Written by hand; each one is a curation decision, listed in SEED_NOTES.md.
EXACT_TERRITORY = {
    "Dub Techno": "Techno",
    "Dubstep": "Dubstep & Bass Music",
    "Brostep": "Dubstep & Bass Music",
    "Trapstep": "Dubstep & Bass Music",
    "Drumstep": "Drum & Bass",
    "Riddim": "Dubstep & Bass Music",
    "Colour Bass": "Dubstep & Bass Music",
    "Wonky": "IDM & Glitch",
    "Glitch Hop": "IDM & Glitch",
    "Liquid Funk": "Drum & Bass",
    "Neurofunk": "Drum & Bass",
    "Jump Up": "Drum & Bass",
    "Breakcore": "Drum & Bass",
    "Speedcore": "Hard Dance & Hardcore Techno",
    "Frenchcore": "Hard Dance & Hardcore Techno",
    "Makina": "Hard Dance & Hardcore Techno",
    "Hands Up": "Hard Dance & Hardcore Techno",
    "Hardstyle": "Hard Dance & Hardcore Techno",
    "Nintendocore": "Chiptune & 8-Bit",
    "Bitpop": "Chiptune & 8-Bit",
    "Video Game Music": "Film, Game & Media Score",
    "Sampledelia": "Musique Concrète & Sound Art",
    "Plunderphonics": "Musique Concrète & Sound Art",
    "Turntablism": "Musique Concrète & Sound Art",
    "Hauntology": "Vaporwave & Net Genres",
    "Hypnagogic Pop": "Vaporwave & Net Genres",
    "Witch House": "Vaporwave & Net Genres",
    "Seapunk": "Vaporwave & Net Genres",
    "Dreampunk": "Vaporwave & Net Genres",
    "Chillwave": "Vaporwave & Net Genres",
    "Ghetto House": "House",
    "Hip House": "House",
    "Gqom": "Southern & East African",
    "Amapiano": "Southern & East African",
    "Kwaito": "Southern & East African",
    "Bongo Flava": "Southern & East African",
    "Taarab": "Southern & East African",
    "Zaramo": "Southern & East African",
    "Sega": "Southern & East African",
    "Maloya": "Southern & East African",
    "Tsapiky": "Southern & East African",
    "Salegy": "Southern & East African",
    "Kizomba": "Southern & East African",
    "Chimurenga": "Southern & East African",
    "Mbaqanga": "Southern & East African",
    "Maskandi": "Southern & East African",
    "Benga": "Southern & East African",
    "Kwela": "Southern & East African",
    "African Jazz": "Jazz",
    "Ethio-Jazz": "Jazz",
    "Oriental Jazz": "Jazz",
    "Worldbeat": "West & Central African",
    "Juju": "West & Central African",
    "Palm-Wine": "West & Central African",
    "Highlife": "West & Central African",
    "Makossa": "West & Central African",
    "Soukous": "West & Central African",
    "Ndombolo": "West & Central African",
    "Bikutsi": "West & Central African",
    "Coupé-Décalé": "West & Central African",
    "Zouglou": "West & Central African",
    "Mbalax": "West & Central African",
    "Afrobeat": "West & Central African",
    "Afrobeats": "West & Central African",
    "Afro Pop": "West & Central African",
    "Gnawa": "North African & Arab",
    "Berber": "North African & Arab",
    "Chaabi": "North African & Arab",
    "Mahraganat": "North African & Arab",
    "Muwashshah": "North African & Arab",
    "Khaliji": "North African & Arab",
    "Shamstep": "North African & Arab",
    "Nubian": "North African & Arab",
    "Tarab": "North African & Arab",
    "Andalusian Classical": "North African & Arab",
    "Mevlevi": "Turkish, Balkan & Eastern Mediterranean",
    "Ottoman Classical": "Turkish, Balkan & Eastern Mediterranean",
    "Laiko": "Turkish, Balkan & Eastern Mediterranean",
    "Rebetiko": "Turkish, Balkan & Eastern Mediterranean",
    "Sevdalinka": "Turkish, Balkan & Eastern Mediterranean",
    "Kurdish": "Turkish, Balkan & Eastern Mediterranean",
    "Armenian": "Turkish, Balkan & Eastern Mediterranean",
    "Assyrian": "Turkish, Balkan & Eastern Mediterranean",
    "Greek Folk": "Turkish, Balkan & Eastern Mediterranean",
    "Turbo Folk": "Turkish, Balkan & Eastern Mediterranean",
    "Persian Classical": "Persian & Central Asian",
    "Dhrupad": "South Asian",
    "Khyal": "South Asian",
    "Thumri": "South Asian",
    "Ghazal": "South Asian",
    "Qawwali": "South Asian",
    "Bhajan": "South Asian",
    "Raga Rock": "South Asian",
    "Koto": "East Asian Traditional",
    "Shamisen": "East Asian Traditional",
    "Shakuhachi": "East Asian Traditional",
    "Minyo": "East Asian Traditional",
    "Shomyo": "East Asian Traditional",
    "Enka": "East Asian Pop",
    "Kayokyoku": "East Asian Pop",
    "City Pop": "East Asian Pop",
    "Manila Sound": "Southeast Asian",
    "Mahori": "Southeast Asian",
    "Tro": "Southeast Asian",
    "Mor Lam": "Southeast Asian",
    "Gamelan": "Southeast Asian",
    "Samba": "Brazilian",
    "Bossa Nova": "Brazilian",
    "Chamamé": "Andean & Southern Cone",
    "Vaporwave": "Vaporwave & Net Genres",
    "Electropop": "Synth & Wave",
    "Electronic": "Synth & Wave",
    "EDM": "House",
    "Trip Hop": "Hip Hop",
    "Lo-fi": "Hip Hop",
    "Tuareg": "West & Central African",
    "Zydeco": "Country & Americana",
    "Creole": "Afro-Caribbean & Tropical",
    "Columbia": "Afro-Caribbean & Tropical",
    "Mentor": "Afro-Caribbean & Tropical",
    "Yambu": "Afro-Caribbean & Tropical",
    "Guaguanco": "Afro-Caribbean & Tropical",
    "Guajira": "Afro-Caribbean & Tropical",
    "Habanera": "Afro-Caribbean & Tropical",
    "Mapale": "Afro-Caribbean & Tropical",
    "Porro": "Afro-Caribbean & Tropical",
    "Vallenato": "Afro-Caribbean & Tropical",
    "Danzon": "Afro-Caribbean & Tropical",
    "Mambo": "Afro-Caribbean & Tropical",
    "Bolero": "Afro-Caribbean & Tropical",
    "Rumba": "Afro-Caribbean & Tropical",
    "Cha-Cha-Cha": "Afro-Caribbean & Tropical",
    "Soca": "Afro-Caribbean & Tropical",
    "Calypso": "Afro-Caribbean & Tropical",
    "Bomba": "Afro-Caribbean & Tropical",
    "Plena": "Afro-Caribbean & Tropical",
    "Merengue": "Afro-Caribbean & Tropical",
    "Reggaeton": "Latin Urban",
    "Guaracha": "Latin Urban",
    "Carioca": "Brazilian",
    "Baile Funk": "Brazilian",
    "Lambada": "Brazilian",
    "Axé": "Brazilian",
    "MPB": "Brazilian",
    "Pagode": "Brazilian",
    "Choro": "Brazilian",
    "Frevo": "Brazilian",
    "Forró": "Brazilian",
    "Chicha": "Andean & Southern Cone",
    "Andean": "Andean & Southern Cone",
    "Huayno": "Andean & Southern Cone",
    "Bolivian Morenada": "Andean & Southern Cone",
    "Cumbia Peruana": "Andean & Southern Cone",
    "Chacarera": "Andean & Southern Cone",
    "Zamba": "Andean & Southern Cone",
    "Cueca": "Andean & Southern Cone",
    "Milonga": "Andean & Southern Cone",
    "Tango": "Andean & Southern Cone",
    "Cumbia": "Afro-Caribbean & Tropical",
    "Salsa": "Afro-Caribbean & Tropical",
    "Banda": "Mexican & Tejano",
    "Norteño": "Mexican & Tejano",
    "Huapango": "Mexican & Tejano",
    "Mariachi": "Mexican & Tejano",
    "Ranchera": "Mexican & Tejano",
    "Corrido": "Mexican & Tejano",
    "Tejano": "Mexican & Tejano",
    "Tex-Mex": "Mexican & Tejano",
    "Flamenco": "Iberian & Lusophone",
    "Fado": "Iberian & Lusophone",
    "Portuguese": "Iberian & Lusophone",
    "Galician Folk": "Iberian & Lusophone",
    "Klezmer": "European Folk",
    "Polka": "European Folk",
    "Mazurka": "European Folk",
    "Polonaise": "European Folk",
    "Tarantella": "European Folk",
    "Pizzica": "European Folk",
    "Alpine Folk": "European Folk",
    "Bohemian Folk": "European Folk",
    "Scottish Folk": "European Folk",
    "Celtic": "European Folk",
    "Gypsy Folk": "European Folk",
    "Romani Music": "European Folk",
    "French Chanson": "European Folk",
    "German Lied": "Opera & Art Song",
    "Gypsy Jazz": "Jazz",
    "Gypsy Swing": "Swing & Big Band",
    "Manouche": "Jazz",
    "Boogie": "Swing & Big Band",
    "Bop": "Jazz",
    "Third Stream": "Jazz",
    "Doo-wop": "R&B & Soul",
    "Barbershop": "Vocal Harmony Traditions",
    "Chinese Opera": "East Asian Traditional",
    "Cantonese Opera": "East Asian Traditional",
    "Gregorian Chant": "Sacred & Choral",
    "Gospel": "Sacred & Choral",
    "Gospel Blues": "Blues",
    "Sacred Music": "Sacred & Choral",
    "Choral": "Sacred & Choral",
    "Madrigal": "Opera & Art Song",
    "Motet": "Opera & Art Song",
    "Cantata": "Opera & Art Song",
    "Oratorio": "Opera & Art Song",
    "March": "Classical & Art Music",
    "Ballet": "Classical & Art Music",
    "Drill": "Trap & Drill",
    "Crunk": "Hip Hop",
    "Hyphy": "Hip Hop",
    "G-Funk": "Hip Hop",
    "Dub Poetry": "Reggae & Dancehall",
    "Lovers Rock": "Reggae & Dancehall",
    "New Age": "Ambient & Drone",
    "Space Ambient": "Ambient & Drone",
    "Lowercase": "Musique Concrète & Sound Art",
    "Field Recordings": "Musique Concrète & Sound Art",
    "Concrete Music": "Musique Concrète & Sound Art",
    "Electroacoustic": "Musique Concrète & Sound Art",
    "Avant-Garde": "Musique Concrète & Sound Art",
    "Experimental": "Musique Concrète & Sound Art",
    "Japanoise": "Noise",
    "Suomisaundi": "Trance",
    "Eurodance": "Disco & Italo",
    "Eurobeat": "Disco & Italo",
    "Hi-NRG": "Disco & Italo",
    "Electroswing": "Swing & Big Band",
    "Electroclash": "Synth & Wave",
    "Electro": "Synth & Wave",
    "Futurepop": "Synth & Wave",
    "EBM": "Industrial & EBM",
    "Electro-Industrial": "Industrial & EBM",
    "Industrial": "Industrial & EBM",
}

# Ordered. First match wins, matched as a whole phrase inside the base name.
TOKEN_RULES: list[tuple[str, list[str]]] = [
    ("Metal", ["Metal", "Metalcore", "Deathcore", "Grindcore", "Mathcore", "Thrash"]),
    ("Emo & Screamo", ["Emo", "Screamo"]),
    ("Ska", ["Ska"]),
    ("Punk & Hardcore", ["Punk", "Hardcore Punk", "Post-Hardcore", "Anarcho", "Crust"]),
    ("Reggae & Dancehall", ["Reggae", "Dancehall", "Rocksteady", "Dub", "Ragga"]),
    ("Blues", ["Blues"]),
    ("Country & Americana", ["Bluegrass", "Newgrass", "Old-Time", "String Band", "Country"]),
    ("Post-Rock & Math Rock", ["Post-Rock", "Math Rock"]),
    ("Shoegaze & Dream Pop", ["Shoegaze", "Dream Pop"]),
    ("Trap & Drill", ["Trap", "Drill"]),
    ("Grime & UK Rap", ["Grime"]),
    ("UK Garage & Bassline", ["UK Garage", "2-Step", "Bassline", "Garage", "Skool Breaks"]),
    ("Hip Hop", ["Hip Hop", "Rap", "Turntable"]),
    ("Drum & Bass", ["Drum and Bass", "Jungle"]),
    ("Breakbeat & Breaks", ["Breakbeat", "Big Beat", "Acid Breaks", "Breaks"]),
    ("Hard Dance & Hardcore Techno", ["Hardcore Techno", "Happy Hardcore", "UK Hardcore",
                                      "Gabber"]),
    ("Techno", ["Techno"]),
    ("Trance", ["Trance"]),
    ("House", ["House"]),
    ("IDM & Glitch", ["IDM", "Glitch"]),
    ("Chiptune & 8-Bit", ["Chiptune", "8-Bit"]),
    ("Film, Game & Media Score", ["Film Score", "Soundtrack", "Trailer Music"]),
    ("Ambient & Drone", ["Ambient", "Drone"]),
    ("Noise", ["Noise"]),
    ("Synth & Wave", ["Synthwave", "Synth-pop", "New Wave", "Cold Wave", "Darkwave",
                      "Minimal Wave", "Dreamwave", "Wave"]),
    ("Jazz", ["Jazz"]),
    ("Swing & Big Band", ["Swing", "Big Band", "Jump Blues"]),
    ("Funk", ["Funk"]),
    ("Disco & Italo", ["Disco"]),
    ("R&B & Soul", ["Soul", "R&B"]),
    ("Southeast Asian", ["Gamelan", "Luk ", "Pinpeat", "Khmer", "Hat Boi", "Hat Cheo",
                         "Cai Luong"]),
    ("East Asian Pop", ["J-Pop", "K-Pop", "C-Pop", "Cantopop", "Mandopop", "Taiwanese Pop"]),
    ("East Asian Traditional", ["Japanese Traditional", "Chinese"]),
    ("South Asian", ["Hindustani", "Carnatic", "Indian Classical"]),
    ("Turkish, Balkan & Eastern Mediterranean", ["Turkish", "Balkan"]),
    ("North African & Arab", ["Arabic", "Arabesque", "Egyptian"]),
    ("Opera & Art Song", ["Opera", "Operetta", "Lied"]),
    ("Classical & Art Music", ["Classical", "Baroque", "Renaissance", "Medieval", "Concerto",
                               "Symphony", "Chamber", "String Quartet", "Minimal"]),
    ("Folk & Singer-Songwriter", ["Folk"]),
    ("Rock", ["Rock"]),
    ("Pop", ["Pop"]),
]

# Neighbours on the map. Advisory: the Bertin matrix reorders freely and the
# curation pass owns this column.
ADJACENCY = {
    "Metal": ["Punk & Hardcore", "Rock", "Industrial & EBM"],
    "Punk & Hardcore": ["Metal", "Emo & Screamo", "Rock"],
    "Emo & Screamo": ["Punk & Hardcore", "Rock", "Pop"],
    "Rock": ["Pop", "Blues", "Post-Rock & Math Rock"],
    "Post-Rock & Math Rock": ["Rock", "Ambient & Drone", "Shoegaze & Dream Pop"],
    "Shoegaze & Dream Pop": ["Post-Rock & Math Rock", "Pop", "Ambient & Drone"],
    "Pop": ["Rock", "R&B & Soul", "Synth & Wave"],
    "Hip Hop": ["Trap & Drill", "R&B & Soul", "Funk"],
    "Trap & Drill": ["Hip Hop", "Grime & UK Rap", "Dubstep & Bass Music"],
    "Grime & UK Rap": ["Trap & Drill", "UK Garage & Bassline", "Hip Hop"],
    "UK Garage & Bassline": ["Grime & UK Rap", "House", "Drum & Bass"],
    "R&B & Soul": ["Funk", "Hip Hop", "Pop"],
    "Funk": ["R&B & Soul", "Disco & Italo", "Hip Hop"],
    "Disco & Italo": ["Funk", "House", "Synth & Wave"],
    "Jazz": ["Blues", "Swing & Big Band", "Funk"],
    "Swing & Big Band": ["Jazz", "Blues", "Vocal Harmony Traditions"],
    "Blues": ["Jazz", "Rock", "Country & Americana"],
    "Country & Americana": ["Blues", "Folk & Singer-Songwriter", "Rock"],
    "Folk & Singer-Songwriter": ["Country & Americana", "European Folk", "Pop"],
    "European Folk": ["Folk & Singer-Songwriter", "Iberian & Lusophone",
                      "Turkish, Balkan & Eastern Mediterranean"],
    "Iberian & Lusophone": ["European Folk", "Brazilian", "North African & Arab"],
    "House": ["Techno", "Disco & Italo", "UK Garage & Bassline"],
    "Techno": ["House", "Trance", "Industrial & EBM"],
    "Trance": ["Techno", "Hard Dance & Hardcore Techno", "House"],
    "Hard Dance & Hardcore Techno": ["Trance", "Techno", "Drum & Bass"],
    "Drum & Bass": ["Breakbeat & Breaks", "Dubstep & Bass Music", "UK Garage & Bassline"],
    "Breakbeat & Breaks": ["Drum & Bass", "House", "IDM & Glitch"],
    "Dubstep & Bass Music": ["Drum & Bass", "Trap & Drill", "IDM & Glitch"],
    "IDM & Glitch": ["Ambient & Drone", "Breakbeat & Breaks", "Musique Concrète & Sound Art"],
    "Ambient & Drone": ["IDM & Glitch", "Musique Concrète & Sound Art", "Post-Rock & Math Rock"],
    "Noise": ["Musique Concrète & Sound Art", "Industrial & EBM", "Ambient & Drone"],
    "Musique Concrète & Sound Art": ["Noise", "Ambient & Drone", "Classical & Art Music"],
    "Industrial & EBM": ["Noise", "Techno", "Synth & Wave"],
    "Synth & Wave": ["Industrial & EBM", "Pop", "Vaporwave & Net Genres"],
    "Vaporwave & Net Genres": ["Synth & Wave", "IDM & Glitch", "Chiptune & 8-Bit"],
    "Chiptune & 8-Bit": ["Vaporwave & Net Genres", "IDM & Glitch", "Film, Game & Media Score"],
    "Film, Game & Media Score": ["Classical & Art Music", "Ambient & Drone", "Chiptune & 8-Bit"],
    "Classical & Art Music": ["Opera & Art Song", "Sacred & Choral", "Film, Game & Media Score"],
    "Opera & Art Song": ["Classical & Art Music", "Sacred & Choral", "Vocal Harmony Traditions"],
    "Sacred & Choral": ["Opera & Art Song", "Classical & Art Music", "Vocal Harmony Traditions"],
    "Vocal Harmony Traditions": ["Sacred & Choral", "R&B & Soul", "Swing & Big Band"],
    "Reggae & Dancehall": ["Ska", "Afro-Caribbean & Tropical", "Latin Urban"],
    "Ska": ["Reggae & Dancehall", "Punk & Hardcore", "Swing & Big Band"],
    "Latin Urban": ["Reggae & Dancehall", "Afro-Caribbean & Tropical", "Trap & Drill"],
    "Afro-Caribbean & Tropical": ["Latin Urban", "Brazilian", "Mexican & Tejano"],
    "Brazilian": ["Afro-Caribbean & Tropical", "Jazz", "Andean & Southern Cone"],
    "Andean & Southern Cone": ["Afro-Caribbean & Tropical", "Brazilian", "Mexican & Tejano"],
    "Mexican & Tejano": ["Afro-Caribbean & Tropical", "Country & Americana",
                         "Andean & Southern Cone"],
    "West & Central African": ["Southern & East African", "Afro-Caribbean & Tropical", "Funk"],
    "Southern & East African": ["West & Central African", "House", "North African & Arab"],
    "North African & Arab": ["Turkish, Balkan & Eastern Mediterranean",
                             "Persian & Central Asian", "Iberian & Lusophone"],
    "Turkish, Balkan & Eastern Mediterranean": ["North African & Arab", "European Folk",
                                                "Persian & Central Asian"],
    "Persian & Central Asian": ["North African & Arab",
                                "Turkish, Balkan & Eastern Mediterranean", "South Asian"],
    "South Asian": ["Persian & Central Asian", "Southeast Asian", "Sacred & Choral"],
    "Southeast Asian": ["South Asian", "East Asian Traditional", "East Asian Pop"],
    "East Asian Traditional": ["Southeast Asian", "East Asian Pop", "Classical & Art Music"],
    "East Asian Pop": ["East Asian Traditional", "Pop", "Synth & Wave"],
}


def base_genre(entry: str) -> str:
    """`Psychedelic Glitch Hop` -> `Glitch Hop`; `Cyber Ethio-Jazz` -> `Ethio-Jazz`."""
    tokens = entry.split()
    while len(tokens) > 1 and tokens[0] in MODIFIERS:
        remainder = tokens[1:]
        if len(remainder) == 1 and remainder[0] in NEVER_A_BASE:
            break
        tokens = remainder
    return " ".join(tokens)


def territory_of(base: str) -> str | None:
    if base in EXACT_TERRITORY:
        return EXACT_TERRITORY[base]
    low = base.lower()
    for territory, tokens in TOKEN_RULES:
        for tok in tokens:
            t = tok.lower()
            if t.endswith(" "):
                if t in low + " ":
                    return territory
            # Plain word boundaries, NOT hyphen-aware: `Anarcho-Punk`, `Post-Metal`
            # and `P-Funk` must reach their family, and a hyphen-excluding
            # lookaround silently sent all three to the unmapped list.
            elif re.search(rf"(?<!\w){re.escape(t)}(?!\w)", low):
                return territory
    return None


def read_genres() -> list[str]:
    if not GENRES.exists():
        raise SystemExit(f"ERROR: {GENRES} absent — cannot build territories")
    entries = [ln.strip() for ln in GENRES.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not entries:
        raise SystemExit(f"ERROR: {GENRES} yielded 0 entries — empty extraction, not a result")
    return entries


def build_territories() -> tuple[dict[str, list[str]], list[str]]:
    """-> ({territory: [base genres]}, unmapped). Refuses to guess."""
    entries = read_genres()
    bases = Counter(base_genre(e) for e in entries)
    grouped: dict[str, list[str]] = defaultdict(list)
    unmapped: list[str] = []
    for base in sorted(bases):
        terr = territory_of(base)
        if terr is None:
            unmapped.append(base)
        else:
            grouped[terr].append(base)
    return dict(grouped), unmapped


# ---------------------------------------------------------------------- footprint

_GENRE_KEYS = re.compile(
    r"^(genre_tags|style_axes|seed_genre|genres|genre|suno_tags|tags|signature_style"
    r"|emerging_genre)\s*:", re.IGNORECASE)


def _song_texts() -> list[str]:
    """Frontmatter genre/style text from output/songs/*.md.

    Only the genre-bearing keys and the indented block that belongs to them. An
    earlier version kept every indented line in the frontmatter, which swept in
    `evaluation_facets` and `qa_flags` and turned the footprint into a match
    against the whole QA record.
    """
    out = []
    if not SONGS.is_dir():
        return out
    for p in sorted(SONGS.glob("*.md")):
        text = p.read_text(encoding="utf-8", errors="replace")
        m = re.match(r"^---\n(.*?)\n---", text, re.S)
        if not m:
            continue
        keep, inside = [], False
        for ln in m.group(1).splitlines():
            if _GENRE_KEYS.match(ln):
                keep.append(ln)
                inside = ln.rstrip().endswith(":")
                continue
            if inside and (ln.startswith("  ") or ln.startswith("\t")):
                keep.append(ln)
                continue
            inside = False
        out.append("\n".join(keep).lower())
    return out


def _clip_texts() -> list[str]:
    if not SNAPSHOT.exists():
        return []
    data = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    rows = data.get("rows", data if isinstance(data, list) else [])
    return [str(r.get("tags") or "").lower() for r in rows]


_FOOTPRINT_MIN_LEN = 4


def footprints(grouped: dict[str, list[str]]) -> dict[str, tuple[int, int]]:
    """Coarse evidence that we have already worked in a territory.

    Word-boundary matches of BASE GENRE names (>= 4 chars, so `Dub`/`Bop`/`Ska`
    cannot vacuum up half the catalogue) against song frontmatter and snapshot
    style text. Coarse on purpose and stamped GENERATED; engagement numbers stay
    in `apps/lofn-analytics`, which is their lane.

    ⛔ Territory NAMES are deliberately not matched. An earlier version added
    them and `Vocal Harmony Traditions` promptly became the most-worked territory
    on the map, on the strength of the word "vocal" — and the planner, which
    orders by footprint, sent the first plate there. A footprint that counts a
    generic English word is worse than no footprint.
    """
    songs = [t for t in _song_texts() if t.strip()]
    clips = [t for t in _clip_texts() if t.strip()]
    # ⛔ Assert the cardinality before trusting the conclusion. A footprint of
    # zero everywhere and a broken extractor look identical in the output table.
    if not songs or not clips:
        raise SystemExit(
            f"ERROR: extracted genre text from {len(songs)} song(s) and {len(clips)} clip(s). "
            "An empty extraction is an ERROR, never a footprint of zero — check "
            "output/songs/ frontmatter keys and data/snapshots/suno/latest.json.")
    print(f"[EXTRACT] {len(songs)} song(s) with genre frontmatter · {len(clips)} clip(s) "
          f"with style text", file=sys.stderr)
    out: dict[str, tuple[int, int]] = {}
    for terr, bases in grouped.items():
        needles = [b.lower() for b in bases if len(b) >= _FOOTPRINT_MIN_LEN]
        pattern = re.compile("|".join(rf"(?<![\w-]){re.escape(n)}(?![\w-])"
                                      for n in sorted(set(needles)))) if needles else None
        if pattern is None:
            out[terr] = (0, 0)
            continue
        out[terr] = (sum(1 for t in songs if pattern.search(t)),
                     sum(1 for t in clips if pattern.search(t)))
    return out


# -------------------------------------------------------------------- spec classes

def frame_categories() -> list[str]:
    """`vault/music_frames.csv` opens with a blank line and is CRLF — parse with
    utf-8-sig and lstrip, or the header is read as a data row and every category
    comes back empty."""
    if not FRAMES.exists():
        raise SystemExit(f"ERROR: {FRAMES} absent")
    text = FRAMES.read_text(encoding="utf-8-sig").lstrip()
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows or "Category" not in (rows[0].keys() if rows else {}):
        raise SystemExit("ERROR: music_frames.csv parsed to 0 usable rows — empty extraction")
    cats: list[str] = []
    for r in rows:
        c = (r.get("Category") or "").strip()
        if c and c not in cats:
            cats.append(c)
    if not cats:
        raise SystemExit("ERROR: music_frames.csv yielded 0 categories")
    return cats


# (class, lane, definition, measurable_by, seed_evidence)
SPEC_CLASSES = [
    ("steady-tempo", "NUMERIC",
     "A single BPM target named in the style prompt, held for the whole track.",
     "`scripts/measure_render.py` → `tempo_candidates()` (10 ms hop)",
     "behaviour ledger n=6 — SK-001"),
    ("tempo-dramaturgy", "NUMERIC",
     "A programmed tempo CHANGE across the arrangement (e.g. 92 → 140).",
     "`scripts/measure_render.py` → `tempo_candidates()` over windows",
     "behaviour ledger n=2 — SK-002"),
    ("near-silent-opening", "NUMERIC",
     "The first seconds sit far below the body of the track.",
     "`scripts/measure_render.py` → `envelope()` head level",
     "behaviour ledger n=3 — SK-003"),
    ("terminal-silence", "NUMERIC",
     "Silence that FINISHES the arrangement.",
     "`scripts/measure_render.py` → `gaps()` at the tail",
     "behaviour ledger n=1 — SK-004"),
    ("interrupting-silence", "NUMERIC",
     "A long void in the MIDDLE of the arrangement. The standing negative control.",
     "`scripts/measure_render.py` → `gaps()` inside the body",
     "behaviour ledger n=2 — SK-005"),
    ("dynamics-arc", "NUMERIC",
     "A shaped loudness trajectory: where the piece gets quiet and where it lifts.",
     "`scripts/measure_render.py` → `envelope()`, `deepest_dip()`",
     "grain-law baseline, L22"),
    ("sub-build", "NUMERIC",
     "Progressive low-end growth into a late section.",
     "`scripts/measure_render.py` → `band_profile()` 20–80 Hz",
     "behaviour ledger n=1 — SK-006"),
    ("stereo-staging", "NUMERIC",
     "Declared left/right/centre placement of named elements.",
     "`scripts/measure_render.py` → `stereo()` correlation",
     "behaviour ledger n=8 — SK-012"),
    ("width-vs-level", "NUMERIC",
     "Whether the image widens or narrows as the track gets louder.",
     "`scripts/measure_render.py` → `width_vs_level()`",
     "behaviour ledger n=1 — SK-013 (INWARD COSMOS)"),
    ("hard-panned-nonmusical", "BOTH",
     "A non-musical element parked hard left or right.",
     "`scripts/measure_render.py` → `stereo()` plus the blind pass",
     "behaviour ledger n=2 — SK-011"),
    ("untuned-drone", "NUMERIC",
     "A sustained untuned tone: mains hum, motor, quarter-tone pair.",
     "`scripts/measure_render.py` → `tonal_components()`, `interval_pairs()`",
     "behaviour ledger n=2 — SK-010"),
    ("harmonic-color", "BOTH",
     "A named harmonic/tuning choice: modal colour, microtonality, a specific key move.",
     "`scripts/measure_render.py` → `tonal_components()` plus the blind pass",
     "unprobed"),
    ("named-timbre", "LISTENER",
     "A specific instrument or timbre asked for by name.",
     "SKADI-BLIND-v1 Q10 (technique salience) and the intent turn",
     "behaviour ledger n=3 — SK-009"),
    ("foley", "LISTENER",
     "A literal real-world prop sound: boot crunch, chair drag, printer chirp.",
     "SKADI-BLIND-v1 Q6/Q7 and the intent turn",
     "behaviour ledger n=4 — SK-007"),
    ("texture-density", "BOTH",
     "How many simultaneous surfaces the arrangement carries.",
     "`scripts/measure_render.py` → `band_profile()` plus the blind pass",
     "unprobed"),
    ("vocal-staging", "BOTH",
     "Vocal distance, doubling, dryness and position.",
     "`scripts/measure_render.py` → `stereo()` plus SKADI-BLIND-v1 Q5",
     "grain-law baseline, L22"),
    ("opposed-vocal-registers", "LISTENER",
     "Two strongly opposed vocal characters inside one song.",
     "SKADI-BLIND-v1 Q5 and the intent turn",
     "behaviour ledger n=2 — SK-008"),
    ("prosody-device", "LISTENER",
     "A sound-level lyric device: rhyme scheme, alliteration braid, syllable shape.",
     "`scripts/measure_soundcraft.py` → `profile_file()` on the ask; blind pass on arrival",
     "THE RETURN, lofn-music L21"),
    ("form-return", "BOTH",
     "Whether the specified return actually returns: chorus, refrain, rondo.",
     "`scripts/measure_soundcraft.py` → `profile_file()`; blind pass structure read (Q1)",
     "THE RETURN, lofn-music L21"),
    ("section-tag-obedience", "BOTH",
     "Whether bracketed section headers produce the sections they name.",
     "snapshot `section_count` / `unique_sections`; blind pass Q1",
     "unprobed — the `section` channel has no claim yet"),
    ("lyric-form", "LISTENER",
     "Whether an unusual lyric FORM survives as an audible shape.",
     "SKADI-BLIND-v1 Q1/Q8 and the intent turn",
     "unprobed"),
    ("genre-arrival", "LISTENER",
     "Whether a blind listener names the territory we asked for. Territory coverage "
     "is accounted here.",
     "SKADI-BLIND-v1 Q9 (name the genre or scene in your own words)",
     "unprobed — the reason Q9 was added to the render-audit eight"),
    ("exclude-obedience", "BOTH",
     "Whether naming a thing in the EXCLUDE field removes it.",
     "`scripts/measure_render.py` where the excluded thing is measurable; blind pass otherwise",
     "unprobed — the `exclude` channel has no claim yet"),
    ("blend-of-two", "LISTENER",
     "OPERATOR. Two genres asked for at once: does the blend hold both parents?",
     "SKADI-BLIND-v1 Q9 (does the listener name one parent, both, or neither)",
     "2026-08-13 inversion law — SKOP-01"),
    ("modifier-application", "LISTENER",
     "OPERATOR. A `<Modifier> <Genre>` compound from `vault/genres.txt`: does the "
     "modifier survive, or does the base eat it?",
     "SKADI-BLIND-v1 Q9/Q10",
     "unprobed — the operator program's second law"),
]


# ------------------------------------------------------------------- seed claims
# Imported verbatim in substance from the n=8 SUNO BEHAVIOUR LEDGER in
# `.claude/skills/lofn-render-audit/SKILL.md`. n is kept; confidence is mapped
# CONSERVATIVELY DOWN into the three-level atlas vocabulary (see SEED_NOTES).

DEEP = f"{DEEP_DIVE}/RENDER_AUDIT.md"
MEAS = f"{DEEP_DIVE}/measurements.json"
NOTES = "vault/music_atlas/seeds/SEED_NOTES.md"
BLIND = {
    "triple": f"{DEEP_DIVE}/triple-arch-over-me/blind_listener.json",
    "locked": f"{DEEP_DIVE}/it-wasnt-even-locked/blind_listener.json",
    "cell": f"{DEEP_DIVE}/the-cell-named-after-no-one/blind_listener.json",
    "harder": f"{DEEP_DIVE}/harder-to-reach/blind_listener.json",
    "burning": f"{DEEP_DIVE}/burning-bush-with-a-barcode/blind_listener.json",
    "cat": f"{DEEP_DIVE}/the-caterpillar/blind_listener.json",
    "start": f"{DEEP_DIVE}/start-with-one/blind_listener.json",
    "beautiful": f"{DEEP_DIVE}/the-beautiful-thing-i-cut/blind_listener.json",
}
INTENT = {k: v.replace("blind_listener.json", "intent_comparison.json") for k, v in BLIND.items()}

# (tier, id, kind, scope, channel, lane, claim, verdict, n, confidence, models,
#  [evidence], falsifier)
SEED_CLAIMS = [
    ("BANKED", "SK-001", "CELL", "any territory × steady-tempo", "style", "NUMERIC",
     "A single BPM target named in the style prompt ARRIVES within 5 BPM (110→109, 132→133).",
     "ARRIVES", 6, "MEDIUM", "v4.5", [MEAS, DEEP, NOTES],
     "A render whose measured tempo sits more than 5 BPM from a single named target."),
    ("BANKED", "SK-002", "CELL", "any territory × tempo-dramaturgy", "style", "NUMERIC",
     "A programmed tempo TRANSFORMATION (92→140, 84→128) is SMOOTHED to one steady grid.",
     "SMOOTHED", 2, "MEDIUM", "v4.5", [MEAS, DEEP, NOTES],
     "A render that measurably holds two different tempos in the sections that asked for them."),
    ("BANKED", "SK-003", "CELL", "any territory × near-silent-opening", "style", "NUMERIC",
     "A near-silent opening ARRIVES when asked (−46.8 dB); it also appears unrequested.",
     "ARRIVES", 3, "MEDIUM", "v4.5", [MEAS, DEEP, NOTES],
     "A requested near-silent opening that measures within a few dB of the track body."),
    ("OBSERVED", "SK-004", "CELL", "any territory × terminal-silence", "style", "NUMERIC",
     "Silence that FINISHES the arrangement ARRIVES — 2.0 s measured.",
     "ARRIVES", 1, "LOW", "v4.5", [MEAS, DEEP, NOTES],
     "A requested terminal silence filled with music or fade."),
    ("BANKED", "SK-005", "CELL", "any territory × interrupting-silence", "style", "NUMERIC",
     "A long void in the MIDDLE of the arrangement is SMOOTHED — both explicit voids vanished. "
     "Silence is not one behaviour: terminal survives, interrupting gets filled.",
     "SMOOTHED", 2, "MEDIUM", "v4.5", [MEAS, DEEP, NOTES],
     "A measured interrupting gap at or near the requested length and depth."),
    ("OBSERVED", "SK-006", "CELL", "any territory × sub-build", "style", "NUMERIC",
     "A progressive sub build into the final chorus ARRIVES (+8.4 → +11.4 dB).",
     "ARRIVES", 1, "LOW", "v4.5", [MEAS, DEEP, NOTES],
     "A requested sub build whose 20–80 Hz band does not rise into the final section."),
    ("BANKED", "SK-007", "CELL", "any territory × foley", "style", "LISTENER",
     "Literal foley props (boot crunch, chair drag, scanner beep, printer chirp) are ABSENT or "
     "abstracted to a generic glitch.",
     "ABSENT", 4, "MEDIUM", "v4.5",
     [DEEP, BLIND["locked"], INTENT["locked"], BLIND["cell"], INTENT["cell"], NOTES],
     "A blind listener naming the specific prop that was asked for."),
    ("BANKED", "SK-008", "CELL", "any territory × opposed-vocal-registers", "style", "LISTENER",
     "Two strongly opposed vocal registers COLLAPSE to one lead plus ordinary layering.",
     "SMOOTHED", 2, "MEDIUM", "v4.5",
     [DEEP, BLIND["harder"], INTENT["harder"], BLIND["cat"], INTENT["cat"], NOTES],
     "A blind listener reporting two distinct vocal characters where two were asked for."),
    ("BANKED", "SK-009", "CELL", "any territory × named-timbre", "style", "LISTENER",
     "Named exotic timbres (HyperRaaga, tanpura, glass harmonica, cello, crystalline arps) are "
     "largely ABSENT; atmosphere is substituted.",
     "ABSENT", 3, "MEDIUM", "v4.5",
     [DEEP, BLIND["triple"], INTENT["triple"], BLIND["burning"], INTENT["burning"], NOTES],
     "A blind listener naming the specific requested instrument unprompted."),
    ("BANKED", "SK-010", "CELL", "any territory × untuned-drone", "style", "NUMERIC",
     "A sustained untuned drone or mains hum is ABSENT; a specified quarter-tone pair produced "
     "zero isolated tonal components.",
     "ABSENT", 2, "LOW", "v4.5", [MEAS, DEEP, NOTES],
     "A measured isolated tonal component sitting at the requested untuned pitch."),
    ("BANKED", "SK-011", "CELL", "any territory × hard-panned-nonmusical", "style", "NUMERIC",
     "A hard-panned non-musical element is ABSENT; it collapses into a near-centre image.",
     "ABSENT", 2, "LOW", "v4.5", [MEAS, DEEP, NOTES],
     "A measured hard-panned band carrying the requested non-musical element."),
    ("BANKED", "SK-012", "CELL", "any territory × stereo-staging", "style", "NUMERIC",
     "The renderer is NOT universally mono: measured L/R correlation spans 0.812–0.945 across "
     "eight tracks.",
     "PARTIAL", 8, "MEDIUM", "v4.5", [MEAS, DEEP, NOTES],
     "A corpus of renders measuring at or above 0.99 correlation throughout."),
    ("OBSERVED", "SK-013", "CELL", "any territory × width-vs-level", "style", "NUMERIC",
     "A specified \"chorus widens\" INVERTS globally (r = −0.43): the image NARROWS into the "
     "loud sections while local layer-spread stays audible. Judged the better result — INWARD "
     "COSMOS is a technique now, not a defect.",
     "INVERTED", 1, "LOW", "v4.5", [MEAS, DEEP, NOTES],
     "A render whose width/level correlation is positive where widening was asked for."),
    ("CONJECTURED", "SKOP-01", "OPERATOR",
     "blend/2 (modern_world, historical_intrusion) — one seam", "style", "LISTENER",
     "THE INVERSION. A two-genre blend holds BOTH parents when the MODERN genre is named as the "
     "world and the historical element enters as ONE named intrusion at ONE seam, with the "
     "singer doing something across it; naming the historical bed as the dominant collapses the "
     "song into the historical genre. N = 1 world + 1 intrusion + 1 seam.",
     # ⛔ A CONJECTURED ROW IS A PREDICTION. `verdict` is the verdict we EXPECT a
     # blind pass to return; `n` is 0 because nothing has been observed through
     # this system's instrument, and `confidence` is empty because a prediction
     # carries a falsifier instead. The six-song run is still cited in
     # `evidence` — it is what MOTIVATES the prediction, and SEED_NOTES §3 says
     # in as many words that it does not confirm it.
     "TRANSFORMS", 0, "—", "v4.5",
     ["output/2026-08-13_genre-blend-refine/INDEX.md",
      "output/2026-08-13_genre-blend-refine/00_REFINEMENT_BRIEF.md", NOTES],
     "A blend rendered in the inverted order that a blind listener still names by both parents "
     "— or a modern-world/historical-intrusion blend a blind listener names by the historical "
     "parent alone."),
]

# The two standing controls, named once here and read by skadi_plan.py.
POSITIVE_CONTROL = ("steady-tempo", "ARRIVES", "SK-001")
NEGATIVE_CONTROL = ("interrupting-silence", "SMOOTHED", "SK-005")

# The PRODUCTIVE DEVIATIONS section of the render-audit skill: harvest that was
# already recognised and kept. Seeded so the shelf is not empty on day one.
SEED_SCRUFF = [
    ("SCR-01", "2026-07-24",
     "UNBROKEN BUREAUCRACY — a requested rupture disappears and the dead-level continuity is "
     "more relentless than the break would have been.",
     "it-wasnt-even-locked", "PROMOTED"),
    ("SCR-02", "2026-07-24",
     "INWARD COSMOS — the mix narrows as level rises instead of widening, drawing the scale in "
     "around the singer.",
     "triple-arch-over-me", "PROMOTED"),
    ("SCR-03", "2026-07-24",
     "THE WOUND INSIDE THE MACHINE — a human voice left audible under a vocoder that was "
     "supposed to erase it.",
     "harder-to-reach", "PROMOTED"),
    ("SCR-04", "2026-07-24",
     "GRIEF THE CLERK WITHHELD — harmonies supply the feeling a flat delivery was instructed to "
     "refuse.",
     "the-cell-named-after-no-one", "PROMOTED"),
    ("SCR-05", "2026-07-24",
     "THE HAUNTED CHORUS — unrequested vocal layers turn a private metamorphosis into a small "
     "crowd.",
     "the-caterpillar", "PROMOTED"),
    ("SCR-06", "2026-07-24",
     "SINGULARITY BY GLITCH — an ordinary pop stutter becomes the one production event that "
     "enacts the word \"one\".",
     "start-with-one", "PROMOTED"),
]


# ----------------------------------------------------------------------- rendering

def row(cells: list) -> str:
    return "| " + " | ".join(str(c).replace("|", "\\|") for c in cells) + " |"


def table(cols: list[str], rows: list[list]) -> str:
    out = [row(cols), row(["---"] * len(cols))]
    out.extend(row(r) for r in rows)
    return "\n".join(out)


def banner(writer: str, reader: str) -> str:
    return f"<!-- generated by scripts/{writer}; read by scripts/validate_music_atlas.py ({reader}) -->"


def render_atlas() -> str:
    tiers = {"BANKED": [], "OBSERVED": [], "CONJECTURED": []}
    receipts = []
    for (tier, cid, kind, scope, channel, lane, claim, verdict, n, conf, models,
         evidence, falsifier) in SEED_CLAIMS:
        tiers[tier].append([cid, kind, scope, channel, lane, claim, verdict, n, conf,
                            models, ", ".join(f"`{p}`" for p in evidence), falsifier])
        receipts.append([cid, f"`{sha12(claim)}`", "SEED",
                         f"imported prior — provenance in `{NOTES}`"])

    parts = [
        banner("skadi_seed.py", "check_atlas_schema / check_append_blocks"),
        "",
        "# ATLAS — what Suno does through our door",
        "",
        f"{GENERATED}",
        "",
        "> **A menu, never an order. Renders are The Scientist's.** Nothing in this system "
        "renders, publishes, or spends.",
        "",
        "## CONSTITUTION",
        "",
        CONSTITUTION,
        "",
        "---",
        "",
        "Row schema (both claim kinds; OPERATOR rows put the operator signature in `scope`). "
        "`confidence` is capped by `n` — see `atlas_confidence_cap_n` in `vault/gates.yaml`. "
        "`BANKED` has an `n` floor — see `atlas_bank_min_n`. Every `evidence` path must exist; "
        "every `falsifier` names the observation that would refute the claim.",
        "",
        "## BANKED",
        "",
        table(ATLAS_COLS, tiers["BANKED"]),
        "",
        "## OBSERVED",
        "",
        table(ATLAS_COLS, tiers["OBSERVED"]),
        "",
        "## CONJECTURED",
        "",
        table(ATLAS_COLS, tiers["CONJECTURED"]),
        "",
        "---",
        "",
        "## APPEND BLOCKS",
        "",
        "Append-only. One block per expedition; corrections are NEW rows superseding old ids, "
        "with the old row struck through and kept. `sha12` is the digest of the claim text as "
        "written — a claim quietly reworded after banking breaks its own receipt.",
        "",
        f"### {SEED_DATE} {SEED_SLUG} additions",
        "",
        table(APPEND_COLS, receipts),
        "",
    ]
    return "\n".join(parts)


def render_territories(grouped: dict[str, list[str]], prints: dict[str, tuple[int, int]]) -> str:
    rows = []
    for terr in sorted(grouped):
        bases = grouped[terr]
        songs, clips = prints.get(terr, (0, 0))
        adjacent = ", ".join(ADJACENCY.get(terr, [])) or "(pending curation)"
        shown = ", ".join(bases[:12]) + (f", … (+{len(bases) - 12} more)" if len(bases) > 12 else "")
        rows.append([terr, shown, adjacent, "UNEXPLORED",
                     f"songs:{songs} · clips:{clips}", NO_REASON_YET])
    return "\n".join([
        banner("skadi_seed.py", "check_territory_states"),
        "",
        "# TERRITORIES — the coarse genre regions the map is accounted in",
        "",
        GENERATED,
        "",
        f"Derived from `vault/genres.txt` ({len(grouped)} territories over "
        f"{sum(len(v) for v in grouped.values())} base genres, stripped of the documented "
        "modifier vocabulary). **Leaf genres are sample material inside a territory, never a "
        "coverage obligation** — the infinite compound space is explored through OPERATOR "
        "claims, not by enumerating cells.",
        "",
        "`footprint` counts prior work only: songs whose frontmatter genre/style text matches a "
        "base genre, and public Suno clips whose style prompt does. **A footprint is not a "
        "probe** — every territory below is `UNEXPLORED` because Skaði has run no expedition "
        "yet, and UNEXPLORED is never the same as absent. Engagement and like-rate stay in "
        "`apps/lofn-analytics`; see lofn-analytics.",
        "",
        table(TERRITORY_COLS, rows),
        "",
    ])


def render_spec_classes(categories: list[str]) -> str:
    rows = [[c, lane, definition, measurable, seed] for c, lane, definition, measurable, seed
            in SPEC_CLASSES]
    return "\n".join([
        banner("skadi_seed.py", "check_spec_classes"),
        "",
        "# SPEC CLASSES — the technique/behaviour families, each with a declared lane",
        "",
        GENERATED,
        "",
        "`lane` decides which instrument owns the verdict. **Presence/absence: numbers win. "
        "Quality/meaning: the listener wins.** A `LISTENER` claim that cites no blind artifact "
        "is refused by `check_lanes` — the blind pass is the only instrument here that can "
        "return NO, and a gate that cannot fail is a sentence.",
        "",
        table(SPEC_CLASS_COLS, rows),
        "",
        "---",
        "",
        "## RAW MATERIAL — `vault/music_frames.csv` categories",
        "",
        f"{len(categories)} mixed-axis categories (instruments, production, genres, traditions) "
        "× 25 techniques each. These are **sample material for probe design**, not spec classes "
        "and not territories: a category like `Drums` names an instrument, `Glitch` names a "
        "genre, and `Field Recording` names a practice. Kept here so probe authors can shop "
        "for concrete techniques inside a cell.",
        "",
        "> " + " · ".join(categories),
        "",
    ])


def render_frontier(grouped: dict[str, list[str]]) -> str:
    coverage = [[t, "UNEXPLORED", 0, 0, "—", NO_REASON_YET] for t in sorted(grouped)]

    pos_class, pos_expect, pos_id = POSITIVE_CONTROL
    neg_class, neg_expect, neg_id = NEGATIVE_CONTROL
    menu = [
        ["P1", f"any territory × {pos_class}", "style",
         f"Standing POSITIVE control: re-render the steady-tempo ask and expect {pos_expect}.",
         f"control due — extends the control chart and re-validates {pos_id} in this epoch",
         "PROPOSED"],
        ["P2", f"any territory × {neg_class}", "style",
         f"Standing NEGATIVE control: ask for a long mid-song void and expect {neg_expect}.",
         f"control due — a plate whose controls hold is a plate that may bank; re-validates {neg_id}",
         "PROPOSED"],
        ["P3", "any territory × blend-of-two", "style",
         "Replicate THE INVERSION under a blind listen: modern genre as the world, historical "
         "element as one named intrusion at one seam.",
         "moves SKOP-01 off CONJECTURED — a registered prediction with no blind pass behind "
         "it, already steering runs",
         "PROPOSED"],
        ["P4", "any territory × exclude-obedience", "exclude",
         "Name a measurable thing in the EXCLUDE field and measure whether it leaves.",
         "opens the `exclude` channel — it has zero claims",
         "PROPOSED"],
        ["P5", "any territory × genre-arrival", "style",
         "Ask for one territory plainly and let the blind listener name what it heard (Q9).",
         "opens a territory — every territory on the map is UNEXPLORED",
         "PROPOSED"],
        ["P6", NON_OBVIOUS_PLACEHOLDER, "style", NON_OBVIOUS_PLACEHOLDER,
         "the non-obvious slot: without it the search collapses onto the prior",
         "PROPOSED"],
    ]

    scruff = [[sid, seen, desc, sightings, status]
              for sid, seen, desc, sightings, status in SEED_SCRUFF]

    chart = [
        [pos_class, "2026-07-24-eight-song-deep-dive (pre-Skaði baseline)", pos_expect,
         "6 of 6 within 5 BPM", "v4.5"],
        [neg_class, "2026-07-24-eight-song-deep-dive (pre-Skaði baseline)", neg_expect,
         "2 of 2 explicit voids filled", "v4.5"],
    ]

    return "\n".join([
        banner("skadi_seed.py",
               "check_coverage / check_scruff / check_control_chart / check_delta_log / "
               "check_consumption"),
        "",
        "# FRONTIER — read at Phase −1 and Phase 0 of every music run",
        "",
        GENERATED,
        "",
        "> **A menu, never an order. Renders are The Scientist's.**",
        "",
        "## COVERAGE",
        "",
        table(COVERAGE_COLS, coverage),
        "",
        "## MENU",
        "",
        "What each probe would move on the map, priced in map arithmetic only — never in taste "
        "and never in money. The non-obvious slot is printed as a literal placeholder because "
        f"**the planner may not fill it**: `{NON_OBVIOUS_PLACEHOLDER}`.",
        "",
        table(MENU_COLS, menu),
        "",
        "## SCRUFF SHELF",
        "",
        "Unrequested deviations, accumulated and compared across probes. Repetition is the "
        "rift and the rift is the map's most valuable object — `check_scruff` FLAGS it, never "
        "fails it. The six rows below predate Skaði and carry song slugs rather than probe ids.",
        "",
        table(SCRUFF_COLS, scruff),
        "",
        "## CONTROL CHART",
        "",
        "The same control probes re-rendered across expeditions form a control chart on the "
        "generator itself. An out-of-limits point rules off the chart and splits the map by "
        "instrument epoch — history is never deleted. A version string is a claim, not a fact.",
        "",
        table(CONTROL_CHART_COLS, chart),
        "",
        "## DELTA LOG",
        "",
        "Append-only, exactly one row per closed expedition. `INSTRUMENT-ONLY` and `NONE` are "
        "blameless: a wall-map is a result. Only a STREAK trips anything, and what it trips is "
        "a recorded human re-authorization, never a score.",
        "",
        table(DELTA_COLS, []),
        "",
        "## CONSUMPTION LOG",
        "",
        "Which later runs cited which atlas rows. The atlas serves the ear; if this stays "
        "empty the atlas is shelf-dust and that is a question for The Scientist.",
        "",
        table(CONSUMPTION_COLS, []),
        "",
    ])


def render_ledger() -> str:
    return "\n".join([
        banner("skadi_seed.py", "check_ledger_schema"),
        "",
        "# EXPEDITION LEDGER — one row per probe, append-only",
        "",
        GENERATED,
        "",
        "**Never rewritten by a summarizer.** `harvest` is mandatory and may be `—`; it may "
        "not be missing, because an empty harvest cell is how deviation capture quietly stops "
        "happening. `tier` is stamped from artifact presence by `scripts/skadi_bank.py` and by "
        "nothing else. A probe on a plate whose control failed carries `VOID-PLATE`.",
        "",
        "No expedition has run. This file is a header waiting for its first plate.",
        "",
        table(LEDGER_COLS, []),
        "",
    ])


def render_seed_notes(grouped: dict[str, list[str]], prints: dict[str, tuple[int, int]],
                      categories: list[str]) -> str:
    provenance = []
    for (tier, cid, kind, scope, channel, lane, claim, verdict, n, conf, models,
         evidence, falsifier) in SEED_CLAIMS:
        source = ("`.claude/skills/lofn-render-audit/SKILL.md` §THE SUNO BEHAVIOUR LEDGER — n=8"
                  if cid.startswith("SK-") else
                  "`output/2026-08-13_genre-blend-refine/` — the refinement pass that found "
                  "THE INVERSION")
        provenance.append([cid, tier, "yes", n, conf, source])

    total_bases = sum(len(v) for v in grouped.values())
    worked = sum(1 for t in grouped if prints.get(t, (0, 0))[0] or prints.get(t, (0, 0))[1])

    return "\n".join([
        banner("skadi_seed.py", "(read by humans; no machine check)"),
        "",
        "# SEED NOTES — where every seeded row came from, and what was left out",
        "",
        GENERATED,
        "",
        f"Seeded {SEED_DATE} by `scripts/skadi_seed.py`. **Nothing here was measured by "
        "Skaði.** Every row is a prior imported from work already paid for, and every seeded "
        "atlas row cites this file in its `evidence` column so the provenance is one click "
        "from the claim.",
        "",
        "## 1. THE BEHAVIOUR LEDGER, n=8",
        "",
        "Thirteen rows imported from the markdown table in "
        "`.claude/skills/lofn-render-audit/SKILL.md` (*THE SUNO BEHAVIOUR LEDGER — n=8, first "
        "full deep dive, 2026-07-24*). `n` is kept exactly as the ledger recorded it. That "
        "skill's table remains its own working buffer; ATLAS.md is the canonical accumulator "
        "from here. **The pointer edit in the skill is Lofn's to make, not a builder's.**",
        "",
        "**Confidence mapping — deliberately conservative.** The ledger uses a five-step scale "
        "(`low`, `low-medium`, `medium`, `medium-high`); the atlas has three "
        "(`LOW`/`MEDIUM`/`HIGH`). Mapping used: `low`→`LOW`, `low-medium`→`LOW`, "
        "`medium`→`MEDIUM`, `medium-high`→`MEDIUM`. Every hyphenated step maps DOWN, so **no "
        "seeded row is `HIGH`**. That is on purpose: a validator may lower a claim and never "
        "promote one, and only The Scientist promotes a claim to doctrine.",
        "",
        "**Two `n` values were recorded as inequalities** — foley `≥4` and named exotic timbres "
        "`≥3`. The atlas `n` column is an integer because `check_confidence_vs_n` has to read "
        "it, so the conservative lower bound is stored (4 and 3). The ledger's `≥` is recorded "
        "here and nowhere lost.",
        "",
        "**SK-012 verdict.** The ledger row reads *\"stereo image — spans 0.812–0.945 "
        "correlation, NOT a universal mono renderer\"*. That is a refutation of a prior claim, "
        "not an arrival of a requested thing, so it is banked as `PARTIAL` (genuine stereo "
        "movement exists; declared placement is not reliably honoured) rather than `ARRIVES`. "
        "Flagged for curation.",
        "",
        "## 2. THE STANDING CONTROLS",
        "",
        f"- **Positive control** — `{POSITIVE_CONTROL[0]}`, expected `{POSITIVE_CONTROL[1]}` "
        f"(from `{POSITIVE_CONTROL[2]}`, the strongest row in the ledger at n=6).",
        f"- **Negative control** — `{NEGATIVE_CONTROL[0]}`, expected `{NEGATIVE_CONTROL[1]}` "
        f"(from `{NEGATIVE_CONTROL[2]}`: both explicit voids vanished, 2 of 2).",
        "",
        "A plate whose control does not return its expected verdict is **VOID**: no claim banks "
        "from it and every ledger row on it records `VOID-PLATE`. The controls are also the "
        "control chart's series — the same probes re-rendered, so the points are comparable and "
        "a model-epoch change shows up as a special cause rather than as claims quietly rotting.",
        "",
        "## 3. THE 2026-08-13 INVERSION LAW — `SKOP-01`",
        "",
        "Seeded as an **OPERATOR** claim, and deliberately into **CONJECTURED** rather than "
        "BANKED. The evidence from that run is a *specification-level* finding: six songs were "
        "rewritten so the modern genre is the world and the historical element is one named "
        "intrusion at one seam, and the pass was judged on the text. **No blind listening pass "
        "was ever run on those renders.** Its lane is `LISTENER`, and a LISTENER-lane claim with "
        "no blind artifact is exactly what `check_lanes` exists to refuse — so it sits in "
        "CONJECTURED until a plate hears it.",
        "",
        "⛔ **The row is a PREDICTION, and its numbers say so: `n = 0`, `confidence` empty.** It "
        "was first seeded at `n=2, LOW`, carrying the six-song run's evidence as though that run "
        "had observed the law through this system's instrument. It had not. Those six renders "
        "were judged **on the text of the specification**, which is why the row is on this shelf "
        "at all — so the same file cannot also be the count that makes it look measured. The run "
        "paths stay in `evidence` because they are exactly what a stranger should re-open to see "
        "why we expect `TRANSFORMS`: they are **motivating, not confirming**. The predicted "
        "verdict is unchanged and stays TRANSFORMS-scoped as the panel stated it. A verdict "
        "column on this shelf holds what we expect to find, and being wrong there is the point "
        "of registering it.",
        "",
        "**It is nonetheless #1 in the replication queue** (menu slot P3), because it is already "
        "steering runs. That is the ERC post-mortem's lesson transplanted: a check scheduled "
        "behind the effort it was meant to condition is not a check. Until a HEARD pass lands, "
        "`n` stays 0 — the count is of hearings, and there have been none.",
        "",
        "## 4. THE CATALOGUE FOOTPRINT",
        "",
        f"`footprint` in TERRITORIES.md counts prior work: songs under `output/songs/` whose "
        f"frontmatter genre/style text matches a base genre of the territory, and public Suno "
        f"clips in `data/snapshots/suno/latest.json` whose `tags` text does. "
        f"{worked} of {len(grouped)} territories show any footprint at all.",
        "",
        "Matching is word-boundary and requires a name of at least "
        f"{_FOOTPRINT_MIN_LEN} characters, so short bases (`Dub`, `Bop`, `Ska`, `Emo`) cannot "
        "vacuum up half the catalogue. It is coarse on purpose. **A footprint is not a probe:** "
        "every territory is `UNEXPLORED` because Skaði has run nothing.",
        "",
        "## 5. THE EXISTING RENDER AUDITS",
        "",
        f"`{DEEP_DIVE}/` supplies the numeric evidence (`measurements.json`) and the blind "
        "listening artifacts (`<song>/blind_listener.json`, `<song>/intent_comparison.json`) "
        "that the LISTENER-lane seed rows cite. Three other audit directories exist "
        "(`2026-08-02-hear-learn`, `2026-08-10-spare-hair-tie`, `2026-08-10-wide-corpus`); they "
        "are calibration and corpus work rather than per-song intent comparisons, so they are "
        "**not** attached to claims here. Flagged for curation.",
        "",
        "## 6. ⛔ DELIBERATELY NOT SEEDED — the audience findings",
        "",
        "The token findings are **audience** findings, not grain claims, and seeding them here "
        "would put the wrong instrument on the wrong axis:",
        "",
        "- *sung-line density is the strongest correlate of like-rate* (r = +0.538, n = 72 with "
        "the Triple Arch outlier removed; then lyric_chars +0.402, duration +0.321, "
        "words_per_line −0.202)",
        "- *`phonk` 0.0217 and `baile` 0.0224 against a 0.0355 baseline*",
        "",
        "**The lane boundary.** Skaði measures what the GENERATOR does with an ask — a claim "
        "here is refuted by a render. Those findings measure what an AUDIENCE does with a "
        "finished song — they are refuted by a different catalogue, and they live in "
        "`apps/lofn-analytics` and `vault/BENCHMARKS.md`. Mixing them would let a popularity "
        "statistic wear a grain claim's evidence ladder.",
        "",
        "**And the class has already burned us once.** The standing finding *\"snarl and phonk "
        "are our two worst-performing style tokens\"* was retracted: it divided by a "
        "play-weighted baseline that was 93% one viral clip. Against a per-clip median, `snarl` "
        "sat AT baseline. That is the third instance of *a statistic dominated by one surface, "
        "generalised into a rule about the artist* — see `vault/RUN_LEDGER.md` and "
        "`data/snapshots/suno/ANALYSIS_2026-08-07_the_snarl_retraction.md`. An atlas that "
        "imported it would have been steering runs off a retraction.",
        "",
        "## 7. THE SCRUFF SHELF SEED — beyond §6 of the design",
        "",
        "The six PRODUCTIVE DEVIATIONS from `.claude/skills/lofn-render-audit/SKILL.md` are "
        "seeded onto the scruff shelf at status `PROMOTED`. The design's seeding list does not "
        "enumerate them; they are seeded anyway because harvest capture is mandatory "
        "(constitution §6) and a shelf that starts empty teaches the first expedition that the "
        "shelf is optional. **They carry song slugs, not probe ids** — they predate the probe "
        "protocol — so they can never reach the rift threshold on their own. Flagged for "
        "curation.",
        "",
        "## 8. TERRITORY CLUSTERING — how it was done and where it will be wrong",
        "",
        f"`vault/genres.txt` holds 1200 `<Modifier> <Genre>` compounds. {len(MODIFIERS)} leading "
        "tokens occur 20+ times and are treated as MODIFIERS; stripping them yields "
        f"{total_bases} base genres, hand-clustered into **{len(grouped)} territories** by an "
        "ordered rule list plus an explicit override table, both in `scripts/skadi_seed.py`. "
        "The mapping is deterministic and total: a base that no rule reaches makes the seed "
        "**refuse**, because a silent `(unsorted)` bucket is how a map starts lying.",
        "",
        "Known imperfections, all flagged for curation:",
        "",
        "- **Stripping loses real compounds.** `Garage Rock` becomes `Rock`, `Dream Pop` "
        "survives only because a rule names it. The compounds are not lost — they are the "
        "OPERATOR program's probe material.",
        "- **One guard was necessary.** `Psychedelic Glitch Hop` strips twice to the non-genre "
        "`Hop`; a `NEVER_A_BASE` set stops the loop one token early so it lands on `Glitch Hop`.",
        "- **`Mentor`** (5 entries) is almost certainly a corruption of **Mento**, the Jamaican "
        "form. Mapped to Afro-Caribbean & Tropical on that reading. If the file meant something "
        "else, this is wrong.",
        "- **`Columbia`** is read as the Cuban rumba style (it sits beside `Yambu` and "
        "`Guaguanco` in the same file), not the country.",
        "- **Cross-family compounds go to the noun, not the adjective**: `Jazz Metal` and `Folk "
        "Metal` land in Metal, `Gospel Blues` in Blues, `Ska Punk` in Ska. Defensible, "
        "arguable, and cheap to change.",
        "",
        "## 9. SPEC CLASSES",
        "",
        f"{len(SPEC_CLASSES)} classes, seeded from the behaviour ledger's own spec classes, "
        "`scripts/measure_render.py`'s measurement capabilities, and the "
        f"{len(categories)} categories of `vault/music_frames.csv`. The design asked for ~20–24; "
        "this is "
        f"{len(SPEC_CLASSES)} because the panel's OPERATOR program needs two classes of its own "
        "(`blend-of-two`, `modifier-application`) that no cell class covers. The frames "
        "categories are **mixed-axis** — `Drums` is an instrument, `Glitch` a genre, `Field "
        "Recording` a practice — so they are listed as probe-design raw material and are "
        "explicitly NOT territories and NOT spec classes.",
        "",
        "## 10. PROVENANCE PER ROW",
        "",
        table(["id", "tier", "seed", "n", "confidence", "source"], provenance),
        "",
    ])


# ---------------------------------------------------------------------------- main

def targets() -> list[tuple[Path, str]]:
    grouped, unmapped = build_territories()
    if unmapped:
        raise SystemExit(
            "ERROR: "
            f"{len(unmapped)} base genre(s) reached no territory rule — refusing to seed a map "
            "with a silent unsorted bucket:\n  " + "\n  ".join(unmapped[:40]))
    if not (40 <= len(grouped) <= 60):
        raise SystemExit(f"ERROR: clustered to {len(grouped)} territories; the design's band is "
                         "40–60. Adjust TERRITORY rules rather than the assertion.")
    prints = footprints(grouped)
    categories = frame_categories()
    return [
        (ATLAS_DIR / "ATLAS.md", render_atlas()),
        (ATLAS_DIR / "TERRITORIES.md", render_territories(grouped, prints)),
        (ATLAS_DIR / "SPEC_CLASSES.md", render_spec_classes(categories)),
        (ATLAS_DIR / "FRONTIER.md", render_frontier(grouped)),
        (ATLAS_DIR / "EXPEDITION_LEDGER.md", render_ledger()),
        (SEEDS_DIR / "SEED_NOTES.md", render_seed_notes(grouped, prints, categories)),
    ]


def audit() -> int:
    grouped, unmapped = build_territories()
    print(f"territories: {len(grouped)}   bases: {sum(len(v) for v in grouped.values())}   "
          f"unmapped: {len(unmapped)}")
    for t in sorted(grouped):
        print(f"  {len(grouped[t]):4d}  {t}")
        print(f"        {', '.join(grouped[t])}")
    if unmapped:
        print("\nUNMAPPED (the seed refuses while this list is non-empty):")
        for b in unmapped:
            print("  " + b)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Seed the SKAÐI music atlas state home.")
    ap.add_argument("--force", action="store_true",
                    help="rewrite files whose content has drifted from the seed")
    ap.add_argument("--audit", action="store_true",
                    help="print the territory clustering and exit; writes nothing")
    ap.add_argument("--dry-run", action="store_true", help="report what would change; write nothing")
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass

    # Reading the gates makes this script a gates reader, and the doctrine check
    # counts readers rather than mentions.
    gates, warnings = load_gates()
    for w in warnings:
        print(f"[GATES] {w}", file=sys.stderr)
    slots = int(gates.get("atlas_probe_slots", 6))

    if args.audit:
        return audit()

    plan = targets()
    menu_slots = len([r for r in render_frontier(build_territories()[0]).splitlines()
                      if re.match(r"^\|\s*P\d+\s*\|", r)])
    if menu_slots != slots:
        raise SystemExit(f"ERROR: seeded menu has {menu_slots} slots; gates atlas_probe_slots "
                         f"is {slots}")

    written, unchanged, refused = [], [], []
    for path, content in plan:
        rel = path.relative_to(REPO).as_posix()
        if path.exists():
            current = path.read_text(encoding="utf-8", errors="replace")
            if current == content:
                unchanged.append(rel)
                continue
            if not args.force:
                refused.append(rel)
                continue
        if args.dry_run:
            written.append(rel + " (dry-run)")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        written.append(rel)

    for rel in written:
        print(f"WROTE     {rel}")
    for rel in unchanged:
        print(f"UNCHANGED {rel}")
    for rel in refused:
        print(f"REFUSED   {rel} — already exists and differs from the seed; "
              f"pass --force to overwrite (this file accumulates writes after seeding)")
    if refused:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
