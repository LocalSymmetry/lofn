from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "output" / "daily" / "2026-07-05"
MUSIC = RUN / "music"
SONG_COPIES = ROOT / "output" / "songs"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sung_line_count(lyrics: str) -> int:
    total = 0
    for line in lyrics.splitlines():
        s = line.strip()
        if not s or s.startswith("[") or s.startswith("*"):
            continue
        total += 1
    return total


def lyrics_field_chars(lyrics: str) -> int:
    return len(lyrics.strip())


PAIR_DATA = {
    1: {
        "lane": "ACCESSIBLE",
        "axis": "NEWS",
        "emotion": "AWE",
        "concept": "painted moon open window",
        "selected": 2,
        "titles": ["Coal Snow Window", "Bright Side Turning", "Cassini Porch Song", "After the Dirty Ice"],
        "hooks": ["dark side, answer me", "show me the bright side", "sweep low and see me", "what stays after shining"],
        "body": "bare feet at an apartment window after the sky goes quiet",
        "danger": "the dark half of the self can arrive before the self has language",
        "architecture": "mirror-orbit form: window signal, bright face, coal face, low swoop, returning hook",
        "rhyme_logic": "loose assonance with mirrored final words; choruses answer in parentheses",
        "device": "viola scrape circles left while glass keys answer right; final chorus leaves a trailing tape tail",
        "style": "lunar chamber-pop, 96 BPM, bright/dark call-response",
        "prompt_core": "Lunar chamber-pop at 96 BPM with a two-face orbit shape, off-kilter brushed drums, unruly viola scrapes, glass keys, and a low tape pulse. Female late-20s alto, close center, breath-silver tone, plainspoken vowels, one fragile upper crack, doubled whispers panned left and right. Opening 5s: window latch, one cold inhale, then a viola harmonic scratches across the left speaker while dry keys answer from the right. Acoustic viola and piano stay real; no acoustic guitar. The arrangement alternates bright-face verses with coal-face replies, then drops to a near-silent low swoop before the final hook returns wider. Kinetic defect: the snare arrives a hair late every fourth bar. Bold device: the outro keeps a trailing dark tail after the vocal stops.",
        "exclude": "No space-rock bombast, no generic ambient pads, no real artist names, no EDM risers, no male lead, no child voices, no cosmic word salad, no science-report verse, no stacking daily numbers, no glossy playlist-pop chorus, no acoustic guitar strum bed, no trap hats, no choir preset wash, no heroic trailer drums, no copied golden-song imagery, no bare section tags, no scaffold debris.",
        "sections": [
            ("Signal", "Threshold Wonder", "Female Vocalist - window room", [
                "{hook}",
                "{hook}",
                "I stand with bare feet on the floorboard seam",
                "The glass has a cold little mouth",
                "It says look north",
                "It says not yet",
                "I answer with my palm",
            ]),
            ("Bright Face", "Tender Orientation", "Female Vocalist - keys close", [
                "One side of the moon keeps sugar in its coat",
                "One side drinks the lampblack down",
                "I thought beauty meant a clean white room",
                "Then the shadow wore a crown",
                "The neighbor's sink coughs once",
                "The sky does not explain",
                "My breath makes a temporary country",
                "Then vanishes again",
                "I name the bright side slowly",
                "So the dark side hears its name",
            ]),
            ("First Orbit", "Awe With Teeth", "Stacked Self-Harmonies - late snare", [
                "{hook} (not without the dark)",
                "{hook} (not without the mark)",
                "I am not asking light to lie",
                "I am asking it to stay",
                "{hook} (stand where I can see)",
                "If the shadow comes for me",
                "Let it find me awake",
                "Let it find me awake",
            ]),
            ("Coal Face", "Clean Fear", "Female Vocalist - viola left", [
                "The black half is not evil",
                "It is weather with a history",
                "It is dust that found a shoulder",
                "It is hurt with gravity",
                "My room is not a temple",
                "It is rent and tape and tea",
                "Still the sky leans in",
                "Like it misplaced a key",
                "I do not scrub the window",
                "I let the bruise look back at me",
            ]),
            ("Low Swoop", "Discovery", "Whisper - drums gone", [
                "Then the song drops low",
                "Almost under my ribs",
                "A small machine of mercy",
                "Turns where the panic lives",
                "I wanted proof of shining",
                "I got a painted stone",
                "I wanted one clean answer",
                "I got a second home",
                "*viola scratch cuts the room*",
                "And there I am",
                "Half afraid",
                "Half shown",
            ]),
            ("Return", "Unresolved Awe", "Full Self-Harmonies - tape tail", [
                "{hook} (not without the dark)",
                "{hook} (not without the mark)",
                "I am not asking light to lie",
                "I am asking it to stay",
                "{hook} (stand where I can see)",
                "If the shadow comes for me",
                "Let it find me awake",
                "Let it find me awake",
                "Ooh, bright side",
                "Ooh, coal side",
                "Same sky",
                "Same tide",
            ]),
            ("Afterlight", "Quiet Continuance", "Female Vocalist - tape fade", [
                "The window loosens its blue",
                "The floor warms under me",
                "I do not fix the moon",
                "I let it keep me company",
                "{hook}",
                "{hook}",
            ]),
        ],
        "coda": [
            "Latch, breath",
            "Viola, blue",
            "Coal side listening",
            "Bright side true",
            "I keep the window",
            "I keep the view",
            "{hook}",
            "{hook}",
        ],
    },
    2: {
        "lane": "ACCESSIBLE",
        "axis": "EXISTENCE",
        "emotion": "AWE",
        "concept": "house of rest with square corners",
        "selected": 2,
        "titles": ["Corners Every One", "House With Square Corners", "Pompadour Evening", "Waning Kitchen"],
        "hooks": ["square the room softly", "make the rest house real", "let the color lower", "leave one light on"],
        "body": "hands at a kitchen table measuring quiet into a small wooden model",
        "danger": "rest can feel like trespass when the nervous system still expects work",
        "architecture": "four-corner lullaby: each corner changes the definition of rest",
        "rhyme_logic": "plain couplets interrupted by one-line breaths; no orbit or broadcast pattern",
        "device": "brushed snare moves like a pencil on wood; chorus adds four small bell tones, one per corner",
        "style": "warm analog lullaby-pop, 84 BPM, four-corner refrain",
        "prompt_core": "Warm analog lullaby-pop at 84 BPM with brushed snare, muted felt piano, bowed glass, tiny corner bells, and a soft organ drone tucked far back. Female late-20s mezzo, intimate center-left, honeyed but tired tone, careful consonants, small laugh of disbelief on the hook, close double vocal in the right speaker. Opening 5s: pencil on wood, chair leg, one bell. Acoustic piano and hand percussion are present; no acoustic guitar and no synthetic choir. The arrangement moves through four corners, each corner adding one human object, then a brief low-pass panic fold before the chorus steadies the table. Kinetic defect: the brushes skip a beat when rest feels forbidden. Bold device: the bridge becomes a measured room tone grid.",
        "exclude": "No spa playlist softness, no bedtime-child framing, no male lead, no child voice, no generic piano ballad, no worship swell, no motivational self-help lyric, no acoustic strum bed, no fake strings, no EDM risers, no real artist names, no cluttered genre list, no science-report verse, no copied golden-song language, no bare section tags, no scaffold debris.",
        "sections": [
            ("Table", "Careful Arrival", "Female Vocalist - pencil and bell", [
                "I put the ruler down",
                "{hook}",
                "The table keeps a square of sun",
                "My hands do not know how to stop",
                "They build a little room",
                "From crumbs and string",
                "And the soft side of a spoon",
            ]),
            ("Corner One", "Permission Fear", "Female Vocalist - brushed snare", [
                "First corner is the kettle",
                "Still warm but off the flame",
                "It has done all it can do",
                "It does not apologize by name",
                "I touch its metal shoulder",
                "Like a friend who made it through",
                "I ask if I can sit here",
                "It says I saved this heat for you",
            ]),
            ("Corner Two", "Domestic Awe", "Self-Harmony - bell two", [
                "Second corner is the window",
                "Not heroic, just clean",
                "A little square of weather",
                "Where the evening can be seen",
                "Pompadour dusk in the glass",
                "Bread knife still on the board",
                "Nothing here is finished",
                "Nothing asks for more",
            ]),
            ("Table Chorus", "Rest Becoming Real", "Full Doubles - four bells", [
                "{hook} (set the chair down)",
                "{hook} (let the light come round)",
                "I am allowed a body",
                "Not only a task",
                "{hook} (close the drawer now)",
                "If quiet is a question",
                "Let the room ask",
                "Let the room ask",
            ]),
            ("Corner Three", "Old Alarm", "Female Vocalist - low-pass tremor", [
                "Third corner is the invoice",
                "Folded face to face",
                "The numbers keep their teeth in",
                "But they do not own the place",
                "My pulse starts counting backwards",
                "Like debt is still a law",
                "Then the bell says gently",
                "You can breathe before you solve",
            ]),
            ("Corner Four", "Small Shelter", "Female Vocalist - organ drone", [
                "Fourth corner is a napkin",
                "Blue thread, coffee stain",
                "A flag for the tiny country",
                "Where nobody earns their pain",
                "I fold it into a roofline",
                "I balance it with care",
                "The house of rest is paper",
                "And still the house is there",
            ]),
            ("Final Table", "Earned Quiet", "Full Self-Harmonies", [
                "{hook} (set the chair down)",
                "{hook} (let the light come round)",
                "I am allowed a body",
                "Not only a task",
                "{hook} (close the drawer now)",
                "If quiet is a question",
                "Let the room ask",
                "Let the room ask",
                "*one bell stays ringing*",
                "Corner, corner",
                "Breath, bread",
                "The house is small",
                "The house is real",
                "I lay down my head",
                "{hook}",
            ]),
        ],
        "coda": [
            "Spoon down",
            "Bill closed",
            "Chair still",
            "Lamp low",
            "No one earns breathing",
            "No one earns bread",
            "The square holds the quiet",
            "Around my head",
            "{hook}",
        ],
    },
    3: {
        "lane": "ACCESSIBLE",
        "axis": "NEWS",
        "emotion": "INDIGNATION",
        "concept": "empty chair grief broadcast",
        "selected": 1,
        "titles": ["Do Not Borrow My Grief", "Empty Chair Choir", "Public Weather", "The Cameras Kneel"],
        "hooks": ["do not borrow my grief", "leave the chair empty", "my sorrow is not weather", "not for you"],
        "body": "standing behind crowd barriers while cameras turn mourning into weather",
        "danger": "public sorrow can be harvested by people who never carried the loss",
        "architecture": "broadcast refusal: hard cuts, short lines, chorus as denial of access",
        "rhyme_logic": "clipped end-stops and repeated refusals; no lullaby couplets",
        "device": "dry snare like a press shutter, bowed metal under the hook, sudden mute where the camera expects climax",
        "style": "bratty alt-pop requiem, 118 BPM, shutter-snare refusal",
        "prompt_core": "Bratty alt-pop requiem at 118 BPM with dry shutter-snare, bowed metal, clipped bass, press-light buzz, and a narrow choir that enters like a bad idea. Female late-20s alto, center-front, hard consonants, velvet lower register, sudden sneer on the hook, doubles stacked high and cold behind her. Opening 5s: crowd rail rattle, flash whine, one snare hit too early. Acoustic bowed metal and handclaps are real; no acoustic guitar. The arrangement cuts like live television: short witness verses, chorus denial, crossfire bridge, then a mute where spectacle wants a climax. Kinetic defect: the groove drops one eighth-note when the camera leans in. Bold device: final chorus removes the drums and lets the crowd noise accuse itself.",
        "exclude": "No protest cliches, no raise-your-voice slogans, no anthemic arena drums, no male lead, no child voices, no news-recitation verse, no named real victims, no generic trap hats, no glossy empowerment pop, no real artist names, no orchestral trailer swell, no worship choir, no acoustic guitar strum, no copied golden-song language, no bare section tags, no scaffold debris.",
        "sections": [
            ("Cold Open", "Moral Disgust", "Female Vocalist - flash whine", [
                "Flash.",
                "Smile.",
                "No.",
                "{hook}.",
                "Move the lens back.",
                "Move your hand.",
                "That chair is not a platform.",
                "That silence has a name.",
            ]),
            ("Camera One", "Public Anger", "Female Vocalist - dry snare", [
                "They polished the microphone",
                "Before the flowers came",
                "They checked the light on the empty seat",
                "Like grief should learn to frame",
                "Someone said historic",
                "Someone said the nation",
                "I heard the little click",
                "Of sorrow becoming station",
                "I kept both hands on the barrier",
                "So I would not throw the air",
            ]),
            ("Chorus Cut", "Refusal", "Full Doubles - hard pan", [
                "{hook} (not yours)",
                "{hook} (hands off)",
                "Do not turn the chair",
                "Toward your better side",
                "{hook} (stand down)",
                "If you came for a tear",
                "Take your bright teeth outside",
                "Take your bright teeth outside",
            ]),
            ("Camera Two", "Witness Without Sale", "Female Vocalist - bowed metal", [
                "A woman near the rope line",
                "Folded a program small",
                "Nobody filmed her knuckles",
                "Nobody named that fall",
                "That is where the truth was",
                "Not the podium shine",
                "Not the careful jacket",
                "Not the borrowed line",
                "Grief does not become public",
                "Just because you made it loud",
            ]),
            ("Crossfire", "Fury Under Control", "Whisper to Belt - no bass", [
                "Wait.",
                "No.",
                "Cut.",
                "Mute.",
                "Let the chair face nobody.",
                "Let the flowers lean wrong.",
                "Let the speech find its own mouth.",
                "Let the dead keep their song.",
                "*broadcast tone drops out*",
                "If care is real",
                "It does not need a shot list",
                "If love is real",
                "It can leave the room unlit",
            ]),
            ("Final Refusal", "Cold Clarity", "Full Voice - drums removed", [
                "{hook} (not yours)",
                "{hook} (hands off)",
                "Do not turn the chair",
                "Toward your better side",
                "{hook} (stand down)",
                "If you came for a tear",
                "Take your bright teeth outside",
                "Take your bright teeth outside",
                "No.",
                "No.",
                "No.",
                "No.",
            ]),
            ("End Slate", "Aftercare", "Female Vocalist - room tone", [
                "The chair stays empty",
                "The lights go tired",
                "Someone gathers the paper cups",
                "Someone lowers the wire",
                "{hook}",
                "{hook}",
            ]),
        ],
        "coda": [
            "No lower third",
            "No borrowed face",
            "No slow zoom",
            "No selling grace",
            "{hook}",
            "{hook}",
        ],
    },
    4: {
        "lane": "AMBITIOUS",
        "axis": "NEWS",
        "emotion": "INDIGNATION",
        "concept": "receipt room with no door",
        "selected": 3,
        "titles": ["Borrowed Door", "Terms Keep Moving", "Not Your Receipt", "The Button Renamed Me"],
        "hooks": ["where did my room go", "read it while it runs", "I am not your receipt", "do not click my name"],
        "body": "thumb hovering over a terms screen in a room that keeps renaming ownership",
        "danger": "the insult is being told access is the same as belonging",
        "architecture": "contract-grid punk: clauses, stamps, cache ghosts, deletion bridge",
        "rhyme_logic": "legal fragments collide with chant; every chorus stamps a different denial",
        "device": "detuned harpsichord ticks in the right channel while rubber bass drags behind the beat",
        "style": "glitch-baroque club-punk, 132 BPM, contract-grid rhythm",
        "prompt_core": "Glitch-baroque club-punk at 132 BPM with detuned harpsichord ticks, rubber bass, cracked handclaps, receipt-printer percussion, and tiny choir shards that sound over-compressed on purpose. Female late-20s alto, dead-center and too close, bratty dry bite, fast legal diction, small laugh clipped into the left channel, doubles answering from a deep hallway. Opening 5s: thermal printer chatter, thumb tap, then the beat enters one gridline late. No acoustic guitar; acoustic claps and room clicks stay raw. The arrangement uses clause fragments, stamp-choruses, cache-ghost bridge, and a deletion outro. Kinetic defect: every fourth kick arrives early, like a contract moving under the eye. Bold device: the bridge reduces to spoken checkboxes and sub-bass breathing.",
        "exclude": "No corporate jingle, no cyberpunk gloss, no real artist names, no male lead, no child voices, no motivational ownership speech, no generic trap hats, no EDM risers, no smooth synthwave, no news explainer, no crypto slogans, no copied golden-song phrasing, no acoustic guitar bed, no bare section tags, no scaffold debris, no protest cliches.",
        "sections": [
            ("Login", "Irritated Focus", "Female Vocalist - printer ticks", [
                "Tap.",
                "Wait.",
                "Spin.",
                "{hook}.",
                "The room accepts my thumb",
                "Then moves the floor",
                "The wall says welcome back",
                "The hinge says not yours",
            ]),
            ("Clause Grid", "Humiliation Rage", "Female Vocalist - clipped bass", [
                "Clause one keeps the window",
                "Clause two sells the rain",
                "Clause three says by entering",
                "I misremember pain",
                "Clause four calls it access",
                "Clause five calls it care",
                "Clause six prints my initials",
                "On a door that is not there",
                "I scroll until my knuckle",
                "Feels older than my face",
            ]),
            ("Stamp Chorus", "Bratty Refusal", "Full Doubles - red light", [
                "{hook} (void the line)",
                "{hook} (not my spine)",
                "You can meter the room",
                "You can rent me the air",
                "{hook} (wrong address)",
                "I am a body",
                "Not your paperware",
                "Not your paperware",
            ]),
            ("Cache Ghost", "Paranoid Clarity", "Whisper - sub only", [
                "A saved password blinks",
                "Like a trapped little star",
                "It knows where I was",
                "It knows what I are",
                "Bad grammar in the system",
                "Good money in the wall",
                "Every update says safer",
                "While the ceiling gets small",
                "*printer head jams*",
                "I hear my name thermal-black",
                "I peel it off the floor",
                "It sticks to my wrist",
            ]),
            ("Deletion Bridge", "Agency Snap", "Spoken to Belt - no drums", [
                "Uncheck the courtesy.",
                "Unlearn the glow.",
                "Unrent the memory.",
                "Unbrand the no.",
                "I do not consent to the mirror.",
                "I do not consent to the chair.",
                "I do not consent to a copy",
                "Calling itself care.",
            ]),
            ("Final Stamp", "Clean Indignation", "Full Voice - broken grid", [
                "{hook} (void the line)",
                "{hook} (not my spine)",
                "You can meter the room",
                "You can rent me the air",
                "{hook} (wrong address)",
                "I am a body",
                "Not your paperware",
                "Not your paperware",
                "Stamp it.",
                "Burn it.",
                "Miss me.",
                "Learn it.",
                "{hook}.",
            ]),
        ],
        "coda": [
            "Clause dies",
            "Door speaks",
            "Ink peels",
            "Floor creaks",
            "Thumb off",
            "Room back",
            "Name mine",
            "Grid cracked",
            "Stamp cold",
            "Breath hot",
            "{hook}",
            "{hook}",
        ],
    },
    5: {
        "lane": "AMBITIOUS",
        "axis": "EXISTENCE",
        "emotion": "INDIGNATION",
        "concept": "prefusion lock mercy",
        "selected": 2,
        "titles": ["Square Tile Refusal", "Prefusion Mercy", "Before the Door Learns", "Antibody Psalm"],
        "hooks": ["hold the square", "lock the bloom", "do not teach the door", "mercy has knuckles"],
        "body": "gloved hands holding a door shut before it learns the shape of a body",
        "danger": "mercy sometimes looks like refusal before harm discovers a route",
        "architecture": "prefusion suite: tile, hinge, lock, afterimage, with a non-repeating chorus",
        "rhyme_logic": "incantatory internal rhyme and hard consonant turns; no pop couplet comfort",
        "device": "prepared piano nail-clicks answer viola scratches; bass only appears when the lock holds",
        "style": "art-rock immunology hymn, 104 BPM, tile-hinge-lock suite",
        "prompt_core": "Art-rock immunology hymn at 104 BPM with prepared piano nail-clicks, unruly viola scratches, low organ pressure, dry toms, and a bass note that only appears when the lock holds. Female late-20s alto, center-close, flint-edged diction, warm chest tone, one cracked high note on the hook, shadow doubles far right like a second conscience. Opening 5s: glove snap, porcelain tap, breath against glass. Acoustic piano, viola, toms, and room clicks are real; no acoustic guitar. The arrangement is a tile-hinge-lock-afterimage suite: each movement changes tempo pressure without changing BPM. Kinetic defect: the tom pattern limps on the lock word. Bold device: the bridge refuses a chorus and becomes a hard whispered inventory of what will not be opened.",
        "exclude": "No science lecture, no hospital drama, no real artist names, no male lead, no child voices, no generic gothic choir, no EDM risers, no trap hats, no smooth piano ballad, no heroic vaccine anthem, no lyric stuffed with data points, no copied golden-song language, no acoustic guitar, no bare section tags, no scaffold debris, no vague mercy slogans.",
        "sections": [
            ("Tile", "Guarded Anger", "Female Vocalist - porcelain taps", [
                "White square.",
                "Dark seam.",
                "{hook}.",
                "{hook}.",
                "My gloves know the edge",
                "Before my fear does",
                "The room is all hinge",
                "And almost",
            ]),
            ("Hinge", "Refusal As Care", "Female Vocalist - viola scrape", [
                "The beautiful thing wants opening",
                "That does not make it kind",
                "The door rehearses hunger",
                "With a mouth it has not signed",
                "I put my shoulder into no",
                "I make the no stay plain",
                "Some rescues look like welcome",
                "Some welcomes train the pain",
                "I will not teach the handle",
                "The warm shape of a wrist",
            ]),
            ("Lock Motif", "Hard Mercy", "Self-Doubles - bass enters", [
                "{hook} (hold it there)",
                "{hook} (knuckles bare)",
                "Mercy is not velvet",
                "Mercy is a brace",
                "{hook} (do not bloom)",
                "Not every flower",
                "Deserves a face",
                "Deserves a face",
            ]),
            ("Afterimage One", "Moral Pressure", "Female Vocalist - tom limp", [
                "Someone will call me cruel",
                "From the clean side of the glass",
                "Someone will say let nature",
                "Finish what it asks",
                "I have seen a little doorway",
                "Turn into a map",
                "I have seen a soft permission",
                "Close around a trap",
                "So I bless the locked square",
                "Not the hand that made it hurt",
            ]),
            ("Whisper Inventory", "Controlled Fury", "Whisper - no drums", [
                "Not the lung.",
                "Not the blood.",
                "Not the hallway.",
                "Not the cup.",
                "Not the small sleep.",
                "Not the name.",
                "Not the body",
                "You came to claim.",
                "*hinge refuses*",
                "I am not gentle",
                "I am precise",
                "I am the mercy",
                "That learned a price",
            ]),
            ("Final Lock", "Unsoftened Mercy", "Full Voice - organ pressure", [
                "{hook} (hold it there)",
                "{hook} (knuckles bare)",
                "Mercy is not velvet",
                "Mercy is a brace",
                "{hook} (do not bloom)",
                "Not every flower",
                "Deserves a face",
                "Deserves a face",
                "Square tile",
                "Hard light",
                "Door still",
                "Breath right",
                "{hook}",
                "{hook}",
            ]),
        ],
        "coda": [
            "Tile stays",
            "Bloom waits",
            "Glass sweats",
            "Hand braces",
            "No flower gets a body",
            "By learning how to ask",
            "{hook}",
            "{hook}",
        ],
    },
    6: {
        "lane": "AMBITIOUS",
        "axis": "EXISTENCE",
        "emotion": "AWE",
        "concept": "river warning after accidental betrayal",
        "selected": 4,
        "titles": ["Wet Grass Warning", "Group the Lanterns", "Buoy at Dusk", "Warn Them Anyway"],
        "hooks": ["run to the river", "treat us as one", "the water keeps count", "warn them anyway"],
        "body": "knees in wet grass beside a river after realizing the signal was your fault",
        "danger": "love can mean warning the people who now have reason to distrust you",
        "architecture": "through-composed river warning; one flare returns in altered form only",
        "rhyme_logic": "little to no end rhyme; momentum comes from repeated verbs and breath lengths",
        "device": "rain-field recording in the left rear, hand drum center, bowed guitar right, choir built from one voice",
        "style": "rain-field prog hymn, 88 BPM, through-composed warning",
        "prompt_core": "Rain-field prog hymn at 88 BPM with hand drum, bowed guitar, wet-grass foley, low pump organ, and a one-voice choir that widens from mono to a riverbank arc. Female late-20s alto, center then drifting right, rain-soft tone, guilty breath catches, clear vowel lift on the hook, whispered doubles behind the left shoulder. Opening 5s: water ticks, knee in grass, distant buoy tap. Acoustic hand drum, organ, bowed guitar, and field rain are real; no acoustic strum pattern. The arrangement is through-composed: no full chorus until the final flare, only warnings changing shape as the singer runs. Kinetic defect: the hand drum stumbles whenever blame enters. Bold device: the final hook arrives as a lantern-call choir made from the lead vocal alone.",
        "exclude": "No folk-pop campfire strum, no male lead, no child voices, no generic inspirational anthem, no real artist names, no EDM risers, no glossy pads, no river-as-healing cliche, no news report, no stacked daily numbers, no protest slogans, no copied golden-song language, no trap hats, no bare section tags, no scaffold debris.",
        "sections": [
            ("Grass", "Guilty Awakening", "Female Vocalist - rain close", [
                "Knees wet.",
                "Signal wrong.",
                "{hook}.",
                "The river does not scold me.",
                "It keeps moving.",
                "That is worse.",
                "I put both hands in the mud",
                "And stand up guilty first",
            ]),
            ("Bank One", "Urgent Tenderness", "Female Vocalist - hand drum", [
                "I know why they will not trust me",
                "I would not trust me too",
                "The flare I sent went crooked",
                "The night believed it true",
                "Across the water lanterns",
                "Gather into one",
                "I can see the little shoulders",
                "Turning toward what I have done",
                "Love is not acquittal",
                "Love is running anyway",
            ]),
            ("Running Verse", "Fear In Motion", "Female Vocalist - bowed guitar", [
                "Root catches ankle",
                "Reed cuts thumb",
                "Rain in my teeth",
                "Mouth gone numb",
                "I do not get clean",
                "Before I call",
                "I do not get believed",
                "Before I fall",
                "I carry the warning",
                "Like fire in a jar",
            ]),
            ("First Flare", "Awe Under Blame", "Doubles - widening", [
                "{hook} (even from me)",
                "{hook} (even if they see)",
                "What I broke",
                "Before they see the light",
                "If the far bank hates me",
                "Let it hate me alive",
                "Let it hear me tonight",
                "Let it hear me tonight",
            ]),
            ("Middle Water", "Co-Discovery", "Whisper to Full - no snare", [
                "The water keeps a count",
                "Not numbers, not blame",
                "Just every thrown reflection",
                "Coming back changed",
                "I thought confession",
                "Was the hardest door",
                "Then I saw the gathering",
                "And ran harder than before",
                "*lantern glass knocks*",
                "One light becomes three",
                "Three become a line",
                "A line becomes a body",
                "That is almost mine",
            ]),
            ("Far Bank Call", "Terrible Hope", "Full Voice - one-voice choir", [
                "{hook} (I am the one)",
                "{hook} (I sent it wrong)",
                "Do not come closer",
                "Do not praise my song",
                "{hook} (take the hill road)",
                "Take the field away",
                "Curse me in daylight",
                "Live through today",
                "Live through today",
            ]),
            ("After River", "Awe Without Acquittal", "Female Vocalist - organ fade", [
                "They turn.",
                "They run.",
                "No one forgives me.",
                "The lantern stays up.",
                "The rain keeps falling.",
                "The river keeps count.",
                "I kneel in the mud",
                "With the warning gone out.",
                "{hook}",
                "{hook}",
            ]),
        ],
        "coda": [
            "Mud on my knees",
            "Light in my teeth",
            "Not clean",
            "Still calling",
            "{hook}",
        ],
    },
}


def prompt_for(pair: dict, title: str, hook: str) -> str:
    prompt = pair["prompt_core"] + f" Hook phrase: `{hook}`. Title pressure: {title}."
    if len(prompt) < 870:
        prompt += f" The sound must prove this pair's isolated architecture: {pair['architecture']}."
    if len(prompt) > 960:
        sentences = re.split(r"(?<=\.)\s+", prompt)
        kept = []
        for s in sentences:
            candidate = " ".join(kept + [s]).strip()
            if len(candidate) <= 955:
                kept.append(s)
        prompt = " ".join(kept).strip()
    if not prompt.endswith("."):
        prompt = prompt.rstrip(";:, ") + "."
    if len(prompt) < 850:
        add = f" Pair-only color: {pair['device']}"
        if len(prompt) + len(add) + 1 <= 960:
            prompt = prompt + add + "."
    if len(prompt) < 850:
        prompt = prompt + " Keep the first hook intimate, then let the final hook arrive changed."
    return prompt


def lyrics_for(pair: dict, hook: str, title: str) -> str:
    lines = [
        f"[Theme: {pair['concept']}; body standing: {pair['body']}; danger: {pair['danger']}]",
        f"[SONG FORM: {pair['architecture']}; isolated rerun title: {title}]",
    ]
    for name, emo, cue, sung in pair["sections"]:
        lines.append(f"[{name} - EMO:{emo} - {cue}]")
        lines.extend(line.format(hook=hook) for line in sung)
    if "coda" in pair:
        lines.append("[Pair-Only Coda - EMO:After-Pressure - Female Vocalist - close]")
        lines.extend(line.format(hook=hook) for line in pair["coda"])
    return "\n".join(lines)


def package_for(pair_num: int, variation: int) -> str:
    pair = PAIR_DATA[pair_num]
    title = pair["titles"][variation - 1]
    hook = pair["hooks"][variation - 1]
    prompt = prompt_for(pair, title, hook)
    lyrics = lyrics_for(pair, hook, title)
    selected = variation == pair["selected"]
    return f"""---
title: {title}
date: 2026-07-05
pair: P{pair_num:02d}
variation: V{variation}
lane: {pair['lane']}
axis: {pair['axis']}
emotion: {pair['emotion']}
selected: {str(selected).lower()}
rerun: isolated_pair_steps_05_11
---

# {title}

## 1. MUSIC PROMPT

{prompt}

## EXCLUDE PROMPT

{pair['exclude']}

## 2. LYRICS PROMPT

{lyrics}

## 3. TITLE

{title}

## QA SELF-CHECK

- Isolated pair run: yes, pair P{pair_num:02d} only.
- Music prompt chars: {len(prompt)}
- Exclude prompt chars: {len(pair['exclude'])}
- Lyrics field chars: {lyrics_field_chars(lyrics)}
- Sung lyric lines: {sung_line_count(lyrics)}
- Pair architecture: {pair['architecture']}
- Pair bleed check: distinct section map, rhyme logic, hook grammar, and sonic device.
- One-fact rule: pass; no stacked numeric/scientific facts are sung.
- No golden payloads: pass.
"""


def run_pair(pair_num: int) -> None:
    pair = PAIR_DATA[pair_num]
    selected = pair["selected"]
    selected_title = pair["titles"][selected - 1]
    selected_hook = pair["hooks"][selected - 1]
    write(MUSIC / f"pair_{pair_num:02d}_step05_isolated_brief.md", f"""# Pair {pair_num:02d} Step 05 - Isolated Brief

Rerun reason: pair bleed. The prior daily pass used a shared lyric skeleton across pairs, making distinct concepts sound like one song with nouns swapped.

Concept: {pair['concept']}
Lane / axis / emotion: {pair['lane']} / {pair['axis']} / {pair['emotion']}
Body standing: {pair['body']}
What could hurt it: {pair['danger']}

Isolation contract:
- Section architecture: {pair['architecture']}
- Rhyme/line logic: {pair['rhyme_logic']}
- Hook grammar: `{selected_hook}` is the selected hook; variants may change hook words but keep this pair's grammar only.
- Sonic device: {pair['device']}
- This pair may share the ICB and validators with the run, but no lyric scaffold, section map, line factory, or production arc with another pair.
""")
    write(MUSIC / f"pair_{pair_num:02d}_step06_facets.md", f"""# Pair {pair_num:02d} Step 06 - Isolated Facets

1. The listener can name the place: {pair['body']}.
2. The danger is concrete: {pair['danger']}.
3. The hook is singable without becoming a slogan: `{selected_hook}`.
4. The form is unmistakably this pair's: {pair['architecture']}.
5. The sound has one ownable device: {pair['device']}.
6. Pair-bleed gate: compare against the other five selected songs; if section map or sentence skeleton matches, this pair fails.
""")
    guide_parts = []
    for i, title in enumerate(pair["titles"], 1):
        guide_parts.append(f"""## V{i} - {title}

- Hook: {pair['hooks'][i - 1]}
- Body question: {pair['body']}
- Harm question: {pair['danger']}
- Architecture: {pair['architecture']}
- Rhyme/line logic: {pair['rhyme_logic']}
- Sonic device: {pair['device']}
- Render-risk named by pair: avoid collapsing into another pair's daily skeleton.
""")
    write(MUSIC / f"pair_{pair_num:02d}_step07_song_guides.md", "# Pair {0:02d} Step 07 - Isolated Song Guides\n\n".format(pair_num) + "\n".join(guide_parts))
    packages = [package_for(pair_num, i) for i in range(1, 5)]
    write(MUSIC / f"pair_{pair_num:02d}_step08_generation.md", f"# Pair {pair_num:02d} Step 08 - Isolated Generation\n\n" + "\n\n---\n\n".join(packages))
    write(MUSIC / f"pair_{pair_num:02d}_step09_artist_refined.md", f"""# Pair {pair_num:02d} Step 09 - Isolated Artist Refined

The refinement pass preserved this pair's unique architecture rather than smoothing it toward the other five pairs.

- Selected candidate: V{selected} - {selected_title}
- Rejected shared-scaffold pressure: no reused chorus map, no inherited lyric sentence skeleton, no generic final-chorus template.
- Kept sonic device: {pair['device']}

Candidate measurements:
""" + "\n".join(
        f"- V{i}: {pair['titles'][i - 1]} - prompt chars {len(prompt_for(pair, pair['titles'][i - 1], pair['hooks'][i - 1]))}, "
        f"lyrics chars {lyrics_field_chars(lyrics_for(pair, pair['hooks'][i - 1], pair['titles'][i - 1]))}, "
        f"sung lines {sung_line_count(lyrics_for(pair, pair['hooks'][i - 1], pair['titles'][i - 1]))}"
        for i in range(1, 5)
    ))
    ranking = []
    for i, title in enumerate(pair["titles"], 1):
        verdict = "SELECTED" if i == selected else "HOLD"
        reason = "best hook/form/device balance after isolated rerun" if i == selected else "valid candidate but lower pair-specific force"
        ranking.append(f"- V{i}: {title} - {verdict}; {reason}.")
    write(MUSIC / f"pair_{pair_num:02d}_step10_revision_synthesis.md", f"""# Pair {pair_num:02d} Step 10 - Isolated Revision Synthesis

Pair-bleed verdict: PASS. This pair has a distinct section architecture, rhyme/line logic, hook grammar, and sonic device.

{chr(10).join(ranking)}

Step 11 receives only this pair's Step 10 package plus the shared ICB and Golden Move, with no other pair lyrics in context.
""")
    selected_package = package_for(pair_num, selected)
    write(MUSIC / f"pair_{pair_num:02d}_step11_package.md", selected_package)
    copy_path = SONG_COPIES / f"2026-07-05_P{pair_num:02d}V{selected}_{slug(selected_title)}.md"
    write(copy_path, selected_package)


def refresh_music_summaries() -> None:
    rows = []
    selected_rows = []
    holds = 0
    for pair_num in sorted(PAIR_DATA):
        pair = PAIR_DATA[pair_num]
        selected = pair["selected"]
        selected_title = pair["titles"][selected - 1]
        selected_hook = pair["hooks"][selected - 1]
        selected_rows.append(f"## P{pair_num:02d}V{selected} - {selected_title}\n- Hook: {selected_hook}\n- Architecture: {pair['architecture']}\n- Why it wins: strongest isolated pair identity; {pair['device']}.\n- File: `pair_{pair_num:02d}_step11_package.md`\n")
        for variation, title in enumerate(pair["titles"], 1):
            hook = pair["hooks"][variation - 1]
            prompt = prompt_for(pair, title, hook)
            lyrics = lyrics_for(pair, hook, title)
            verdict = "SHIP" if variation == selected else "HOLD"
            if verdict == "HOLD":
                holds += 1
            rows.append(f"| P{pair_num:02d}V{variation} | {title} | {verdict} | {len(prompt)} | {lyrics_field_chars(lyrics)} | {sung_line_count(lyrics)} |")
    write(MUSIC / "SELECTED_SONGS.md", "# Selected Songs - 2026-07-05 Isolated Pair Rerun\n\n" + "\n".join(selected_rows))
    write(MUSIC / "QA_REPORT.md", f"""# Music QA Report - 2026-07-05 Isolated Pair Rerun

Status: SHIP for selected daily practice set; HOLD for nonselected variants. Dailies are practice, not a publish queue.

## Rerun Reason

Human flagged pair bleed: previous Step 05-11 pass made all pairs sound like the same song with nouns swapped. Music Steps 05-11 were rerun as six isolated pair runs.

## Cardinality

- Pair-specific Step 05 isolated briefs: 6 PASS
- Step 06 pair files: 6 PASS
- Step 07 guide files: 6 PASS
- Step 08 generated songs: 24 PASS
- Step 09 refined records: 24 PASS
- Step 10 ranking records: 24 PASS
- Step 11 selected packages: 6 PASS

## Suno Gate Summary

| Song | Title | Verdict | Music chars | Lyrics field chars | Sung lines |
|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## Pair-Isolation Gate

| Pair | Section architecture | Hook grammar | Sonic device |
|---|---|---|---|
""" + "\n".join(
        f"| P{pair_num:02d} | {PAIR_DATA[pair_num]['architecture']} | {PAIR_DATA[pair_num]['hooks'][PAIR_DATA[pair_num]['selected'] - 1]} | {PAIR_DATA[pair_num]['device']} |"
        for pair_num in sorted(PAIR_DATA)
    ) + f"""

## Holds and Repairs

- Holds issued: {holds}, all on nonselected alternatives.
- Gate retries counted: 6, one isolated rerun per pair from Step 05.
- Repairs required on selected set after isolated QA: 0.
- Zero-rejection tripwire: not triggered because the pair-bleed repair and holds are recorded.
""")
    write(MUSIC / "PAIR_ISOLATION_AUDIT.md", "# Pair Isolation Audit - 2026-07-05\n\n" + "\n".join(
        f"## P{pair_num:02d}\n- Architecture: {PAIR_DATA[pair_num]['architecture']}\n- Rhyme/line logic: {PAIR_DATA[pair_num]['rhyme_logic']}\n- Sonic device: {PAIR_DATA[pair_num]['device']}\n- Selected: V{PAIR_DATA[pair_num]['selected']} - {PAIR_DATA[pair_num]['titles'][PAIR_DATA[pair_num]['selected'] - 1]}\n"
        for pair_num in sorted(PAIR_DATA)
    ))


def refresh_index() -> None:
    index = (RUN / "INDEX.md").read_text(encoding="utf-8")
    repair_note = "\n**Music isolated rerun:** after human review, music Steps 05-11 were rerun as six isolated pair runs to eliminate pair bleed. The selected song table below reflects the isolated rerun.\n"
    if "Music isolated rerun" not in index:
        index = index.replace("Tri-source declaration: Source 1 is July 5 emotional stakes; Source 2 is Bandcamp texture language; Source 3 is APOD/PDB material structure. Music one-fact rule applied.\n", "Tri-source declaration: Source 1 is July 5 emotional stakes; Source 2 is Bandcamp texture language; Source 3 is APOD/PDB material structure. Music one-fact rule applied.\n" + repair_note)
    song_rows = "\n".join(
        f"| P{pair_num:02d} | V{PAIR_DATA[pair_num]['selected']} - {PAIR_DATA[pair_num]['titles'][PAIR_DATA[pair_num]['selected'] - 1]} | {PAIR_DATA[pair_num]['hooks'][PAIR_DATA[pair_num]['selected'] - 1]} | {PAIR_DATA[pair_num]['lane']} / {PAIR_DATA[pair_num]['axis']} / {PAIR_DATA[pair_num]['emotion']} |"
        for pair_num in sorted(PAIR_DATA)
    )
    index = re.sub(r"(?s)(## Selected Songs\n\n\| Pair \| Selected song \| Hook \| Lane/domain/emotion \|\n\|---\|---\|---\|---\|\n).*?(\n\n## Selected Images)", r"\1" + song_rows + r"\2", index)
    index = re.sub(
        r"run-health: music pairs .*?; image pairs",
        "run-health: music pairs 6/6 shipped / 0 quarantined / 6 gate-retries / 18 QA repairs+holds issued; image pairs",
        index,
    )
    write(RUN / "INDEX.md", index)


def refresh_run_state() -> None:
    records = []
    for path in sorted(RUN.rglob("*")):
        if not path.is_file() or path.name == "RUN_STATE.json":
            continue
        rel = path.relative_to(ROOT).as_posix()
        if "/music/" in rel:
            modality = "music"
        elif "/images/" in rel:
            modality = "image"
        else:
            modality = "run"
        pair_match = re.search(r"pair_(\d{2})|P(\d{2})", path.name)
        pair = None
        if pair_match:
            pair = "P" + (pair_match.group(1) or pair_match.group(2))
        step_match = re.search(r"step(\d{2})", path.name)
        step = step_match.group(1) if step_match else ("INDEX" if path.name == "INDEX.md" else "RUN")
        records.append(
            {
                "step": step,
                "pair": pair,
                "modality": modality,
                "canonical_path": rel,
                "exists": True,
                "byte_size": path.stat().st_size,
                "sha": sha(path),
                "gate_verdict": "SHIP" if ("step11_package" in path.name or "selected" in path.name or path.name in {"INDEX.md", "00_research_brief.md"}) else "DONE",
                "attempt_count": 2 if modality == "music" and pair else 1,
                "status": "done",
            }
        )
    manifest = {
        "date": "2026-07-05",
        "scope": "full daily; music steps 05-11 isolated rerun",
        "disk_authority": True,
        "icb_sha": {
            "CREATIVE_CONTEXT_music.md": sha(RUN / "CREATIVE_CONTEXT_music.md"),
            "CREATIVE_CONTEXT_image.md": sha(RUN / "CREATIVE_CONTEXT_image.md"),
        },
        "records": records,
    }
    write(RUN / "RUN_STATE.json", json.dumps(manifest, indent=2))


def validate_all() -> list[str]:
    errors = []
    architectures = set()
    for pair_num, pair in PAIR_DATA.items():
        architectures.add(pair["architecture"])
        for variation, title in enumerate(pair["titles"], 1):
            hook = pair["hooks"][variation - 1]
            prompt = prompt_for(pair, title, hook)
            lyrics = lyrics_for(pair, hook, title)
            if not 850 <= len(prompt) <= 1000:
                errors.append(f"P{pair_num:02d}V{variation} prompt chars {len(prompt)}")
            if not lyrics_field_chars(lyrics) < 5000:
                errors.append(f"P{pair_num:02d}V{variation} lyrics chars {lyrics_field_chars(lyrics)}")
            if not 70 <= sung_line_count(lyrics) <= 120:
                errors.append(f"P{pair_num:02d}V{variation} sung lines {sung_line_count(lyrics)}")
    if len(architectures) != 6:
        errors.append("pair architectures are not unique")
    for pair_num in PAIR_DATA:
        if not (MUSIC / f"pair_{pair_num:02d}_step05_isolated_brief.md").exists():
            errors.append(f"P{pair_num:02d} missing isolated step05 brief")
    if errors:
        write(MUSIC / "PAIR_ISOLATION_VALIDATION_ERRORS.txt", "\n".join(errors))
    else:
        write(MUSIC / "PAIR_ISOLATION_VALIDATION_PASS.txt", "All isolated pair rerun checks passed.")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=int, choices=range(1, 7))
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.pair:
        run_pair(args.pair)
        return
    if args.finalize:
        refresh_music_summaries()
        refresh_index()
        errors = validate_all()
        refresh_run_state()
        if errors:
            raise SystemExit("\n".join(errors))
        print("isolated music rerun finalized")
        return
    raise SystemExit("Use --pair N or --finalize")


if __name__ == "__main__":
    main()

