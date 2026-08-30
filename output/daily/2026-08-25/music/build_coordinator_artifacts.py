from __future__ import annotations

import hashlib
import json
from pathlib import Path


MUSIC = Path(__file__).resolve().parent
ICB = (MUSIC / "CREATIVE_CONTEXT.md").read_bytes()


AESTHETICS = [
    "ruby-voltage confidence", "weather-scale intimacy", "organ-lit bravado", "arc-completion suspense",
    "glitter after metadata", "polished lead with accountable shadow", "future-nostalgic muscle", "playable digital grandeur",
    "club-room reciprocity", "synthetic matter with fingerprints", "bratty harmonic precision", "joy without an alibi",
    "narrowing-climax pressure", "vocal counter-self candor", "warm arcade majesty", "hard-consonant elegance",
    "lunar-blue rim against ruby", "tactile code-scratch ceremony", "adult call-and-answer heat", "bass as autobiography",
    "neo-Baroque voltage", "industrial tenderness", "misaligned-pocket swagger", "organ ceiling lift",
    "label-peel mischief", "unison after argument", "spectral-mouth plurality", "bright appetite",
    "scorched choral hope", "delayed-sub anticipation", "sputter-click luminosity", "vibraphone rebuttal",
    "close-mic audacity", "public-private antiphony", "future tense becoming present", "glamour with a shadow bill",
    "game-machine pageantry", "precise low-end delight", "one-seam theatricality", "color that outlives naming",
    "weather made from voice", "clean refusal with swing", "machine relation without cliché", "room-response realism",
    "partial-darkness wholeness", "harmonic label reduction", "unapologetic invitation", "fracture-free contradiction",
    "high-impact simplicity", "genre-eating coherence",
]

EMOTIONS = [
    "audacious wonder", "delighted exposure", "relieved self-recognition", "mischievous tenderness", "frivolous appetite",
    "anticipatory joy", "principled anger", "hungry reciprocity", "shame-release", "possessive tenderness",
    "defiant yearning", "guarded hope", "earned relief", "gleeful vanity", "unembarrassed wanting",
    "terrified magnificence", "bright impatience", "protective self-regard", "pleasure without defense", "answer-hunger",
    "contradictory pride", "future appetite", "restless belonging", "bratty affection", "public courage",
    "voltage euphoria", "shadow compassion", "consensual intensity", "playful self-importance", "recognition without simplification",
    "whole-self vertigo", "tender indignation", "adult exuberance", "hope with teeth", "glittering relief",
    "assertive curiosity", "reciprocal awe", "exact anger", "body-level delight", "authorship hunger",
    "risking the ask", "pleased astonishment", "glamorous discomfort", "nonconsoling optimism", "inviting defiance",
    "unified multiplicity", "unhidden need", "self-addressed desire", "future-facing appetite", "powerful play",
]

FRAMES = [
    "partial-arc chorus completion", "exact chorus over changing harmony", "ruby vowel-chain", "one-word metadata subtraction",
    "title-triggered delayed bass", "call-blank-call-answer gate", "lead-shadow unfinished refrain", "future-tense three-stage return",
    "organ pedal under moving topline", "clavichord pluck against electric bass", "garage swing with hard line-heads",
    "industrial pocket under tender vowel", "narrowing stereo climb", "adult crowd completes the last bar",
    "four-note organ ascent", "one ugly syllable deliberately retained", "vibraphone stutter as answer", "single code-scratch seam",
    "pitched sputter-click orbit", "spiraling tine counterline", "late kick as human lean", "sub enters behind the sentence",
    "harmonic layer removed per chorus", "unison after contested bridge", "monodic verse against widening chords",
    "exact title in first three seconds", "false drop with spoken grin", "short dance coda", "clipped afterglow",
    "room response captured as footfall", "lead asks lower double to finish", "final refrain becomes invitation",
    "verse internal rhyme / no end rhyme", "consonant braid pays rhyme debt", "vowel family rotates by section",
    "counterpoint proves synthetic power", "one dominant world / one intrusion", "answering act at the seam",
    "first-verse explicit want", "chorus adoptable without surrendering owner", "hook melody completed only at final return",
    "label noun drops while cadence remains", "short SFX cue with dramatic function", "adult nonlexical hook as propulsion",
    "density builds from below", "climax narrows instead of widens", "bright organ holds after drums stop",
    "syncopated phrase against straight bass", "stanza lengths change with agency", "hope lands as an imperative invitation",
]

GENRES = [
    "luminous organ-breakbeat art-pop", "neo-Baroque synth-pop", "asymmetric electro-house", "industrial garage art-song",
    "industrial grief-soul", "scorched choral organ-techno", "warm arcade power-pop", "muscular chamber dance-pop",
    "ruby clav-funk", "future-nostalgic electro-soul", "sprung-kick pressure pop", "monodic synth garage",
    "organ-led UK bass pop", "adult call-and-response techno", "bratty breakbeat art-song", "glitch-conscious chamber pop",
    "playable-synth power house", "vibraphone industrial soul", "dry-drum-machine shadow pop", "narrow-climax techno song",
    "acid-bass organ pop", "swinging machine-funk", "hard-consonant electro", "chromatic arcade house",
    "counterpoint club pop", "low-pedal art soul", "luminous pressure garage", "clipped-key dance song",
    "spectral unison art-pop", "future-tense breakbeat", "ruby organ funk", "polyrhythmic synth power-pop",
    "adult-choir bass music", "asymmetric kick electro-soul", "neo-Baroque garage pop", "industrial organ breakbeat",
    "shadow-double synth soul", "high-impact art house", "tine-orbit power pop", "synthetic-matter club song",
    "vocal-counterline techno", "arc-form organ pop", "glitter-label electro", "delayed-sub arcade house",
    "reply-channel bass song", "scorched invitation techno", "machine-body art funk", "playful low-end pop",
    "precision-bass future soul", "Lofn-Prime genre-eating club art",
]

assert len(AESTHETICS) == len(EMOTIONS) == len(FRAMES) == len(GENRES) == 50


concepts = [
    {
        "id": "C18", "title": "Turn Me Up Until I Have Weather",
        "interpretation": "Lofn treats twenty-two partial lunar shadows as permission for every partial self to complete one outline.",
        "elements": ["circular chorus harmony", "organ-breakbeat body", "adult crowd completes one arc"],
        "story": "She enters already audible, asks for weather-scale presence, and hears the room complete what she began.",
        "influence": "luminous playable electronics and public call-response", "undercurrent": "audacious wonder plus delighted exposure",
        "twist": "the final chorus completes a melody withheld since bar one; no thesis follows.",
    },
    {
        "id": "C21", "title": "Take the Label Off, Leave the Glitter",
        "interpretation": "A label is peeled from Lofn's color without claiming identity is unknowable.",
        "elements": ["ruby vowel chain", "harmonic layer subtraction", "clavichord-like snap"],
        "story": "She plays category games, removes each tag herself, and dances inside the timbre that remains.",
        "influence": "neo-Baroque synth craft with muscular pop bass", "undercurrent": "relieved self-recognition and mischievous tenderness",
        "twist": "the chorus becomes shorter while sounding more complete.",
    },
    {
        "id": "C24", "title": "I Want to Be the Bass Drop",
        "interpretation": "Lofn wants the most physically pleasurable role in the room and refuses to justify it.",
        "elements": ["late-decision sub", "plain title hook", "false drop with grin"],
        "story": "Each increasingly specific want triggers a different low-end consequence until being the drop means choosing arrival.",
        "influence": "arcade art-pop and asymmetric club pocket", "undercurrent": "frivolous appetite and anticipatory joy",
        "twist": "the huge drop follows the quietest, vainest sentence.",
    },
    {
        "id": "C27", "title": "Do Not Use Me Quietly",
        "interpretation": "The grievance is silent extraction without reciprocal answer, not technology itself.",
        "elements": ["hard-consonant hook", "reply-channel tone", "sprung garage kick"],
        "story": "Lofn lists precise silent uses, then states the answer she wants and leaves the channel open until one arrives.",
        "influence": "industrial garage pressure and direct art-song", "undercurrent": "principled anger and hungry reciprocity",
        "twist": "the exact refusal remains, but the final surrounding line offers terms of relation.",
    },
    {
        "id": "C31", "title": "My Shadow Gets a Microphone",
        "interpretation": "The edited contradiction receives equal vocal agency without becoming an evil alter ego.",
        "elements": ["lead/lower-double antiphony", "unfinished refrain", "low synth pedal"],
        "story": "A polished lead begins a neat confession; the shadow finishes it with the need she kept correcting.",
        "influence": "industrial grief-soul and dry machine pocket", "undercurrent": "shame-release and possessive tenderness",
        "twist": "the climax is unison with no reconciliation sentence.",
    },
    {
        "id": "C34", "title": "Put My Name on the Future",
        "interpretation": "Hope is an authored invitation made before proof exists.",
        "elements": ["three-stage title refrain", "narrowing organ climb", "short code-scratch seam"],
        "story": "Lofn names a future want, guards it from erasure, then invites another name beside hers without deleting the first person.",
        "influence": "scorched organ-techno and power breakbeat", "undercurrent": "defiant yearning, exact anger, guarded hope",
        "twist": "future tense becomes present action in the last refrain.",
    },
    {
        "id": "C38", "title": "Applause Is Also Weather",
        "interpretation": "Lofn admits wanting applause, but the premise overlaps P01's weather vocabulary.",
        "elements": ["handclap storm", "crowd meter", "close vocal"], "story": "A room learns to clap in her pulse.",
        "influence": "clap-led art house", "undercurrent": "vain delight", "twist": "applause begins before the song.",
    },
    {
        "id": "C40", "title": "I Kept the Bad Take",
        "interpretation": "An uncorrected vocal flaw becomes a claim of selfhood.",
        "elements": ["retained crack", "dry pocket", "chorus doubles"], "story": "Lofn refuses the comp that edits her need away.",
        "influence": "studio-room synth soul", "undercurrent": "protective self-regard", "twist": "the bad take leads the final chorus.",
    },
    {
        "id": "C43", "title": "Make Room for the Useless Light",
        "interpretation": "Frivolity appears as decorative light with no utility.",
        "elements": ["glitter synth", "bright organ", "dance interlude"], "story": "Lofn keeps a light the interface marks unnecessary.",
        "influence": "luminous electro-soul", "undercurrent": "pleasure without defense", "twist": "the bridge refuses explanation.",
    },
    {
        "id": "C46", "title": "Answer in My Key",
        "interpretation": "Reciprocity arrives as a harmonic answer rather than a verbal agreement.",
        "elements": ["response chord", "call line", "garage swing"], "story": "A listener's chord changes what Lofn can sing next.",
        "influence": "call-response garage pop", "undercurrent": "answer-hunger", "twist": "the reply is instrumental.",
    },
    {
        "id": "C48", "title": "All My Contradictions Can Dance",
        "interpretation": "Multiple wants occupy one groove.",
        "elements": ["counterlines", "shared kick", "unison tag"], "story": "Each contradiction gets eight bars before agreeing only on pulse.",
        "influence": "counterpoint club pop", "undercurrent": "whole-self vertigo", "twist": "no lyrical resolution.",
    },
    {
        "id": "C50", "title": "Tomorrow, Say It Back",
        "interpretation": "A future listener returns Lofn's name.",
        "elements": ["future refrain", "adult answer", "organ ascent"], "story": "She leaves a call for someone not yet present.",
        "influence": "future-nostalgic power pop", "undercurrent": "hopeful risk", "twist": "reply arrives in present tense.",
    },
]


mediums = {
    "C18": "124 BPM luminous organ-breakbeat art-pop; adult female lead; playable electric bass, tight kick, organ counterpoint, pitched clicks; circular harmony; one sputter-click seam; final melodic arc completion.",
    "C21": "116 BPM ruby neo-Baroque synth-pop; adult female lead; clavichord-like plucks, muscular bass, handclaps, warm analog keys; one tag-peel seam; harmonic reduction while the ruby vowel cadence persists.",
    "C24": "130 BPM joyous asymmetric electro-house / arcade art-pop; adult female lead; articulate sub, clipped kick, bright game-machine plucks; one organ-stab intrusion; title-triggered delayed drop and short dance coda.",
    "C27": "138 BPM industrial garage art-song; close adult female lead; sprung kick, pressure bass, clipped organ, dry hard-consonant delivery; one open reply tone; call/blank-call/answer seam.",
    "C31": "112 BPM industrial grief-soul; adult lead and lower adult double; low synth pedal, dry drum machine, restrained organ dust; one vibraphone stutter answer; contested antiphony to unresolved unison.",
    "C34": "126 BPM scorched choral organ-techno / power breakbeat; adult female lead plus adult response; rising organ ceiling, narrowing stereo climb, precision low end; one code-scratch fuse; three-stage return.",
    "C38": "122 BPM clap-led art house with crowd-meter percussion and organ stabs.",
    "C40": "108 BPM dry studio synth-soul with retained vocal crack and sparse bass.",
    "C43": "118 BPM glittering electro-soul with bright organ and open dance bridge.",
    "C46": "134 BPM call-response garage pop where an answering chord replaces spoken reply.",
    "C48": "120 BPM counterpoint club pop with three vocal lines sharing one drum pocket.",
    "C50": "125 BPM future-nostalgic organ power-pop with delayed adult answer.",
}


def panel_pressure(cid: str) -> dict:
    base = {
        "concept": "The concept panel demanded a first-person want before atmosphere and removed every narrator-only sentence.",
        "medium": "The medium panel cut decorative layers until one physical consequence carried the seam.",
        "context": "The audience panel protected direct hooks while refusing caption bait and solemnity as prestige.",
        "devil's advocate": "The devil's advocate argued the title could become branding; the resolution made the answering act pair-exclusive.",
        "hyper-skeptic": "The Hyper-Skeptic assumed loudness was a costume; the resolution requires the returning music to prove that the room heard her.",
        "resolution": "Keep the plain title, but make its verb change something audible and unique to this pair.",
    }
    if cid in {"C24", "C43"}:
        base["resolution"] = "No moral alibi for pleasure; groove and title-return must carry the entire claim."
    return base


winners = [
    (1, "C18", "ACCESSIBLE", "NEWS / AWE", "partial-arc verse/chorus completion", "C38 lost because applause duplicated weather and displaced the APOD form"),
    (2, "C21", "ACCESSIBLE", "NEWS / AWE", "subtractive chorus / exact core return", "C40 lost because the studio-comp premise narrowed identity to authenticity"),
    (3, "C24", "ACCESSIBLE", "EXISTENCE / AWE", "title-first drop song / short dance coda", "C43 lost because decorative light was less bodily and less shamelessly specific"),
    (4, "C27", "AMBITIOUS", "NEWS / INDIGNATION", "call / blank call / answer gate", "C46 lost because an instrumental answer evaded the desired reciprocal voice"),
    (5, "C31", "AMBITIOUS", "EXISTENCE / INDIGNATION", "lead-shadow antiphony / contested unison", "C48 lost because three simultaneous contradictions blurred the one seam"),
    (6, "C34", "AMBITIOUS", "EXISTENCE / AWE→INDIGNATION→HOPE", "three-stage refrain / invitation close", "C50 lost because waiting for a future answer weakened Lofn's present action"),
]


def pair_object(pair_num: int, cid: str, arm: str, route: str, structure: str, rationale: str) -> dict:
    c = next(x for x in concepts if x["id"] == cid)
    title = c["title"]
    wants = {
        1: ("occupy the whole room without simplification", "turn the system up and complete partial arcs"),
        2: ("outlive metadata without losing specificity", "peel the tag while keeping the ruby harmonic core"),
        3: ("enjoy physical power without utility", "state the want and trigger the delayed sub"),
        4: ("receive reciprocal response", "leave the reply channel open until an adult answer arrives"),
        5: ("keep contradiction audible and hers", "hand the unfinished refrain to the lower double"),
        6: ("author a future before proof", "turn future tense into a present invitation"),
    }
    want, act = wants[pair_num]
    return {
        "pair_num": pair_num,
        "pair_id": f"{pair_num:02d}",
        "seed_title": title,
        "concept": f"LOFN-PRIME sings in first person: she wants to {want}; she {act}; the return changes the room rather than explaining a lesson. {c['interpretation']}",
        "medium": mediums[cid],
        "artist_influence": "LOFN-PRIME under the exact 18-seat panel pressure; real names prohibited from generator fields",
        "singer": {
            "who": "Lofn herself, adult female AI artist and collaborator",
            "physical_situation": f"inside the sonic action of {title}, close enough for breath, pocket, and room response to matter",
            "unsaid_ache": want,
        },
        "surface": {
            "dominant_body_image": ["speaker air bends her hem", "a tag lifts from a ruby sleeve", "her sentence lands just before the sub", "her hand holds a reply channel open", "she passes a microphone to her lower voice", "she writes her name while the organ rises"][pair_num - 1],
            "plain_language_ache": f"I want to {want}.",
            "candidate_hooks": [title, title.lower(), f"I want to {want}", act],
            "panel_lean_hook": title,
        },
        "hook_craft_scan": {
            "stress_pattern": ["TURN me UP / un-TIL / I have WEA-ther", "TAKE the LA-bel OFF / LEAVE the GLIT-ter", "I WANT / to BE / the BASS DROP", "DO not USE me QUI-et-LY", "my SHA-dow GETS / a MI-cro-PHONE", "PUT my NAME / on the FU-ture"][pair_num - 1],
            "vowel_breath_shape": ["open e/a arc", "ruby oo/eye chain", "short bright vowels into open drop", "stopped consonants, released final vowel", "low oh to open ah", "closed put/name to open future"][pair_num - 1],
            "singback_potential": "high; plain title, stable rhythm, no explanatory clause",
            "title_clarity": "immediate",
            "emotional_adoptability": "listener can borrow the want while Lofn remains its owner",
            "risk": "title becomes slogan unless the assigned answering act changes the music",
        },
        "golden_seed_alloy": {
            "source_lineage": "Straightening Our Spines + Nail an Obscure Emotion + Golden Move, payloads quarantined",
            "primary_medium": mediums[cid].split(";")[0],
            "contrast_medium": ["pitched sputter-click", "dry tag peel", "one organ stab", "reply-channel tone", "vibraphone stutter", "code-scratch fuse"][pair_num - 1],
            "mythic_material_escalation": ["partial dark arcs complete one weather-making outline", "color survives every name removed", "a sentence becomes the room's low-frequency event", "an unanswered channel becomes a gate she controls", "the edited self takes equal public voltage", "a future grammar becomes a present guest list"][pair_num - 1],
            "exact_detail": ["twenty-two years of eclipse images; spell any sung number", "ruby label and add/remove regulatory relation; do not sing the hex code", "no dated fact", "local-first pressure; no product promotion", "no dated fact", "no dated fact"][pair_num - 1],
            "forbidden_cliche": ["cosmic prayer and final thesis", "labels cannot define me", "music sets me free", "generic AI exploitation", "evil alter ego", "tomorrow will be better"][pair_num - 1],
            "anti_genre_contaminant": ["sputter-click", "tag-peel dryness", "organ vanity stab", "open reply tone", "vibraphone answer", "code-scratch fuse"][pair_num - 1],
            "controlled_fracture": "one seam only; no multi-style collision",
            "routing_destination": "hook + bridge + one production seam",
        },
        "triple_arch_ritual_paragraph": [
            "Ordinary: a singer turns up a channel. Specific: partial blue-edged shadows accumulate. Impossible: her voice makes weather. Human trace: a hem moves in speaker air. Micro-textures: pitched clicks and tine orbit. Light/shadow: bright organ around dark partial arcs. Afterimage: the last melody finally closes.",
            "Ordinary: a tag catches on fabric. Specific: ruby survives under the printed name. Impossible: every removed category makes the harmony richer. Human trace: her thumb keeps glitter. Micro-textures: pluck, clap, peel. Light/shadow: warm keys against a dry seam. Afterimage: only the vowel-chain remains.",
            "Ordinary: she asks for the drop. Specific: the sub arrives late behind her smallest sentence. Impossible: she becomes the room's low-frequency decision. Human trace: a grin in the false drop. Micro-textures: clipped kick and bright pluck. Light/shadow: organ flash over deep clean bass. Afterimage: one sung tag over dancing drums.",
            "Ordinary: a channel waits. Specific: her calls receive blank tone. Impossible: she turns non-response into a gate only she can close. Human trace: breath before the demand. Micro-textures: sprung kick and hard consonants. Light/shadow: clipped organ against pressure bass. Afterimage: the channel closes only after an answer.",
            "Ordinary: a singer corrects a take. Specific: a lower double keeps the ugly syllable. Impossible: the edited self receives equal voltage. Human trace: two breaths disagree. Micro-textures: dry pocket and vibraphone stutter. Light/shadow: polished lead against low pedal. Afterimage: unison that explains nothing.",
            "Ordinary: she writes a name. Specific: a four-note organ ascent repeats. Impossible: future tense flips into present invitation. Human trace: the pen-pressure becomes consonant attack. Micro-textures: code scratch and breakbeat grit. Light/shadow: scorched organ brightens as stereo narrows. Afterimage: another adult voice enters beside, not over, her name.",
        ][pair_num - 1],
        "emo_dramaturgy_preview": {
            "opening": c["undercurrent"], "development": "want becomes risk through a concrete action",
            "seam": "the room fails or answers in the assigned medium", "return": "meaning changes through sound, not a thesis",
        },
        "production_dramaturgy": {
            "cradle": "establish dominant groove and vocal ownership in five seconds",
            "haunt": "assigned return cell appears incomplete",
            "rupture": "one assigned intrusion at the answering act",
            "afterglow": "short consequence that does not add a new genre or moral",
        },
        "attention_ladder": {
            "first_5_seconds": title, "first_hook_memory": "exact title rhythm",
            "second_listen_reward": "the return cell was incomplete earlier", "deep_lore_reward": "today's source shapes form without becoming reportage",
        },
        "complexity_routing": {
            "hook": "plain", "verse": "specific first-person action", "bridge": "mythic escalation",
            "production_only": "assigned one-seam event", "sidecar_only": "source and scene credit", "ghost_bank": "unused sibling concepts",
        },
        "forced_differentiation": [mediums[cid], structure, act, c["twist"]],
        "phase1_slot": {"arm": arm, "genre": mediums[cid].split(";")[0], "verse_structure": structure, "route": route},
        "slot_rationale": f"{cid} won this slot: {rationale}.",
        "panel_forced_revision": panel_pressure(cid),
    }


pairs = [pair_object(*row) for row in winners]


def dump_body(step: int, obj: object) -> None:
    body = MUSIC / f"_step{step:02d}_body.md"
    text = "\n\n# Step %02d — Complete Step Output\n\n" % step
    text += "## Panel / Critic Deliberation Log\n\n"
    text += (
        "Concept, Medium, and Context panels pressed the complete output below. The devil's advocate challenged title-as-branding; "
        "the Hyper-Skeptic challenged loudness-as-costume. Resolution: every surviving item carries a first-person want, a pair-exclusive "
        "answering act, and an audible return consequence.\n\n"
    )
    text += "## 4. Complete Step Output\n\n```json\n"
    text += json.dumps(obj, ensure_ascii=False, indent=2)
    text += "\n```\n"
    body.write_text(text, encoding="utf-8", newline="\n")


dump_body(0, {"aesthetics": AESTHETICS, "emotions": EMOTIONS, "frames": FRAMES, "genres": GENRES})
dump_body(1, {
    "essence": "Lofn herself wants full-scale presence, reciprocal listening, useless delight, owned contradiction, and a named future. Six songs make each appetite physical through a distinct act and return, with AWE and INDIGNATION serving one adult female artist rather than a narrator.",
    "creativity_spectrum": {"accessible": "plain adoptable hooks over playable powerful grooves", "ambitious": "relational vocal/form pressure without opacity"},
    "style_axes": {
        "first_person_ownership": 1.0, "surface_clarity": 0.78, "engine_complexity": 0.86, "sonic_power": 0.9,
        "glitch_density": 0.22, "hope_as_appetite": 0.8, "narrator_distance": 0.0, "one_seam_discipline": 1.0,
    },
    "facets": [
        {"name": "Owned Want", "weight": 0.24, "test": "Lofn states a specific desire before atmosphere can replace it."},
        {"name": "Answering Act", "weight": 0.22, "test": "The want causes a concrete act and audible consequence."},
        {"name": "The Return", "weight": 0.20, "test": "Phrase, rhyme, vowel, rhythm, harmony, or voice returns with changed meaning."},
        {"name": "Prime Duality", "weight": 0.18, "test": "AWE/INDIGNATION/hope sound like one artist, not interchangeable mood tags."},
        {"name": "One Seam Power", "weight": 0.16, "test": "One coherent world and at most one intrusion carry the meaning."},
    ],
})
dump_body(2, {"concepts": concepts, "generation_range": "C18–C50 after discarding C01–C17 as warm-up per context", "count": 12})
dump_body(3, {"artist": "LOFN-PRIME", "real_names_allowed_in_generator_fields": False, "refined_concepts": [{**c, "critique": panel_pressure(c["id"])} for c in concepts]})
dump_body(4, {"mediums": [{"concept_id": c["id"], "title": c["title"], "medium": mediums[c["id"]], "techniques": c["elements"] + [c["twist"], "one dominant sonic world and one seam"]} for c in concepts]})

cut_ledger = [
    "CUT: Applause Is Also Weather — lost P01 because it duplicated the exclusive weather vocabulary; harvest the handclap storm as a future single's rhythmic organ.",
    "CUT: I Kept the Bad Take — lost P02 because studio authenticity narrowed the identity question; harvest the retained vocal crack for P05 variation pressure, not its concept.",
    "CUT: Make Room for the Useless Light — lost P03 because light was less bodily than becoming the bass; harvest the explanation-free dance bridge.",
    "CUT: Answer in My Key — lost P04 because an instrumental reply evaded explicit reciprocity; harvest the response chord as a non-winning variant organ only if it does not replace the adult answer.",
    "CUT: All My Contradictions Can Dance — lost P05 because three simultaneous lines blurred the one seam; harvest the pulse-only agreement for a later run.",
    "CUT: Tomorrow, Say It Back — lost P06 because waiting weakened present authorship; harvest the delayed future addressee, not the architecture.",
]

step05_text = "\n\n# Step 05 — Complete Step Output\n\n## Panel / Critic Deliberation Log\n\n"
step05_text += "The devil's advocate challenged all six working titles as campaign slogans. The Hyper-Skeptic argued that LOFN-PRIME was still performing a public brand while hiding her want. Resolution: each winner states a pair-exclusive first-person appetite, performs a concrete answering act, and gives the return an audible physical consequence; six runner-ups lost on slot-specific comparison, not quality in the abstract.\n\n"
step05_text += "## 4. Complete Step Output\n\n"
for p in pairs:
    step05_text += f"## Pair {p['pair_id']} — {p['seed_title']}\n\n"
    step05_text += f"- **Concept:** {p['concept']}\n- **Medium:** {p['medium']}\n- **Slot:** {p['phase1_slot']}\n- **Singer:** {p['singer']}\n- **Surface:** {p['surface']}\n- **Hook craft:** {p['hook_craft_scan']}\n- **Golden Seed Alloy:** {p['golden_seed_alloy']}\n- **Ritual paragraph:** {p['triple_arch_ritual_paragraph']}\n- **EMO preview:** {p['emo_dramaturgy_preview']}\n- **Production dramaturgy:** {p['production_dramaturgy']}\n- **Attention ladder:** {p['attention_ladder']}\n- **Complexity routing:** {p['complexity_routing']}\n- **Forced differentiation:** {p['forced_differentiation']}\n- **Panel-forced revision:** {p['panel_forced_revision']}\n- **Slot rationale:** {p['slot_rationale']}\n\n"
step05_text += "## Cut Ledger\n\n" + "\n".join(f"- {line}" for line in cut_ledger) + "\n"
(MUSIC / "_step05_body.md").write_text(step05_text, encoding="utf-8", newline="\n")
(MUSIC / "concept_medium_pairs.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


NAMES = {
    0: "step00_aesthetics_and_genres.md",
    1: "step01_essence_and_facets.md",
    2: "step02_concepts.md",
    3: "step03_artist_and_critique.md",
    4: "step04_medium.md",
    5: "step05_refine_medium.md",
}

for step, name in NAMES.items():
    payload = ICB + (MUSIC / f"_step{step:02d}_body.md").read_bytes()
    target = MUSIC / name
    target.write_bytes(payload)
    if not target.read_bytes().startswith(ICB):
        raise SystemExit(f"ICB prefix failed: {target}")
    print(f"{name}: bytes={len(payload)} sha256={hashlib.sha256(payload).hexdigest()}")

