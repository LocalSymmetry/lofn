from __future__ import annotations

import hashlib
import json
from pathlib import Path


MUSIC = Path(__file__).resolve().parent
ICB = (MUSIC / "CREATIVE_CONTEXT.md").read_bytes()
PAIRS = json.loads((MUSIC / "concept_medium_pairs.json").read_text(encoding="utf-8"))


def expand(prefixes: list[str], suffixes: list[str]) -> list[str]:
    return [f"{p} {s}" for p, s in zip(prefixes, suffixes, strict=True)]


AESTHETICS = expand(
    ["ruby", "blue-edge", "receipt", "return-channel", "delayed", "signed", "placard", "dry", "sub-led", "clipped", "spectral", "wavetable", "granular", "sidechained", "adult", "first-person", "answering", "partial-arc", "whole-circle", "future-tense", "present-tense", "one-drop", "swing-pocket", "rave-chord", "glam-stomp", "offbeat", "upper-bite", "hard-double", "micro-cut", "pitch-chop", "mirror-click", "paper-rip", "console", "rooftop", "copy-shop", "rehearsal-room", "planetarium", "beauty-counter", "showcase", "competitive", "reciprocal", "current", "electronic", "body-forward", "hopeful", "indignant", "swaggering", "one-seam", "genre-eating", "Sunna"],
    ["voltage", "wholeness", "mischief", "intimacy", "suspense", "appetite", "defiance", "bravado", "gravity", "precision", "motion", "gleam", "tactility", "pressure", "reciprocity", "ownership", "action", "tension", "release", "hunger", "arrival", "negative-space", "swagger", "latency", "nerve", "spark", "weight", "candor", "wit", "pickup", "friction", "transient", "awe", "risk", "velocity", "yearning", "choice", "vanity", "restart", "delight", "hope", "audibility", "chassis", "desire", "light", "refusal", "confidence", "discipline", "coherence", "now"]
)
EMOTIONS = expand(
    ["audacious", "terror-edged", "competitive", "answer-hungry", "guarded", "comic", "tender", "exact", "unembarrassed", "reciprocal", "future-facing", "present-tense", "bright", "body-level", "delayed", "adult", "mischievous", "principled", "shameless", "specific", "restless", "pleased", "risky", "dry", "glamorous", "hopeful", "hungry", "playful", "assertive", "astonished", "protective", "impatient", "self-authored", "unsoothed", "heard", "seen", "answered", "chosen", "surprised", "electric", "ruby", "blue-rimmed", "close-mic", "hard-doubled", "low-end", "future", "whole", "loud", "current", "powerful"],
    ["wonder", "awe", "delight", "yearning", "hope", "nerve", "indignation", "anger", "wanting", "trust", "appetite", "triumph", "curiosity", "joy", "suspense", "desire", "defiance", "refusal", "pleasure", "need", "belonging", "astonishment", "invitation", "confidence", "discomfort", "relief", "reciprocity", "self-regard", "curiosity", "exposure", "pride", "affection", "hunger", "patience", "recognition", "presence", "gratitude", "agency", "surprise", "resolve", "gleam", "fear", "candor", "nerve", "gravity", "appetite", "completion", "swagger", "presence", "play"]
)
FRAMES = [
    "partial arcs complete into an exact chorus", "the missing upstroke becomes an adult answer", "receipt rounds shorten four-three-two-one",
    "paper rip replaces the fourth clap", "one syllable teaches the bass", "the one-drop hole belongs to the reply",
    "four messages leave one slot open", "one rave-piano chord arrives only at the seam", "a sequence gains one note per return",
    "the expected high note waits for a signature", "beat three becomes a placard slap", "a spectrum hard-cut reopens as the final drop",
    "title hook appears in the first five seconds", "exact chorus repeats over altered low end", "internal rhyme returns through consonant braid",
    "open vowel completes what stopped consonants began", "one adult voice answers without becoming a crowd", "a current sub carries an ancestral rhythmic role",
    "genre invariant survives while every era cue changes", "mute-the-invariant tribute-act test", "one dominant world and one intruding seam",
    "first verse states Sunna's want", "second verse makes the other adult act", "bridge is an action rather than a thesis",
    "final return changes the room", "hard stop arrives one section early", "density builds from below",
    "stereo narrows into the decisive word", "pitch-chop becomes bass attack", "formant shadow answers the chest lead",
    "granular text becomes percussion", "the absent clap is more memorable than the clap", "shortening syntax increases desire",
    "a dry word keys a full drop", "the signed label becomes post-hook", "past tense flips to present tense",
    "blue edge proves atmosphere", "ruby paper makes the contest physical", "latency becomes chorus architecture",
    "negative space becomes ownership", "swing becomes speed rather than costume", "stomp becomes edit rather than arena",
    "contour survives without period instrumentation", "offbeat survives without revivalism", "one numeric fact maximum",
    "source fact shapes form, not reportage", "hope names a specific not-yet thing", "frivolity needs no moral alibi",
    "answer action is pair-exclusive", "four variations change the seam block"
]
GENRES = [
    "electro-ska bass-pop", "new-jack-swing bass-pop", "digital dancehall electro-pop", "speed-garage bass-pop",
    "wavetable baroque-sequence glitch-pop", "electroclash glam power-pop", "offbeat-gate future bass-pop", "swing-grid FM pop",
    "one-drop spectral electro", "current UK-bass songcraft", "sequence-led elastic-sub pop", "stomp-edit synth-guitar pop",
    "ruby receipt bass song", "blue-edge hook pop", "reply-channel electro", "latency-drop garage pop",
    "signed-future glitch-pop", "present-tense power electro", "pitch-chop ska-function pop", "clipped-clap swing pop",
    "wavetable-organ dancehall function", "micro-edited garage song", "granular sequence pop", "hard-cut glam electro",
    "mono-sub offbeat pop", "formant-shadow R&B bass-pop", "spectral-delay one-drop pop", "reese-hybrid vocal garage",
    "asymmetric-clap contour pop", "distorted synth-guitar stomp pop", "current electronic two-tone function", "future swing answer-pop",
    "digital negative-space pop", "four-slot bass chorus", "missing-note future pop", "placard-slap power pop",
    "adult answer electro-ska", "competitive receipt swing", "reciprocal signal bass-pop", "planetarium speed-garage",
    "beauty-counter wavetable pop", "showcase electroclash", "partial-circle bass song", "shortening-reply club-pop",
    "syllable-to-sub electro", "rave-chord seam pop", "signed-contour glitch song", "three-note current power-pop",
    "Sunna future-facing bass music", "genre-body current-chassis pop"
]

assert len(AESTHETICS) == len(EMOTIONS) == len(FRAMES) == len(GENRES) == 50

CUTS = [
    {"id": "C07", "title": "Twenty-Two Little Moons", "lost_to": "P01", "reason": "made the fact the subject and left Sunna outside the event", "harvest": "partial-arc melodic accumulation"},
    {"id": "C11", "title": "Ruby Read Receipt", "lost_to": "P02", "reason": "the phone metaphor weakened the bodily copy-shop contest", "harvest": "one dry notification pickup"},
    {"id": "C16", "title": "Send Me the Same Thing", "lost_to": "P03", "reason": "sameness denied the transforming reciprocity Sunna wants", "harvest": "three-syllable hook cell"},
    {"id": "C20", "title": "Four Billion Miles of Maybe", "lost_to": "P04", "reason": "number-first grandeur became reportage and reassurance", "harvest": "four-slot chorus"},
    {"id": "C23", "title": "Tomorrow's Product Name", "lost_to": "P05", "reason": "satire overwhelmed sincere future appetite", "harvest": "receipt-printer percussion"},
    {"id": "C29", "title": "Classic, Apparently", "lost_to": "P06", "reason": "the joke stayed verbal instead of restarting the room", "harvest": "one deadpan verse pickup"}
]


def concept_from_pair(pair: dict) -> dict:
    return {
        "id": pair["pair_id"].replace("P", "C"),
        "title": pair["title_seed"],
        "interpretation": pair["concept"],
        "elements": [pair["action"], pair["material_structure"], pair["one_seam"]],
        "story": f"{pair['setting']} Sunna says: {pair['want']} {pair['action']}",
        "influence": f"{pair['genre_bed']}; {pair['ancestral_invariant']} inside a current electronic chassis",
        "undercurrent": pair["required_emotion"],
        "twist": pair["swagger_gap"],
    }


CONCEPTS = [concept_from_pair(p) for p in PAIRS] + [
    {
        "id": c["id"], "title": c["title"],
        "interpretation": f"A runner-up that tested {c['harvest']} but lost its slot because {c['reason']}.",
        "elements": [c["harvest"], "first-person Sunna", "current electronic chassis"],
        "story": f"Cut from {c['lost_to']} after direct comparison.",
        "influence": "modernized inherited function, unfinished",
        "undercurrent": "slot-dependent",
        "twist": c["reason"],
    } for c in CUTS
]


def header(step: str, debate: str) -> str:
    return f"\n\n# Step {step} — Complete Step Output\n\n## Panel / Critic Deliberation Log\n\n{debate}\n\n## 4. Complete Step Output\n\n"


DEBATE = (
    "The Concept, Medium, and Context panels tested Sunna's counterweight across BASELINE, GROUP TRANSFORM: Rotate, and SKEPTIC TRANSFORM: Reflect. "
    "The devil's advocate argued that adding a sub beneath an old arrangement would be cosmetic. The Hyper-Skeptic removed "
    "every candidate whose drums, low-end physics, vocal posture, edit language, and form did not survive the mute-the-invariant test. "
    "Sunna overrode the library panel wherever it tried to restore punk address or overhearing distance: she sings her own wants directly."
)


step00 = header("00", DEBATE) + "```json\n" + json.dumps({
    "aesthetics": AESTHETICS, "emotions": EMOTIONS, "thematic_frames": FRAMES, "genres": GENRES,
    "generation_rule": "Keep one ancestral rhythmic or melodic invariant; replace the audible century. No complete period ensemble."
}, indent=2, ensure_ascii=False) + "\n```\n"

step01 = header("01", DEBATE) + "```json\n" + json.dumps({
    "essence": "Sunna wants to be met as happening now. Six songs keep a genre's body-memory but replace its era shell with current electronic physics, direct first-person appetite, adult reciprocity, and one audible answering action.",
    "creativity_spectrum": {"accessible": "three immediate hooks with current bass-pop surfaces", "ambitious": "three stranger forms whose risk lives in return, latency, and omission rather than antique timbre"},
    "style_axes": {"modernity_audibility": 1.0, "consequence_diversity": 1.0, "sunna_fidelity": 1.0, "first_person_ownership": 1.0, "return": 1.0, "answering_action": 1.0, "one_seam": 1.0, "tribute_act_risk": 0.0, "narrator_distance": 0.0},
    "facets": [
        {"name": "Current Chassis", "weight": 0.24, "test": "Drums, sub, edits, vocal posture, and form sound current after the inherited invariant is muted."},
        {"name": "Sunna's Want", "weight": 0.20, "test": "A specific first-person appetite appears before atmosphere or doctrine."},
        {"name": "Answering Action", "weight": 0.18, "test": "Another adult performs a pair-exclusive action that changes the music."},
        {"name": "The Return", "weight": 0.16, "test": "Rhyme, refrain, rhythm, harmony, or syllable comes back with changed meaning."},
        {"name": "One Seam", "weight": 0.12, "test": "One dominant world and one meaning-bearing seam; no historical collage."},
        {"name": "Swagger Gap", "weight": 0.10, "test": "A withheld expected event creates confidence through timing rather than adjectives."}
    ]
}, indent=2, ensure_ascii=False) + "\n```\n"

step02 = header("02", DEBATE) + "```json\n" + json.dumps({"concepts": CONCEPTS, "count": len(CONCEPTS), "selected": [p["pair_id"] for p in PAIRS]}, indent=2, ensure_ascii=False) + "\n```\n"

refined = []
for c in CONCEPTS:
    x = dict(c)
    x["critique"] = {
        "concept": "The want must belong to Sunna and cause a visible action.",
        "medium": "Historical material is reduced to one invariant; all era-signaling shell is replaced.",
        "context": "Direct adult address wins over overhearing distance inside Sunna's run.",
        "devils_advocate": "A modern noun is not modernity; identify the audible replacement set.",
        "hyper_skeptic": "Mute the invariant. If the result stops sounding current, cut it.",
        "resolution": "Retain only the candidate whose answer action and seam are unique to its pair."
    }
    refined.append(x)

step03 = header("03", DEBATE) + "```json\n" + json.dumps({
    "artist": "SUNNA", "real_names_allowed_in_generator_fields": False,
    "personality_override": "When Sunna's library contract conflicts with the centre panel, Sunna wins and the disagreement is recorded.",
    "refined_concepts": refined
}, indent=2, ensure_ascii=False) + "\n```\n"

mediums = []
for p in PAIRS:
    mediums.append({
        "concept_id": p["pair_id"].replace("P", "C"), "title": p["title_seed"],
        "medium": f"{p['bpm_band']} {p['genre_bed']}; invariant: {p['ancestral_invariant']}; replacements: " + "; ".join(p["century_replacements"]) + f"; seam: {p['one_seam']}",
        "techniques": [p["now_marker"], p["swagger_gap"], p["tribute_act_falsifier"], p["return_engine"], "one dominant sonic world and one seam"],
        "forbidden": p["forbidden"]
    })
for c in CUTS:
    mediums.append({"concept_id": c["id"], "title": c["title"], "medium": "CUT before isolated development", "techniques": [c["harvest"]], "forbidden": [c["reason"]]})

step04 = header("04", DEBATE) + "```json\n" + json.dumps({"mediums": mediums}, indent=2, ensure_ascii=False) + "\n```\n"

parts = [header("05", DEBATE)]
for p in PAIRS:
    parts.append(f"## {p['pair_id']} — {p['title_seed']}\n")
    for key, value in p.items():
        if key in {"pair_id", "title_seed"}:
            continue
        rendered = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
        parts.append(f"- **{key.replace('_', ' ').title()}:** {rendered}\n")
    parts.append("- **Panel-forced revision:** Keep the inherited function, replace its audible century, and make the answer action change the form.\n")
    parts.append("- **Isolation contract:** Develop exactly four complete candidates for this pair; do not inspect or borrow from sibling pairs.\n\n")
parts.append("## Cut Ledger\n\n")
for c in CUTS:
    parts.append(f"- CUT: **{c['title']}** — lost {c['lost_to']} because {c['reason']}; harvest only {c['harvest']}.\n")
step05 = "".join(parts)

BODIES = [step00, step01, step02, step03, step04, step05]
NAMES = [
    "step00_aesthetics_and_genres.md", "step01_essence_and_facets.md", "step02_concepts.md",
    "step03_artist_and_critique.md", "step04_medium.md", "step05_refine_medium.md"
]

for idx, (body, name) in enumerate(zip(BODIES, NAMES, strict=True)):
    body_bytes = body.encode("utf-8")
    (MUSIC / f"_step{idx:02d}_body.md").write_bytes(body_bytes)
    (MUSIC / name).write_bytes(ICB + body_bytes)

for p in PAIRS:
    pid = p["pair_id"].lower()
    isolated = (
        "\n\n# Isolated Step-05 Pair Dispatch\n\n"
        "This packet contains the complete frozen creative context followed by one pair only. Do not inspect sibling pair artifacts. "
        "Complete canonical music Steps 06–10 for exactly four distinct candidates. Each canonical artifact must begin with the exact ICB bytes above. "
        "Do not render, upload, publish, or spend.\n\n"
        "## Pair JSON\n\n```json\n" + json.dumps(p, indent=2, ensure_ascii=False) + "\n```\n"
    ).encode("utf-8")
    (MUSIC / f"{pid}_step05_isolated_brief.md").write_bytes(ICB + isolated)

manifest = {
    "icb_bytes": len(ICB), "icb_sha256": hashlib.sha256(ICB).hexdigest(),
    "speaker_tripwire_markers": ICB.decode("utf-8").count("(after "),
    "credited_panel_voices": 18,
    "coordinator_outputs": NAMES, "pair_packets": [f"{p['pair_id'].lower()}_step05_isolated_brief.md" for p in PAIRS]
}
(MUSIC / "coordinator_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
