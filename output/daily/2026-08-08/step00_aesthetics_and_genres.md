# STEP 00 — AESTHETICS, EMOTIONS, FRAMES, GENRES
## `2026-08-08-daily-music` · THE WRONG INVENTORY

**Continuity Payload Used:** frozen ICB `CREATIVE_CONTEXT.md`, LF-sha `9b538e91…`, 142,900 B — 18 baseline panel voices, 15 **Special Flairs**, LOFN-PRIME DNA inlined (27,796 B), Golden Seed, metaprompt, pair assignments.
**Step file:** `skills/music/steps/00_Generate_Music_Aesthetics_And_Genres.md`
**Cardinality gate:** 50 / 50 / 50 / 50 (`gates.yaml → taxonomy_cardinality: 50`).

> ⛔ **Nothing here is a menu the pairs must pick from.** This is the branching vocabulary the tree expands over. Values are chosen **against** the measured house monoculture (`00_research_brief.md` §4) — Glitch-Baroque, HyperRaaga, dry-close-mic and flat-single-dynamic are absent by construction, not by omission.

---

## PAYLOAD (valid JSON)

```json
{
  "run": "2026-08-08-daily-music",
  "archetype": "THE WRONG INVENTORY",
  "aesthetics": [
    "back-office fluorescent", "end-of-shift", "laminate and biro", "van interior at night",
    "kitchen table after dark", "corridor between two doors", "flat late-afternoon field light",
    "a room used all day", "honey-warm and slightly woozy", "smoke without a fire",
    "cheap electric warmth", "brushed and propulsive", "live-room bleed", "one take, kept",
    "unbolted", "tape running past the end", "handwriting under pressure", "shift-swap admin",
    "plastic tub on a shelf", "a lid going back on", "a plus-one", "an unread column",
    "a rounding error", "a threshold not cleared", "a rota with one gap", "a name written second",
    "a spare key to a sold car", "a call sign coming off", "concrete pad in a field",
    "photograph of something washed away", "the good light gone", "an amplifier still on",
    "a chair scraping", "someone's cough on tape", "sandwiches in foil", "an alarm set for four",
    "a folding chair", "a filter in a box", "two mugs, one cold", "the last row of a form",
    "unspent", "misfiled and undiminished", "competent and tired", "mid-task",
    "slightly irritated", "not moved", "finishing up", "going home",
    "the gesture, repeated", "nobody finds out"
  ],
  "emotions": [
    "Composure", "Ennui", "Detachment", "Equanimity", "Apathy", "Numbness", "Listlessness",
    "Resignation", "Disillusionment", "Disenchantment", "Dismay", "Regret", "Recognition",
    "Appreciation", "Warmth", "Kindness", "Tenderness", "Serenity", "Tranquility", "Ease",
    "Contentment", "Satisfaction", "Fulfillment", "Absorption", "Fascination", "Captivation",
    "Wonder", "Marvel", "Elevation", "Sublimity", "Dignity", "Accomplishment", "Confidence",
    "Empowerment", "Indignation", "Outrage", "Contempt", "Exasperation", "Irritation",
    "Impatience", "Defiance", "Playfulness", "Mirth", "Silliness", "Whimsy", "Anticipation",
    "Expectation", "Loneliness", "Isolation", "Alienation"
  ],
  "frames": [
    "two lines in one register crossing once", "counterpoint that never resolves",
    "byte-identical chorus", "wordless vocable hook", "spoken burden under a sung line",
    "call-and-answer with a fixed three-word answer", "end-rhyme chain on one vowel",
    "one-word refrain re-meaning by position", "shrinking stanzas 8-6-4-2",
    "two simultaneous independent strophes", "AABA 32-bar", "hymn verse plus refrain, no bridge",
    "through-composed 7/8 with one 4/4 refrain", "shouted pre-chorus that does not enlarge",
    "two-bar tag ending", "cold open on a room tone", "entry by instrument, not by downbeat",
    "the hook stated before the first word", "the hook restated after the last word",
    "a part swap mid-phrase", "a semitone pair mistaken for one voice",
    "a minor third crossing", "a major second beating", "octave self-doubling",
    "unison refused", "no key change", "one fixed tempo", "a modal lift with no drop",
    "a brightening mid-phrase", "a stop that is not a stop", "answer arriving early",
    "answer arriving late", "a line finished by the wrong person", "an interrupted sentence",
    "an unfinished sentence kept", "second person accusation", "second person affection",
    "first person procedural", "a form being filled in", "handwriting as rhythm",
    "trade nouns only", "no metaphor at all", "hymn diction on plastic objects",
    "playground taunt meter", "overheard half-sentences", "a burden in another mouth",
    "the same words much later", "a gesture described twice", "a wince the singer misses",
    "a fact responded to, not recited"
  ],
  "genres": [
    "Ethio-jazz-function instrumental soul", "close-harmony gospel-function folk",
    "fuzz-organ garage stomp", "odd-meter chamber prog", "spacious modal chamber-jazz ballad",
    "industrial folk", "AABA song-form soul", "hymn-shaped communal folk", "beat-group stomp",
    "spiritual-jazz devotional", "brushed-and-propulsive soul-funk", "chamber-folk with a Hammond bed",
    "drone-and-drum ballad", "unaccompanied close singing", "library-instrumental soul",
    "psych-folk with a combo organ", "desert-blues-adjacent slow rock", "second-line-adjacent shuffle",
    "boogaloo-adjacent mid-tempo", "clavinet funk", "Wurlitzer soul ballad", "tenor-led slow burn",
    "vibraphone-led lounge", "brushed bossa-adjacent", "waltz-time folk", "sea-shanty-adjacent burden song",
    "plainsong-adjacent monody", "gospel-clap communal stomp", "handclap-and-shaker folk",
    "acoustic guitar and one voice", "piano-and-voice, no drums", "upright bass and voice",
    "cello-and-voice chamber song", "brass-band-adjacent slow march", "reed-organ hymn",
    "melodica-led lilt", "mbira-adjacent cyclic pattern", "kalimba-and-tape rhythm",
    "hand-percussion circle", "tabletop percussion", "found-object percussion", "struck metal ballad",
    "somatic-bass minimal", "sub-and-voice austerity", "wind-and-drone ambient song",
    "field-recording-anchored folk", "post-punk two-chord drive", "art-rock guitar chorale",
    "krautrock-adjacent motorik", "long-form Afrobeat-length groove"
  ]
}
```

---

## COUNTS (asserted, then verified in `step01`)
`aesthetics 50` · `emotions 50` · `frames 50` · `genres 50` · **total 200.**

## BANNED BY CONSTRUCTION (measured, `00_research_brief.md` §4 · `core_seed.md` §7)
⛔ Glitch-Baroque · HyperRaaga · `relentless` · `explosive` · `battle` · `brutal` · `raw` · `aggressive`.
⚠️ `dry / close-miked` survives as a value but is **capped at one pair**; single-dynamic is **capped at two**.

**Self-critique.** The genre axis is the strongest here — fifty entries, none of them the house's recent five. The **emotion** axis is the weakest: it leans hard on the Composure/Ennui/Detachment cluster because the locked mood demands it, and that is a deliberate narrowing that costs range. It is defensible only because the *listener's* emotion (Dismay, Recognition) is carried on a separate axis from the singer's — if a pair collapses those two onto one voice, this taxonomy will not have warned it. D1 does.
