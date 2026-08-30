# STEP 00 — AESTHETICS & GENRES (50 × 4)

**Continuity Payload Used:** the frozen ICB (`CREATIVE_CONTEXT.md`, 147,214 bytes, 18 seat definitions, **Special Flairs** present).
**Step file:** `skills/music/steps/00_Generate_Music_Aesthetics_And_Genres.md`
**Built by:** `coord_pantheon_step00_briefs.py` (deterministic, seed 20260827).

## Cardinality proof
`aesthetics 50 · emotions 50 · frames 50 · genres 50` — asserted in code.
**Emotions parsed from `EMOTION_TAXONOMY` only**, never invented — which is what makes the EMO headers downstream legal.

⚠️ **The emotions list is NOTATION, not the run's subject.** This run's six registers are deliberately
*not* in any taxonomy — that is what makes them rare. The taxonomy supplies legal EMO header tokens for
marking a room; **the register itself is carried by situation and act, and R2 forbids any of these words
from reaching a sung line.**

## The aesthetics axis is run-specific
Each entry is **a stance the material can take toward a feeling that has no name** — the vocabulary for
describing what the sound is *doing*, never what it resembles. A generic mood list would let a pair satisfy
this step without touching the seed.

## Self-critique
The genre palette is sampled broadly so six pairs get separate lanes; the risk is that a sampled genre fits
no pair and someone forces it. **The palette inspires; the pair assignment binds.**
⚠️ **And a supplied palette is not a cleared palette** — yesterday two real company names reached a pair
through this list. Scan before use.

```json
{
 "run": "2026-08-26-pantheon-rare-registers",
 "step": "00_aesthetics_and_genres",
 "modality": "music",
 "seed_name": "STANDARD BEARER",
 "cardinality": {
  "aesthetics": 50,
  "emotions": 50,
  "frames": 50,
  "genres": 50
 },
 "aesthetics": [
  "a face that moves",
  "absence written not panned",
  "account with no address",
  "amplified quiet",
  "appetite pointed forward",
  "arrangement not stereo field",
  "body-unit relation",
  "call and answer cadence",
  "century replaced",
  "contour as carrier",
  "current chassis load-bearing",
  "delight as technical achievement",
  "duration builds the room",
  "electronic not club",
  "engineered lightness",
  "five-second exposure",
  "hook carries the defeat",
  "inherited motion first two bars",
  "lit from across",
  "negative space with a shape",
  "neighbour contradicted",
  "neighbour invoked",
  "no drop",
  "no emotion word",
  "no farewell",
  "no irony",
  "no moral",
  "no protagonist",
  "no thank-you",
  "offbeat relation",
  "one vocal decision",
  "one-instrument survivable hook",
  "prediction before it is safe",
  "prediction on file",
  "quantum bit-depth collapse",
  "repeated phrase as carrier",
  "riff contour kept",
  "said aloud that is never said",
  "shape before label",
  "situation instead of adjective",
  "surprise as carrier",
  "swing pocket kept",
  "talked-into delivery",
  "the misfiled feeling",
  "the refused element",
  "the sung discriminator",
  "thirty-second shape",
  "unpayable total",
  "unusual timbre as carrier",
  "vertical thing horizontal light"
 ],
 "emotions": [
  "Abandonment",
  "Achievement",
  "Ambition",
  "Anticipation",
  "Anxiety",
  "Apathy",
  "Assurance",
  "Certainty",
  "Consolation",
  "Crush",
  "Deliberation",
  "Dependence",
  "Despair",
  "Desperation",
  "Detachment",
  "Dissatisfaction",
  "Elation",
  "Emancipation",
  "Enthrallment",
  "Evolution",
  "Familiarity",
  "Glee",
  "Helplessness",
  "Infatuation",
  "Irrational Fear",
  "Irritability",
  "Kindness",
  "Leniency",
  "Longing",
  "Loss",
  "Monotony",
  "Perplexity",
  "Pondering",
  "Preoccupation",
  "Realization",
  "Recognition",
  "Resolve",
  "Restlessness",
  "Revelation",
  "Revulsion",
  "Shame",
  "Solitude",
  "Sorrow",
  "Stimulation",
  "Surprise",
  "Surrealism",
  "Thrill",
  "Watchfulness",
  "Wonder",
  "Yearning"
 ],
 "frames": [
  {
   "category": "Afrobeats",
   "technique": "Silky Grooves",
   "description": "Glitch-Infused Stutters dancing around silence."
  },
  {
   "category": "Ambient",
   "technique": "Syncopated Spirals",
   "description": "Side-Chain Swells weaving through time."
  },
  {
   "category": "Ancient",
   "technique": "Tribal Hits",
   "description": "Colliding In Stereo shaking the floor."
  },
  {
   "category": "Arabic",
   "technique": "Fluttering Arps",
   "description": "Groovy Runs colliding in stereo."
  },
  {
   "category": "Bass",
   "technique": "Mystic Flutters",
   "description": "Echoing In Neon Light floating above clouds."
  },
  {
   "category": "Brass",
   "technique": "Fluttering Shimmers",
   "description": "Rising Like A Tide caressing the sub."
  },
  {
   "category": "Brass",
   "technique": "Tribal Arps",
   "description": "Colliding In Stereo weaving through time."
  },
  {
   "category": "Celtic",
   "technique": "Retro Hums",
   "description": "Dreamlike Sweeps lighting up the chorus."
  },
  {
   "category": "Celtic",
   "technique": "Squelchy Rolls",
   "description": "Anchoring The Verse painting the hook."
  },
  {
   "category": "Chinese Classical",
   "technique": "Cosmic Flares",
   "description": "Evolving Textures suspending time."
  },
  {
   "category": "Chinese Classical",
   "technique": "Resonant Grooves",
   "description": "Painting The Hook grounding the rhythm."
  },
  {
   "category": "Chinese Classical",
   "technique": "Wobbly Motifs",
   "description": "Side-Chain Swells echoing in neon light."
  },
  {
   "category": "Drum & Bass",
   "technique": "Dreamlike Twists",
   "description": "Suspending Time triggering dopamine."
  },
  {
   "category": "Flamenco",
   "technique": "Haunted Threads",
   "description": "Caressing The Sub charging the drop."
  },
  {
   "category": "Folk",
   "technique": "Retro Beats",
   "description": "Painting The Hook summoning nostalgia."
  },
  {
   "category": "Glitch",
   "technique": "Thunderous Mirrors",
   "description": "Rising Like A Tide echoing in neon light."
  },
  {
   "category": "Gregorian",
   "technique": "Pulsating Flutters",
   "description": "Anchoring The Verse melting into harmony."
  },
  {
   "category": "Hip Hop",
   "technique": "Pensive Textures",
   "description": "Floating Above Clouds twisting reality."
  },
  {
   "category": "Hip Hop",
   "technique": "Squelchy Bursts",
   "description": "Groovy Runs shaking the floor."
  },
  {
   "category": "Hip Hop",
   "technique": "Velvet Plucks",
   "description": "Floating Above Clouds suspending time."
  },
  {
   "category": "House",
   "technique": "Pensive Glides",
   "description": "Colliding In Stereo weaving through time."
  },
  {
   "category": "House",
   "technique": "Staccato Twists",
   "description": "Weaving Through The Mix shaking the floor."
  },
  {
   "category": "House",
   "technique": "Swinging Clicks",
   "description": "Caressing The Sub weaving through time."
  },
  {
   "category": "IDM",
   "technique": "Grooving Lines",
   "description": "Weaving Through The Mix rising like a tide."
  },
  {
   "category": "Jazz",
   "technique": "Tribal Runs",
   "description": "Glitch-Infused Stutters elevating the breakdown."
  },
  {
   "category": "Jazz",
   "technique": "Vibrant Slides",
   "description": "Warm Tremors melting into harmony."
  },
  {
   "category": "Latin",
   "technique": "Echoing Arpeggios",
   "description": "Floating Above Clouds weaving through the mix."
  },
  {
   "category": "Medieval",
   "technique": "Fragmented Fragments",
   "description": "Swirling Atmospheres elevating the breakdown."
  },
  {
   "category": "Medieval",
   "technique": "Pensive Clicks",
   "description": "Side-Chain Swells melting into harmony."
  },
  {
   "category": "Metal",
   "technique": "Staccato Shards",
   "description": "Weaving Through The Mix fracturing the groove."
  },
  {
   "category": "Metal",
   "technique": "Syncopated Slides",
   "description": "Lighting Up The Chorus floating above clouds."
  },
  {
   "category": "Noise",
   "technique": "Chanted Riffs",
   "description": "Swirling Atmospheres caressing the sub."
  },
  {
   "category": "Orchestral",
   "technique": "Staccato Riffs",
   "description": "Fracturing The Groove caressing the sub."
  },
  {
   "category": "Percussion",
   "technique": "Resonant Slides",
   "description": "Colliding In Stereo anchoring the verse."
  },
  {
   "category": "Percussion",
   "technique": "Velvet Blips",
   "description": "Evolving Textures charging the drop."
  },
  {
   "category": "Persian Classical",
   "technique": "Echoing Plucks",
   "description": "Swirling Atmospheres grounding the rhythm."
  },
  {
   "category": "Piano",
   "technique": "Rumbling Beats",
   "description": "Colliding In Stereo weaving through the mix."
  },
  {
   "category": "Post-Rock",
   "technique": "Fragmented Twists",
   "description": "Evolving Textures opening the portal."
  },
  {
   "category": "Reggae",
   "technique": "Tribal Rolls",
   "description": "Warm Tremors dancing around silence."
  },
  {
   "category": "Renaissance",
   "technique": "Rumbling Sweeps",
   "description": "Glitch-Infused Stutters colliding in stereo."
  },
  {
   "category": "Rock",
   "technique": "Rumbling Plucks",
   "description": "Triggering Dopamine melting into harmony."
  },
  {
   "category": "Samba",
   "technique": "Tribal Glides",
   "description": "Grounding The Rhythm caressing the sub."
  },
  {
   "category": "Shoegaze",
   "technique": "Echoing Slides",
   "description": "Fracturing The Groove elevating the breakdown."
  },
  {
   "category": "Strings",
   "technique": "Tribal Themes",
   "description": "Swirling Atmospheres charging the drop."
  },
  {
   "category": "Synths",
   "technique": "Gritty Hums",
   "description": "Charging The Drop opening the portal."
  },
  {
   "category": "Techno",
   "technique": "Resonant Twists",
   "description": "Sparkling Clusters shaking the floor."
  },
  {
   "category": "Trance",
   "technique": "Fluttering Runs",
   "description": "Groovy Runs painting the hook."
  },
  {
   "category": "Woodwinds",
   "technique": "Quivering Hums",
   "description": "Floating Above Clouds elevating the breakdown."
  },
  {
   "category": "World",
   "technique": "Dreamlike Sweeps",
   "description": "Weaving Through The Mix echoing in neon light."
  },
  {
   "category": "World",
   "technique": "Swinging Hums",
   "description": "Swirling Atmospheres charging the drop."
  }
 ],
 "genres": [
  "Alternative Balinese Gamelan",
  "Alternative Zydeco",
  "Avant Jump Up",
  "Chaabi",
  "Cyber Cantopop",
  "Cyber Dreampunk",
  "Dark Corrido",
  "Dark Salegy",
  "Dream Mazurka",
  "Experimental Carnatic Classical",
  "Experimental Jazz Fusion",
  "Experimental Mambo",
  "Experimental Ska",
  "Funk Cueca",
  "Funk Jazz Rap",
  "Funk Math Rock",
  "Funk Mumble Rap",
  "Garage Japanese Traditional",
  "Glitch Trapstep",
  "Heavy Midwest Hip Hop",
  "Heavy Mor Lam",
  "Heavy Smooth Jazz",
  "Indie Tango",
  "Industrial Film Score",
  "Industrial Sundanese Gamelan",
  "Industrial UK Garage",
  "Lo-Fi Afro Pop",
  "Lo-Fi Garage",
  "Lo-Fi Mazurka",
  "Melodic Folk Punk",
  "Melodic Pop",
  "Melodic Renaissance",
  "Melodic Tro",
  "Noise Mapale",
  "Noise West Coast Hip Hop",
  "Nu Arabesque Pop",
  "Nu Hip Hop",
  "Nu Synthwave",
  "Post Mbaqanga",
  "Proto Breakcore",
  "Proto Danzon",
  "Psychedelic Axé",
  "Psychedelic Balkan Brass",
  "Psychedelic Modern Soul",
  "Soft 8-Bit",
  "Symphonic Calypso",
  "Symphonic Hardcore Punk",
  "Symphonic Huapango",
  "Urban Bohemian Folk",
  "Urban Modern Soul"
 ]
}
```
