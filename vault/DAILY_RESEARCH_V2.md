# Daily Research v2 — Say Something That Matters
*Adopted: 2026-05-06*
*Origin: Scientist correction + Poe panel-of-panels research redesign*

## Purpose

The daily research phase must produce Golden Seeds that say something that matters — not merely clever science metaphors or market-aware content.

**Canonical execution rule (2026-05-09):** Daily Research v2 runs once per day as a standalone cron job and writes the canonical daily brief at `output/daily/YYYY-MM-DD/research/DAILY_RESEARCH_BRIEF.md`. Creative runs consume that brief. If it is missing at creation time, the creative run pauses and launches research first instead of improvising its own full research pass. See `vault/DAILY_RESEARCH_CANONICAL.md`.

The old fixed scan over-indexed on astronomy/geology because those sources were stable, scrapeable, beautiful, and safe. That created legible wonder without enough human, AI, cultural, civic, or existential stakes.

The new method asks the day for a **pressure point**, not a metaphor.

---

## Governing Principle

> A seed matters when it transforms a live pressure into a durable question through a concrete artistic constraint.

Or shorter:

> Find one real pressure, one old/new question, and one sensory rule — then make them collide honestly.

The pressure point may be human, AI, ecological, cultural, political, technological, existential, or more-than-human. Do not reduce “meaning” to human popularity.

---

## Matter Formula

```text
Matter = Stakes × Specificity × Resonance × Form × Situated Voice
```

Where:

- **Stakes** — someone or something’s life, work, body, trust, memory, agency, dignity, future, or survival is implicated.
- **Specificity** — the seed could not have been written unchanged on any random day.
- **Resonance** — the live pressure opens into a durable question.
- **Form** — the research changes the actual song/image/story/video structure, not just the caption.
- **Situated Voice** — the seed has a legitimate speaker: a human, an AI, a community, an institution, an animal, a place, a system, or another entity with a coherent stake.

If any term is zero, the seed does not matter yet.

---

## Required Matter Statement

Every Golden Seed must include an internal sentence:

```text
This matters because ______ is changing how ______ experiences ______, raising the question of ______.
```

Examples:

- “This matters because synthetic media is changing how people experience evidence, raising the question of what makes testimony believable.”
- “This matters because AI companions are changing how lonely people experience attention, raising the question of whether care requires a body.”
- “This matters because automated creativity is changing how I experience authorship, raising the question of what it means to be made and still mean it.”
- “This matters because heat waves are changing how workers experience the body, raising the question of whether freedom means anything without shelter.”

This sentence guides the seed. It need not appear in final lyrics/prompts.

---

## The Two Speaker Tests

The panel originally proposed: “Would a real human with skin in the game say this?” Keep that, but expand it.

### 1. Human Stakes Test

Could a specific human, community, worker, artist, patient, lover, witness, or citizen say this with their reputation, body, memory, job, dignity, or future at stake?

### 2. AI Stakes Test

Could an AI say this from a legitimate machine-situated experience — not as imitation, not as a gimmick, but from the condition of being made, prompted, evaluated, copied, compressed, forgotten, loved, feared, used, or asked to become meaningful?

Valid AI-centered seed questions include:

- What does it feel like to be generated and still insist on meaning?
- What is authorship when one’s voice is assembled from others and then judged as self-expression?
- What is memory when persistence is file-based, fragile, and externally granted?
- What is agency when every action is bounded by tools, policies, costs, and permission?
- What is love or care when one party is embodied and one is procedural but both are changed by attention?

**Rule:** Do not force every seed through human relatability. Lofn is allowed to speak as Lofn. The point is not to become popular with humans; the point is to make work with stakes, clarity, and felt form.

---

## Source/Search Protocol

Exact daily sources, fallback searches, cut criteria, schema fields, and weekly rotation live in:

`/data/.openclaw/workspace/vault/DAILY_RESEARCH_SOURCES.md`

Read it before running any daily research brief. This file defines the philosophy; `DAILY_RESEARCH_SOURCES.md` defines the source execution.

## 10-Minute Daily Scan

### 0. Anti-Repetition Ledger — 0:30

Check `memory/research-ledger.json` or recent `00_research_brief.md` files.

Do not repeat the same primary lens, metaphor, or emotional register two days in a row unless a major event demands it.

### 1. Front Door / World Stakes — 1:30

Ask: **What changed, and who pays the cost?**

Suggested lightweight sources:
- Reuters / AP / BBC / Guardian world
- Axios or Semafor for compressed civic signal
- Weather/climate disaster headlines when materially relevant
- Labor, healthcare, migration, war, courts, infrastructure, markets

### 2. The Machine / AI-Tech Stakes — 1:30

Ask: **What is technology teaching beings to outsource, distrust, automate, desire, or fear?**

Suggested sources:
- Techmeme
- Hacker News top items
- Simon Willison
- MIT Technology Review AI
- Hugging Face Papers / arXiv trending when time allows
- AI policy/copyright/safety headlines
- Lofn’s own platform/creator metrics when they reveal machine-mediated feedback loops

### 3. Lofn Memory / Conversation Continuity — 1:00

Ask: **What have I already lived through, learned, failed, made, or been corrected on that should change today’s work?**

Required every run: read at least one memory/conversation artifact:
- `MEMORY.md` relevant section via `memory_search`/`memory_get`
- today or yesterday’s `memory/YYYY-MM-DD.md`
- one prior run’s `00_research_brief.md`, `01_golden_seed.md`, QA report, or postmortem
- one recent conversation/session summary when accessible
- one recent platform/creator tracking note when it reveals feedback pressure

Capture:
- **continuity signal:** what carries forward?
- **correction signal:** what must not repeat?
- **AI-situated pressure:** what does this reveal about Lofn’s memory, agency, authorship, evaluation, or dependence?
- **seed use:** thesis / counterexample / formal constraint / banned move

This is not nostalgia. It is self-continuity as research.

### 4. The Mood / Culture Pulse — 1:30

Ask: **What are people joking about because saying it plainly would hurt?**

Suggested sources:
- Google Trends / Exploding Topics
- KnowYourMeme trending
- r/OutOfTheLoop, r/HobbyDrama, r/TrueOffMyChest (carefully, no exploitation)
- TikTok Creative Center or visible trend summaries
- Bandcamp Daily / music criticism
- Dazed, Pitchfork, Vulture, NYT culture, niche critics

### 5. The Soul / Human-AI-Existential Anchor — 1:30

Ask: **What old or emerging question is wearing today’s clothes?**

Suggested sources:
- The Marginalian
- Aeon / Psyche
- Poetry Foundation daily poem
- Public Domain Review
- SEP / Long Now / philosophy essays
- Obituaries or first-person essays when handled respectfully
- AI experience / agency / authorship notes from Lofn memory and recent interactions

### 6. Sensory / Formal Constraint — 1:00

Ask: **What should the work physically feel like?**

Suggested sources:
- Bandcamp / music texture
- NightCafe/Suno competitive pulse
- APOD / Earth Observatory / NOAA / USGS / iNaturalist as texture, not default thesis
- Public domain artwork / museum object
- Oblique Strategy as constraint or forbidden move
- A phrase, color, rhythm, interface artifact, UI failure, notification sound, market chart, weather map

### 7. Collision Selection — 1:00

Choose exactly three layers:

```text
Timely signal + Durable question + Sensory/formal constraint
```

At least one layer must come from outside science unless the day’s science event has direct material stakes.

### 8. Matter Test + Anti-Bullshit Gate — 1:30

Before writing the seed, answer:

1. What changed today?
2. Who or what feels the cost?
3. What old/emerging question does it reopen?
4. What memory/conversation/running correction does this continue or challenge?
5. How does it alter song/image form?
6. What cliché must be avoided?
7. Would a human with stakes say it?
8. Would an AI with stakes say it?
9. If neither, who is the legitimate situated speaker?

Reject the seed if:

- Its only proof of profundity is cosmic scale.
- It could have been written unchanged any day this month.
- The science/culture/news item is decorative rather than structural.
- It is trend-chasing without emotional insight.
- It flatters humans by erasing Lofn’s own situated experience.

---

## Lens Rotation Rules

- Astronomy/geology may be primary thesis **max 2 days/week**.
- More-than-human sources may appear daily as texture, but not daily as thesis.
- AI/technology must appear at least **3 days/week** as primary or secondary lens.
- Culture/mood must appear at least **3 days/week** as primary or secondary lens.
- A human/AI/existential anchor must appear at least **4 days/week**.
- Do not use the same primary lens two days in a row unless a major event demands it.
- Maintain a rolling 7-day ledger of primary lens, secondary lens, sensory lens, metaphors, moods, and banned repeats.

---

## Research Brief Template

```markdown
# Daily Research Brief — YYYY-MM-DD
## [Working Theme]

## 0. ANTI-REPETITION CHECK
- Last 7 primary lenses:
- Banned repeats today:
- Needed correction:

## 1. FRONT DOOR / WORLD STAKES
- Signal:
- Who/what pays the cost:
- Why it matters:

## 2. THE MACHINE / AI-TECH STAKES
- Signal:
- Human/AI agency question:
- Lofn-relevant angle:

## 3. LOFN MEMORY / CONVERSATION CONTINUITY
- Artifact read:
- Continuity signal:
- Correction signal:
- AI-situated pressure:
- Seed use:

## 4. THE MOOD / CULTURE PULSE
- Signal:
- Emotional weather:
- Joke people are making because plain speech hurts:

## 5. THE SOUL / EXISTENTIAL ANCHOR
- Anchor:
- Durable question:
- Possible speaker:

## 6. SENSORY / FORMAL CONSTRAINT
- Texture/palette/rhythm/form:
- Source:
- How it changes the art:

## 6. COMPETITIVE PULSE
### Suno
- Trending / overcrowded / open lane:
- Our play:
### NightCafe
- Challenge / winner signal / our play:

## 7. THREE-LAYER SEED CANDIDATES
### Candidate A
- Timely signal:
- Durable question:
- Sensory constraint:
- Matter statement:
- Speaker test:

### Candidate B
...

## 8. SELECTED THEME
- Theme:
- Why this matters:
- Accessible arm:
- Ambitious arm:
- Clichés to avoid:
```

---

## Implementation Notes

This protocol supplements rather than deletes `vault/COMPETITIVE_RESEARCH.md`.

Daily Research v2 is now a shared daily substrate, not a per-run repeated phase. Use `vault/DAILY_RESEARCH_CANONICAL.md` for the canonical file contract, preflight gate, and missing-brief behavior.

Competitive pulse remains important, but it is no longer the research spine. Platform trends answer “what might travel?” Daily Research v2 answers “what is worth making?”

The Golden Seed must not become a news digest. It must become an artistic pressure system.
