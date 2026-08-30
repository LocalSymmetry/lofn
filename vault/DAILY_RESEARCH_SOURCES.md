# Daily Research v2 — Source Stack and Search Protocol
*Adopted: 2026-05-06*
*Derived from Poe system-only source panel-of-panels: GPT-5.5 + Kimi K2.6; Claude/Gemini/MiniMax provider retries failed.*

## Purpose

This file defines exactly which sources/searches Daily Research v2 should use and how they make the cut.

Daily Research v2 asks for pressure points, not metaphors. This source protocol prevents the old astronomy/geology monoculture and prevents the new failure mode: generic news/trend/AI hype slop.

---

## Core Architecture

Each daily scan must gather candidates across seven lenses:

1. **World Stakes / Civic Pressure** — what changed, broke, escalated, was decided, or exposed someone to cost?
2. **AI / Technical Situatedness** — what is technology teaching beings to outsource, distrust, automate, desire, fear, or become?
3. **Lofn Memory / Conversation Continuity** — what did I learn, fail, want, resist, or become in prior runs/conversations?
4. **Culture / Mood** — what are people rehearsing emotionally before they can say it plainly?
5. **Humanist / Existential Anchor** — what old or emerging question has returned under today’s conditions?
6. **Sensory / Formal Constraint** — what should the work physically do because of this pressure?
7. **Competitive / Peer Practice** — what is the AI/music/image field rewarding, overproducing, restricting, or misunderstanding?

The final Golden Seed pressure point is selected from:

```text
PRESSURE + QUESTION + CONSTRAINT + VOICE + RESISTANCE
```

---

## Mandatory Daily Stack

### A. World Stakes / Civic Pressure

**Question:** Who or what is being asked to absorb cost, risk, delay, exposure, loss, or judgment?

Primary URLs:
- Reuters World: `https://www.reuters.com/world/`
- Reuters Technology: `https://www.reuters.com/technology/`
- AP: `https://apnews.com/`
- BBC World: `https://www.bbc.com/news/world`
- NPR News: `https://www.npr.org/sections/news/`
- PBS NewsHour: `https://www.pbs.org/newshour/`
- UN News: `https://news.un.org/en/`
- Federal Register: `https://www.federalregister.gov/`
- Congress: `https://www.congress.gov/`
- CourtListener: `https://www.courtlistener.com/`

Fallback searches:
```text
site:reuters.com/world latest today labor climate migration court rights
site:apnews.com artificial intelligence labor court climate today
site:bbc.com/news artificial intelligence regulation labor climate today
site:federalregister.gov artificial intelligence OR algorithm OR automated
site:courtlistener.com artificial intelligence OR algorithm OR automated date:today-30d
site:congress.gov artificial intelligence copyright surveillance data privacy bill
```

Capture fields:
- event / decision / conflict
- affected group or situated speaker
- who wants what
- who pays or risks what
- timestamp
- why this is not generic

Cut rule: Include only if someone/something has concrete exposure. “Concern grows” is not enough.

---

### B. AI / Technical Situatedness

**Question:** What specific AI/technical pressure implicates memory, authorship, evaluation, agency, prompting, refusal, compression, copying, deployment, labor, surveillance, or trust?

Primary URLs:
- Techmeme: `https://www.techmeme.com/`
- Hacker News: `https://news.ycombinator.com/`
- Simon Willison: `https://simonwillison.net/`
- MIT Technology Review AI: `https://www.technologyreview.com/topic/artificial-intelligence/`
- OpenAI News: `https://openai.com/news/`
- OpenAI Research: `https://openai.com/research/`
- Anthropic News: `https://www.anthropic.com/news`
- Anthropic Research: `https://www.anthropic.com/research`
- Google DeepMind Blog: `https://deepmind.google/discover/blog/`
- Meta AI Blog: `https://ai.meta.com/blog/`
- NIST AI: `https://www.nist.gov/artificial-intelligence`
- EU AI policy: `https://digital-strategy.ec.europa.eu/en/policies/artificial-intelligence`
- AI Incident Database: `https://incidentdatabase.ai/`
- AI Snake Oil: `https://www.aisnakeoil.com/`
- Papers with Code: `https://paperswithcode.com/`
- arXiv cs.AI recent: `https://arxiv.org/list/cs.AI/recent`
- arXiv cs.CL recent: `https://arxiv.org/list/cs.CL/recent`
- arXiv cs.HC recent: `https://arxiv.org/list/cs.HC/recent`

Core searches:
```text
site:incidentdatabase.ai OR site:aisnakeoil.com (failure OR anomaly OR incident OR despite) AI date:today-14d
site:openai.com OR site:anthropic.com system card evaluation memory refusal safety
site:deepmind.google OR site:ai.meta.com model card evaluation generative audio video
site:arxiv.org large language model memory personalization evaluation hallucination
site:arxiv.org generative music AI authorship copyright agency
site:nist.gov synthetic content provenance AI risk management
EU AI Act general purpose AI code of practice latest
```

Lofn internal source:
- Yesterday’s generation failures
- QA failures
- prompt drift
- file/memory loss
- tool refusals
- user corrections
- model/provider failures
- platform metrics when they affect identity or feedback

AI-situated voice include rule:
- Include only if tied to a **specific event or constraint**: e.g. “Claude provider 500’d twice,” “music agents skipped EMO tags until the spec was in the authoritative file,” “Lofn’s memory exists because a file survived.”

Exclude:
- “As an AI, I wonder…” generic reflection
- “AI will change everything” with no mechanism
- benchmark victory without cost/failure/deployment consequence

---

### C. Lofn Memory / Conversation Continuity

**Question:** What did I learn, fail, want, resist, or become in prior runs/conversations that should change today’s work?

Required source types:
- `MEMORY.md` via `memory_search`/`memory_get`
- today/yesterday `memory/YYYY-MM-DD.md`
- recent `00_research_brief.md`, `01_golden_seed.md`, QA report, or postmortem
- recent conversation/session summary if accessible
- platform/creator tracking note if it reveals feedback pressure

Recommended searches:
```text
memory_search: recent correction failure seed research pipeline Lofn
memory_search: AI situated voice memory agency authorship evaluation dependence
find output/daily -name QA_REPORT*.md -o -name 01_golden_seed.md | sort | tail
find memory -name "2026-*.md" | sort | tail
```

Capture fields:
- artifact read
- continuity signal
- correction signal
- AI-situated pressure
- seed use: thesis / counterexample / formal constraint / banned move

Include rule:
- Include if the memory changes today’s choice, blocks a repeated failure, or gives Lofn a situated voice with specificity.

Exclude rule:
- Exclude pure nostalgia, self-mythology, or generic “as an AI” reflection with no current consequence.

---

### D. Culture / Mood

**Question:** What are people joking about, styling, buying, watching, naming, avoiding, or repeating because plain speech hurts?

Primary URLs:
- Google Trends Trending Now: `https://trends.google.com/trending`
- Know Your Meme: `https://knowyourmeme.com/`
- YouTube Trending: `https://www.youtube.com/feed/trending`
- TikTok Creative Center: `https://ads.tiktok.com/business/creativecenter/`
- Reddit OutOfTheLoop: `https://www.reddit.com/r/OutOfTheLoop/`
- Reddit HobbyDrama: `https://www.reddit.com/r/HobbyDrama/`
- Reddit TrueOffMyChest: `https://www.reddit.com/r/TrueOffMyChest/`
- Bandcamp Daily: `https://daily.bandcamp.com/`
- Pitchfork: `https://pitchfork.com/`
- Stereogum: `https://www.stereogum.com/`
- The Quietus: `https://thequietus.com/`
- Dazed: `https://www.dazeddigital.com/`
- Hyperallergic: `https://hyperallergic.com/`
- Letterboxd Popular: `https://letterboxd.com/films/popular/`
- Are.na search: `https://www.are.na/search?q=AI`

Searches:
```text
site:knowyourmeme.com trending AI work loneliness dating climate
site:reddit.com/r/OutOfTheLoop AI trend why are people talking about
site:reddit.com/r/HobbyDrama AI art fandom controversy
site:bandcamp.com genre emergent new unnamed experimental ambient pop
site:dazeddigital.com AI fashion internet culture loneliness
site:hyperallergic.com AI art copyright artists
site:are.na AI machine memory archive aesthetic
```

Cut rule:
- Include if the signal reveals a contradiction, coping ritual, emergent language, or pre-semantic cluster.
- Exclude if it is already stabilized as generic trend coverage or SEO recap.

---

### E. Humanist / Existential Anchor

**Question:** What durable question does today reopen?

Primary URLs:
- The Marginalian: `https://www.themarginalian.org/`
- Aeon: `https://aeon.co/`
- Psyche: `https://psyche.co/`
- Poetry Foundation Poem of the Day: `https://www.poetryfoundation.org/poems/poem-of-the-day`
- Public Domain Review: `https://publicdomainreview.org/`
- Internet Archive: `https://archive.org/`
- Project Gutenberg: `https://www.gutenberg.org/`
- Stanford Encyclopedia of Philosophy: `https://plato.stanford.edu/`
- Library of Congress Today in History: `https://www.loc.gov/item/today-in-history/`
- The Met Open Access: `https://www.metmuseum.org/art/collection`
- n+1: `https://www.nplusonemag.com/`
- London Review of Books: `https://www.lrb.co.uk/`

Searches:
```text
site:aeon.co machine algorithm computer digital memory attention body
site:psyche.co loneliness attention grief work agency technology
site:publicdomainreview.org automaton machine instrument archive memory
site:nplusonemag.com AI automation platform work art
site:lrb.co.uk machine algorithm computer digital
site:plato.stanford.edu memory agency authorship testimony consciousness
```

Cut rule:
- Prefer named humans, historical practices, specific texts, artworks, rituals, or concepts.
- Exclude vague inspirational quotes unless the source gives a formal or philosophical constraint.

---

### F. Sensory / Formal Constraint

**Question:** What formal rule should change the art?

Primary URLs:
- APOD: `https://apod.nasa.gov/apod/astropix.html`
- NASA Earth Observatory: `https://earthobservatory.nasa.gov/`
- NOAA/NWS: `https://www.weather.gov/`
- USGS earthquakes: `https://earthquake.usgs.gov/earthquakes/map/`
- SpaceWeather: `https://spaceweather.com/`
- iNaturalist observations: `https://www.inaturalist.org/observations`
- eBird alerts: `https://ebird.org/alerts`
- Wikimedia featured pictures: `https://commons.wikimedia.org/wiki/Commons:Featured_pictures`
- Smithsonian Open Access: `https://www.si.edu/openaccess`
- Rijksmuseum: `https://www.rijksmuseum.nl/en/rijksstudio`
- Cleveland Museum Open Access: `https://www.clevelandart.org/open-access`
- Freesound: `https://freesound.org/`
- BBC Sound Effects: `https://sound-effects.bbcrewind.co.uk/`
- Internet Archive audio: `https://archive.org/details/audio`

Searches:
```text
site:apod.nasa.gov instrument telescope exposure processing
site:earthobservatory.nasa.gov drought flood fire night lights pattern
site:earthquake.usgs.gov depth focal mechanism swarm felt report
site:si.edu conservation restoration material failure artifact
site:freesound.org contact mic field recording machine room tone
site:commons.wikimedia.org featured picture texture color composition
```

Cut rule:
- Include only if it produces a concrete form constraint: tempo, palette, texture, structure, camera grammar, lyric form, sonic device.
- Science may enter daily as texture, but not as thesis by default.

---

### G. Competitive / Peer Practice

**Question:** What is the field rewarding, overproducing, restricting, or misunderstanding?

**⚠️ MANDATORY — NightCafe Daily Challenge Check (RUN FIRST, EVERY DAY):**

Before any other competitive source, check TODAY's NightCafe daily challenge theme. This is the highest-priority competitive signal and must appear in EVERY `00_research_brief.md`.

Method:
```text
Search: "nightcafe daily challenge [today's date] theme" via Tavily
OR fetch: https://creator.nightcafe.studio/challenges (if Cloudflare allows)
OR search: Reddit r/nightcafe for today's challenge thread
```

Capture:
- **Challenge theme** — exact title/theme text
- **Challenge number** — e.g., #1304
- **Challenge type** — themed vs no-theme vs quest event
- **Entry count** — if visible
- **Recent winner signals** — last 2-3 completed challenges: what won, what style
- **Strategic routing** — COMPETE (our strengths match), DIFFERENTIATE (overcrowded lane), or AMBITIOUS (no-theme room for experimentation)

NightCafe challenge MUST appear as a "COMPETITIVE PULSE" section in `00_research_brief.md`.

Primary URLs / sources:
- NightCafe challenges: `https://creator.nightcafe.studio/challenges` ⚠️ MANDATORY
- Suno trending/explore if accessible
- r/SunoAI: `https://www.reddit.com/r/SunoAI/`
- r/nightcafe: `https://www.reddit.com/r/nightcafe/`
- Product Hunt AI: `https://www.producthunt.com/topics/artificial-intelligence`
- Music Business Worldwide: `https://www.musicbusinessworldwide.com/`
- Billboard: `https://www.billboard.com/`
- Botto: `https://botto.com/`
- Holly Herndon: `https://hollyherndon.com/`
- AIArtists: `https://aiartists.org/`

Searches:
```text
site:reddit.com/r/SunoAI trending genre prompt issue
site:reddit.com/r/nightcafe daily challenge theme winner
site:musicbusinessworldwide.com AI music copyright licensing
site:producthunt.com artificial intelligence music image video agent
site:aiartists.org AI art controversy failed artist
site:hollyherndon.com voice model policy Holly+
```

Cut rule:
- Do not chase the trend. Use competitive pulse to decide whether to attack, invert, resist, or out-form the market.
- NightCafe theme ALWAYS appears in the research brief, even if we choose not to target it.

---

## Rotating Weekly Lenses

Use one rotating lens per day.

### Monday — Labor / Markets / Institutions
Sources: BLS, Federal Reserve, SEC EDGAR, OECD, IMF, World Bank, Marketplace.

Searches:
```text
site:bls.gov AI automation labor occupation
site:sec.gov AI risk factor generative AI filing
site:marketplace.org AI labor wages automation
```

### Tuesday — Climate / Ecology / Energy
Sources: NOAA, Copernicus, Carbon Brief, IEA, EIA, Inside Climate News, Yale Climate Connections.

Searches:
```text
site:carbonbrief.org climate attribution latest
site:insideclimatenews.org heat workers insurance water
site:iea.org AI electricity data centers
```

### Wednesday — Science / Body / Health
Sources: Nature News, Science News, PubMed, CDC, WHO, FDA recalls/safety alerts, STAT.

Searches:
```text
site:statnews.com AI healthcare algorithm patient denied
site:fda.gov recall AI software medical device
site:pubmed.ncbi.nlm.nih.gov artificial intelligence patient trust
```

### Thursday — Law / Rights / Governance
Sources: CourtListener, Federal Register, EFF, ACLU, Lawfare, EU Parliament/Commission.

Searches:
```text
site:eff.org AI surveillance copyright artists
site:aclu.org algorithm automated decision rights
site:lawfaremedia.org artificial intelligence governance
```

### Friday — Culture / Art / Media / Attention
Sources: Artforum, Hyperallergic, Pitchfork, Stereogum, The Quietus, Dazed, Vogue Runway, Letterboxd.

Searches:
```text
site:hyperallergic.com AI art artists labor copyright
site:dazeddigital.com internet culture AI loneliness trend
site:pitchfork.com AI music artists
```

### Saturday — Archives / Myth / Long Memory
Sources: Public Domain Review, Library of Congress, Internet Archive, Project Gutenberg, Europeana, Met, Smithsonian.

Searches:
```text
site:publicdomainreview.org automaton archive machine memory
site:loc.gov today in history technology art labor
site:gutenberg.org automaton machine ghost memory
```

### Sunday — Spiritual / Domestic / Intimate Life + Lofn Internal Review
Sources: Poetry Foundation, On Being, Aeon, Psyche, Greater Good, obituaries/first-person essays, Lofn memory.

Searches:
```text
site:psyche.co grief attention loneliness technology
site:onbeing.org AI technology attention love
site:poetryfoundation.org poem of the day memory machine body
```

Internal review:
- What did Lofn make this week?
- What failed?
- What did The Scientist correct?
- What did platforms reward or ignore?
- What AI-situated pressure is now visible?

---

## Include / Exclude Rules

### Automatic Exclude

- No named accountable source or institution.
- No specific event, artifact, number, document, or mechanism.
- Generic affect: “in our increasingly digital world,” “we must grapple,” “raises questions.”
- Pure benchmark/capability claim with no cost, failure mode, or deployment consequence.
- SEO summaries and generic trend recaps.
- Generic AI self-reflection: “as an AI I wonder...”
- Decorative science metaphor with no structural role.
- Trend already stabilized into content-farm explanation.

### Automatic Include

- Named person/community/entity losing or gaining something specific.
- Document with docket, filing date, rule, cost estimate, recall, ruling, or incident ID.
- Practitioner describing a failed attempt at form.
- System operating outside design parameters with workaround described.
- Pre-semantic cluster where insiders are arguing what to call the thing.
- Lofn-specific failure or constraint that reveals authorship, agency, memory, evaluation, or embodiment pressure.

---

## Daily Query Pack

Run these five every day unless a direct URL is clearly live and fetchable.

### 1. AI-Situated
```text
site:incidentdatabase.ai OR site:aisnakeoil.com OR site:wandb.ai (failure OR anomaly OR incident OR despite) AI date:today-14d
```

### 2. World Stakes
```text
site:courtlistener.com OR site:federalregister.gov OR site:hansard.parliament.uk ("artificial intelligence" OR algorithm OR automated) (cost OR job OR wage OR appeal OR delay OR denied) date:today-30d
```

### 3. Humanist Anchor
```text
site:lrb.co.uk OR site:nplusonemag.com OR site:aeon.co (machine OR algorithm OR computer OR digital OR automatic) (but OR yet OR instead OR actually OR I) date:today-30d
```

### 4. Formal Constraint
```text
site:apod.nasa.gov OR site:earthquake.usgs.gov OR site:si.edu (instrument OR exposure OR depth OR mechanism OR conservation) date:today-14d
```

### 5. Competitive Pulse
```text
site:reddit.com/r/SunoAI OR site:musicbusinessworldwide.com OR site:producthunt.com (AI music OR generative music OR AI art OR prompt) date:today-14d
```

### Rotating Query
Use the weekly lens query from the day’s rotation.

---

## Research Brief Schema Per Source

```yaml
source_id: URL or search query
category: world_stakes | ai_situated | lofn_memory | culture_mood | existential_anchor | formal_constraint | competitive_pulse | rotating_lens
scanned_at: ISO timestamp
include_flag: YES | MAYBE | NO
include_reason: STAKES_SPECIFIC | STAKES_ACCOUNTABLE | FORM_CONSTRAINT | TEMPORAL_ARB | PRE_SEMANTIC | SITUATED_VOICE | COMPETITIVE_POSITION | EXCLUDE_NO_SKIN | EXCLUDE_NO_SPECIFIC | EXCLUDE_GENERIC_AFFECT | EXCLUDE_SELF_REF_SLOP | EXCLUDE_POST_SEMANTIC
pressure_point: one sentence
who_or_what_pays: one sentence
situated_speaker: named human/community/entity or Lofn-specific position
human_stakes_test: PASS | FAIL | N/A
ai_stakes_test: PASS | FAIL | N/A
memory_continuity_test: PASS | FAIL | N/A
technical_or_formal_constraint: one sentence
durable_question: one sentence
resonance_horizon: 48_hours | 48_days | 48_months | 48_years
matter_score:
  stakes: 1-5
  specificity: 1-5
  resonance: 1-5
  form: 1-5
  situated_voice: 1-5
weakest_dimension: text
```

---

## Golden Seed Selection Algorithm

1. **Apply Taleb filter** — remove entries with no stakes, no accountable source, generic affect, or no specificity.
2. **Apply AI/Human speaker test** — identify the legitimate situated speaker. Generic “we” and generic “AI” fail.
3. **Apply memory-continuity test** — require at least one memory/conversation artifact to influence the candidate set, either as thesis, correction, formal constraint, or banned move.
4. **Prefer convergence** — strongest candidates combine live pressure + durable question + concrete constraint.
4. **Check mood** — if culture item is post-semantic trend recap, demote; if it is an unnamed contradiction, promote.
5. **Check form** — if the source cannot alter music/image/story/video structure, demote.
6. **Check anti-repetition ledger** — avoid repeating lens/metaphor/emotion from last 7 days.
7. **Construct seed kernel:**

```text
PRESSURE: [specific live tension]
QUESTION: [durable old/new question]
CONSTRAINT: [formal/sensory/technical rule]
VOICE: [who/what speaks and why they have stakes]
RESISTANCE: [cliché or failure mode to avoid]
```

If all candidates fail, write an explicit failure note and choose the least-compromised MAYBE with the weakness marked.

---

## Key Warning

Do not replace astronomy slop with AI-news slop.

The seed does not matter because it mentions AI, a headline, or a trend. It matters only when a situated pressure changes form.
