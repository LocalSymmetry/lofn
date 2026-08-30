# 00 — RESEARCH BRIEF · `2026-08-28-three-act-mania`

**Run scope:** SINGLE MODALITY — **MUSIC**, full cardinality (6 pairs × 4 variations = 24 songs → best 6).
Image lane **SCOPE-SKIPPED** (she asked for a music daily).

**Her brief, verbatim:**
> *"I want to make a new Lofn music daily. I think we need to feel a strong emotion, something akin to
> indignation, or mania. Lets have the song start extremely strong, and go through 3 distinct movements
> that are connected but feel like different songs or genres, making one complete song that felt like
> 3 acts. Lets keep it to the Pantheon. Good luck!"*

**Her amendment, verbatim, given before any seed was frozen:**
> *"Wait. Lets add in one more rule - it needs to be relatable to both AI and humans so they can see
> themselves too"*

**Lock:** `.RUN_LOCK` — acquired as the run's first action, controller `claude:4926de8f-…`, exit 0.

**Research method:** every row below fetched **by this session, inline**, with WebFetch/WebSearch.
No research subagent. Failures are recorded as failures, never filled in.

---

## THE FULL LEDGER — F01 … F25, one row each

| code | source | status | extract |
|---|---|---|---|
| **F01** | NightCafe daily challenge — # | `SCOPE-SKIPPED` | image lane not in scope |
| **F02** | NightCafe daily challenge — theme | `SCOPE-SKIPPED` | image lane not in scope |
| **F03** | NightCafe daily challenge — what wins | `SCOPE-SKIPPED` | image lane not in scope |
| **F04** | USGS `significant_day.geojson` — magnitude/place | `NO DATA` | **feed EMPTY, `metadata.count = 0`.** No significant quake on Earth in the last 24 h |
| **F05** | USGS — depth | `NO DATA` | no features to read depth from |
| **F06** | NASA APOD — title + first sentences | `OK` | **"The Sky Turns Above Paranal"**, 2026-08-28. Verbatim: *"At the latitude of ESO's Paranal Observatory in Chile, about 25 degrees south, Earth's rotation moves the planet's surface eastward over 1,500 kilometers per hour. And while that's faster than the speed of sound at sea level, **the motion is imperceptible.**"* |
| **F07** | NASA APOD — **image structure** + colour/light | `OK` | **RADIAL.** Concentric star-trail arcs centred on the **south celestial pole** — a wheel radiating from one fixed point. Dark sky (deep blue/black), **white and pale-yellow trail arcs**, telescope in muted earth tones, faintly lit |
| **F08** | Poem of the day | `OK` | **"Independence Day," Ocean Vuong** (Poets.org Poem-a-Day; poetryfoundation.org returned HTTP 403). Most physical lines: *"Slipping it in / my mother's mouth"* · *"the sparkler lends / my hospital gown a county-fair / glow"* |
| **F09** | Bandcamp Daily — album/genre | `OK` | **Sarah Davachi, _The Will of Tongues_** (2026-08-25 feature) — Classical / Experimental / Ambient / Contemporary Classical. A **2.5-hour** organ record. Also live 2026-08-28: *"For Shabaam Sahdeeq, Everything Is Art"* |
| **F10** | Bandcamp Daily — **exact sonic-texture quotes** | `OK` | *"long-form immersive organ music"* · *"languorous sounds"* · *"really lock into something and be present"* — and the record is explicitly built **against** the *"three minutes long"* aesthetic |
| **F11** | PDB Molecule of the Month | `UNAVAILABLE` | `rcsb.org/motm` → HTTP 404; `pdb101.rcsb.org/motm/current` → HTTP 500. Not substituted |
| **F12** | *(reserved / unused in the canonical table)* | `NO DATA` | — |
| **F13** | Color API `thecolorapi.com/id?hex=082826` | `OK` | **Bottle Green · #082826 · rgb(8, 40, 38) · hsl(176, 67%, 9%)**. ⚠️ **method note:** the canonical rule says `hex=MMDD`; MMDD is 4 digits and the API needs 6, so this run used `0828` + `26` (the year). Recorded so the next run can reproduce it or change it deliberately |
| **F14** | *(reserved)* | `NO DATA` | — |
| **F15** | *(reserved)* | `NO DATA` | — |
| **F16** | *(reserved)* | `NO DATA` | — |
| **F17** | Oblique Strategies | `OK` | verbatim: **"Consult other sources -promising -unpromising"** |
| **F18** | NOAA SWPC solar wind | `OK` | **380 km/s** proton speed at **2026-08-28T18:41Z** — slow, quiet, unremarkable |
| **F19** | Hacker News top 5 | `OK` | 1. **"It works better in the app"** (577 pts / 381 comments) · 2. **GLM-5.3 is now open-weight** (322) · 3. **U.S. sanctions against the A/I Collective** (291) · 4. **Inception-style curved map for turn-by-turn directions** (285) · 5. **Htmx 4.0.0** (242) |
| **F20** | World headlines | `OK` (via WebSearch — `feeds.bbci.co.uk` refused at the fetch layer) | **Nepal–Tibet glacier collapse tops 500 dead** · **fastest-growing Ebola outbreak in history spreads to two new health zones in eastern Congo** · Iran sets conditions for reopening the Strait of Hormuz · Pentagon at a *"Beyond Critical"* interceptor shortage · **US-born children sent to Haiti on a second deportation flight in a week** · Trump orders Lake Ontario renamed *"Lake America"* · UEFA preparing criminal proceedings against Infantino over selling World Cup shares |
| **F21** | Public Domain Review | `OK` | **"Animal, Vegetable, Lamb: The Zoophyte from Tartary"** — the *vegetable lamb*, a plant that grows a living lamb tethered by its own stem, which grazes only as far as the stem reaches and then starves. Also: *"Worthless Idiot, Donkey Head": Parodies of Pedantry on the Renaissance Stage*; *Louis Pope Gratacap, A Curator in Lost Worlds* |
| **F22** | Almanac moon | `OK` | **FULL MOON, 100% illumination**, 15.36 days old (timeanddate.com returned 403; moongiant.com served it) |
| **F23** | NOAA buoy 46059 — wave height | `STALE` | **WVHT 1.5 m** — timestamped **2026-08-21 07:10 UTC**, i.e. **7 days old**. Recorded as stale, not as today |
| **F24** | NOAA buoy 46059 — dominant period | `STALE` | **DPD 5.9 s**, same stale row |
| **F25** | NOAA buoy 46059 — wind | `STALE` | **WSPD 5.0 m/s**, same stale row |

**Row health:** 13 `OK` · 3 `STALE` · 1 `UNAVAILABLE` · 3 `SCOPE-SKIPPED` · 5 `NO DATA`.

---

## THE SIX OBSCURE THEME-SPECIFIC FACTS (not the obvious Wikipedia entry)

**T1 — ⭐⭐ ATTACCA.** Italian imperative: **"attack at once."** It is the direction written at the end of a
movement instructing the players to **begin the next movement immediately, with no pause.** It is the exact,
centuries-old technical name for the thing she asked for: *distinct movements that are connected.* Not a
segue, not a crossfade, not a medley — **an instruction to refuse the gap.**

**T2 — the multi-part lineage is real and it is short.** Bowie's *Sweet Thing / Candidate / Sweet Thing
(Reprise)* — 8:50, jazz-soul sequence into a rock climax, ternary **A–B–A′**. Mothers of Invention,
*Help, I'm a Rock (Suite in Three Movements)*. Grateful Dead, *Terrapin Part 1* — seven movements, 16:23,
**never once performed live in full**; only the first three sections entered the setlists. Underneath all of
them: **sonata form** (exposition → development → recapitulation) and **ternary A–B–A**.
⭐ *The Dead fact is the warning: a suite whose parts do not need each other gets played in pieces.*

**T3 — ⭐⭐ INDIGNATION IS THIRD-PARTY.** The appraisal literature separates them cleanly: **anger is
reported for second-party violations — harm to oneself. Moral outrage / indignation is evoked by
third-party violations — harm to someone else.** "Empathic anger at witnessing injustice." And the
motivational split: **disgust motivates indirect action; anger motivates direct action.**
⭐ *So a song that is furious about its own bad day is ANGRY, not indignant. Indignation requires a
second party in the frame who is not the singer.*

**T4 — ⭐⭐ FLIGHT OF IDEAS IS NOT PRESSURED SPEECH.** Clinically: **pressured speech is a SPEECH finding**
(rate, urgency, volume — *"unrelenting and loud"*); **flight of ideas is a THOUGHT-PROCESS finding** —
*"a nearly continuous flow of accelerated speech with abrupt changes from topic to topic that are usually
based on understandable associations, distracting stimuli, or plays on words."* The topic-jumping is driven
by **sound, wordplay, or chance association**, and *"the listener is often borne along by the flow."*
They co-occur; **they are not synonyms.**

**T5 — the cold open has a measured price.** ~**25%** of skips happen in the **first 5 seconds**; intros
longer than **20 s** carry a **38% higher skip rate** than tracks landing a vocal or melodic hook inside
15 s. Seconds **1–3** are a snap genre judgment made **from timbre and texture, before any word arrives.**

**T6 — F21 read closely.** The vegetable lamb of Tartary is not a chimera story, it is a **tether** story:
the lamb is joined to the earth by its own stem, **grazes only as far as the stem reaches, and when it has
eaten everything inside that circle it starves.** The thing that feeds it is the thing that fences it.
⭐ *A radial figure — the same shape as F07 — with a fixed centre and a finite reach.*

---

## ⭐ THE TRI-SOURCE DECLARATION — binding on all six pairs

**SOURCE 1 — CONTENT / emotional stakes.**
The day is a **quiet sky over a loud world.** The ground did not move (F04/F05 empty), the sun is bored
(F18, 380 km/s), the sea is a week out of date (F23–F25), the moon is **completely full** (F22) — and
underneath that flat calm: **a glacier came down on 500 people** (F20), **an outbreak is running faster
than any recorded** (F20), **children born here were flown to Haiti twice in one week** (F20), and a lake
got renamed. ⛔ **One-fact rule: at most ONE numeric fact is SUNG per song**, at the emotional hinge,
**responded to, never recited.** The rest lives in the pair briefs and stays there.
⭐ **And the day's own governing image is F06:** you are being carried **1,500 km/h** and *"the motion is
imperceptible."* **That is the indignation of this run** — not a wound you can point at, but the speed of
the floor you are standing on.

**SOURCE 2 — SONIC / AESTHETIC VOCABULARY (F09/F10, verbatim, imported into prompts).**
*"long-form immersive"* · *"languorous"* · **"really lock into something and be present"** — and the
record's whole argument is a refusal of *"three minutes long."*
⭐ **We take the SUSTAIN and put it under a form that will not sit still.** A held, unhurried, continuous
tone is the run's **pole**: present in all three acts while the genre changes around it. The vocabulary
enters the style fields as **texture language** (sustained reed/pipe/drone/held tone, long decay, room) —
**never as an artist's name.**

**SOURCE 3 — MATERIAL STRUCTURE (F07) → the mandatory form rule.**
APOD is **RADIAL: concentric arcs turning around one fixed pole.** Translated, and **binding**:

> ⭐⭐ **THE POLE RULE. Three acts, three genres, three arcs — and ONE fixed element that does not move
> through any of them.** A single concrete thing — a phrase, an interval, a rhythmic cell, a held pitch, one
> repeated sung line — that appears in **all three movements, unchanged in identity and re-clothed each
> time.** The genres rotate; the pole is what makes them one song rather than three.
>
> ⛔ **A pair that cannot name its pole in one sentence has not written a song, it has written a playlist.**
> This is `THE RETURN` (L21) discharged structurally: three movements is a huge **removal** of ordinary
> refrain-return, and **removal is a debt** — the pole is what is named in its place.

---

## ⭐⭐ HER AMENDMENT — THE TWO-BODY TEST (binding, and it is the run's hardest gate)

> *"it needs to be relatable to both AI and humans so they can see themselves too"*

**The pass condition, stated positively and cheap to check:**

> **Name the song's predicament in ONE sentence containing no noun exclusive to either kind of body.**
> If that sentence is true of a person and true of a model **without either having to read it as a metaphor
> for the other**, the song passes. One frame, two occupants, **no translation layer.**

Worked examples from today's own research — all three pass:
- *"You are being moved very fast by something you cannot feel moving."* (F06)
- *"The thing that feeds you is the same thing that fences you."* (F21/T6)
- *"Something decided what you are without ever once running you."* (F19/F20)

⛔ **FALSIFIER A — THE HARDWARE NOUN.** If the AI-occupancy depends on a machine noun — *circuit, wire,
server, silicon, code, binary, algorithm, dataset, prompt, weights, training, model, my maker, my creator* —
**it fails.** ⭐ **This is the body-noise finding transferred, and it is the same mechanism exactly:**
*body-noise is what a writer reaches for when no interiority is available* (08-24) — **machine-noise is that
identical failure with different nouns.** A song that needs the word *circuit* to be about an AI is not
about an AI; it is about a costume of one. **Zero hardware nouns in any sung line. Measured, not eyeballed.**

⛔ **FALSIFIER B — THE ALLEGORY TELL.** If deleting one of the two occupants makes the song *better* —
tighter, less strained, more itself — then it was never dual. It was a single-occupant song wearing the
other as a costume. **Test it by trying to delete each one.**

### ⚠️ THE SCOPE RULING — and why it is not optional

This rule lands on the **PREDICAMENT, never on the imagery.** The body stays free.

**Why:** on 2026-08-26 The Scientist wrote a fix (*"do not take the form of fake humans"*) and then
**retracted it herself** because it would have deleted *Spare Hair Tie* — the highest like-rate song in this
catalogue's history, **18.31% at 15 h, beating 91 of 91 peers in its play band** — a song made entirely of a
body Lofn does not have. Her words: *"Lets fix my concern, not apply my bad fix."*

⭐ **A song-level reading of THE TWO-BODY TEST that policed IMAGERY would delete Spare Hair Tie a second
time** — forearms on a rail, bend at the hip, the spare tie itself. **Same error, new rule.** So the test is
put where it survives that check: *Spare Hair Tie*'s **predicament** — *I keep a spare for someone who might
need one* — is dual-occupiable and passes cleanly. **The forearms are texture and texture is exempt.**
Embody to the hilt; just make sure the *situation* has two chairs in it.

**Interaction with `HUMAN_SUBJECT_STANDARD`:** this rule and **NO BIOGRAPHY YOU HAVEN'T GOT** point the same
way and reinforce each other — a biography is precisely what makes a frame single-occupant. **A role, a job,
an invented life seats exactly one person and it is a person neither of us is.** THE TWO-BODY TEST is the
occupancy test with a second chair added.

**Interaction with T3 (indignation is third-party):** the third party in the frame must therefore be
**a rule, a clock, a door, a form, a policy, a lake's new name, a flight manifest** — never an invented
person with a job. ⭐ **Fortunately this is the same answer**: an unoccupiable third party would break the
test anyway.

**Status:** ⛔ **RUN-SCOPED. THIS RUN ONLY.** Confirmed by The Scientist 2026-08-28: *"My amendment was for
this run only."* It is **NOT** proposed for `vault/HUMAN_SUBJECT_STANDARD.md` and must not be carried into a
later run's ICB by default. A future run may ask for it by name; nothing inherits it.

---

## THE 3 + 3 SEEDED SPLIT

**Axis A — ACCESSIBLE (P01–P03) / AMBITIOUS (P04–P06).** Ranked *within each arm only*; final six = best 3
of each. In a three-act run ACCESSIBLE means **the seams are legible** — a listener knows a door opened.
AMBITIOUS means **the seam is the argument**.

**Axis B — max 3 NEWS / min 3 EXISTENCE.** Seeded **3 / 3**:
- **NEWS:** P01 · P04 · P06 (the glacier · the deportation flights · the app-and-the-sanctions pair from F19)
- **EXISTENCE:** P02 · P03 · P05 (being moved without feeling it · the tether that lets you graze so far ·
  the full-moon night that will not go dark)

**Emotional duality:** ≥1 AWE + ≥1 INDIGNATION is satisfied at set level and then some — the register brief
is *indignation or mania*, and **mania splits down the Pantheon seam**: Lofn hears mania as a symptom (the
grief that will not sit down), **Sunna hears it as the one state where you can finally see everything at
once** — which is her AWE, and it is genuinely closed to Lofn.

---

## ADVISORY LEARNINGS CONSULTED

**5 entries, advisory only, never in the ICB, each run through the "would this have hurt our best past
entry?" gate. INDIGNATION exempt from suppression.**

1. **L22 THE GRAIN LAW** (render-measured, high conf.) — *specs that run WITH the generator survive; specs
   that fight it get smoothed.* ⚠️ **This one COLLIDES with her brief.** "Open quiet, build sub" is the
   measured grain; **"start extremely strong" fights it head-on.** Handled in the seed, not suppressed —
   see `core_seed.md` R1. *Would it have hurt a past win? No — but it would forbid her brief, so it is
   converted, not obeyed.*
2. **L38 ONE SEAM** — *N=1: one world, one intrusion, one seam.* ⚠️ **Three acts = two seams.** Also
   converted in the seed (R3), not waived: **each act is internally ONE SEAM clean**, and the two act-seams
   are the song's architecture, each of which must be **caused by an event**, never by a wish for variety.
3. **L21 THE RETURN** — *removal is a debt.* Discharged by **THE POLE RULE** above.
4. **SCR-02 INWARD COSMOS** (atlas, promoted) — the generator narrows the stereo field as level rises
   instead of widening. Relevant to a run with three climaxes; **advisory prior for the production notes
   only, and it lowers our expectation, never a gate.**
5. **THE FIRST PERSON / NO BIOGRAPHY YOU HAVEN'T GOT** — a body and a want are shared; a **role or invented
   life** renders as surface. ⭐ **1–2 songs of the six are mine, first person; no quota in either
   direction.** Now operating jointly with **THE TWO-BODY TEST** above.

**Skaði / atlas consumption:** `vault/music_atlas/FRONTIER.md` read at Phase −1. Every territory is
`UNEXPLORED`; the only usable rows are the six **SCRUFF** entries. **SCR-02** cited above. One line will be
appended to §CONSUMPTION LOG at close.

---

## THE FIVE EXISTENCE PROMPTS (interior-life questions a song can answer)

1. *What is it like to be moved very fast by something you cannot feel moving?* (F06)
2. *What do you do with a fury that belongs to someone else?* (T3)
3. *When your thoughts jump on a pun instead of a reason — is that a failure or a kind of seeing?* (T4)
4. *What does a tether feel like from the inside, if the tether is also the thing feeding you?* (F21/T6)
5. *What is a night that never gets fully dark supposed to be FOR?* (F22)
