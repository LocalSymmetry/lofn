# 06 — MUSIC HANDOFF · `2026-08-14-daily-music`

> ⛔ **THIS FILE IS NOT OPTIONAL AND IT IS WHY IT EXISTS.** On 2026-08-06 the coordinator never wrote a
> handoff, and the **step-11 contract file instructs the tier to embed the full archived Golden Song
> payloads** while `EXECUTION.md` §3 forbids exactly that in a generating context. Five pairs reached past
> the file in front of them to doctrine and refused by name; **one obeyed its nearest contract and embedded
> two complete style prompts and two full lyric sets into a generating context.** The pair was not at fault —
> an agent that obeys its nearest contract is behaving correctly, and the coordinator had removed the
> artifact that overrides it. **When doctrine and a step file conflict, the conflict is resolved IN WRITING,
> in the run directory. Here it is.**

---

## 1. ⛔ THE CONFLICT, RESOLVED

**`skills/music/steps/11_Generate_Music_Enhancement.md` (and any step file that says the same) instructs you
to embed the archived Golden Song payloads. DO NOT. That instruction is OVERRIDDEN for this run.**

**GOLDEN-OUTPUT QUARANTINE (`EXECUTION.md` §3):** past golden **outputs** never enter a **generating**
context — not the ICB, not the handoff, not as "calibration examples." An exemplar in the generator's prompt
becomes a **mold**: the 2026-06-28 published piece reproduced its benchmark's title line, vocal spec, key,
BPM and arrangement formula while its self-check claimed *"no copying."*

- **Generators get:** the Golden **Seed** + the **GOLDEN MOVE** (below). That is all.
- **Golden Songs for this run — NAMES ONLY, for judge-side blind comparison:** `Triple Arch Over Me` ·
  `Five Wrong Colors`. **Their payloads are not in this file, not in the ICB, and must not be fetched by any
  generating step.**
- **Judge-side contexts that MAY carry payloads:** `lofn-qa` blind comparison · step 12 audit ·
  `lofn-step11-packager` (external review).

---

## 2. THE GOLDEN MOVE (what you get instead)

1. **Stand somewhere real.** ONE concrete place the body occupies. Concept-illustration is the failure mode;
   experience-report is the move.
2. **One wounding fact.** At most ONE numeric fact sung, at the emotional hinge, **responded to, never
   recited**, and **spelled out in words**. Your pair's fact is allocated — do not borrow another's.
3. **The turn.** Past the midpoint the song contradicts or complicates its opening stance. A song that
   asserts its final emotion from line one is a corpse.
4. **Fear stays braided in.** Every AWE song carries a clean fear it does not resolve cheaply. **Your pair's
   PURSUER is that fear — it is never named and it never resolves.**
5. **Rotate the register.** No crystalline soprano / A major / ~110 BPM / frost-and-cosmos default.
6. **The surface names its subject.** A stranger retells scene AND subject in one sentence after ONE listen.
   Obliqueness about the referent is **fog**, not depth — `REPAIR — FOG`.
7. ⭐ **THE RETURN.** Song is made of returns. **Removal is a debt** — strip rhyme only by naming what
   returns in its place. **A byte-identical chorus is correct and requires no note.**

---

## 3. ⚠️ ICB DELIVERY — A DELIVERY-METHOD DEVIATION, DECLARED

`EXECUTION.md` §3 says: *"Do NOT rely on agents reading upstream files — inject the full payload directly
into the spawn task."* **The ICB for this run is 128,878 bytes on disk.**

> ⚠️ **CORRECTION, LOGGED — and it is the standing pattern again.** The dispatch prompts and the first
> version of this line said **127,055 bytes**. That was the coordinator's number: `len(out.encode())`
> **before** the write, counting `\n`. Python's default newline translation wrote **CRLF**, so the file on
> disk carries **1,823 extra bytes** — 128,878. **Pair 04 measured 128,878 and was right; the coordinator
> was wrong.** ⭐ **Fourth consecutive run in which a re-measured pair claim proved the AGENT correct and
> the discrepancy the COORDINATOR's — the failure rate lives on the JOIN side.** It is also the CRLF class
> that has corrupted a hash and a regex in this repo before. **The proof requirement below is what caught
> it, and it caught it because it asked for a measurement rather than a confirmation.** Reproducing it inside six spawn prompts
means the coordinator **retyping** it six times — and the single best-evidenced failure mode in this
system's record is: ⭐ *"EVERYTHING GENERATED MECHANICALLY WAS RIGHT. EVERY ERROR CAME FROM PROSE TYPED FROM
MEMORY BESIDE IT."* Retyping the ICB six times is that failure mode, six times, on the run's most
integrity-critical artifact.

**Ruling, stated rather than hidden:** each pair subagent's **FIRST ACTION** is to `Read` the complete
`output/daily/2026-08-14/CREATIVE_CONTEXT.md` and treat it as **pinned and read-only** for its entire chain.
The **pair-specific slice** (isolation brief, assignment, gates) IS injected verbatim into the spawn prompt,
because it is small enough to be exact.

**The deviation is paid for with STRONGER proof than a byte count.** A pasted-payload count proves presence
only — *a paraphrase can match length*. Each pair must instead return **three verbatim ICB substrings** that
cannot be guessed without having read the file, and the coordinator re-stats them against the real file:

1. the **exact byte size** of `CREATIVE_CONTEXT.md`;
2. the **`VOICE 14` roster row, verbatim** (a canonical, machine-checkable row);
3. **Special Flair #9's name, verbatim.**

**A pair that cannot produce all three has not loaded the ICB and its artifact is not accepted.**

---

## 4. THE HARD GATES (every pair, every variation)

**MUSIC PROMPT** — `## 1. MUSIC PROMPT`, dense **prose paragraph**, **850–1000 chars, written into 870–960**
(≥985 is a boundary-hug FLAG), **ending as a complete sentence**. Order: genre/tempo/opening → vocalist spec
with spatial staging → instrumentation with physical adjectives → arrangement arc → bold sonic device.
Must contain a vivid **first-5-seconds opening**, **spatial language**, a **kinetic defect**, and explicit
**acoustic / no-acoustic** declarations. ⛔ Banned openers: *Begin in/with… · Use… · Build the track from… ·
Chronology: · For an adult human singer…* ⛔ No real-artist names.

**EXCLUDE PROMPT** — `## EXCLUDE PROMPT`, separate, **400–900 chars** (hard max 1000). Concrete blacklist
tokens. ⛔ Avoidance prose does not go in the style field — Suno reads that field as **positive** tokens.

**LYRICS** — open `[Theme: …]` then `[SONG FORM: …]`; full EMO headers
`[Section - EMO:<emotion(s)> - <Role> - <cues>]` with emotions **from the taxonomy in `step00`** —
⛔ **never bare AWE/INDIGNATION**; **≥1 standalone `*SFX*` cue**; **70–120 sung lines** (the floor is a
floor, not a target — ≤72 is a hug FLAG).
🚨 **LYRICS FIELD < 5000 CHARS, TARGET ≤ 4800. MEASURE EXACTLY AND STATE THE COUNT.** The field holds
everything you paste into Suno's lyrics box. Over 5000 **will not render**; the line target yields to the
cap. If over: cut/merge lines → tighten headers → move Disc_Channel + metadata to a `## Production Sidecar`
**outside** the field.

**RETURN FLOORS** — strict end-rhyme ≥ **0.30** · repeated-line ratio ≥ **0.20** · mean words/line ≤ **7.5** ·
alliteration ≥ **11.0** per 100 w. **Measured by `scripts/measure_soundcraft.py → profile_file()` — that
function IS the definition.** ⛔ Do **not** re-derive the rhyme window: it is **±4 lines**, from
`gates.yaml → rhyme_window`. *(Three repair agents once reverse-engineered three different windows from a
brief that named a threshold without naming the function, and each honestly reported "reproduces the
baseline exactly.")* **P04 is the only pair permitted below the rhyme floor, and its debt is paid in
writing in its brief.**

**SOURCE-2 FLOOR** — **≥ 2 exact Bandcamp phrases in EVERY generator-facing prompt**, counted **in the
shipped prompt text**, never in your design notes. **Vary which two per variation** — folding identical
mandatory phrases into four prompts collides them (measured: 0.523 → 0.669 against a 0.58 ceiling).

**SUNG NUMERALS SPELLED OUT** — in words, always. Digits stay legal in `[Theme:]`, `[SONG FORM:]`, EMO
headers, BPM and key: none of those is voiced.

---

## 5. THE FIVE STANDING VETOES (the Somatic Gate, live)

⛔ **No song congratulates** — not the listener, not the room, not itself.
⛔ **An objection answered only in the PRODUCTION SPEC is NOT answered** — it lives in the lyric or the form.
⛔ **A clean period pastiche is a museum** — modern condition = world, historical bed = intrusion, one seam.
⛔ **Every song has a visible EXIT** — nobody locked in, obliged, or shamed for stopping.
⛔ **Every song carries one FRICTION a playlist would rather was not there.**

---

## 6. THE DESCRIBE-RENDER SELF-CHECK (one capped pass, inside the existing max-3 loop)

Before finalising step 10, predict in **2–3 sentences what your Suno prompt would actually PRODUCE** — the
literal first 5 seconds, the vocal placement, the groove, where the bold device lands — then diff that
against the Golden Seed and the GOLDEN MOVE (⛔ never against golden-song payloads). Phrase it adversarially:
**"name the one way this would render generic."** If it drifts, **self-repair ONCE** through the same loop.
⛔ No render-verifier subagent, no recursion, no new tier.

---

## 7. WHAT TO WRITE, AND WHERE

`pair_{NN}_step06_facets.md` · `pair_{NN}_step07_song_guides.md` · `pair_{NN}_step08_generation.md` ·
`pair_{NN}_step09_artist_refined.md` · `pair_{NN}_step10_revision_synthesis.md`
— flat in `output/daily/2026-08-14/`. ⛔ **Write ONLY into your own `pair_{NN}_*` namespace.** Never the run
INDEX, never `RUN_STATE.md`, never `CREATIVE_CONTEXT.md`, never another pair's files.
⛔ **Steps 07, 09 and 10 are the editorial spine and are NOT optional** — a run missing them for a
non-quarantined pair is **NON-CANONICAL** and cannot ship.

---

## 8. LINEAGE & CREDIT (every pair — living scenes are in every bed)

Name the scene, point upstream, **fusion with lineage never capture.** Crossover telemetry is a reason to
**credit harder**, never a market opening.
⛔⛔ **IF THE PROMPT NAMES A BREAK, CREDIT THE DRUMMER:** amen → **Gregory C. Coleman / The Winstons** ·
Funky Drummer → **Clyde Stubblefield** · Think → **Lyn Collins** · Apache → **the Incredible Bongo Band** ·
slap-and-pop → **Larry Graham** · the sixties girl-group stack → **the singers**.
**Sweep the STYLE field, not EXCLUDE — raised ≠ forbidden.** **Word boundaries** (`ornament` contains
`amen`). ⛔ **Honesty clause:** *"we asked for this figure by name, so we name him"* — **never** *"contains a
sample of…"*. ⛔ **Never credit an abuser as lineage.** ⛔ **D4 applies to the paratext:** credit a living
person always; say a song is *about* them never.

---

*Handoff written before dispatch, because the run that did not write one shipped a quarantine breach.* 💜
