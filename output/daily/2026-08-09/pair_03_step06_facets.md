# PAIR 03 — STEP 06 · SCORING FACETS · `2026-08-09_daily_music_genz`

**Pair:** P03 — **HALF A SECOND** · ACCESSIBLE · EXISTENCE · INDIGNATION (low-burn)
**Emotion to nail:** **exulansis** — *the moment you stop trying to explain because the explaining is worse
than the silence.* *(Target drawn from the run's obscure-emotion vocabulary; coinage credited to John Koenig,
`The Dictionary of Obscure Sorrows` — his definitions are not reproduced anywhere in this pair.)*
**Lane (A5):** pluggnb, ~140 BPM half-time. **Person (A4):** two people on a call with the audio a half-second
out of sync, both apologising over each other. **Archetype:** THE LAST WITNESS, INVERTED.

**Continuity Payload Used:** the full frozen `CREATIVE_CONTEXT.md` ICB — **173,669 bytes**, sha256
`297941561ca6880d38c323dcc0fdd739aa6fd970e7293fd7e98e38fb0b882f4b`, read in full and pinned at the head of
every step in this chain: Golden Seed, metaprompt, the complete **104,422-byte** LOFN-PRIME personality YAML,
all **18** panel voices with their objections, the **15 Special Flairs**, the seed genre palette and the seed
music-frames palette.
**Independent integrity proof:** `python3 output/daily/2026-08-09/verify_icb.py` → **`VERDICT: PASS`**
(14/14 checks; 18 speaker tags; personality present as an unbroken 104,422-byte substring; flairs marker present).

**Contract for this step:** produce the rubric that later steps are scored against. `vault/gates.yaml →
step06_min_facets: 8`; the legacy step file's "exactly 5" yields to the gate (2026-07 OVERRIDE / `EXECUTION.md`
§4). **Twelve** facets below, weighted. Each one is specific to *this* pair's assigned devices — nothing here
would score another pair's song, which is the `step06_max_pair_similarity: 0.50` requirement doing its job at
the source rather than at the audit.

---

## THE TWELVE FACETS (weighted — the rubric steps 08–11 are judged against)

| # | facet | weight | what a failing prompt looks like |
|---|---|---:|---|
| **F1** | ⭐ **THE LAG IS PERFORMED, NOT PROCESSED.** Two continuous voices held at a **fixed one-line distance** by actual sung material — the pitched-up ad-lib channel carries the tail of the lead's previous line under every line she sings. The interval never varies and never resolves until the hinge. It must survive a phone speaker with the sub rolled off. | **0.98** | the lag is specified as a delay setting, a slapback, a "1/8 dotted echo," or any mix instruction. THE GRAIN LAW: the renderer has been measured smoothing exactly that class. A delay is a plug-in; a canon is a song. |
| **F2** | ⭐ **HALF-LINE ECHO, EVERY HOOK LINE, EXACTLY ONE WORD SWAPPED.** The back half of each chorus line repeats the front half's rhythm with precisely one token changed — four times per chorus, no exceptions, no line where two words move. | **0.95** | a "roughly parallel" second half; two words changed; a line where the echo is only semantic and not rhythmic. This is the device that **pays the end-rhyme debt** — a loose echo is an unpaid debt. |
| **F3** | ⭐ **THE FIRST SOUND IS THE BIGGEST HOOK.** Bar one is the full chorus. No count-in, no pad, no bell arpeggio, no breath, no code-scratch, no room tone establishing itself. | **0.92** | anything before the hook. Even two bars of bells is a failure of A3 — and `HOOK AT ZERO` is one of only three duration grammars in the run that the renderer has been measured **keeping** (near-silent openings and terminal silence survive; mid-song voids do not). |
| **F4** | ⭐ **THE MID-WORD SPILL AT EVERY STANZA BOUNDARY.** Each verse-family stanza ends on a word broken in half; the next stanza opens with the completion. No other pair in this run enjambs across a stanza boundary. | **0.90** | a stanza that ends on a whole word, or a break placed at a syllable boundary so tidy it reads as a line-break rather than an interruption. Break **against** the syllable where possible (`sec—/—ond`, `sen—/—tence`). |
| **F5** | ⭐ **THE SPEAKER SWAP IS THE HINGE, AND IT IS A SUBSTITUTION.** At the top of the final chorus the second voice completes her broken word and **keeps the song**; simultaneously the wide continuous layer is **replaced** by that dry close voice. Once. Never twice. | **0.90** | a hinge built from a hole (D10 breach); a hinge that only happens in the production spec; a second voice that merely joins in rather than taking over. |
| **F6** | ⛔ **THE THEME IS NEVER NAMED.** The words *connection · connect · reach · signal · distance · apart · together · understand · communicate · get through* and every synonym are absent. **The lag is the theme; naming it is the fog failure.** | **0.88** | one abstract noun doing the job the lag is supposed to do. The Somatic Gate calls this `REPAIR — FOG`, and the Critic Pass took this word off this pair by name at step 03. |
| **F7** | **THE HUMILIATING SPECIFIC IS PRESENT AND SMALL.** She has already told this story once tonight, and it landed on the beat of an apology. Named plainly, once, without commentary. | **0.85** | a general feeling of embarrassment with no object attached to it. The Angst Cartographer's whole objection. |
| **F8** | **THE UNANSWERED CHECK.** One small, specific, humiliating verification whose **outcome is never stated** — and it is a *different* verification in each of the four variations. | **0.80** | stating the outcome; or performing the check as a metaphor rather than as a thing the hands do. |
| **F9** | ⭐ **LEVEL, NOT ACCRETION.** Each section **swaps** a layer — hats out and triangle in, bells thinned to one note, sub held rather than raised. The arrangement at the last chorus has no more elements than the arrangement at bar one. | **0.78** | "and then the strings come in." Six of six pairs accreted last run, voluntarily. This pair is one of three assigned LEVEL and it is the assignment, not a preference. |
| **F10** | ⛔ **WEATHER, NOT VILLAIN.** The call is not the antagonist and the lag is not a metaphor for modern life. It is a half-second. It is written the way rain is written. | **0.75** | any line where the device, the network, the feed or the century is blamed. **D2** is sharpest on this pair in the whole run. |
| **F11** | **ONE NUMERAL, SPELLED, SUNG ONCE, ANSWERED.** *"half a second"* — in words, at the hinge, and the second voice **responds** to it rather than repeating it. No other quantity is sung anywhere: no *twice*, no *three times*, no *half a beat*. | **0.72** | a second countable slipping into a verse; the numeral recited as a fact; digits anywhere in a sung line. |
| **F12** | **RETURN FLOORS BEATEN WITHOUT LEANING ON END-RHYME.** `rhyme_return ≥ 0.30 · line_return ≥ 0.20 · words_per_line ≤ 7.5 · alliteration_per_100w ≥ 11.0`, measured with `scripts/measure_soundcraft.py → profile_file()`, never by eye. The return is carried by the half-line echo, the byte-identical chorus, the overlapping couplets and the canon. | **0.70** | hitting the floors by writing a rhyming quatrain, which would silently discard the pair's declared rhyme posture and make this pair sound like P02. |

**Distribution check.** Every facet is a *test a prompt can fail*, not a mood. Eleven of the twelve name a
device that exists only in this pair's assignment (fixed lag, half-line echo, hook at zero, mid-word spill,
speaker-swap hinge, the unanswered check, the half-second). F12 is the only shared one and it is a run-wide
floor, not a pair signature.

---

## PANEL PRESSURE ON THIS PAIR — the three Hyper-Skeptics, on P03 specifically

> **THE DYNAMIC RANGE AUDITOR (after Katz) — the one that changed the facets.**
> *"You have written 'two voices at a fixed lag' and you are about to hand me a delay time. I measured this
> class: a specified four-second stop rendered at 0.40 s, long internal voids smoothed two of two, hard-panned
> non-musical elements collapsed to near-mono. A delay is a mix decision and mix decisions are exactly what
> gets normalised out. If the lag is not in the **phrase-lengths** — if it is not something a human being
> could stand in a room and sing — you do not have a fixed lag, you have a note in a document."*
> **Consequence:** F1 was rewritten from *"a fixed interval between two voices"* to *"performed, not
> processed,"* and the interval was moved into the ad-lib channel the genre already runs — the pitched-up
> plugg ad-lib. **The lane's native element became the load-bearing structure.** Building with the grain.

> **THE COHORT ABOLITIONIST (after Cohen) — killed a facet outright.**
> *"Your draft rubric had a facet reading 'the loneliness of talking through a device.' Delete every collective
> pronoun from that sentence and there is nothing left standing. That is demography with a melody. There is no
> cohort in a kitchen at eleven at night — there is one person, one bottle label, one other person, and a
> half-second. Score **that**."*
> **Consequence:** the facet was deleted and F7 (the humiliating specific) took its slot. **Kill count on the
> rubric: one facet.** He also imposed the line-by-line application recorded in the step-10 self-check.

> **THE HARDCORE ELDER (after Rollins) — dissent recorded, NOT resolved.**
> *"Both of them apologising, both of them kind, nobody rude, and a warm second voice that finishes her word
> for her. That is a warm thing where a cold thing should be. The cold version is that she stops trying and
> nobody notices she stopped. You are going to write the version where somebody notices, because that is the
> version that is nice. Ask me again when there are lines on the page."*
> **Consequence:** carried forward **unresolved** into step 07 as a standing test, and it is the reason
> **V4 exists** — a variation in which nothing is resolved, both are warm, and it is fine, and it isn't.
> He is asked again at step 10 with lines in front of him. *Recorded rather than settled, per method.*

> **THE ANGST CARTOGRAPHER (after Rodrigo)**, non-skeptic, from the step-03 artist pass, retained as F7:
> *"the humiliating specific is that she has already told this story once tonight and it landed on the beat of
> an apology."* Taken verbatim as a hard requirement, not a suggestion.

> **THE MINIMALIST ATMOSPHERIST (after O'Connell)**, retained as F1's survival clause: *"the lag lives in the
> arrangement's phrase-lengths, so it survives a small speaker."*

> **THE SKATE-PUNK BLENDER (after Feldmann)**, applied pre-emptively: nothing in this pair may depend on
> stereo width or on the kick. The lag rides a **vocal**, which is the one element a handset reproduces well.

---

## WHAT THIS PAIR IS NOT — recorded before drafting

- Not a song about phones. **D2.** The call is weather.
- Not a song about a generation. **D3/D6.** One kitchen, one counter, two people.
- Not a song where anybody is taught anything. **D1.** Both are the same age and neither has the moral.
- Not a song that knows how the call ends. **D5.** The singer is inside it at every line.
- Not a song addressed to the listener. **D8.** ⭐ **Declared plainly, because this pair is the run's edge
  case:** the second person in these lyrics is the **other character on the call**, who is present in the
  fiction and is being spoken to inside it. The listener is never addressed, never included, never comforted.
  A phone call overheard from one side is the *purest* available realisation of "overhearing, not addressing" —
  and it is the assigned angle (V1: *"the entire song is two people saying 'sorry, you go' in different
  words"*). **First-person plural — `we`, `us`, `our` — is banned outright in this pair** and is the line the
  cohort gate is enforced on.
- ⛔ Not opened with `[Object. State.]`. Banned for this pair. Every song here opens on the **hook**, which
  makes the ban free: there is no establishing shot because there is no establishing.

---

## Provenance & self-critique

**Step file:** `skills/music/steps/06_Generate_Music_Facets.md`, with its ≤5-facet output schema superseded by
`vault/gates.yaml → step06_min_facets: 8` and `EXECUTION.md` §4 (2026-07 OVERRIDE). Golden-output quarantine
observed: `golden_songs_index.md` was **not opened**; this chain has the Golden Songs' **names only**
(*"Five wrong colors"*, *"The Blue Screen Breathes"*) and the **GOLDEN MOVE** block, per `06_music_handoff.md`
(L30). Scratch namespace: `_work/pair_03/`. Files written: `pair_03_*` only.

**Self-critique — the honest one.** F1 and F5 are the same physical event measured at two moments (the lag
running, the lag being taken over), and a rubric that scores one event twice will over-reward a song that
nails the gimmick and under-reward one that is actually moving. I am keeping both, deliberately, because the
failure modes are opposite: F1 fails by being a plug-in and F5 fails by being a hole, and a single merged
facet would let a draft pass by being neither. But the weighting is the guard — **F7 (the humiliating
specific) at 0.85 must not lose to two 0.9s describing the same trick**, and if a step-09 draft scores high on
F1/F5 and thin on F7, the trick has eaten the song and the correct action is repair, not a higher score.

Second: the Hardcore Elder is probably right about V1 and I have not fixed it, I have **variated around it.**
That is on the record here so that step 10 cannot pretend it was answered.
