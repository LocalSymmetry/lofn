# P06 — STEP 09 · ARTIST REFINEMENT PASS
**Run:** `2026-08-15_daily_music_the_resolution` · **ICB sha:** `7d4dd66f2a6e` · **Pair:** P06
**Tri-source declaration:** SOURCE 1 the day's resolution facts (scale calibration only) · SOURCE 2 two
André Cymone phrases case-exact in the generator-facing field · SOURCE 3 the Composite Rule, rendered
literally as **THE VISIBLE JOIN AT SINGLE-BAR SCALE.**

> ⭐ Every number below was **measured in code** against `scripts/measure_soundcraft.py` and `vault/gates.yaml`
> before this file was written. Nothing here is eyeballed. Three of the six repairs were caught only by the
> measurement, and the sixth was caught by re-reading a standard rather than a gate.

---

## REPAIR 1 — THE LYRICS FIELD WAS 800–1,000 CHARACTERS OVER, AND THE HEADERS WERE THE CAUSE
**Measured, revision 1:** V1 5,828 · V2 5,055 · V3 5,131 · V4 5,104. **Hard cap 5,000, target ≤4,800.**
Every one over. The sung lines were **not** the problem — the **EMO headers were**, at ~90 characters each
across 28 sections, about 2,500 characters of overhead per song.

**Repair:** one header per *pair or triple* of tercets instead of one per tercet, and every header
compressed to a four-part form of ~55 characters — `[Verse 3 - EMO:practised-calm - guitar thin left -
clips, inverter, log]`. **The full EMO header contract is intact** (section · specific emotion · role ·
cues), the emotions are still never bare, and the structure of the song is unchanged: merging headers
merged nothing musical, because the tercet was always the real unit.
**Measured, revision 3:** V1 4,438 · V2 4,433 · V3 4,570 · V4 4,514. ✅ All under target.

⭐ **The transferable bit:** the field cap is a cap on *the whole pasted field*, and this pair's assigned
form (one object per stanza) generates more section boundaries than any other form in the run. **A form can
fail a character gate through its metadata alone.** Nothing about the writing was wrong.

## REPAIR 2 — SUNG LINES UNDER THE FLOOR, AND THE FIX HAD TO BE OBJECTS, NEVER PADDING
**Measured, revision 2:** V1 75 · V2 72 · V3 72 · **V4 69 — below the hard floor of 70.** V2 and V3 sat
exactly on `sung_lines_floor_hug: 72`, which is a boundary-hugging FLAG in its own right.

⛔ The wrong repair was available and obvious: add a stanza that comments on the list. **Facet 5 forbids it
— the list IS the emotion, and a stanza that explains it is the single most damaging edit available here.**
So every added line is **another object, with its own wear**, obeying ABA and asyndeton:
- **V1** — the keys: *"keys on a ring, the boot key worn down / the same key every time, the same turn /
  the lock gives to the hand it has known."*
- **V2** — the boot lock, the pencil worn to the wood.
- **V3** — the stairs counted by the feel of the nosing; the spare reel *"kept for the night the first one fails."*
- **V4** — the hall rug, the split-soled shoe that exists only to hold a door, the clip order done blind.
**Measured, revision 3: 78 / 78 / 78 / 78.** ✅ Floor cleared, hug flag cleared, inside the 78–110 target.

## REPAIR 3 — A NEGATION INSIDE A STYLE FIELD (V3)
V3's prompt read *"mono in the middle, **no width at all**."* ⛔ ICB ban 7: Suno reads a negation as
positive tokens — *"no width"* asks for width. **Repair:** *"width collapsed to a point."* Same instruction,
stated positively. **Measured across all four prompts: zero occurrences of no / not / never / without /
nothing / none / avoid.** The anti-prettiness bans were already living in EXCLUDE, which is where they belong.

## REPAIR 4 — THREE PROMPTS UNDER THE 850 FLOOR
**Measured, revision 1:** V2 835 · V3 842 · V4 830 — under the band. V1 880, fine.
**Repair:** spatial and kinetic detail added, never adjectives — *"dry enough to hear the concrete"*,
*"felt at the sternum before it is heard"*, *"staying there"* on the early hat.
**Revision 3: 880 / 897 / 888 / 870.** ✅ All inside 850–1000 **and** inside the 870–960 target, none at or
above the 985 hug ceiling. All four end as complete sentences; none opens with a banned opener.

## REPAIR 5 — WORDS PER LINE AT THE CEILING
V1 measured 7.83 then 7.46; V4 measured 7.70 then **exactly 7.50, sitting on the ceiling.** Boundary-sitting
is the thing `gates.yaml` explicitly calls a flag, so nine lines were tightened — *"nothing on this landing
needs it more tonight"* → *"nothing here needs it more tonight"*; *"hook by the door takes the coil, takes
the weight"* → *"the hook takes the coil, takes the weight."*
**Revision 3: 7.42 / 7.37 / 7.24 / 7.40.** ✅ All under 7.5. ⭐ Every trim made the asyndeton harder, which
is the direction this pair should always be pushed anyway.

## ⭐⛔ REPAIR 6 — THE ONE A MEASUREMENT COULD NOT HAVE CAUGHT: **A MISSING PLAYER CREDIT**
The first dedication drafts said, in all four packages: *"No break or named figure is asked for, so no
player is credited."* **That was false, and it read as diligence.**

The Source-2 phrase this pair is required to ship is **"furiously slapping and popping his bass"** — and it
is in the **positive STYLE field**, not EXCLUDE. **That is a named technique being asked for by name.**
`vault/DEDICATION_STANDARD.md`: *"a named historical medium/procedure → the tradition AND the specific
practitioner where one exists."* Slap-and-pop has a practitioner: **Larry Graham**, thumb and pluck, Sly &
the Family Stone. **A missing player credit is a REPAIR, not a note.**

Both of the standard's legitimate branches were considered. **DROP THE CREDIT** was wrong here — unlike
P04's V3 last run, this prompt genuinely wants the technique and would be damaged by removing it. So:
**credit the invention, hedge the referent.** The shipped line is
> *"Slap-and-pop is asked for by name: the technique is **Larry Graham's invention** (thumb and pluck, Sly
> & the Family Stone), **never his recording**."*

⛔ Checked against the standard's four traps: **implied-contents grammar** — avoided, it credits the
*invention*, never *"whose slap this is"*. **A claim about a living person's life or standing** — none; it
states a fact about a technique, and it does **not** say the song is about him. **Raised vs forbidden** —
the phrase is in STYLE (raised) in all four, verified by string count, so the credit has a real trigger.
**Honesty clause** — *"Suno generates, it does not sample"* ships in all four.
And **André Cymone is credited for words, not for playing**: the two phrases are quoted prose from a
Bandcamp Daily piece, so the dedication names the piece. Crediting him as a *player* would be the false
credit this standard exists to prevent — his recording is not invoked, only his describer's vocabulary.
**Measured: 488 / 487 / 491 / 491 characters. ✅ All ≤500.** Each dedication opens with its own song's
leftover object, so no block is pasted across four songs.

---

## REFINEMENTS THAT WERE NOT REPAIRS *(taste, made explicit)*

**a) *"the note"* → *"the hum".*** The returning join line first read *"the note drops half a step, comes
back."* ⛔ **Nothing in this lyric is music, because nothing in this lyric is heard by anybody** — a "note"
smuggles a performance into a song that is banned from depicting one. **"the hum"** is the fridge and the
inverter. Same rhythm, correct universe.

**b) The one-fact rule is per SONG, not per stanza.** Draft V1 carried *"forty feet"*, *"four winters"* and
*"eleven hours"*. Two had to go. **Kept:** *eleven hours*, at the hinge, answered by *"crank out. the answer
is the same."* **A number that gets a feeling in reply is a lecture; a number that gets a hand in reply is
this song.** Measured per song: V1 *eleven* ×1 · V2 *fourth* ×1 · V3 *forty* ×2 (**one fact, returned at the
hinge**) · V4 *two* ×2 (same fact, same return). ⚠️ **FLAG, declared not hidden:** *both* (V1×1, V3×3, V4×1)
and *one* (V3×1) are determiners, not facts — reported so a reader can rule on them rather than discover them.

**c) The first person was counted, not assumed.** V1 and V4 measure **I × 8** — that is the title line
appearing twice per refrain across four byte-identical refrains, **and nowhere else.** V2 and V3 measure
**I × 0.** ⭐ **Zero *you*, zero *we*, zero *us*, zero *our*, zero *your* in the sung lines of all four**,
word-boundary matched. LIST WITH NO ADDRESSEE holds.

**d) Asyndeton was measured, not trusted.** *and / but / or / nor / so / yet* = **0 across all four sung
sets.** One draft line, *"and the cord runs down the hall"*, died for a single word.

**e) The returns are above floor without a single apology.** strict end rhyme **0.603 / 0.513 / 0.487 /
0.538** against a floor of 0.30; line return **0.295** against 0.20; alliteration **21.9 / 16.9 / 19.3 /
18.0** against 11.0. The refrains are byte-identical and **no justification is filed for that**, per
`chorus_repetition_requires_no_justification`.

**f) The ending was checked four times for the failure this pair is most likely to commit.** V1 the crank
goes back under the jack · V2 *"the grip fits the hand that wore it"* · V3 *"towel folded. terminals clean.
the floor clear."* · V4 ⭐ *"nothing happened. again."* **None thanks anybody. None admires anybody.** The
register of admiration — *keeps going, still standing, makes do, somehow* — is absent by measurement:
*somehow* = 0, *thank / thanks / grateful / gratitude* = 0.

---

## WHAT STILL WORRIES ME, GOING INTO STEP 10 *(carried forward as FLAGS, not resolved silently)*
1. ⚠️ **The sag is the whole bold device and it is the thing most likely to be smoothed.** The Deviation
   Clause names what to keep in advance; the EXCLUDE field now also blocks its three fake substitutes
   (sidechain filter sweep, glissando pitch bend, smoothed pitch correction). It can still come back ironed.
2. ⚠️ **V3 is the variation most at risk of rendering *atmospheric*** — it is the interior one, and a
   kitchen at night is an aesthetic waiting to happen. Its EXCLUDE carries `domestic warmth` for exactly
   this reason, and its lyric answers it in the writing rather than the spec: the fridge stanza is about
   ranking food by what spoils, not about a nice kitchen.
3. ⚠️ **`song` appears once per lyrics field** — inside the ICB-mandated `[SONG FORM: …]` tag. It is a
   structural tag, never a sung line. Declared rather than quietly excluded from the sweep.
