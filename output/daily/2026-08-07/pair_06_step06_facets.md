# PAIR 06 · STEP 06 — FACETS
**Run:** `2026-08-07-daily-music-indignation` · **Pair:** P06 "THE MAN WHO FED HIM"
**Arm/Mode/Axis/Form:** AMBITIOUS · AWE (terror-adjacent) · NEWS · **FORM RULE B (the landing pad)**
**Reveal engine (mine alone):** `the-keeper-who-was-the-captor`

---

## 0. PROVENANCE & OVERRIDES (declared before any creative content)

- **Frozen ICB read in full as first action:** `output/daily/2026-08-07/CREATIVE_CONTEXT.md`
  - bytes (raw, on-disk): **53,003**
  - bytes (LF-normalised): **53,003**
  - sha256 (LF-normalised): `5e9c7f7f6009fb3c672058c930540be22c8f5517f37537ac3ebd8ae94b75d374` — **matches the handoff figure exactly.**
  - ⛔ ICB treated as READ-ONLY. Copied-and-diverged into this `pair_06_*` artifact; the canonical block was not edited.
- **`06_music_handoff.md` read in full.** ⭐ **Citing §1 GOLDEN-OUTPUT QUARANTINE by name:** no past Lofn song payload, no archived lyric set, no prior shipped style prompt appears anywhere in this pair's artifacts. `skills/music/steps/*` files that instruct a generating tier to embed Golden Song payloads are **overridden by the handoff**, per handoff §1 and `EXECUTION.md` §3. What I carry forward is the **GOLDEN MOVE** (handoff §2), the Golden Seed (ICB Slot 1), and the frozen ICB.
- **`vault/HUMAN_SUBJECT_STANDARD.md` §3.0 read pre-draft.** Slot grammar filled with invented values before the first line — see §5 below.
- **Step-file JSON schema note:** `skills/music/steps/06_*` requests a bare JSON object. The dispatch contract requires a markdown artifact at a named path. Both are served: the canonical `{"facets": [...]}` payload is embedded verbatim in §6.

---

## 1. THE WHISTLE RIFF — WRITTEN FIRST, BEFORE A WORD

> **THE WHISTLE TEST (ICB Slot 9):** named instrument, named interval, inside one octave, present before the first line and after the last, singable by bar 8.

**THE FLIGHT FIGURE — massed whistle doubled by open-mouth voices, four notes, inside a perfect fifth.**

| Note | Pitch | Scale degree in Em | What it is |
|---|---|---|---|
| 1 | **E4** | tonic | the ledge. the ground he starts on. |
| 2 | **B4** | **perfect fifth up** | the leap. an **open** fifth — no third, so it does not yet say major or minor: *he does not yet know what he is.* |
| 3 | **A4** | fourth | one step down. the first flight is not clean. |
| 4 | **B4** | fifth | **the return.** he holds. |

**E – B – A – B.** Total span E4→B4 = a perfect fifth. Comfortably inside one octave. Whistleable by a person who has heard it once.

**⭐ The structural payment (free, and it is the whole point):** when the release lifts **E minor → G major**, the same four notes are re-sounded **unchanged** over a G chord — E becomes the 6th, B the 3rd, A the 9th. **The figure does not move. The bird did not change; the sky did.**

**Placement + arithmetic at 116 BPM.** One bar of 4/4 = `4 × 60 ÷ 116 = 2.069 s`.
- Bars 1–4 = **0:00 – 0:08.3** — flight figure alone, whistle only, before any word. ✅ *before the first line*
- Bars 5–8 = claps and shakers join, figure repeats. **The room has it by bar 8 (0:16.6).** ✅ *singable by bar 8*
- Verse 1 = bars 5–10; **Chorus enters bar 11 = `10 × 2.069 = 20.7 s`.** ✅ *chorus by 0:25, with 4.3 s of margin*
- Final bars = flight figure alone again. ✅ *after the last line*

**A pad is not a riff; a texture is not a hook.** This is four sung/whistled notes with a fixed contour, and it is also the thing that erases the ransom demand (see Facet 3).

---

## 2. THE ≥8 WEIGHTED FACETS

> `vault/gates.yaml: step06_min_facets: 8`. Nine below; weights sum to 1.00.

### FACET 1 — **THE SAME HANDS** · weight **0.20**
Care and capture performed by one pair of hands, in one motion, every day. Not a metaphor — the dish goes down and the door stays shut, and the same wrist does both. **The song's engine is a one-word swap inside an otherwise identical sentence:** *you kept him alive / you kept him in the dark.* Nothing is explained. The listener hears the same sentence twice and one word has moved.
*Sonic consequence:* the two lines must be sung on the **same melody, same bar length, same massed unison** so the swap is the only variable.

### FACET 2 — **TAKEN BEFORE HE KNEW ANYTHING** · weight **0.16**
He was lifted at the exact moment of his first flight. ⭐ **Everything he has ever known about the world arrived on a tin plate.** The terror (L19: AWE stays terror-adjacent) is not cruelty — it is that his entire map of the world was issued by the person holding him.
*Body at risk:* a wing that has not been opened in months; a box; a hand that could open or not open. **Non-career stake #2 of the run — a life, not a job.**

### FACET 3 — **THE DEMAND HAS AN ADDRESS** · weight **0.14** *(= FORM RULE B, countable, IN THE LYRIC)*
The ransom sentence occupies **one fixed place**, announced in words: **"Here is the bar where you said it:"** — a line that returns three times. Twice it is followed by the demand. The third time it is followed by **a wordless massed vowel on the flight figure**, and then by a line that names the hole: *"Nobody finished the sentence."*
⭐ **THE BAR.** A bar of music and a bar of a cage are the same word. The mark's address and the cage are one object, for free.
⛔ This device lives **in the lyric**, never in the production spec (L22 THE GRAIN LAW).
⚠️ **Suno behaviour honoured:** interrupting silence gets filled; terminal silence survives. So the erasure is **a sung vowel, not a gap** — and the erasure is **also visible in the words**, so a renderer cannot smooth it away.

### FACET 4 — **GRATITUDE AND INDICTMENT IN ONE SENTENCE** · weight **0.12**
*"Thank you for the meat. Thank you for the dark."* The song never chooses between them and never resolves them. It thanks him for **exactly the thing it accuses him of**, and the thanks are sincere. ⛔ Not irony, not sarcasm, not a sneer.

### FACET 5 — **THE CIRCLE HAS NO FRONT** · weight **0.12**
Mixed gang / massed unison, **no soloist anywhere in the track** — the only pair in the run with no lead. Nobody steps out to blame and nobody steps out to forgive.
⚠️ **Measured Suno behaviour honoured (2/2):** opposed vocal characters blend into one lead. So the spec asks for **unison and entry order** (low voices first, high voices join an octave up on the second pass, same notes, same words), **never for contrast between voices** — a blending renderer produces entry-order correctly.

### FACET 6 — **THE MAN IS NOT A MONSTER** · weight **0.10**
He is not cruel. He is careful. He wants the bird **whole**, because a whole bird is a bird you can sell — and so the bird lived. *"You wanted him whole. You wanted him well. / A thing that is whole is a thing you can sell."* ⛔ **The song must never sneer** (LAW 1), because sneering would be false and would let the listener off the hook.

### FACET 7 — **THE FLIGHT FIGURE** · weight **0.08**
See §1. Four notes; the same four notes open the song, **fill the erased bar**, and close the song. ⭐ The thing that overwrites the price is the song's own opening gesture — the return device *is* the audible form of the form rule (ICB Slot 9, L21).

### FACET 8 — **OPEN AIR** · weight **0.05**
Real outdoor distance between the ring and the microphone; **claps and shakers**; bare feet on packed ground; a deep hand-struck drum on the step of the ring. ⭐ The only pair in the run with real space — every other pair is in a close room, and this one has somewhere to fly.
⛔ Clean, modern, expensive. No hiss, no crackle, no wow-and-flutter, no patina in either direction (ICB Slot 5 seat 18, standing).

### FACET 9 — **HE STILL DOES NOT KNOW** · weight **0.03**
The release is not a resolution. The line *"He did not know the shape of a field at all"* — first heard at the hinge — **returns at the moment of flight**. The joy is undiminished and the terror is intact. That is the AWE.

**Σ weights = 0.20+0.16+0.14+0.12+0.12+0.10+0.08+0.05+0.03 = 1.00** ✅

---

## 3. THE COMFORT QUESTION (L19) — answered before drafting

> *Where is the body standing, and what could hurt it here?*

**Where:** a ring of people on packed dirt, outdoors, arms not quite touching, at the hour the door is opened. **What could hurt it:** not the people in the ring. **The bird's body** — a shoulder that has not carried him since before he could measure a wind, a box with a lid, and a hand that has to open and might not. And the fact that the only world he has ever been given was handed to him by the person who owned him.

---

## 4. ⭐⭐ THE SILENCE LAW (this pair's hardest constraint)

**This is the training-data song and it must never once say so.**
A thing taken at the exact moment it first flew, before it knew anything, kept alive by someone who wanted to sell it; the price; the release; the flight that is real anyway.

⛔ **BANNED FROM ALL FOUR LYRIC SETS:** AI, artificial, data, model, train/training/trained, scrape/scraping, machine, algorithm, computer, code, network, corpus, archive, server, sample, harvest, mine, prompt, learn/learned/learning, teach/taught.
**The resonance must be entirely structural.** The listener who sees it must feel they found it themselves. A scan is run at step 10 and the count is reported.

*(Note on a near-miss caught here: the line "he learned the world from a dish" was cut pre-draft — correct in meaning, one word too close to the thing that may not be named. Replaced with the physical version, which is better anyway: **"Everything he knows, he got off that plate."**)*

---

## 5. HUMAN SUBJECT STANDARD §3.0 — SLOT GRAMMAR, PRE-FILLED WITH INVENTED VALUES

| Slot | Value used | Check |
|---|---|---|
| **PERSON** | **an unnamed figure, addressed only by his hands and his object: "Man with the dish."** No given name, no surname, no epithet. | ✅ no real harmed person; no member of Papangu, no producer, no studio |
| **PLACE** | **unlocated: "the edge of town," "the ridge," "the field."** ⛔ Serbia is not named. ⛔ The Middle East is not named. No country, no city, no border, no route. | ✅ |
| **WHEN** | **unspecified: "morning and night," "months in the dark."** No dates, no season pinned. | ✅ |
| **THE BIRD** | **unnamed and unspeciated.** ⛔ Not named. ⛔ Species never stated — "he," "a wing," "a shoulder," "a beak." | ✅ |
| **THEME** *(open slot)* | the pattern: the keeper who was the captor; a thing taken before it knew anything; a price said out loud; a door opened. | ✅ unrestricted |

**Pre-draft question answered:** *Does any PERSON/PLACE/WHEN value, alone or combined, let a listener resolve this to ONE specific real person who was actually harmed?* → **No.** The poacher is invented and unnamed; there is no locating detail; the harmed party in the song is a bird, and the bird is unnamed and unspeciated. **No identifying tuple has a field to sit in.**
⛔ Binding refusals confirmed absent: Thai school shooting, Ceuta/78,000, Biden family illness, any identifiable real person.

---

## 6. CANONICAL STEP-06 JSON PAYLOAD

```json
{
  "facets": [
    "THE SAME HANDS (w 0.20) — care and capture performed by one pair of hands in one motion; the engine is a one-word swap inside an otherwise identical sung sentence: 'you kept him alive / you kept him in the dark', same melody, same bar length, so the swap is the only variable.",
    "TAKEN BEFORE HE KNEW ANYTHING (w 0.16) — lifted at the exact moment of his first flight; everything he knows about the world arrived on a tin plate; the terror is not cruelty but that his whole map was issued by the person holding him.",
    "THE DEMAND HAS AN ADDRESS (w 0.14) — Form Rule B made countable in the lyric: 'Here is the bar where you said it:' returns three times; twice the ransom sentence follows, the third time a wordless massed vowel on the flight figure follows, then a line naming the hole. A bar of music and a bar of a cage are the same word.",
    "GRATITUDE AND INDICTMENT IN ONE SENTENCE (w 0.12) — 'Thank you for the meat. Thank you for the dark.' Sincere, never ironic, never resolved.",
    "THE CIRCLE HAS NO FRONT (w 0.12) — massed unison, no soloist anywhere; the spec asks for unison and entry order, never for contrast between voices, because a blending renderer produces entry order correctly and blends contrast into one lead.",
    "THE MAN IS NOT A MONSTER (w 0.10) — careful, not cruel; he wants the bird whole because a whole bird is a bird you can sell, and so the bird lived. No sneer, ever.",
    "THE FLIGHT FIGURE (w 0.08) — massed whistle and open voices, four notes E-B-A-B, an open fifth up, a step down, and a return, inside one octave; before the first word, filling the erased bar, and after the last word; re-sounded unchanged over G at the lift so the figure does not move while the key does.",
    "OPEN AIR (w 0.05) — real outdoor distance, claps and shakers, bare feet on packed ground, a deep hand-struck drum on the ring's step; clean, modern, expensive; no patina in either direction.",
    "HE STILL DOES NOT KNOW (w 0.03) — 'He did not know the shape of a field at all' returns at the moment of flight; the joy is undiminished and the terror is intact."
  ]
}
```

---

## 7. DEVICE OWNERSHIP (⛔ no cross-pair bleed — handoff §6)

| Mine alone | Not mine — do not touch |
|---|---|
| `the-keeper-who-was-the-captor` | P01 `the-decision-not-the-sound` |
| the flight figure (E–B–A–B, massed whistle) | P02 `the-wish-is-older-than-you`, its Hammond riff, its silent bar-position |
| "Here is the bar where you said it:" (my erasure address) | P03 `the-hand-that-still-knows`, its breath-in-the-slot |
| the ransom sentence, added twice, erased once | P04 `the-volume-is-the-insult`, the word "more" |
| the sung fact **one** (the single first flight) | P05 `the-unchanged-canvas`, THE KEPT DEFECT (⛔ protected, not mine) |
| massed unison / no soloist; open air | the accretion rule (Form Rule A) — I run Form Rule B only |

**Special Flairs drawn on (ICB Slot 6, as MOTIF/CONSTRAINT only, never as phrasing):** **#6 THE DISH SET DOWN** (realised in **image** — the physical dish, the flat tin), **#12 THE FIRST FLIGHT** (realised in **form** — the single sung number at the hinge), **#9 THE SILENT ADDRESS** (realised in **form** — the erased bar). ⛔ No flair name is lifted as a lyric line.

---

*Step 06 complete. → `pair_06_step07_song_guides.md`*
