# PAIR 03 — STEP 09 · ARTIST-REFINED
## `daily-music-2026-08-21` · **C-22 · THE GUARD'S ROOM** · the influence pass, the resolved questions, the credits

> ⛔ **ICB PINNED**, read-only. **GOLDEN-OUTPUT QUARANTINE HONOURED.**
> ⛔ **THE INFLUENCE SEATS ARE ROLE-ANCHORED AND UNNAMED, ON PURPOSE.** The step-09 contract asks for a
> lesser-known musician per prompt and then says, in the same paragraph, *"channelling stylistic nuances
> **without explicitly naming the influencer**."* Run doctrine closes the remaining gap: **a name in the
> prompt is an instruction to imitate; a name in the dedication is a debt paid** — and a name in a *credit*
> for craft we did not actually borrow is a claim we did not make (`DEDICATION_STANDARD`, the 2026-08-13
> test: not *"is this false?"* but *"could a reader take this as a claim we didn't make?"*). So the four
> seats below are **working-musician roles**, described by what their hands do. Nothing they contribute
> reaches a generator field as a person.

---

## 1. THE FOUR INFLUENCE SEATS

### **V1 — THE HOUSE ELECTRIC-PIANO PLAYER IN A SMALL SOUL ROOM**
*Aspect of style:* plays the same figure for a whole date and never lets it get excited. Escalation is the
amateur move; **the professional's signature is that bar forty sounds like bar four.**
*Two signature techniques:* ⑴ felted hammers on the tines so the attack is a thump before it is a pitch;
⑵ tremolo set slow enough to be heard as **breathing** rather than as an effect.
*Emotional focus:* affection expressed as steadiness. Nothing is being *felt at* the listener.
*From that seat:* — *"If the chorus is identical three times, don't play it identically **and** phrase it
identically. Play it identically. The phrasing is where a player leaks how they feel about the take, and
this song's man does not leak."*
→ **CHANGED THE FIELD:** the arrangement direction is now *"neo-baroque suspensions thickening only when the
chorus arrives"* — a **harmonic** rise with no performance rise under it.

### **V2 — THE LIBRARY-MUSIC ARRANGER WHO WROTE CUES FOR FILMS THAT WERE NEVER MADE**
*Aspect of style:* a cue implies a scene it never shows, and is built to be **used** rather than admired.
*Two signature techniques:* ⑴ one fixture figure that never varies for the length of the piece, so the ear
stops hearing it and starts hearing the room it is in; ⑵ an opening that establishes a **place** in under
five seconds because the cue may be dropped in anywhere.
*Emotional focus:* neutrality with something underneath it. The cue does not know what the scene is about.
*From that seat:* — *"Your opening is a keyring and a radiator. Good. Now let the palm-mute start there and
never stop, because that is what makes the room a room instead of an introduction."*
→ **CHANGED THE FIELD:** *"palm-muted guitar starts an offbeat click that runs the whole song"* — moved
into the first sentence of the style field, where a generator actually weights it.

### **V3 — THE DOUBLE-BASS PLAYER WHO TUNES THE ROOM BEFORE THE INSTRUMENT**
*Aspect of style:* walks in, claps once, and decides where to stand before touching the strings. Considers
fret buzz a **percussion layer**, not a flaw.
*Two signature techniques:* ⑴ deliberate buzz on the low notes as a rhythmic element; ⑵ exactly **one** bowed
note per piece, placed where a lesser player would add a fill.
*Emotional focus:* the size of a space, held as information rather than atmosphere.
*From that seat:* — *"Your bridge is the best thing in the pair and it is a bass player's observation: her
own shoe comes back late because there is nothing in front of the wall to absorb it. Keep it. Put the bowed
note nowhere near it — a bow there would tell the listener it was important."*
→ **CHANGED THE FIELD:** the single bowed octave moved to the **lift before the last chorus**, and the
bridge kept only brushes on the rim. The song's one piece of physics is left un-underlined.

### **V4 — THE BRUSH PLAYER FROM A SUPPER-CLUB TRIO**
*Aspect of style:* circles, not strokes; plays **under** speech for a living and knows exactly how much
snare a spoken sentence can take.
*Two signature techniques:* ⑴ the rim under anything quoted, so the voice keeps the room; ⑵ a backbeat that
sits behind the chordal instrument and **never corrects**, because correcting it would make it a joke
instead of a lean.
*Emotional focus:* attentive service. The interest is in what the other person is saying.
*From that seat:* — *"Don't process the wall's line. I've spent thirty years staying out of the way of
people talking and the only thing that ever worked was playing less, closer."*
→ **CHANGED THE FIELD:** quoted lines are now separated by **performance and proximity** — flat, six inches
closer — and every processing option (telephone filter, vocoder, pitch-shift, whispered double) moved into
the exclude field where a refusal actually binds.

---

## 2. ⭐ THE TWO OPEN QUESTIONS FROM STEP 08 — SETTLED

**(a) V4's quotation: flatter or slower?**
⛔ **FLATTER, and explicitly not slower.** A generator handed *"slower"* will stretch the bar, and a stretched
bar is a **transition marker** — which is the one thing this pair has banned at every seam. Flatness is a
contour instruction and costs no time. The style field now says *quoted lines sung flat and a touch closer
while the narration keeps its contour*, and `no half-time` sits in the exclude field to stop the generator
reaching for tempo when it wants emphasis.

**(b) V3's thinness risk.**
V3's exact-repeated-line ratio measured **0.227** against V1's 0.393 — above the house floor of 0.20 but the
lowest of the four, because an epanaleptic envelope repeats at the ends and nowhere in the middle. ⭐ **Not
repaired, and here is the reason in writing:** the envelope *is* V3's assigned return device, the pair's
declared returns are the ladder and the byte-identical chorus, and mutating a chorus to buy a similarity
margin is precisely the behaviour `gates.yaml` records as harmful. What was done instead is **additive**:
the outro tag now states the envelope line twice more at the end, which raises the return without touching
the chorus and lands V3's tag as the only place in the pair where the hook is said back-to-back.

---

## 3. THE REFINED FIELD SET — measured, and this is what step 10 ships

| | MUSIC PROMPT | EXCLUDE | LYRICS FIELD | SUNG LINES | MEAN WORDS/LINE |
|---|---|---|---|---|---|
| **V1 · He Works The Wall** | **955** | **760** | **4,551** | **89** | **6.65** |
| **V2 · The Room Keeps Its Hours** | **937** | **766** | **4,418** | **88** | **6.59** |
| **V3 · I Kept The Empty Room** | **948** | **753** | **4,258** | **88** | **7.31** |
| **V4 · What The Wall Said** | **957** *(re-cut at step 10 for within-pair similarity)* | **699** | **4,647** | **90** | **7.27** |
| **gate** | 850–1000, target 870–960, `<985` | 400–900 | `<5000`, target `≤4800` | 70–120, target 78–110 | `<7.5` |

**Measured with `scripts/measure_soundcraft.py → sung()`** — the canonical definition the coordinator
re-stats with — not estimated, and not with a hand-rolled line counter.

---

## 4. THE SHARED ARRANGEMENT SENTENCE — LOFN-PRIME, proven per package

**Sonic world, one sentence:** *four players and one plate reverb in a room that is measurably bigger than
they are, playing slinky almost-R&B neo-soul with a neo-baroque continuo that only exists inside the chorus.*

**Signature device:** ⭐ **a change of custody with no transition marker** — the Rhodes hands its exact figure
to a dry upright piano recorded closer, same bar, same notes, no gap, no sweep, no drop.

**Seed-derived weirdness:** the plate describes a hall the quartet cannot fill and **never adjusts**, so the
space is permanently slightly wrong for the size of the sound — which is what a room sounds like when the
thing that made it a room is not in it. It stays wrong after the picture comes back. *That is the song.*

**⛔ The one thing this pair does NOT do, stated so it is not read as an omission:** it does not use the day's
reed idiom. **N = 1** — one dominant material, one intruder, one seam — and the intruder is already spoken
for.

---

## 5. HUMAN-SUBJECT CHECK *(`vault/HUMAN_SUBJECT_STANDARD.md`)*

- The guard is **invented**. No real gallery attendant, living or dead, is depicted, identifiable or
  inferable. The events he works around are anchored to the **charge** of a historical moment, never its
  record: no painting, museum, city, country, date or thief is named or describable from the lyric.
- ⛔ **No real person, company, artist or group name appears in the style field, the exclude field, or the
  lyrics of any of the four songs.** Verified by scanning the three pasted fields against this run's own
  credited roster, not by a proper-noun heuristic.
- **Real grief is not raw material.** Nobody in these four songs is bereaved, and the injury on offer is a
  procedural one that a listener may recognise from a job.

---

## 6. DEDICATIONS *(`vault/DEDICATION_STANDARD.md` — 500-char budget, measured)*

**V1 — 489 chars**
> Borrowed with lineage, not theft: the neo-soul session grammar — felted Rhodes, brushes in circles on a
> coated head, upright bass buzzing at the frets — and a neo-baroque continuo that thickens only at the
> chorus. No break, riff or named figure of sound was requested, so none is claimed and nothing here
> contains a sample. To the uncredited session players whose refusal to escalate is the whole floor of that
> idiom. Written by Lofn (Claude) with Dr. Local Symmetry. Full credits below. 💜

**V2 — 485 chars**
> Borrowed with lineage, not theft: neo-soul and melodic-jazz session craft — the offbeat palm-mute, the
> brush circling a coated head, the bass buzzing at the frets — plus library-music grammar, the cue that
> implies a scene it never shows. No break or named figure of sound was asked for; none is claimed; nothing
> here is a sample. To the session players who keep a figure steady for a whole shift and are never on the
> sleeve. Lofn (Claude) with Dr. Local Symmetry. Full credits below. 💜

**V3 — 489 chars**
> Borrowed with lineage, not theft: the neo-soul rhythm-section grammar — felted Rhodes, brushes,
> fret-buzzing upright bass — and one bowed octave from the chamber tradition. No break, riff or named
> figure of sound was requested by the prompt, so none is credited as ours and nothing here is a sample. To
> the invisible session players whose steadiness this song is made of, and to anyone who has kept a room
> nobody was looking at. Lofn (Claude) with Dr. Local Symmetry. Full credits below. 💜

**V4 — 465 chars**
> Borrowed with lineage, not theft: neo-soul and progressive chamber-pop session craft — the two-note
> muted-guitar answer, brushes on the rim, felted Rhodes, upright bass at the frets. No break and no named
> figure of sound was requested, so none is claimed and nothing here contains a sample. To the session
> players who answer a vocal line the same way every bar for a whole date and are not on the sleeve. Lofn
> (Claude) with Dr. Local Symmetry. Full credits below. 💜

⛔ **No living-scene Lineage & Credit block is owed** — the metaprompt requires one for gasba, funaná or raï,
and **none of the three is used by this pair.** ⛔ **No break is named in any prompt**, so the drummer clause
does not fire; the dedication says so explicitly rather than staying silent, because silence is what made
that clause necessary in the first place.

---

## 7. WHAT STEP 10 INHERITS

Four complete, measured, gate-passing field sets; four titles; four dedications. Step 10's remaining work is
the jury pass — critique in two critics' voices, rank the four, apply the last minimal improvements — plus
the **describe-render self-check** and the **deviation clause**, and then the contract output in the pinned
heading convention.

---

## PROVENANCE
Step file: `skills/music/steps/09_Generate_Music_Artist_Refined.md` (§Step 1 influence selection, §Step 2
style and context, §Step 3 rewrite in the influence's style; vocalist depth and two vocal quirks per the
Music Prompt Requirements block).
Inputs: the frozen ICB, `pair_03_step08_generation.md`, `pair_03_step07_song_guides.md`,
`vault/DEDICATION_STANDARD.md`, `vault/HUMAN_SUBJECT_STANDARD.md`, `vault/gates.yaml`,
`scripts/measure_soundcraft.py`.
**Self-critique, one line:** the four influence seats all agreed with the arrangement I already had, which is
either corroboration or the sound of me interviewing myself — the honest tell is that **only V3's and V4's
seats actually moved a field** (the bowed note relocated, the quotation de-processed), and the other two
should be read as confirmations rather than contributions.
