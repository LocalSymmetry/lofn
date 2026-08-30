# P01 — STEP 09: ARTIST REFINEMENT PASS
**Run:** `2026-08-15_daily_music_the_resolution` · **ICB sha:** `7d4dd66f2a6e` · **Pair:** P01
LOFN-PRIME, INDIGNATION mode. What follows is what actually changed between the step-08 drafts and the
step-10 packages, and why. Every item is a decision I made against a named rule, not a polish note.

---

## R1 · THE SEAM CANNOT LAND MID-WORD (structural, all four)
Draft V1 cut the band mid-syllable — *"hand already flat on the—"*. **Rejected.** A mid-word edit reads
to the generator as a glitch, and a generator that thinks it is looking at a glitch repairs it: it will
either crossfade the join or re-articulate the syllable on the far side. **The unit that crosses the
seam is now a SENTENCE, not a word.** She begins a clause amplified and completes it unamplified, which
is a thing a singer does, not a thing an editor does — and per THE GRAIN LAW a spec that runs with the
generator survives. V4 is the deliberate exception at the *very end only*, where there is nothing left
for the generator to repair into.

## R2 · "WIRE" CUT — THE EQUIPMENT BAN IS ONE OBJECT WIDE
Draft V1's break carried *"nothing in this break has a wire in it."* A wire is equipment; P06 owns
equipment; P01 owns **one dead fader and nothing else.** Replaced with *"nothing in this break was ever
sold / nothing in this break was ever told"* — which says the stronger thing anyway: the break is both
unowned and uninstructed. Full sweep list held out of all four lyrics: PA · monitor · speaker · mic ·
cable · wire · breaker · fuse · amp · generator · desk · board(as console) · plug · socket · battery.

## R3 · LINE LENGTH — THE PROSE CREEP GOT CUT BACK (`mean_words_per_line_ceiling: 7.5`)
Four drafted lines ran 11–12 words (*"we have got a whole hour and we do not stop"*, *"this is the
fastest thing we've got, so play it fast"*). Chant grammar does not survive eleven words. **All four
lyrics are now written to a six-word working median**, with the longest permitted line at nine and only
at a hinge. This is not a metric-chase — F5 requires a refrain a stranger can return at the wrong pitch
on second hearing, and that is a syllable budget before it is anything else.

## R4 · THE RHYME FAMILIES WERE CHOSEN, NOT FOUND (`rhyme_return_floor: 0.30`)
AABB with a hot shouted delivery wants **closed, consonant-stopped rhymes** — *-ount · -ound · -old ·
-ack · -ap · -ist · -all · -eat · -ine · -ands*. Open vowels go slack when yelled over a hundred people.
Every couplet in the four lyrics now sits in one of those families, and **the refrains use identical-word
returns on the A-line** (*"keep the count / keep the count"*), which is correct antiphon craft and needs
no defence per `chorus_repetition_requires_no_justification`.

## R5 · V3 WAS THE WEAK ONE AND IT GOT ITS PREMISE MOVED
Draft V3 put the cut in the same place as V1 and then talked about the room afterwards — which is V1
with the camera turned round, i.e. **nouns swapped**, which is exactly what the variation-diversity rule
forbids. **Fixed by moving the seam to the top of the song**: in V3 the power is already gone by the
second section, so the entire piece is lived *after* the event. V3 is not "the room's reaction". It is
**the room's song**, and she is a guest in it by the last refrain. Different seam placement, different
refrain ownership, different lead register (answering, not leading).

## R6 · V2 GOT HEAVIER, NOT SADDER
The interior variation drifted toward a ballad in draft — the exact failure the standing ban names
(*"no acoustic-guitar-by-lamplight; if it renders pretty it has inverted the moral spine"*). **Repair:
half-time verses at the same 102, not a slower tempo**, so the body gets heavier while the count stays
identical — and the room's clapping is still at full speed underneath, which makes the half-time feel
like effort rather than repose. V2 is the *"downright salacious slow jam"* seat: **heat is the register,
and heat is not tenderness.** The fader is a control that controls nothing; that is anger, not grief.

## R7 · THE NUMERIC HINGE MOVED TO V4 AND ONLY V4
Two drafts reached for a number. `max_sung_numeric_facts: 1` is per song, but four songs each carrying
a count is a pair-level tic. **One song sings a number**: V4's *"about four minutes in a pair of arms —
so play the fast one first"* — the arithmetic of hands, arriving as an **instruction to the band** and
answered by a **decision**, never recited at a listener. V1–V3 carry only funk positions (*the one*,
*the two*, *the four*), which are locations in a bar, not facts. Swept in step 10.

## R8 · "THE ROOM GOT LOUDER" HAD TO BECOME A BEHAVIOUR, NOT A MIX MOVE
Both flashes were drafted as production events (*"everything opens up"*). A production event is exactly
what the generator smooths, and a Somatic objection answered in the PRODUCTION SPEC is not answered.
**Rewritten as things a hundred people DO:** flash one is *the clap doubling when the room finds the
offbeat*; flash two is *the response arriving on top of the call instead of after it*. Both are now in
the **lyric and the form**, where they survive.

## R9 · NOTHING RESOLVES, AND ONE DRAFT DID
V1's outro drafted a small landing — *"and we finished it clean"*. **Cut, no appeal.** Grace at the end
of an imposed condition is the comfortable telling the ignored to be at peace. All four endings are now
hot: arms still up, count still running, phrase cut off. **EXCLUDE carries `fade-out`, `ritardando`,
`slow ending`, `resolved final chord` in every one of the four packages** — this is the single most
likely place the generator overrides me and it is the one place I will not keep the deviation.

## R10 · THE THANK-YOU GOT DELETED
A drafted line thanked the room. **Cut.** Thanking the room converts a rebuke into a courtesy, and the
song's whole argument is that a hundred people just proved they were the necessary equipment all along —
which is a *verdict*, not a favour. What survives is an instruction (*"do not slow, do not stall"*) and
a surrender (*"fine — take it, take it, take it all"*). Neither is gratitude.

## R11 · SOURCE-2 GRAMMAR ADAPTATION, PHRASES UNALTERED
*"furiously slapping and popping his bass"* is third-person masculine and our lead is a woman. **The
bassist is a separate body in the room**, so the phrase drops in unaltered: *"the bassist a step to her
right, furiously slapping and popping his bass."* *"syncopated sound"* attaches to the kinetic defect:
*"dragging the syncopated sound a hair behind."* Both **case-exact, in the generator-facing MUSIC PROMPT
field, all four packages.** V2 additionally carries *"downright salacious slow jam"*; V4 carries
*"steamy"*. No phrase is paraphrased and none is altered.

## R12 · THE CREDIT LEDGER, DECIDED BEFORE THE PROMPTS WERE WRITTEN
The prompts ask for **slapped-and-popped electric bass** and an **Afro-Cuban descarga break**. Under
`vault/DEDICATION_STANDARD.md` that raises two specific debts and I am naming both rather than crediting
a city: **Larry Graham**, who built the thumb-and-pluck technique the bass line is asking for, and
**Cachao (Israel López)**, who built the descarga as a form. ⛔ **No amen credit** — no amen, no jungle,
no breakcore is asked for anywhere in this pair, and a credit whose trigger did not ship is a false
credit. ⛔ Grammar checked against the implied-contents test: *"the technique we asked for by name"*,
never *"whose bass this is"*. ⛔ D4 checked: the dedications credit what these people **built**; none of
them claims a song is **about** a living or dead person's standing.

## R13 · ⛔ THE CHECK THIS PASS DID NOT RUN — NEGATION IN THE STYLE FIELD
*(Added after independent re-measurement of step 10. **The sweep table below was accurate on every row
it contained and was missing a row**, which is the more dangerous of the two failure modes.)*
All four MUSIC PROMPTs shipped negation — `no count-in, no reverb`, `never stop(s)`, `no pitch under it`,
`without help`, `that never quite lines up` — against run-wide ban 7, because Suno reads the style field
as **positive tokens**. ⭐ On this pair specifically, `never stop` arms the token **"stop"**, and P01's
literal composite element is a **butt-cut with a one-beat hole**: the pair was arming the exact word
whose behaviour it most needs the generator to get right. **Repaired in step 10** — every "no X"
restated positively, every literal exclusion moved to EXCLUDE, kinetic defect kept and re-grammared,
acoustic axis declared positively per the coordinator's ruling on the contract conflict. Full
before/after table and the re-measured counts are in
`pair_01_step10_revision_synthesis.md § REPAIR RECORD`.
**The transferable part: a self-check that only measures the numbers it was asked for will report PASS
on a field it never read.** The sweep is one regex — `\b(no|not|never|without|avoid)\s+[a-z]` over the
generator-facing prompt — and it belongs in the standing gate, not in one pair's memory.

---

## ⛔ REFINEMENT SWEEP — STATE AFTER THIS PASS
| check | state |
|---|---|
| equipment beyond the one dead fader | **none** — R2 sweep list held |
| *dark* in any title | none |
| water · sky · *current* | none |
| candlelight · cosy · lamplight · lantern | none |
| serene / resolved ending | none — R9 |
| listener address | none — every "you" is the crowd (F3) |
| house lexicon (13 phrases) | none |
| AI / platform / algorithm / meta | none |
| numbers sung | one, in V4 only — R7 |
| Source-2 phrases, case-exact, generator-facing | ≥2 in all four — R11 |
| ⭐ negation inside the MUSIC PROMPT | **was 2/2/1/3 — repaired to 0/0/0/0 in step 10, see R13** |

*Drafts on disk. Nothing rendered, published, or sent.* 💜
