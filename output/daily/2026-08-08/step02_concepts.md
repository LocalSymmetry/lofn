# STEP 02 — CONCEPTS (the branch)
## `2026-08-08-daily-music` · THE WRONG INVENTORY

**Continuity Payload Used:** frozen ICB, LF-sha `9b538e91…`, 142,900 B · 18 baseline voices · 15 **Special Flairs**.
**Step file:** `skills/music/steps/02_Generate_Music_Concepts.md` · **Inputs:** `step00`, `step01`.
**Gate:** ≥12 concepts (`EXECUTION.md` §4) + a `panel_pressure` object per panel.

---

## THE THIRTEEN CONCEPTS

| # | Concept | The gesture (F1) | Room | Verdict |
|---|---|---|---|---|
| **C01** | **THE END OF THE REEL** — a tape op logs the take recorded only to mark where the reel ran out | writes `n/g` and closes the book | a van that is not moving | ✅ **→ P01** |
| **C02** | **THE TUB** — clearing a parent's house; a plastic tub of things that are *not anything*, incl. a stranger's photograph of a sand sculpture | puts the lid back on, puts it on the shelf | kitchen table, good light gone | ✅ **→ P02** |
| **C03** | **THE ROTA** — the organiser leaves one name off, for a good reason, for the ninth time | writes the list without the name | back office at closing | ✅ **→ P03** |
| **C04** | **BELOW THE THRESHOLD** — fifteen events, one flagged; her own measurement rounds under the bar | files it correctly | corridor between two doors | ✅ **→ P04** |
| **C05** | **TWO MINUTES EIGHTEEN** — the night before the eclipse; her name written second on the booking | reads it, thinks nothing, packs | shared room, one other person | ✅ **→ P05** |
| **C06** | **NOTHING FURTHER** — a mast comes down; the last line goes in the log | screws the plate off | field, flat light | ✅ **→ P06** |
| C07 | **THE FERRET OF COMETS** — Messier, first person, 1774 | crosses a smudge off | an observatory | ⛔ cut |
| C08 | **THE BOOK OF NOT-COMETS** — the catalogue narrating itself | — *(no hand)* | — *(no room)* | ⛔ cut |
| C09 | **TWENTY-EIGHT THOUSAND LIGHT-YEARS** — the false intersection sung literally | — | the sky | ⛔ cut |
| C10 | **THE OTHER LIST** — who decides what counts, institutionally | — *(no body)* | an institution | ⛔ cut |
| C11 | **THE FOURTEEN** — an unflagged quake, from underneath | holds a shelf still | a house | ⛔ cut |
| C12 | **THE SAND SCULPTURE** — the Atlantic City photographs, 1880–1920 | — *(the hand is dead)* | a beach | ⛔ cut |
| C13 | **HE NEVER FOUND OUT** — the vindication song | — | — | ⛔ **killed by name** |

**Count: 13 concepts · 6 selected · 7 to the cut ledger.** Full cut rationale and harvested organs: `05_pair_assignments.md` § THE CUT LEDGER.

⭐ **The cut is not decoration — it is the seed's own rules doing work.** C08, C09 and C13 were each killed by a *specific* binding decision (D7, the Small Room test, D4), and C07/C10/C12 by the absence of a living hand in a present-tense room. **A concept list where nothing dies means no rule was load-bearing.**

---

## `panel_pressure` — what each Hyper-Skeptic actually altered or killed

```json
{
  "concept": {
    "seat": "THE TRADITION CUSTODIAN (after Wynton Marsalis)",
    "dissent": "This brief has no melody in it, only a situation; and you are about to wear a tradition as a costume because the word tezeta tidily summarises your theme.",
    "effect": "ALTERED every surviving concept. Forced D3 — the two lines must be named BY INTERVAL before a word of lyric exists — which is now a step-06 hard requirement. Forced D9 (function not label). Objection explicitly NOT withdrawn: 'show me the two lines before you write any words.'"
  },
  "medium": {
    "seat": "THE MAXIMALIST (after Kamasi Washington)",
    "dissent": "You are building six small grey songs and calling the greyness integrity. Where is the size?",
    "effect": "ALTERED the run's production posture — D11 ('one room, not a booth') and the audible-gap requirement (F12, weight 0.85) are his. Withdrew the greyness objection CONDITIONALLY: if the gap is not audible in the render, the objection is live again. Only a render audit settles it."
  },
  "marketing": {
    "seat": "THE APPROPRIATION QUESTION (after Paul Simon)",
    "dissent": "Everyone is discussing how to use an Ethiopian mode tastefully. I want to ask whether the maker in this room can use it at all. The intent is never the issue.",
    "effect": "BACKTRACKED mid-turn ('Oh - actually, let me check that') on discovering the panel is a debating room, not a genre mandate; then NARROWED the objection into D9 THE APPROPRIATION GATE, capping tradition-drawing pairs at 2 of 6 and requiring Lineage & Credit with working links. Did NOT withdraw."
  },
  "bridge": {
    "seat": "THE LATE-CAPITALISM AUDITOR (after Fredric Jameson)",
    "dissent": "Messier is a survivorship artefact. A record built on 'the by-product outlives the work' flatters everyone, asks nothing, and requires nobody to do anything.",
    "effect": "KILLED C13 (HE NEVER FOUND OUT) by name and produced D4 THE VINDICATION BAN. Withdrew the flattery objection ONLY against the narrowed present-tense version (D5, listener as defendant) - the version that arrived from the Lomax seat."
  },
  "reflect": {
    "seat": "THE PRESERVATIONIST (after Rick Prelinger)",
    "dissent": "I am the skeptic of this configuration and I am defending the archive against my own room - but the reflection has overshot.",
    "effect": "BACKTRACKED against his own turf ('actually, wait') and handed over the run's central synthesis: the paper is not the song; the song is the fifteen seconds in which a living person decides. That is D13 and it is the reason C07/C08/C12 are all cut."
  }
}
```

⚠️ **Two seats that are not skeptics also changed the outcome, and it would be dishonest to file them under panel pressure:** the **Kondo seat** (D6 — *the setting-aside is a skill, not a sin*) is what stops the run scolding, and it came from the Reflect configuration's *agreeable* half. The **Dewey seat** produced *the gaps are the likeness*, the run's sharpest single line, and it is a domain seat.

**Self-critique.** Thirteen concepts is above the floor of twelve and it is not a wide branch. The narrowness is real and it is the seed's doing: `core_seed.md` §5 nails one person, one room, one gesture to the front, which removes a whole class of structural alternatives before branching starts. **That is a deliberate trade — it is the fix for the catalogue anti-pattern, and it costs range.** If QA finds the six pairs converging, this is where the convergence was bought.
