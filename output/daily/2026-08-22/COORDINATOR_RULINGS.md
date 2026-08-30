# COORDINATOR RULINGS — `daily-music-2026-08-22-sunna-offbeat`

*Questions escalated by pair agents, and how they were answered. One entry per ruling, with the
evidence that closed it. ⛔ No ruling here is closed on the coordinator's say-so — each names a file or
a script output a stranger could re-run.*

---

## RULING 1 — THE DEDICATION CONFLICT ⭐ *(raised by P01; my dispatch was wrong, not the agent)*

### What was escalated
> *"My dispatch bans naming real musicians **anywhere**; the ICB §8 scopes it to lyric/style fields and
> RULING C3 **requires** player credit. I credited in `## DEDICATION` only… Blocks lift out cleanly if
> you read the ban as absolute."* — P01

**The agent is right that these conflict, and it was right to stop rather than guess.**

### The ruling: **CREDIT STAYS. THE DISPATCH WAS OVER-BROAD AND THAT IS MY ERROR.**

My per-slot dispatch line — *"no real musician, band, record, company, official, or legal case named
anywhere"* — was written to protect the **HUMAN SUBJECT STANDARD** (do not fingerprint a real harmed
person). It over-reached into a completely different rule and would have deleted the credit doctrine
requires. The two rules are not in tension once they are read at the right scope:

| | rule | scope |
|---|---|---|
| **HUMAN SUBJECT STANDARD** | do not make a real person identifiable **as a subject** — the banned object is the tuple *(real name × locating detail)* for someone **actually harmed** | the whole artifact |
| **DEDICATION STANDARD / RULING C3** | **credit the player** — required, not optional | the dedication, and **only** the dedication |

⭐ **The distinction that resolves it, and it is already doctrine:**
> **A name in a generator-facing field is an instruction to IMITATE. A name in a dedication is a debt PAID.**

So: **generator-facing fields — `## 1. MUSIC PROMPT`, `## 1B. SUNO EXCLUDE PROMPT`, `## 2. LYRICS`,
`## 3. TITLE` — carry no real names. `## DEDICATION` is where credit is REQUIRED.** P01's handling was
correct as written and nothing lifts out.

⛔ **The dispatch's "anywhere" is struck.** Recorded because a coordinator instruction that contradicts
doctrine is a defect in the coordinator, and the agent that caught it should not have had to.

### The evidence that closes it — ⛔ not my testimony
`credit_sweep.py` extracts every capitalised name from all 24 `## DEDICATION` blocks, then greps all
four generator-facing fields of all 24 variations for each — full name **and surname alone**:

```
distinct credited names: 33
generator-facing fields swept: 4 x 24 variations
LEAKS: 0 - every credited name is confined to its DEDICATION block.
```

⚠️ **Honest note on the instrument:** the name regex over-captures — it also matches song titles
(*Be My Guest*, *My Boy Lollypop*) and sources (*Open Music Theory*, *Beat/Upbeat Ratio*). That is the
**safe** direction of error: over-capturing means *more* strings are checked for leakage, not fewer, so
a zero-leak result is strictly stronger than the name list is clean. It is not a defect in the
dedications.

### ⭐ What the sweep shows we actually did — the credit that matters most
The run's seed is **a deleted beat**. The two players who authored the deletions are both credited:

- **Carlton Barrett** (P03) — the **one drop**, the pattern defined by the kick that does not land on beat one.
- **Lloyd Knibb** (P02) — the **ska stroke**, the upstroke that became an entire music.

Alongside them: Don Drummond, Roland Alphonso, Jackie Mittoo, Lloyd Brevett, Jah Jerry Haynes, Rico
Rodriguez, John Bradbury, King Tubby, Augustus Pablo, Cedric Brooks, Andrew Love and Wayne Jackson,
Bernard Edwards, Brad Delson, Rob Bourdon, Neneh Cherry, Noga Erez and Ori Rousso, Fats Domino,
Barbie Gaye.

⛔ **This is the credit the system is built to forget** — *"living-scene credit is easy, visible and
already gated; the player who made the figure is invisible, usually dead, and asks for nothing."* On
2026-08-13 two songs shipped `amen` in the style field and credited nobody. **This run asked for the
one drop and named Carlton Barrett.** Honesty clause held both ways: these are generated tracks, not
samples — the dedications say *we asked for this figure by name, so we name him*, never *"contains a
sample of."*

---

## RULING 2 — THE MID-RUN TOOL PATCH *(raised by P01 and P05)*

Both agents noticed that `scripts/validate_step.py` changed underneath them and that caveats they had
written into their own artifacts had been made false by it. **Both were right and both re-ran their
measurements rather than leaving a stale claim on disk. That is the correct behaviour and it is noted
approvingly here.**

**Ruling: the patch stands, and it is disclosed rather than absorbed.** Full write-up, root cause and
regression sweep in `INSTRUMENT_LOG.md` §2. It is **shared tooling outside this run's directory**, so it
is surfaced to The Scientist explicitly rather than buried in a run artifact. ⛔ No threshold was
changed; no check was weakened or removed.

---

## RULING 3 — THE UNFENCED LYRICS BLOCK *(declared by P02, P05 and P06 as "unsatisfiable")*

Three agents independently declared the house code fence impossible because the **broken** validator
rejected it. **They reasoned correctly from a faulty instrument** — see `INSTRUMENT_LOG.md` §4, where
this is recorded as the run's most important finding.

**Ruling: the fence is restored, mechanically, and no text was touched.** `restore_lyric_fences.py`
asserts textual identity before writing and **refused to write on its first run** rather than guess.
8 sections restored across P05 and P06; P02 and P03 in an earlier pass. All six packages now
`STEP 10 PASSED`, and all 24 variations re-measured afterwards with **0 hard fails**.

⛔ **Nothing declared "unsatisfiable" by an agent was accepted as unsatisfiable without checking the
instrument first.** In all three cases the instrument was the problem.

---

## RULING 4 — THE TWO-TONE LABEL IN A STYLE FIELD *(raised by P02, decided by the agent, upheld)*

P02 wrote **"ska"** rather than the 2 Tone label in its style field, on the grounds that the label is a
real company and this run bans a real company from a generator-facing field — carrying the body instead
through *tempo, organ, horn stabs and the no-distortion clause.*

**Upheld, and it is the better call.** It is also the sharper craft decision: a label name asks a
generator to imitate a catalogue, while the four production facts ask it to play the music. **The
lineage is paid in the dedication, where P02 credits John Bradbury and the Skatalites players by name.**

---

*Rulings 1–4 issued 2026-08-22. Ruling 1 corrects the coordinator, not the agent.* 💜
