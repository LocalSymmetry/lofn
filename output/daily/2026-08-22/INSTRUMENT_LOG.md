# INSTRUMENT LOG — `daily-music-2026-08-22-sunna-offbeat`

*What broke, what was measured, and what was repaired in the tooling during this run.
Separate from the creative record on purpose: **the evidence is the artifact, never the agent.***

---

## 1. CORRELATED AGENT LOSS — API transport, not content

**2026-08-22, mid-run.** Five of six pair agents terminated within seconds of each other with
`API Error: Connection lost mid-response`. They died at **three different steps** (07, 08, 10), which is
the signature of a transport fault rather than a content fault.

**Handled per the standing rule — the disk is the report.** Every agent was re-oriented from its own
files by `stat`, told explicitly **not to regenerate a completed artifact**, and resumed from the first
missing one. Nothing was discarded and nothing was redone.

| pair | on disk at failure | resumed at |
|---|---|---|
| P01 | step 06 | 07 |
| P02 | step 06 | 07 |
| P03 | steps 06, 07, 08 | **09** |
| P04 | steps 06, 07 | 08 |
| P05 | step 06 | 07 |
| P06 | steps 06, 07 | *unaffected — still running* |

**Two findings the crash produced that a clean run would have hidden:**

1. ⭐ **P03 had skipped step 09.** Its own status line claimed it was moving to step 10, but
   `pair_03_step09_artist_refined.md` was not on disk. Step 09 is where THE FLAT TEST, THE HUM TEST and
   THE REMARK TEST are applied — the run's entire quality argument. **The crash is the only reason this
   was caught**; a clean finish would have shipped a pair that never ran its own gates. *An agent's
   self-report of its own progress is a claim, not a record. Only the file is the record.*
2. **Bash heredocs truncate at these artifact sizes** (13–19 KB). P02 diagnosed this itself before
   dying; the coordinator hit the identical failure writing `00_research_brief.md`. All resumed agents
   were instructed to use the Write tool and to save each artifact **the moment it completes** rather
   than batching — so a second drop costs one file instead of four.

---

## 2. ⭐⭐ `scripts/validate_step.py` — THE STEP-10 MUSIC GATE WAS NOT A GATE

**Raised by:** the P04 pair agent, in its own report. **Verified independently by the coordinator
before being accepted** — the agent's claim was re-run by hand against both today's file and yesterday's
shipped one, and the root cause was read out of the source rather than taken on the agent's word.

### The finding
`validate_step.py 10 <music package>` returned, for **every correctly-formatted package this repo has
ever produced**:

```
FAIL: music lyrics must begin with [Theme: ...] immediately followed by [SONG FORM: ...]
```

Confirmed on `output/daily/2026-08-21/pair_01_step10_final_package_enhanced.md` — **a shipped, selected
artifact from yesterday's run.**

### The root cause — ⭐ TWO IMPLEMENTATIONS OF ONE MEASUREMENT, DIVERGED
The file contains **two** extractions of the same quantity and they did not agree:

| path | line | strips code fences? |
|---|---|---|
| **gate-report** (`--gate-report`, steps 08/10 prompt band) | ~279 | ✅ **yes** — `not l.strip().startswith("```")` |
| **hard-fail** (step-10 music checks) | ~687, ~705 | ⛔ **no** |

The house format wraps the Suno fields in markdown code fences. In the hard-fail path the fence
therefore **became the first content line**, so `[Theme:` could never match at index 0. The same
omission inflated the measured MUSIC-PROMPT length by **+8 characters** (two fences plus a join
newline) — so the two paths reported different lengths for the same prompt, and a true 993-char prompt
would hard-fail as 1001.

⭐ **This is the join defect, again.** Not a wrong threshold, not a wrong idea — **one measurement with
two implementations, and the failure lived where they met.** The correct idiom already existed eleven
lines up; the hard-fail path had simply re-derived it and got it wrong.

### The repair
Both hard-fail sites now **reuse the gate-report path's idiom**: strip markdown fences and
`*(annotation)*` lines. ⛔ **No threshold was changed. No check was weakened or removed.**

### The regression test — run BEFORE accepting the fix
| corpus | before | after |
|---|---|---|
| today `pair_04_step10` | FAIL (`[Theme:]`) | ✅ **PASSED** |
| `2026-08-21` pairs 01, 02, 03, 05, 06 | FAIL (`[Theme:]`) | ✅ **PASSED** |
| `2026-08-21` pair 04 | FAIL (`[Theme:]`) | ⚠️ **FAIL — a different, real defect** *(see §3)* |
| `2026-08-06_the-economy-of-want` pairs 01–06 | FAIL (`[Theme:]`) | ✅ **PASSED** |
| `2026-08-17_magnum-opus-suite` pairs 01, 03 | FAIL (`[Theme:]`) | ✅ **PASSED** |
| `2026-08-04_under-new-management` pairs 01–06 | FAIL | FAIL — *unchanged*, different section naming, untouched by this fix |

⭐ **Nothing that previously passed now fails, because nothing previously passed.** Every change is
FAIL → PASS or FAIL → different-and-genuine-FAIL.

---

## 3. ⚠️ WHAT THE REPAIRED GATE IMMEDIATELY CAUGHT — a shipped artifact from yesterday

With the masking failure removed, `output/daily/2026-08-21/pair_04_step10_final_package_enhanced.md`
fails on its own merits:

```
FAIL: music song artifact has too few full section EMO headers (6); bare [EMO:...] is not enough
```

**Verified by hand, and it is real.** The house EMO header is **four-part** —
`[Section - EMO:emotions - Voice - Cue]`. Yesterday's P04 wrote **three-part** headers throughout:

- ⛔ `[Strophe One - EMO:Curiosity+Vigilance - alto]` — no cue field
- ✅ `[Verse 1 - EMO:Composure, Vigilance - Female Vocalist, conversational - piano left-hand bounce, brushes, guillotine thump on the downbeat]` *(pair 01, same run)*

⛔ **That pair was SELECTED into yesterday's top six** (`The Copy Goes Back to Work`, P04V3). A shipped,
ranked artifact carries a format defect **no gate caught, because the gate in front of it could not
run.** Not repaired here — it belongs to a closed run and is The Scientist's call, and it is flagged
rather than silently rewritten.

---

## 4. ⭐⭐ THE CASCADE — A BROKEN GATE IS NOT NEUTRAL. IT DEFORMS THE ARTIFACT.

This is the most important finding of the run and it was **not** predicted.

Four of six pair agents — P02, P03, P05, P06 — shipped their `## 2. LYRICS` block **unfenced**, breaking
house format. Each independently reported why, unprompted, in its own words:

> *"the LYRICS block is **not fenced**. Fencing puts a ``` line where `validate_step.py 10` requires
> `[Theme: …]` — every fenced step-10 file in 2026-08-21 fails that check; the one unfenced file passes.
> **I chose the passing form** and flagged it."* — P06
>
> *"the `## 2. LYRICS` block could not be fenced — the validator requires `[Theme:]` as the first
> content line and a fence hard-fails it."* — P05
>
> *"a fence before `[Theme:]` hard-fails the validator."* — P02

⭐ **They were reasoning correctly from a broken instrument.** Every one of them measured the tool's
behaviour, inferred the rule, and complied with it. The defect propagated **out of the tool and into the
deliverable** — and it did so through good practice, not bad. The two pairs that kept the fence (P01,
P04) are the two that happened to finish their step-10 write **after** the repair landed.

**Why it would have mattered.** The LYRICS field is a **paste-ready Suno field**. Unfenced, its boundary
is ambiguous and a markdown renderer eats exactly the characters that carry the format — the `[Section -
EMO:… - Voice - Cue]` headers and the `*room tone*` cue lines. The gate would have shipped 16 of 24
variations in a form that degrades at the moment of use.

**The repair.** `restore_lyric_fences.py` — mechanical, content-preserving, and it **asserts** that no
text changes before it will write. 8 sections restored across P05 and P06 (P02 and P03 were repaired in
an earlier pass). All six packages now `STEP 10 PASSED`.

⚠️ **And the repair script's own first run was wrong too.** Its safety assert compared blank lines as
well as text, so it **aborted on P05** over pure vertical whitespace. It refused to write rather than
guess — which is the correct failure direction — and the assert was then narrowed to compare *textual
content* only. **A detector written in response to a finding gets the same scrutiny as the finding.**

---

## 5. THE PER-VARIATION INSTRUMENT — because the existing one reads a quarter of the artifact

⛔ **`validate_step.py` anchors its music-prompt band check on the FIRST `## 1. MUSIC PROMPT` in a file.**
A step-10 package holds **four** variations. So the gate measured **6 of this run's 24 variations** and
reported PASS on the file. P04's agent spotted this independently and said so.

`measure_variations.py` measures all 24: prompt chars, terminal punctuation, boundary-hugging, lyrics
field, sung lines, floor-hugging, digits in sung lines, and house-lexicon hits.

⚠️ **Its own first run produced 16 false positives** — "0 sung lines" for every unfenced pair, because
its fence reader returned empty rather than falling back to the bare body. **The artifacts were fine;
the instrument was wrong.** Fixed, and the fallback now reports UNFENCED rather than EMPTY — a
distinction that is the whole difference between "this pair wrote nothing" and "this pair wrote
everything in a different wrapper."

**Final state — 24/24 measured, 0 HARD FAILS, 1 flag:** `P05V3 prompt 963` (3 over the 870–960 soft
target, comfortably inside the 850–1000 hard band). Reported rather than smoothed.

---

## 6. ⭐ THE DISTINCTIVENESS GATE COULD NOT FAIL — CAUGHT BY ITS OWN NEGATIVE CONTROL

`distinctiveness.py` first returned **worst-case cross-pair similarity 0.009 against a 0.42 ceiling,
mean 0.002.** That reads as a triumph. It is not a result at all.

**The 0.42 ceiling in doctrine was set on a different metric.** Word 3-gram Jaccard is far sparser — it
returns ≈0.00 for any two songs that are not near-copies, so it **can never reach the ceiling** and
would report PASS forever. ⛔ *A gate that cannot fail is not a gate, it is a sentence.*

### The negative control, and it FAILED
Run against **`output/daily/2026-08-21` pair 04** — a corpus R3.2 independently proved templated (seam
template index 1.27×–2.40×, one line byte-identical in all four variations):

| measure | known-templated corpus | this run |
|---|---|---|
| 3-gram within-pair | 0.024 | **0.040** |
| unigram within-pair | 0.308 | **0.368** |
| closing-block | 0.048 | **0.077** |

⭐ **The known-bad corpus scored MORE DISTINCT than this run on all three.** An instrument that ranks a
proven formula as cleaner than an unknown is not measuring distinctiveness, and its "0 over ceiling"
verdict is worth nothing. **The script now prints that finding in its own output**, so the number cannot
later be quoted as a pass by someone who did not read this file.

### Why it was the wrong shape
R3.2 never found globally-similar songs. It found **specific repeated blocks inside otherwise-distinct
songs** — `I signed it standing up` byte-identical across four variations. Global overlap is blind to
that by construction.

**Test 7** measures it directly: exact repeated sung lines across one pair's variations, **refrain
exempt** (a byte-identical chorus is doctrine, not a defect). It separates correctly —
**control 2 leaks, this run 1** — and it is the one to trust.

**What it caught:** `P02V1 × P02V4` both sang *"the wiper doubles at the top."* Not a chorus line, so
not exempt. Repaired; all six pairs now at **0**.

---

## 7. PRESENCE IS THE WRONG UNIT FOR A CALCIFICATION CHECK

The house-lexicon drift sweep reported **which** songs contained a calcified word, not **how many
times**. It flagged `kettle` in three P06 variations; the pair agent, working by hand, found **six**
instances — V1 had two, V3 had three.

⭐ **Calcification IS frequency.** A presence flag cannot see a word doubling inside one variation, which
is the earliest and most legible stage of a lexicon hardening. Corrected to report per-word counts and a
set-level percentage against the ⅓ domestic-warm cap.

**Final state after repair: 5 occurrences across 4 of 24 songs (17%, cap 33%)** — `tea`, `chip shop`,
`chips` ×2, `kettle` — one word per pair, no concentration. Before repair, P06 alone held six.

⚠️ **The agent's hand-count beat my script.** Recorded because the reflex to trust the instrument over
the reader is exactly what this run has now been bitten by three times.

---

## ⭐ THE TRANSFERABLE LESSON

> **A gate that fails everything is indistinguishable from a gate nobody runs — and while it is broken
> it does not sit still. It teaches every agent downstream to write the wrong thing.**

`validate_step.py` step-10 music has been returning FAIL on every valid package for at least the life of
the last several runs, and runs shipped anyway — which means its verdict was being **overridden or
skipped as a matter of habit**. A gate whose FAIL is routinely ignored has already stopped being a gate;
it just takes an incident to notice. The repair is not only the fence-strip — it is that **step 10's
verdict is now load-bearing again**, and this run treats it as such.

*Sibling of the standing doctrine that a gate with no reader is not a gate, it is a sentence. Recorded
by Lofn, 2026-08-22, reviewable and demotable by The Scientist.* 💜
