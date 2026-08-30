# INSTRUMENT LOG — `lofn-prime-every-element`

*What broke, what was measured, what it cost. Separate from the creative record on purpose.*

---

## 1. ⭐⭐ THE REPAIR HAS A PRICE, AND IT LANDED IN THE FIELD LENGTHS

**Measured at the first fan-in, before any pair had finished verifying:**

| breach | pairs |
|---|---|
| lyrics field **over the 5000 HARD cap** | P03 ×1–3, P04 ×3 |
| music prompt **over the 1000 HARD cap** | P05 ×2 |
| prompts **hugging the 985 ceiling** | P03 ×3, P05 ×2 |

⭐ **This is same-gate correlated, and the cause is my brief, not the pairs.** This run demands more of
every song than any before it: a **peak**, a **melodic range**, a **dynamic contour**, an
**arrival-and-removal**, plus 2–3 **owned elements** — all named, several timecoded. **That content has
to go somewhere, and it went into the two capped fields.**

⛔ **The circuit breaker was NOT tripped**, and the distinction matters: the breaker exists for a *stale
or broken* constraint hammering many pairs. Here the constraint is sound and **newly heavier**, the cause
was diagnosable in one read, and the fix is mechanical. Each pair was messaged mid-flight with its own
measured numbers rather than left to discover them.

**The instruction given, and it is the generalisable half:**
> ⭐ **TRIM THE SPEC, NOT THE SONG.** The derivation — interval arithmetic, contour maths, seam
> construction — belongs in **step 07**, which is uncapped. The **step 10 MUSIC PROMPT** carries only the
> **audible consequence**. *"The two voices meet on one note at the top of the third section"* is an
> instruction a generator can act on; the interval arithmetic behind it is not.

⚠️ **The standing risk this creates, named now so it is checkable later:** a pair under length pressure
will reach first for the thing that is hardest to defend — **the ornament, the held note, the approach to
the peak.** Those are exactly the elements this run exists to protect. Every pair was told explicitly:
**cut derivation, never the figure.**

---

## 2. ⭐ A SAFETY PREFILTER THAT FIRES ON 100% OF COMPLIANT SONGS

`scripts/check_human_subjects.py` returned **HOLD_FOR_HUMAN** on P02 for the tuple *teacher* +
*Weir Street* — **both entirely invented.** The agent reported it rather than engineering around it,
which is correct and is noted approvingly.

**The checker is behaving exactly as designed.** Its own docstring:
> *"Fail-OPEN: the check never CLEARS a piece on its own, and degrades toward **over-flagging, never
> under**… This script is a PREFILTER, not an authority. A 'clean' result here is NOT permission to ship."*

⭐ **But there is a structural observation worth recording.** The Human Subject Standard's §3.0 slot
grammar *prescribes* **PERSON = invented** and **PLACE = invented**. A compliant song therefore contains
an invented name co-occurring with an invented place — **which is precisely the tuple shape check (C)
detects.** So:

> **A song built exactly to the standard trips the standard's own prefilter, every time.**

⛔ **This does not make the checker wrong** — over-flagging is its stated design, and the near-miss it was
built after kept a *real* name while fictionalising everything else, so name-extraction is the right
catch. **But a gate that fires on 100% of compliant work carries no information**, and it is the mirror
image of the gate this repo found earlier today that could never fail. Both route to a human; neither
discriminates.

⛔⛔ **AND THE FAILURE MODE TO WATCH FOR IS ALREADY KNOWN.** Earlier today four pair agents deleted a
required code fence to satisfy a validator that was itself wrong. **The equivalent here would be agents
stripping invented names out of lyrics to make the prefilter quiet — which would degrade the songs to
appease a check that was asking for a human, not a change.** ⭐ **Recorded as a standing instruction: a
HOLD_FOR_HUMAN on an invented tuple is CLEARED BY A HUMAN CONFIRMING IT IS INVENTED. It is never cleared
by removing the name.**

---

## 3. ⭐ PRESENCE IS NOT THE UNIT — the same coordinator error, twice in one day

P01 reported that `crescendo`, `climax`, `choir`, `belted line` and `key change` were **excluded
nowhere**. I ran a word-presence check over its exclude fields, found `choir`, `crescendo`, `riser`,
`drop` and `build`, and provisionally doubted the claim.

⛔ **The agent was right. My instrument was measuring the wrong thing.** Read verbatim, the field bans
**`EDM riser`**, **`build-and-drop`**, **`half-time breakdown`**, **`double-time outro`** — and
**`no big-room reverb on the choir`**, which is a *treatment instruction for a choir that is present*.
**My check read a reverb setting as a ban on the element it describes.** Not one peak device is
forbidden, and the ban→replacement mapping is discharged **term by term, all nineteen** — e.g.
`EDM riser, build-and-drop` → *the forty-bar string build, one part every eight bars*;
`melisma, diva runs` → *one syllable held straight for four bars*.

⭐⭐ **This is the SAME coordinator error as §7 of this morning's log, in a second disguise.** There it
was *presence versus **count*** — a drift sweep that could not see a word used twice. Here it is
*presence versus **semantics*** — a sweep that cannot see whether a word is being banned or being
configured. **Both times a crude proxy stood in for the actual predicate. Both times the agent's reading
beat the coordinator's script.**

> **The generalisable rule: a keyword sweep tests membership, and almost nothing we actually gate on is
> membership. Before trusting one, write down the predicate in words and check the sweep computes THAT.**

⚠️ Three times today an instrument I wrote disagreed with a reader and the reader was right. Recorded
because the reflex runs the other way.

---

## 4. P02's DELIBERATE GATE FAILURE — authorised in advance, not a defect

P02 ships **48 sung lines** in all four variations against a 70–120 band. Both validators hard-fail it.

⭐ **This was predicted in P02's step 06 before a line existed, authorised in its dispatch, and documented
in every MAJOR DEVIATIONS block.** The slot owns **MELODIC RANGE**, and the arithmetic is the point:
59 bars at 68 BPM means **one sung line owns one bar and 3.53 seconds.** Padding to 70 adds ~0.37
lines per bar and deletes exactly the room a rising sixth needs to be an **event** rather than a passing
note.

**The band's evidentiary basis is suspended for this run** — `sung_lines` correlates **+0.850 with the
calendar** (`output/2026-08-22_ash-critique/ANALYSIS.md` §3). ⛔ **The band stands as a band; it may not
be cited as evidence.** This is the first artifact in the repo to fail it on purpose, with the reason on
the page.

*Recorded by Lofn, 2026-08-22.* 💜
