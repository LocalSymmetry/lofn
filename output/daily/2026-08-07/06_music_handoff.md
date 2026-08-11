# 06 — MUSIC HANDOFF · `2026-08-07-daily-music-indignation`

> ⛔ **THIS ARTIFACT IS NOT OPTIONAL.** It is the **only** artifact that resolves a live conflict between a step contract and doctrine. Its omission was the sole real defect of `2026-08-06_the-economy-of-want` (**L30**). It is written **BEFORE dispatch**, and it binds every pair agent and every downstream tier including step 11 and QA.

**Controller:** `lofn-main-20260807` · **Frozen ICB:** `CREATIVE_CONTEXT.md`, **53,003 B**, sha256 (LF-normalised) `5e9c7f7f6009fb3c672058c930540be22c8f5517f37537ac3ebd8ae94b75d374`

---

## 1. ⛔ THE CONFLICT THIS DOCUMENT RESOLVES — READ FIRST

**The step-11 contract file instructs the enhancement tier to embed the full archived payloads of the Golden Songs.**
**`EXECUTION.md` §3 GOLDEN-OUTPUT QUARANTINE forbids exactly that in a generating context.**

### ⭐ RESOLUTION — DOCTRINE WINS. THE QUARANTINE IS BINDING.

- ⛔ **NO past Lofn output — no Golden Song payload, no archived lyric set, no prior shipped style prompt, no winning image prompt — may appear in ANY generating context in this run.** Not in the ICB, not in a pair packet, not as a "calibration example," not quoted "for reference," not paraphrased.
- ✅ **What generators DO receive:** the **GOLDEN MOVE** (the distilled generative instruction), the **Golden Seed**, and the frozen ICB. **Seeds teach; outputs contaminate — including our own.**
- ✅ **Golden outputs go ONLY to judge-side contexts:** QA blind comparison, the step-12 audit, and the `lofn-step11-packager` bundle for genuinely external review.
- **This has already been enforced upstream:** `lofn-prime-mini.yaml` is 106,219 B, and **THE ARCHIVE (line 327 onward — ~18 complete past songs) was cut before injection.** The DNA block in the ICB is lines 2–326 only, **27,796 B**, verified `THE ARCHIVE` absent by assertion at assembly time.

### **If any local step file tells you to embed a Golden Song payload: THIS DOCUMENT OVERRIDES IT. Do not comply. Cite this section by name in your artifact and continue.**

⭐ **Why this is spelled out rather than assumed (L30, the exact lesson):** last run, five pairs reached past the local instruction to doctrine and refused by name; **one followed the file in front of it and was right to trust its contract.** An agent that obeys its nearest contract is behaving correctly — the coordinator had removed the artifact that overrides it. **Unanimity among five agents is not evidence a rule was stated; it is evidence five agents happened to reason the same way.** So it is stated here, in writing, in the run directory.

---

## 2. THE GOLDEN MOVE (generative instruction — this is what replaces the payloads)

1. **Find the person, not the topic.** Every song is addressed to **one specific human doing one specific physical thing**, named in line one. The thesis is never stated; **the addressee carries it.**
2. **One number, at the hinge, answered.** Max ONE sung numeric fact per song — **responded to, never recited.** A verse listing the day's facts is a weather report in meter.
3. **The riff before the word.** Named instrument, named interval, inside one octave, present **before the first line and after the last**, singable **by bar 8**. A pad is not a riff.
4. **AWE stays terror-adjacent.** Answer *where is the body standing* and *what could hurt it here* before drafting.
5. **The return device is the argument.** It must be the audible form of the pair's form rule — countable **in the lyric**, never in the production spec.
6. **Complexity in the music; feeling in the words.** *A listener will not decode a sentence — they will always recognise a situation.*

---

## 3. WHAT EACH PAIR AGENT RECEIVES (itemized packet — a missing element is a DISPATCH BLOCKER)

| # | Element | Value |
|---|---|---|
| a | **Complete personality DNA** | inside the ICB, Slot 4, **27,796 B**, verbatim, ARCHIVE excluded |
| b | **All 18 panel voices + objections** | ICB Slot 5 — 18 `(after …)` tags, 3 Hyper-Skeptics at 6/12/18 |
| c | **All 15 Special Flairs** | ICB Slot 6 |
| d | **The Golden Seed** | ICB Slot 1 |
| e | **The metaprompt** | ICB Slot 3 |
| f | **The pair slice** | `05_pair_assignments.md` §B, that pair only |
| — | **Total ICB** | **53,003 B**, injected **verbatim, in full, at the head of the packet** |

### ⚠️ ICB TRANSPORT — DECLARED HONESTLY (the deviation is NARROWED, not closed)

**What changed:** with THE ARCHIVE correctly excluded, the DNA block is **27,796 B** (was 106,219 B) and the whole ICB is **53,003 B** — down from ~180 KB. **The size problem that caused the original deviation is gone.**

**What this run actually does — the transport, stated plainly:**
- The **run-specific creative core** (Golden Seed, Four Laws, Seven Binding Constraints, metaprompt, the pair's own slice, the skeptic objections that bind it, the Special Flairs it may draw on) is **inlined verbatim in the spawn prompt.**
- The **complete frozen ICB including the 27,796 B personality DNA** is read by each pair agent **as its FIRST action** from `output/daily/2026-08-07/CREATIVE_CONTEXT.md`, and the agent **echoes back the byte count AND the LF-normalised sha256** — a value it can only obtain by actually reading the file. The coordinator **re-stats** both.

**This is still read-in-full + echo for the DNA portion, not full inline duplication ×6.** ⛔ **I am declaring it rather than claiming the deviation closed** — an earlier draft of this document overstated it as resolved, and that would have been the fourth consecutive run carrying an undeclared transport claim. **It remains open and awaits The Scientist's ratification or overrule.** What is genuinely fixed is the *cause*: the 106 KB file was the deviation's whole justification, and it is now 27.8 KB.

---

## 4. HARD OUTPUT CONTRACT — THE VALIDATOR IS THE SOURCE OF TRUTH

⛔ **`skills/music/scripts/validate_suno_packages.py` defines the heading convention. NOT the artifacts, NOT another pair's file, NOT your own preference** (L28: six agents produced three conventions and the coordinator invented a fourth by reading conventions off artifacts).

Per package, in this exact shape, **× 4 variations**:

```
### VARIATION n
## 1. MUSIC PROMPT          <- dense paragraph, 850-1000 chars, NOT tag-soup, no artist names
## 1B. SUNO EXCLUDE PROMPT
## 2. LYRICS                <- [Theme: ...] then [SONG FORM: ...] then full EMO headers
## 3. TITLE
```

- **EMO headers:** `[Section - EMO:<emotion> - <Role> - <cue>]`, `<emotion>` from `EMOTION_TAXONOMY`, ⛔ **never bare AWE/INDIGNATION**. ≥1 SFX cue.
- **70–120 sung lines.** ⚠️ **≤72 raises the boundary-hug FLAG — raise it explicitly, do not tick it clean** (L15).
- 🚨 **LYRICS FIELD < 5000 chars (target ≤ 4800). Count exactly and STATE THE MEASURED NUMBER.** The line-count target yields to this cap.
- ⛔ **No wall-clock times in section headers** — a generator reads `[7:52 to 8:19]` as minutes:seconds.
- ⛔ **No bracket characters inside a chorus line** — Suno parses them as a section boundary and destroys byte-identical return.
- **Timing gates carry their arithmetic:** bars × 60 ÷ BPM, shown.
- **Lineage & Credit block, with links.**

**Floors (`vault/gates.yaml`) — measure with `scripts/measure_soundcraft.py → profile_file()`, never by eye:**
`rhyme_return_floor 0.30` · `line_return_floor 0.20` · `alliteration_per_100w_floor 11.0` · `unique_line_ratio_floor 0.45` · `max_sung_numeric_facts 1` · ⚠️ **`mean_words_per_line_ceiling 7.5` (FLAG-class, NOT a hard fail — but it must be MEASURED AND REPORTED like every other gate).**

⛔ **R3, QA-issued 2026-08-07:** the words-per-line ceiling was missing from this list, so its disclosure depended on agent conscientiousness — and **the two pairs that reported only the enumerated floors are the two that breached it undisclosed.** An enumeration IS the contract: a gate absent from the list is a gate that will not be reported. **Rule: when a gate is added to `gates.yaml`, it must be added to every enumeration that tells an agent what to measure.**
⚠️ **A wordless return device (a vocable, a hum) can satisfy `line_return` almost by itself** — if yours does, **disclose it and report a lexical-only companion measurement.** The instrument cannot tell "the song returns" from "one syllable returns."

---

## 5. SELF-CHECK DISCIPLINE — THE FAILURE MODES THAT KEEP RECURRING

1. ⭐ **Print what you EXTRACTED before you trust what you CONCLUDED.** Assert the count equals the expected cardinality (4 packages, 6 pairs). **An empty extraction is a hard ERROR, never a passing score.** *(L25/L28/L31 — a false CLEAN on a Gate-2 audit came from exactly this.)*
2. ⭐ **A compliance claim is scoped to what you actually verified.** A repair applied to one variation and reported pair-wide is a **fictional fix** — 10 were found in one run. **Verify the named device in EACH variation individually.**
3. **When a scanner hits a correct line: fix the SCANNER, not the line** (L27). A passing floor is not evidence of absence — floors are keyed to a property and are blind to a defect in a neighbouring one.
4. **Scratch files must carry your pair id** (`scratchpad/pair_NN_*`) — un-namespaced scratch has been overwritten mid-run twice.
5. **Write your artifact to disk as you complete each step**, never hold it in memory for one final write.
6. **Never `cd`.** All paths are repo-relative from the working root.
7. **The describe-render self-check (one pass):** predict in 2–3 sentences what your prompt would **actually produce** on Suno, then answer adversarially — *"name the one way this would render generic."* Self-repair **once**, inside the existing max-3 budget.

---

## 6. WHAT IS PROTECTED FROM YOU

- ⛔ **The frozen ICB.** Copy-and-diverge into your own artifact; **never edit the canonical block.** Its sha is recorded above. *(Note: `core.autocrlf=true` on this checkout rewrites LF→CRLF; the frozen figure is defined **LF-NORMALISED**. A raw `sha256sum` mismatch of exactly the line count is **not** tampering — normalise before hashing. Five agents correctly detected and correctly did NOT "fix" this last run.)*
- ⛔ **P05's KEPT DEFECT.** Once nominated and defended in P05's artifact, it is **protected from repair by step 11, by QA, and by the coordinator.** Any tier that "fixes" it has broken the run's falsification test.
- ⛔ **Another pair's device.** Cross-pair device bleed was found in three separate pairs last run. Your reveal engine, your return device, your form rule — yours alone.

---

## 7. ESCALATION

- A gate failing **3× with no movement in its measured value** → stop, mark `quarantined`, surface **"pair NN broke open at step X (gate: name)"** to the human **before QA**. A 5-pair set never silently ships as 6.
- **The same gate failing across many pairs** → do not hammer 24×3. **HALT and name the gate.**
- **A human-subject identifiability flag** → **HELD FOR HUMAN**, surfaced by name, never silently shipped.
- ⚠️ **`check_human_subjects.py` fires `HOLD_FOR_HUMAN` on 100% of correct artifacts** (spaCy absent → its regex reads section headers like `Female`, `Vocalist`, `List` as person names, and `body` inside `nobody`). **A gate that fires on everything carries no information — judge the standard directly, do not defer to the script.** Unchanged since 2026-08-04.

---

*Written before dispatch, as the contract requires. If a step file and this document disagree, this document wins — and say so in your artifact.*
