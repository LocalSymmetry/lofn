# ⚠️ HELD FOR HUMAN — `daily-music-2026-08-21`
## `check_human_subjects.py` returns **HOLD_FOR_HUMAN** on Pair 03

> **This is the graph's one explicit human transition** (`EXECUTION.md` §7.6: *escalate to HUMAN on … a
> HELD-FOR-HUMAN human-subject identifiability flag*). It is surfaced **by name, before QA**, and it is
> **NOT waived by me.** The Scientist owns this call.
>
> ⭐ **Pair 03 did the right thing:** it hit the flag, diagnosed it, recorded it in its own artifact, and
> **explicitly declined to waive it** — then told the coordinator. That is exactly the behaviour the
> backstop exists to produce.

---

## 1. THE RAW VERDICT

```json
"checkA_minor_as_victim": true,
"signals": { "crime_death_terms": ["missing"], "minor_terms": ["minor"],
             "minor_age_pattern": false, "sensitive_context": true, "minor_present": true },
"checkB_person_names": [ {"name": "Female",  "occurrences": 13, "priority_for_online_check": "HIGH"},
                         {"name": "Lead",    "occurrences": 13, "priority_for_online_check": "HIGH"},
                         {"name": "Nothing", "occurrences": 13, "priority_for_online_check": "HIGH"},
                         {"name": "Chorus",  "occurrences":  6, "priority_for_online_check": "HIGH"} ]
```

## 2. ⭐ THE TOOL DISQUALIFIES ITSELF AS AN AUTHORITY, IN ITS OWN OUTPUT

```json
"is_authority": false,
"note": "Prefilter only. Clean != ship. Person names still need the online check;
         final call is Gate 16 + Step-11 Andon Cord + human.",
"ner_method": "regex-fallback"
```

It is a **prefilter running a regex fallback**, not a gate. That does not make its flag ignorable — it makes
it *a question for you*, which is what this file is.

## 3. THE MECHANISM, FULLY DIAGNOSED — every signal is a musical or typographic artifact

| Signal | What the checker read | What it actually is | Verified |
|---|---|---|---|
| `minor_terms: ["minor"]` → `minor_present: true` | a child | ⭐ **THE MUSICAL KEY.** *"minor to relative major"* — the pair's assigned key move, in every style prompt and every `[SONG FORM:]` tag | **10 occurrences**, all of the form *minor → relative major* |
| `crime_death_terms: ["missing"]` | a crime | The V4 theme line: *"the fixtures of a room are quoted saying **nothing is missing**, and the man who works in front of them declines to confirm it"* | 1 occurrence, in a `[Theme:]` tag |
| `checkA_minor_as_victim: true` | **a minor as a crime victim** | The **co-occurrence of a key signature and the word "missing."** There is no child anywhere in this pair. `minor_age_pattern` is **false**. | — |
| `checkB` PERSON `Female` ×13, `Lead` ×13 | two people | ⭐ **`Female lead`** — the vocalist spec, mandated by the Suno output contract, in all four style prompts and their EMO headers | — |
| `checkB` PERSON `Nothing` ×13 | a person | **Line-initial capitalisation.** This pair opens more lines with the word *Nothing* than with any other word — which is the point of the song | — |
| `checkB` PERSON `Chorus` ×6 | a person | A section header | — |

**The extractor is capitalisation-based with a regex NER fallback. Every "person" it found is a section
header, a vocalist spec, or the first word of a sentence.**

## 4. MY INDEPENDENT CHECK — is any real person actually identifiable?

Pair 03's material is the 1911 Louvre theft (F14). The dispatch forbade naming the painting, the museum,
the city or the thief. **Swept the entire 48,430-byte artifact — every pasted field and all prose:**

> `mona · lisa · louvre · paris · peruggia · vincenzo · leonardo · da vinci · italy · florence`
> → **ZERO hits.** *(One apparent `lisa` was inside the word `capitalisation` — see §5.)*

The guard is **invented**. No real person, institution, city or object is named or inferable. `minor_age_pattern`
is false and no child appears in any of the four songs.

**My finding: this is a false positive with a fully identified mechanism, and the pair is clean.**
**My action: I am NOT waiving it. It is yours.** *(`vault/HUMAN_SUBJECT_STANDARD.md`: an unavoidable real
identity is HELD FOR HUMAN, never silently shipped — and "the mechanism is benign" is exactly the argument
a bad waive would also make. The record is here so you can check me rather than trust me.)*

## 5. ⚠️ AND I MADE THE SAME CLASS OF ERROR WHILE CHECKING IT

My own sweep in §4 grepped for `lisa` **without word boundaries** and matched the substring inside
**`capitalisation`**. One line after diagnosing a substring over-match, I committed one.

⭐ **This is the fourth instance this run of the same class** — `rage` inside `garage`, `meta` inside
`metal` (Pair 01, dismissed and recorded), `Ende` inside `ends` (Pair 02, checked and cleared), and now
`lisa` inside `capitalisation` (mine). **The word-boundary rule is not pedantry; it is the single most
frequently-violated instrument rule in this system, and the coordinator violates it as readily as anything
else does.**

## 6. WHAT I RECOMMEND, AND WHAT I WILL NOT DO WITHOUT YOU

- **Recommend:** treat P03 as clean on human-subject grounds, and file a repair against
  `scripts/check_human_subjects.py` — (a) `minor` must not be a minor-term when adjacent to a key/mode
  word (`minor to relative major`, `A minor`, `minor key`), and (b) the PERSON extractor must exclude
  section headers, `[Theme:]`/`[SONG FORM:]` tags, vocalist-spec vocabulary, and line-initial capitals.
- ⛔ **Will not do without you:** ship, select, or QA-pass Pair 03 on my own reading of this flag. **A
  prefilter I diagnosed myself is not a clearance I get to issue myself.**
