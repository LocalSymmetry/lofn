# 06 — MUSIC HANDOFF (the Suno contract)

**Run:** `2026-08-27-high-energy-registers` · **Seed:** `SOMETHING TO PUSH AGAINST`

**Four variations per pair. Three pasted fields per variation.** Everything below is the hard contract; thresholds live in `vault/gates.yaml` and are quoted here, never invented.

---

## THE THREE PASTED FIELDS

- **`## 1. MUSIC PROMPT`** — ONE standalone, copy-paste Suno style prompt per song. **Dense prose paragraph, 850–1000 chars; write into 870–960** *(pinning the cap is a boundary-hug FLAG)*, and it **must end as a complete sentence** — a mid-phrase truncation to fit the cap is a hard fail; trim a clause instead. No categorized `key:value` brackets.
  **Mandatory order:** genre/micro-genre + tempo/energy/opening → vocalist spec with spatial staging → instrumentation with physical adjectives → arrangement arc → bold sonic device.
  **Must contain:** a vivid **opening moment (first 5 s)**, **spatial language**, a **kinetic defect** (asymmetric groove), and explicit acoustic / no-acoustic declarations.
  ⛔ **Banned openers:** *Begin in/with…, Use…, Build the track from…, Chronology:, For an adult human singer…*

- **`## 1B. SUNO EXCLUDE PROMPT`** — separate negative field, **400–900 chars (hard max 1000)**. Concrete blacklist terms and failure classes, never avoidance prose in the style field.
  ⭐ **`suno_exclude_chars: [400, 900]` is in `vault/gates.yaml` as of today.** It was prose-only until 2026-08-27, when a pair went looking for it there on the coordinator's instruction and found nothing.
  ⚠️ **We have ZERO measured claims that exclude tokens do anything at all** (`vault/music_atlas`, `exclude` channel: 0 claims). **The forbidden-default list is a discipline for the writer, not a guarantee about the render.** Do not spend your discriminator on a prohibition.

- **`## 2. LYRICS`** — open with `[Theme: <scene-pressure / emotional OS>]`, then `[SONG FORM: <named form & sequence>]`; **full EMO headers** `[Section - EMO:<emotion(s)> - <Role> - <cues>]` with emotions from `EMOTION_TAXONOMY` — ⛔ never bare AWE/INDIGNATION; **≥1 standalone `*SFX*`**; clean sung lines with no prompt or QA debris; **70–120 sung lines** *(this floor is rung 6 of the permission clause and yields — say so if you use it)*.

- 🚨 **LYRICS FIELD < 5000 CHARACTERS, TARGET ≤ 4800.** A Suno render limit, not a style note: the field holds **everything** you paste — `[Theme]` + `[SONG FORM]` + the Disc_Channel block + every section/EMO header + every `*SFX*` + all sung lines. **Measure exactly with a script; never estimate.** Over 5000 will not render. If over: cut/merge sung lines, then tighten headers, then move production metadata to a `## Production Sidecar` *outside* the field.

---

## ⛔ NAMES — THE ONE RULE WITH NO EXEMPTION

⛔ **NO REAL ARTIST, GROUP, PANEL-SOURCE OR COMPANY NAME IN ANY OF THE THREE PASTED FIELDS.** Scene and technique in the prompt; **people in the dedication.**

> ⭐ **A name in the prompt is an instruction to imitate; a name in the dedication is a debt paid.**

⚠️ **The manufacturer/brand-token conflict is STILL OPEN and is The Scientist's** — `vault/DISC_CHANNEL_GUIDE.md` *prescribes* brand tokens under its Instrument Backdoor Principle while this handoff forbids company names in the three fields, **so a pair following the guide is required to commit the violation.** It has now caused defects in two consecutive runs.
⛔ **FOR THIS RUN: keep them out of the three pasted fields.** Use a **mechanism description** — what the thing physically does, not who makes it. *A tape-replay keyboard with audible flutter and an eight-second note ceiling*, not the trade name. Name them freely in your ledger and dedication. **The conflict is escalated, not resolved.**

⭐ **`python3 scripts/check_field_census.py <run-dir>` is run by the coordinator after your package lands**, against the fenced fields themselves — not against your verification row. A hit named in your `## Major Deviations` reports as DECLARED; **the artists class has no exemption at all.**

---

## LINEAGE, CREDIT, DEDICATION

- **Living-scene genres require a `Lineage & Credit` block.**
- ⛔ **IF YOUR PROMPT NAMES A BREAK, CREDIT THE DRUMMER.** The player who made the figure is usually invisible, often dead, and asks for nothing — which is exactly why the system forgets the credit that most needs paying.
- **DEDICATION ≤ 500 characters, MEASURED, never eyeballed** (`vault/DEDICATION_STANDARD.md`). Overflow to a companion credits block. Trim priority 5 upward: never buy space by cutting the raised source-sound or the sharpest session-player credit.
- **Honesty clause both ways:** a generated track is **not a sample.** Say *"we asked for this figure by name, so we name them"* — never *"contains a sample of…"*.

---

## ⛔ MEASURED, NOT THEORISED — binding for this run

| instruction | what actually happened | class |
|---|---|---|
| `Hard_Pan_Right` | **−0.02 dB** and **+0.13 dB** overall balance across two renders; **0/52** and **0/56** windows above +3 dB | ⛔ **non-instruction** |
| a specified ~4-second internal stop | rendered at **0.40 s**; internal voids **filled 2 of 2** | ⛔ **SMOOTHED — measured dead** |
| a near-silent **opening** | preserved | ✅ **SURVIVED** |
| stereo width vs level, our own catalogue | **r = −0.43** — the render narrows at high level whether asked or not | ⚠️ **a condition, not a lever** |

⭐ **THE GRAIN LAW: specs that run WITH the generator survive; specs that fight it get smoothed.** If you want something absent, **write nothing there** — a prohibition is not a carrier.
⭐ **A Somatic or distinctiveness objection answered in the PRODUCTION SPEC is NOT ANSWERED.** Put it in the lyric or the form.

---

## WHAT YOU RETURN

Four variations, each with the three fenced fields, plus: the **register** and the **template you defeated** · **the named resistance quoted from your own sung text, with its verb and its timestamp** (R1) · the **discriminator line** (R3) · the **R6 vocal decision** · your **own chosen refused element** · `LINE BUDGET:` · `## Major Deviations` · the **R7 predictions** (collapse-template filed *before* writing; listener prediction before any render) · **Lineage & Credit + measured dedication**.

⛔ **You author your own form.** Section order, stanza lengths, rhyme scheme, hook grammar and sonic device are deliberately NOT specified — `core_seed.md` §2 carries the measured reason.

⛔ **`vault/AUTONOMY.md` holds:** drafts on disk. **Nothing rendered, published, or spent.**
