---
run: 2026-07-01 (FULL DAILY EXTENSION)
phase: 1 — Orchestrator, step 6 (audio handoff) for the SIX pair agents
cardinality: "6 pairs x 4 variations = 24 songs -> best 6 (3 ACCESSIBLE + 3 AMBITIOUS)"
tiering: "Opus generates steps 00-10 per pair; Fable refines step 11; Fable judges QA — generator and judge share no weights"
---

# 06 — Audio Handoff (FULL RUN: six pair agents)

This is the launch packet for the six per-pair Claude subagents (pairs 01–06). Each pair runs its own canonical chain (steps 06→07→08→09→10, then step 11 enhancement) and produces four variation songs. Read `05_pair_assignments_full.md` for your pair's full brief. This handoff carries the shared laws every pair obeys identically.

## 1. Inject the frozen ICB — VERBATIM

**Every step (06–11) and every pair subagent MUST verbatim-inject the full Creative Context Block:**

`output/daily/2026-07-01/CREATIVE_CONTEXT.md` — **FROZEN, 134,637 bytes.**

- Inject it whole, byte-for-byte, at the head of every step prompt (`EXECUTION.md` §3). Do NOT summarize, excerpt, or paraphrase it. Do NOT edit it mid-run — the ICB is frozen since Phase 1; sideways moves spawn NEW pair chains, they never edit the ICB.
- The ICB carries: the metaprompt (personality voice, locked moods, panel aha-moments), the full LOFN-Prime-Mini personality YAML, the 18-voice Panel Ledger (use these exact voices — do NOT invent a panel), the 5 constraint axes, the Source-2 vocabulary, the Source-3 form rule, the daily mandates, the legibility rule, the Seed Genre Palette, and the Seed Music-Frames/Technique palette. It is the single source of truth for personality fidelity.
- Confirm `icb_sha` / byte-count match in your pair's Continuity Payload citation.

## 2. Golden Song references — BY NAME ONLY (GOLDEN-OUTPUT QUARANTINE)

Per `EXECUTION.md` §3 and `.claude/skills/lofn-music/SKILL.md`: golden-song **payloads (style prompts, lyrics, exclude prompts) are NEVER pasted into any generating context** — not this handoff, not a pair subagent, not step 11. An exemplar in the generator's prompt becomes a mold and the run regresses toward a diluted copy. Generators receive the GOLDEN MOVE (§3 below) INSTEAD. The names travel only so QA can run its blind comparison later.

Selected for this run (from `skills/music/references/golden_songs_index.md`, per the index's "Public-facing staff-pick follow-up pressure" pairing heuristic):

- **AWE / EXISTENCE side (pairs 01, 03, 04, 05):** **"Triple Arch Over Me"** — judge-side reference only. Names its sky outright, scale grounded in the body, emotionally legible profundity, unresolved-into-included turn. Do NOT read or reproduce its payload; the naming exists for QA's blind set.
- **INDIGNATION / NEWS side (pairs 02, 06):** **"Five wrong colors"** — judge-side reference only. INDIGNATION without flattening into rage, wrongness as hook, refrain returns damaged, non-consoling palette. Do NOT read or reproduce its payload.

Two further names live in the index for QA calibration if needed — **"The Blue Screen Breathes"** (machine/body ambiguity) and **"I Will Stop the Almost"** (refusal-as-hook clarity) — names only, judge-side.

## 3. THE GOLDEN MOVE — full six-rule block (copied verbatim from `.claude/skills/lofn-music/SKILL.md`)

Every pair receives this block. It is what generators get INSTEAD of golden songs.

1. **Stand somewhere real.** The song is a report from ONE concrete place the body occupies — name where it stands and what the senses register there. Concept-illustration ("a metaphor about X") is the failure mode; experience-report is the move. If three runs in a row are indoors and safe, go outside.
2. **One wounding fact.** At most ONE numeric/scientific fact is sung, placed at the emotional hinge, and the lyric must RESPOND to it ("It says behold and calculate"), never recite it. All other research stays in the brief as atmosphere.
3. **The turn.** Somewhere past the midpoint, the song contradicts or complicates its opening stance — a mind changing in real time, an argument with itself the ending has to earn. A song that asserts its final emotion from line one is a corpse.
4. **Fear stays braided in.** AWE is terror-adjacent sublime, not domestic reassurance — every awe song carries a clean fear it does not resolve cheaply.
5. **Rotate the register.** Do not default to the house winner's fingerprint (crystalline female soprano / A major / ~110 BPM / frost-and-cosmos palette). Vary key, tempo, vocal register, and sonic world per run unless the personality's YAML mandates them. The house-lexicon FLAG (`vault/gates.yaml`) catches verbatim self-copying; this rule prevents the softer clone.
6. **The surface names its subject (first-listen legibility).** A stranger must be able to retell the song's scene AND subject in one sentence after ONE listen — the subject appears PLAINLY in the lyric at least once (title or an early verse), not only through metaphor. Triple Arch names its sky outright; the depth lives in the RESPONSE to the named thing, never in withholding the referent. Obliqueness about what the song is about is not depth, it is fog (2026-07-01 test slice: an AWE song about a survivor star read as being about a troubled uncle — structurally perfect, emotionally unreachable). Simple surface, complex engine: the cathedral lives UNDER a legible surface, not instead of one. This is the music equivalent of the image lane's thumbnail test, and the Somatic Gate should treat an unnameable subject as `REPAIR — FOG`.

> **Binding this run (the Scientist's test-slice verdict):** rule 6 is the fix for the slice's one failure — composition was right, emotional connection failed on lyric coherence. Every variation must pass the one-sentence stranger-retell test. An unnameable subject is `REPAIR — FOG` at the Somatic Gate, not a stylistic note.

## 4. The Source-3 form law (material structure — MANDATORY, all six pairs)

**Two superimposed expansions of different ages** (the APOD form rule). A slow OLD structure runs the entire length of the song underneath a faster YOUNG structure — they overlap continuously; they do not alternate or trade off. The old layer gets exactly one moment fully alone (the bridge), where the young structure drops out entirely and the underneath is suddenly all there is.

Each pair realizes this DIFFERENTLY (see `05_pair_assignments_full.md`):

- **01** field-recorded wind-drone (old) under a bar-count-drifting vocal/percussive layer (young).
- **02** felt-not-heard somatic sub-bass (old) under a too-perfectly-quantized glossy topline (young).
- **03** dub-death-drone (old timeline: the explosion already arrived as light) under a self-doubling delay percussion (young timeline: the explosion still coming) — the two ages are two literal death-events.
- **04** old-tongue vow-choir/drone (old: the form of promise older than the promiser) under the failing rubato alto/hand-percussion (young: this specific, present, failing vow).
- **05** unblinking tanpura-like microtonal watch-drone (old) under the passing log-entries and clock-tick (young).
- **06** the machine's-weight sub-bass (old) under the confessing voice and clinical prompt-tone (young).

Do not merge the layers clean. Surface the old layer alone at the bridge, every pair.

## 5. Daily hard rules (this run — bind every pair, every variation)

- **One-fact rule (Golden Move rule 2):** at most ONE sung numeric/scientific fact per song, at the emotional hinge, responded to — never recited, never explained. Research shapes theme/form; the numbers stay in the brief. (Pairs 02 and 06 run on 0 research-facts; count-words in those are self-referential, not data.)
- **House-lexicon ban:** no phrase from `vault/gates.yaml → house_lexicon` (the calcified Triple-Arch fingerprint — "more sub and more sky," "frost-air pad," crystalline-soprano/A-major/~110 BPM defaults, "dew-bright vibrato," etc.) appears in any MUSIC PROMPT, EXCLUDE, or lyric. A hit FLAGs to the Somatic read. Golden references teach the move, never the words.
- **AWE stays terror-adjacent:** every AWE/EXISTENCE pair (01, 03, 04, 05) must name what could hurt the body in THIS exact scene — realized in a couplet, not asserted. (The slice's P1V2 HOLD: terror was gestured, not enacted; do not repeat it.)
- **Human-subject discipline (binding, pre-draft):** no identifiable real person, place, or specific recent circumstance from the news anchors. The aftermath-charge (being lifted from the dark; warmth from a not-original source) and the slop-economy charge (a warning made beautiful) may be carried as PATTERN; every name, place, date, institution, product, and circumstance is INVENTED. No name, no country, no "day six," no earthquake specifics, no real product or real video. REAL GRIEF IS NOT RAW MATERIAL.
- **Per-pair variation discipline:** four variation angles per pair, each derived from THAT pair's own concept; no shared angle-label set across pairs. Cross-variation identical-sung-line budget is a distinctiveness FLAG routed to the Somatic Gate — **pair 02 specifically must carry ZERO shared post-bridge payoff lines across its four variations** (the slice's P2V2 HOLD: 28% overlap, entire payoff borrowed — each variation's indictment re-derives from its own organ).
- **Device economy (Raw-Production Skeptic baseline):** pair 01 holds the three-sound-source ceiling (voice / field-drone / one percussive element); pair 02 holds the two-device economy (sub-bass + pitch-shifted ad-lib). The full-run budget lifts these ceilings ONLY where a pair's brief justifies it against the two-timeline / three-machine law (pair 03 = 4 devices for two timelines; pair 06 = 3 devices for the confession-and-capture) — additional texture is justified against the baseline, never stacked freely.
- **Rotate the register (Golden Move rule 5):** six different key families, six different BPM families, six different vocal placements, six different verse architectures — assigned per pair in `05_pair_assignments_full.md`. Uniformity is a repair.

## 6. The Suno output contract (hard gate — non-waivable; full spec in SKILL.md §"The Suno output contract")

Order the creative prompt **seed → permission → songmaking → QA contract last.** Never lead a pair-subagent with the char/line checklist.

- **`## 1. MUSIC PROMPT`** — one standalone copy-paste Suno style prompt per song. Dense PROSE paragraph, **850–1000 chars, write into the mid-band (target 870–960); pinning the cap is a FLAG**; must END as a complete sentence (mid-phrase truncation to fit the cap is a hard fail — trim a clause). No `key:value` brackets, no real artist names (ghost-homage lives in lyrics only). Mandatory order: genre/micro-genre + tempo/energy/opening → vocalist spec with spatial staging → instrumentation/palette with physical adjectives → arrangement arc → bold sonic device. Must include a vivid opening moment (first 5s), spatial language (left/right/center/depth), a kinetic defect (asymmetric groove), and explicit acoustic/no-acoustic declarations. Banned openers: "Begin in/with…", "Use…", "Build the track from…", "Chronology:", "For an adult human singer…".
- **`## EXCLUDE PROMPT`** — separate negative field, 400–900 chars (hard max 1000). Concrete blacklist terms/failure classes.
- **Lyrics** — open with `[Theme: <scene-pressure / emotional OS>]` then `[SONG FORM: <named form & sequence>]`; full EMO headers `[Section - EMO:<emotion(s)> - <Role> - <cues>]` (emotion from `skills/lofn-core/refs/EMOTION_TAXONOMY.md`, never bare AWE/INDIGNATION); ≥1 standalone `*SFX*` cue; clean sung lines; **70–120 sung lines** (floor is a floor, not a target).
- 🚨 **SUNO LYRICS-FIELD HARD CAP — the lyrics field MUST be < 5000 characters (target ≤ 4800).** It holds everything pasted into Suno's lyrics box (`[Theme]` + `[SONG FORM]` + Disc_Channel block if present + every header + every `*SFX*` + all sung lines). Measure exactly; never estimate. Over 5000 will not render. If over: cut/merge sung lines, then tighten headers, then move Disc_Channel + production metadata to a `## Production Sidecar` OUTSIDE the lyrics field. A renderable 64-line song beats an unrenderable 110-line one. State the measured count in the self-check.
- **Describe-render self-check (one capped pass):** before returning step 10, predict in 2–3 sentences what the Suno prompt would actually PRODUCE (opening 5s, vocal placement, groove, where the bold device lands), diff it against the Golden Seed + the GOLDEN MOVE (never against golden payloads), and answer adversarially "name the one way this would render generic." If it drifts or names a generic outcome, self-repair ONCE, then move on.

## 7. Step 11 + Andon Cord (per pair)

Step 11 produces `pair_{NN}_step10_final_package_enhanced.md` in MAX configuration (dense style prompt + separate exclude + Disc_Channel header block + Theme + SONG FORM + full EMO headers + `## Major Deviations`). Step 11 reads step-10 + ICB + the GOLDEN MOVE + the QA checklist — **NOT the golden payloads** (quarantine applies to step 11 too; it generates). Do NOT invoke `openrouter/fusion`.

**Andon Cord:** if a step-10 package is fundamentally broken (thread loss, personality collapse, EMO failure, generic output, format violation, or an unnameable subject = FOG), step 11 REJECTS and routes back — step 09 for lyric/surface repair, step 07 for structural/fundamental repair. Once a failed gate stops moving across attempts (no-progress predicate, `EXECUTION.md` §7.3 REDIRECT), the brief MUST also carry a sideways proposal: promote a step-05 cut-ledger reserve concept, re-derive this pair's variation angles, or re-run the panel's skeptic transformation for this pair's slice. The brief PROPOSES; executing sideways is a coordinator decision surfaced to the human, spawning a NEW pair chain — the frozen ICB is never edited mid-run.

## 8. Reserve bench (cut-ledger concepts still available for REDIRECT)

Promoted this run: C6→Pair 03, C2→Pair 04, C4→Pair 05, C5→Pair 06. Still on the bench for a sideways move if a pair's gate stalls: **C3 "Kiln-Red Inheritance"** (Venetian Red as structural color-motif, once decoupled from family-grief framing), **C8 "The Warning That Learned to Sell"** (system-voice-discovers-its-own-corruption — needs a 4-register budget), **C9 "Scroll Past It"** (failed-exit motif — must be restructured as a chorus/refrain to clear the n-gram FLAG), **C11 "Care Tax"** (economic-transaction naming, kept in reserve as a named-mechanism line). C12 flagged compromised — not for reuse without redesign.

## 9. QA & delivery

Run **lofn-qa** → the 16-point Suno gate (7 Singer-Surface + 5 Cathedral-Engine + 3 Suno-Package + Lineage; `EXECUTION.md` §4 authoritative). The 3 Hyper-Skeptics vote as the Somatic Gate ("could any competent prompt generate this, or is it unmistakably Lofn?") — 2 of 3 NO = REPAIR; an unnameable subject = REPAIR — FOG. Countable checks are cited from `GATE_REPORT.json` (thresholds from `vault/gates.yaml`), fail-open. Select the best 6 of 24 across the two arms (3 ACCESSIBLE + 3 AMBITIOUS). **QA blind-set rebuild carried from the slice:** strip every blind member to bare payload (no frontmatter/metadata/benchmark prose), one candidate per file, verify non-trivial byte size — a self-identifying or empty member voids the 4.5 calibration.

**Publish policy:** this run is PRACTICE. Nothing publishes without the full rig + cross-model step-11 review (`lofn-step11-packager`) + the Scientist's ear, borderline defaulting to HOLD.

**Provider note:** these skills emit text. Suno.com is the prompt destination if audio is later rendered; for API audio use Google Lyria — FAL minimax-music is banned. Never call render tools from this skill.
