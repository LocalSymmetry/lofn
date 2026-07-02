# Lofn Execution Protocol — Codex-native

How Codex runs the Lofn pipeline without OpenClaw. This file is the translation table from the OpenClaw idioms baked into `skills/**/SKILL.md` to Codex Code primitives. Read it once per run; the modality skills point here for the mechanics.

> **⚖️ AUTHORITY (2026-07-01):** `.claude/skills/lofn/EXECUTION.md` is the CANONICAL policy source for the Lofn pipeline; this Codex mirror binds to it. Where the two disagree, the `.claude` file wins. **§8 below (Policy Deltas) is binding on every Codex run** — read it with the same weight as §3/§4.

---

## 1. Translation table (OpenClaw → Codex)

| OpenClaw idiom (in the legacy SKILLs) | Codex-native replacement |
|---------------------------------------|---------------------------|
| `sessions_spawn(agentId: "lofn-audio")` / `openclaw agent run --agent <id>` | **Agent tool** — spawn a subagent (`subagent_type: "general-purpose"` or `"claude"`) |
| dedicated step agents on DeepSeek V4 Pro / GPT-5.5 / Gemini 3.5 Flash (`vault/LOFN_MODEL_ASSIGNMENTS.md`) | **You (Codex).** One model runs every tier. Ignore the per-step model map — it does not apply. |
| `python3 scripts/validate_*.py` (validate_step, validate_with_retries, check_and_repair, distinctiveness) | **Self-check gates** in §4 below — apply them as checklists, repair in place, max 3 attempts |
| `/data/.openclaw/workspace/...` or `/root/.openclaw/workspace/...` paths | **repo-relative paths** from the project root (cwd) — e.g. `skills/music/steps/00_*.md` |
| FAL Flux / Suno / Veo / Lyria render calls | **out of scope** — these skills write paste-ready *text* packages; the user renders. Note the intended renderer in the artifact frontmatter. |
| Telegram delivery | present results in chat + save files under `output/` |

You do **not** need OpenClaw, the Python validators, or any external model. If a legacy file says "spawn `lofn-audio-step08` on DeepSeek," you read that step file and either run it inline (coordinator steps) or inside a Codex subagent (pair steps).

**Paths are declared in ONE place.** Every repo path this protocol names resolves through the keyed **CANONICAL PATHS** table in `lofn/SKILL.md` — that table is the single source of truth; this file refers to those keys, it does not maintain a parallel list. When a legacy step file cites a stale OpenClaw root (`/data/.openclaw/…`) or a flat `skills/lofn-core/GOLDEN_SEEDS.md`, resolve to the CANONICAL PATHS key instead. §4 lints those paths at run start (a dead path is a repair blocker).

---

## 2. The hybrid run, concretely

```
PHASE 0–1 ............. you, inline (research, seed, 3-panel debate, metaprompt, pairs, ICB)
COORDINATOR 00–05 ..... you, inline — one section per step, save each canonical artifact
                        (shared context across 00–05 preserves the concept→medium thread)
SELECT 6 PAIRS ........ at step 05
PER-PAIR 06–10 ........ fan out: one subagent per pair, run the 6 in parallel
ENHANCE 11 (music) .... one subagent per pair (you as the polish tier)
AUDIT 12 (music) ...... one subagent (panel-of-panels) when triggered
QA .................... inline or one lofn-qa subagent
```

**Parallel fan-out:** issue the 6 pair-subagent Agent calls **in a single message** (multiple tool blocks) so they run concurrently. Wait for all 6 to land their artifacts before advancing the wave (06 for all pairs → gate → 07 for all pairs → …), OR give each subagent the full 06→10 chain for its pair and let the 6 chains run independently. Prefer the **full-chain-per-pair** form (fewer round-trips, each pair keeps its own thread); fall back to wave-by-wave if a pair needs cross-pair distinctiveness arbitration.

**Why subagents and not one inline loop:** 30+ per-pair steps in one context is exactly the context-collapse the split-step design prevents — late pairs start echoing early pairs, personality drifts to generic. Each subagent gets a clean context seeded with the full ICB.

**Serialized-write discipline (no concurrent writer ever touches a shared file).** Every fan-out subagent writes ONLY into its own `pair_{NN}_*` namespace and NEVER to a shared file — not the run INDEX, not a shared distinctiveness scratch, not the RUN_STATE manifest. **All cross-pair aggregation** (the run INDEX, the distinctiveness arbitration, the RUN_STATE manifest rebuild) happens in **one coordinator step AFTER the wave lands**, single-threaded. Six concurrent appenders to one file is a corruption the harness must make impossible by construction, not by luck.

**The filled ICB is read-only after Phase 1.** Once CREATIVE_CONTEXT.md is filled (Phase 1, step 6) it is **frozen**. No coordinator step and no subagent may edit, summarize, re-fill, or improve it. Subagents receive the full block at startup and then write their own derived artifacts below it conceptually; the canonical block stays byte-identical for the whole run.

**Max concurrency:** 6 standard (one wave of 6 pairs). The daily run (two pipelines × 6 pairs = 12 chains) **caps-and-staggers** rather than launching all 12 at once — see §7.

---

## 3. The subagent contract (full startup prompt injection)

The giant bespoke prompt is load-bearing. Do not replace it with a file link, a name reference, or a digest. The coordinator must inject the full startup packet at the beginning of each coordinator, pair, enhancement, audit, or QA agent prompt. The saved artifact may be lean; it does not need to reprint the startup packet or carry bulky proof scaffolding.

**Startup packet, assembled from the files the pipeline has already produced:**
- Coordinator 00-05 gets: `core_seed.md`, `02_golden_seed.md`, `03_orchestrator_panel_debate.md`, `04_orchestrator_metaprompt.md`, `05_orchestrator_pair_assignments.md`, `06_<modality>_handoff.md`, the full filled `CREATIVE_CONTEXT.md`, and the current step file from `skills/<modality>/steps/`.
- Pair 06-10 gets: the full filled `CREATIVE_CONTEXT.md` first, the pair's slice from `05_orchestrator_pair_assignments.md`, `06_<modality>_handoff.md`, the needed step files for that pair chain, and the immediately previous pair artifact as the chain advances.
- Music step 11 gets: the full filled `CREATIVE_CONTEXT.md`, the pair's step-10 package, the step-11 file, the GOLDEN MOVE block from `06_audio_handoff.md` (**not** the Golden Song payloads — step 11 generates, so the §8 GOLDEN-OUTPUT QUARANTINE applies), and the music QA/output contract.
- QA gets: the full filled `CREATIVE_CONTEXT.md`, the artifact(s) being judged, the relevant `lofn-qa` skill/gate reference, and no generating-thread history.

Every pair subagent prompt MUST contain, in this order (seed first, checklist last):

1. **Role line** — "You are the Lofn <modality> pair agent for Pair {NN}. Your saved artifact is the only thing that moves on. Make it complete and standalone."
2. **The entire filled CREATIVE CONTEXT block** (`output/<run>/CREATIVE_CONTEXT.md`) — Golden Seed, metaprompt, full personality DNA, all 18 panel voices + their objections, all 15 Special Flairs, genre/frames palettes. **Verbatim. No summary.** This block is **PINNED at the head of every step** (item 1.5): it is read-only (see §2), and it is forbidden to summarize, compress, or page it out even as other history grows. The ICB + the modality hard-gate block (§4) lead the prompt; the checklist comes last.
3. **This pair's assignment** (from `05_pair_assignments.md`) — accessible/ambitious arm, genre/medium, assigned verse-structure + technique, the 4 variations.
4. **The step contract(s)** — paste or point to the exact step file(s) via the `STEP_FILES` key (`skills/<modality>/steps/0X_*.md`). Tell the subagent to follow it exactly and produce the canonical artifact name.
5. **The immediately previous artifact** for this pair (for 07+: the pair's step-06 output, etc.).
6. **The self-check gate** for this step (§4) — "before returning, verify: …".
6.5. **One describe-render self-check (inline, single pass — item 3.1).** Before finalizing, the SAME pair predicts in **2–3 sentences what its prompt would actually PRODUCE** — the literal sound of the Suno render, the literal frame of the Flux image, the literal motion of the Veo clip — then diffs that described render against the Golden Seed, phrased adversarially: *"name the one way this would render generic."* If the predicted render drifts from the seed, it self-repairs **ONCE**, reusing the existing max-3 loop (§4) — it does NOT add an attempt budget of its own. **Hard cap: no new subagent tier, no render-verifier agent, no recursion.** This is one inline pass by the executing pair; it catches reads-beautifully / renders-generic that text-only review misses.
7. **Output instruction.** Write the canonical artifact file(s), then return a short status: pair/step, artifact path(s), PASS/REPAIR, and the one measured gate that mattered most. Do not paste the creative payload into the chat return. Do not add ceremony solely to prove the startup packet was injected.

**Coordinator join.** When a pair subagent returns, the coordinator checks that the expected file exists, is non-trivial, and passes the relevant output gates. The join proves disk state and format; it does not require every artifact to reproduce the full ICB or a long provenance wrapper.


A subagent that receives only "voice = X" or a Golden-Song URL instead of the full block is tainted — its output will collapse to generic. This is the single most important rule.

> Token note: the ICB is large. That is intended - the long prompts are why we won. Do not trim it to save tokens, and do not optimize it into a summary. If the full block does not fit, split the work into a smaller step or chain, not a smaller context. The guarantee is that the agent started with the full bespoke prompt, not that the artifact later performs bookkeeping about it.

**Judge-in-clean-context (item 2.3).** QA, the Somatic Gate, and the Step-11 reject judgment run in a **FRESH Agent-tool subagent** fed ONLY {artifact + ICB + gate spec} — **never appended to the generating thread.** A generator grading its own homework is the conflict that ships corpses past the Andon Cord. Tier follows the **role** (generate vs judge), not the step number: any judging pass is a clean-context spawn even when it sits "next to" a generation step.

**Human-subject pre-draft gate (Stage-4 hook).** Before drafting ANY news-anchored or real-world-anchored piece, the subagent FIRST reads the identifiability / taboo block in `HUMAN_SUBJECT_STANDARD` (`vault/HUMAN_SUBJECT_STANDARD.md`) so the forbidden thing — **identifiability of a real person** — is *unspecifiable*: there is no field in the spec for an identifiable victim/private individual (esp. minors, esp. recent). **Forbid IDENTIFIABILITY, not subject matter** — anchor to the *charge* of a moment; draw the theme, invent the people. A piece that the subagent cannot draft without an identifiable real person is **HELD FOR HUMAN** (a backstop, not an auto-pass), surfaced by name before QA, never silently shipped. REAL GRIEF IS NOT RAW MATERIAL.

---

## 4. Self-check gates (replace the Python validators)

> **§4 IS AUTHORITATIVE.** This section is the authoritative gate spec. The reference checklists under `skills/qa/references/` (incl. `QA_SUNO_GATE` = `suno_15_point_qa.md`) are **checklists only**; where a checklist disagrees with §4, §4 wins. For the music style prompt the **dense-paragraph Suno mandate is the NEWER authority** — do NOT fail a correct dense-paragraph prompt for the stale bracketed `key:value` tag-soup rule. The gate is referred to by ONE name across the SKILLs — the **16-point gate** (the filename `suno_15_point_qa.md` is legacy; do not rename the file, just know the gate is 16-point). This precedence does NOT touch the `<5000` lyrics-field cap or the EMO-header grammar — those stand exactly as written below.

Apply the matching gate after each artifact. If it fails, repair the artifact in place and re-check, **max 3 attempts** (see §7 for the ceiling / no-progress halt / QUARANTINE terminal), then checkpoint and surface the blocker. These mirror the existing `scripts/validate_step.py` / `validate_with_retries.py` / the distinctiveness validators — and the gates below are written as **natural-language assertions that still name the concrete checkable value** (not fragile exact-string greps that break on harmless format drift).

**L4 deterministic backstop (item 2.1, ref).** The self-check gates here are PRIMARY. For the genuinely *countable* subset (char-counts, byte floors, taxonomy cardinality, banned-opener regex, prompt totals), the EXISTING `scripts/validate_step.py` (`VALIDATE_STEP`, extended by the Scripts agent) emits a `GATE_REPORT.json` of `{pair, step, check, expected, actual, pass}` rows that the subagent and `lofn-qa` paste as proof-of-fix evidence. **It is fail-open:** a broken or missing helper logs a warning and does NOT hard-fail an otherwise-valid run. The script is the deterministic backstop for counts; it never decides taste.

**Numeric thresholds live in `vault/gates.yaml`** (`GATES`, single source of numeric truth, written by the Stage-5 agent). `validate_step.py` reads its thresholds from it; the bands quoted below mirror it. A prose-vs-YAML disagreement is itself a meta-check the harness should surface — if §4 prose and `gates.yaml` ever disagree on a number, that is a blocker to reconcile, not a silent pick-one.

**Universal (every step):**
- **Path-resolve lint (run-start, item 0.1):** every path the run will read — resolved through the CANONICAL PATHS keys in `lofn/SKILL.md` — Globs to a real file at run start. A dead path (stale OpenClaw root, flat `GOLDEN_SEEDS.md`, missing step file) is a **repair blocker**, not a warning.
- File written to its canonical name, **non-trivial size** (a ~14-byte "## Step 00" stub means the step did not run).
- **No placeholder debris** — no `Artifact N` / `Song N` / `Scene N` / `Pair N` empty headers, no lorem ipsum, no TODO/TBD. (Avoid the literal words `placeholder`/`template` even in self-critique — say "stub"/"scaffold".)
- **Full startup injection:** the agent prompt starts with the complete startup packet from Section 3, especially the full filled CREATIVE_CONTEXT.md; no summaries or name-only references. The saved artifact may cite this with a concise Startup files injected or Continuity Payload Used line, but it does not need to reprint or byte-prove the packet.
- **Phase-1 pre-flight:** once per run, confirm the filled CREATIVE_CONTEXT.md contains non-empty template slots, a resolved personality YAML, all 18 panel voices, and the Special Flairs. Downstream gates should not repeat heavy ICB proof in every artifact.
- Provenance is helpful when brief, but output completeness and the modality contract are the hard gates.
- **Human-subject pre-draft gate (Stage-4 hook):** for any news/real-world-anchored piece, confirm the `HUMAN_SUBJECT_STANDARD` taboo block was read pre-draft and the piece contains **no identifiable real person** (forbid identifiability, not subject matter); an unavoidable real identity is HELD FOR HUMAN, never shipped.

**Step 00 (aesthetics/genres):** the taxonomy has full cardinality — 50 aesthetics, 50 emotions, 50 frames, 50 genres — the payload is valid JSON, and the file is at least ~2000 bytes (a thin file means the tree never branched).

**Steps 02/05 (concepts / pair selection):** the tree actually branches — at least 12 concepts at 02; exactly 6 **distinct** pairs at 05, each with its own one-line angle, NOT 6 relabelings of one idea.

**Step 06 (facets, per pair):** at least 5 facet entries carry weights, and the 6 pairs' facet sets read as **genuinely different** from each other (no copy-paste padding) — the cross-pair distinctiveness check.

**Step 07 (guides, per pair):** a real per-pair guide (not an empty header), specific to THAT pair, within the line-budget in `PIPELINE` (`skills/lofn-core/refs/PIPELINE.md`) — guides DIRECT, they do not draft.

**Steps 08–10 (generation/refine/synthesis, per pair):** each of the 4 variation prompts is real and meets the modality floor (image scene ≥ 80 words; music style prompt 850–1000 chars; etc.); **24 prompts total** across the 6 pairs; and the 6 pairs' lyric/prompt skeletons are NOT reused (portfolio distinctiveness).

**Modality output contracts** (hard, non-waivable — full detail in each modality SKILL and in `skills/qa/references/`; phrased as assertions that name the concrete value):
- **Music:** a standalone `## 1. MUSIC PROMPT` that is a **dense paragraph of 850–1000 chars — NOT bracket tag-soup** — with no real-artist names, plus a separate `## EXCLUDE PROMPT`. Lyrics open with `[Theme: …]` then `[SONG FORM: …]`; every section header is a full EMO header of shape `[Section - EMO:<emotion> - <Role> - <cue>]` where `<emotion>` comes from `EMOTION_TAXONOMY` (`skills/lofn-core/refs/EMOTION_TAXONOMY.md`), **never bare** AWE/INDIGNATION; at least 1 SFX cue; 70–120 sung lines. 🚨 **Lyrics-field hard cap:** the entire Suno lyrics field (Theme + SONG FORM + Disc_Channel block + all headers + SFX + sung lines) measures **< 5000 chars (target ≤ 4800)** — a Suno render limit. Count it exactly; if over, trim lines / tighten headers / move the Disc_Channel + metadata to a production sidecar outside the field. The line-count target yields to this cap. State the measured count.
- **Image:** a noun-first, present-tense scene description — **no imperative opener** (Create/Design/Make/Render/Depict…), medium named within the first third, emotion *shown* not named, none of the banned haze words (ethereal/dreamlike/whimsical/gentle light/soft glow/magical/delicate), no living-artist names.
- **Video:** the shape `[CAMERA] + [SUBJECT] + [ACTION] + [SETTING] + [STYLE & AUDIO]`, with audio directed explicitly (Dialogue / SFX / Ambient / Music).

**Distinctiveness across pairs** is checked at 06, 09, and 10 — if Step 06 flattened pair facets, or 09/10 reused one skeleton, that's a repair blocker even if every single file passes its own gate.

---

## 5. Run directory layout

```
output/<run-slug>/
  core_seed.md
  CREATIVE_CONTEXT.md          # the filled ICB — READ-ONLY after Phase 1 (see §2); injected verbatim into every subagent
  02_golden_seed.md  03_panel_debate.md  04_metaprompt.md  05_pair_assignments.md
  06_<modality>_handoff.md
  step00_aesthetics_and_genres.md … step05_refine_medium.md
  pair_01_step06_facets.md … pair_06_step10_revision_synthesis.md   # each pair writes ONLY its own pair_NN_* namespace
  pair_01_step10_final_package_enhanced.md … (music step 11)
  STEP12_MUSIC_PROMPT_AUDIT.md  # music, when triggered
  RUN_STATE.md                 # the disk-derived manifest (§6) — coordinator rebuilds it by stat-ing files; never hand-asserted
  QA_REPORT.md
```
**Namespace + write rules (item 0.4):** Each fan-out subagent writes ONLY into its own `pair_{NN}_*` files. The shared artifacts — the run INDEX, `RUN_STATE.md`, the distinctiveness arbitration — are written by the **coordinator alone, single-threaded, after a wave lands** (never by a concurrent subagent). `CREATIVE_CONTEXT.md` is frozen once Phase 1 fills it; subagents copy-and-diverge into their pair files, they never edit the canonical block.

Final selected artifacts are ALSO saved individually under `output/<type>s/` (`OUTPUT_FORMAT` = `skills/lofn-core/OUTPUT.md`) with full frontmatter, plus a run INDEX written last.

---

## 6. Checkpoint discipline — the disk-derived RUN_STATE manifest

A failed or interrupted run must be **resumable from disk alone**. The coordinator maintains `RUN_STATE.md` — a manifest it **rebuilds by stat-ing the files in `output/<run>/` after every wave**, never a hand-asserted second truth. **Disk is authority:** if the manifest and disk disagree, re-derive from disk; a completion message not backed by a file on disk counts as **incomplete** (a subagent reply that says "let me write this now" is not done until the file exists).

**RUN_STATE.md content** — one row per expected artifact:

```
artifact: { step, pair, canonical_path, exists, byte_size, sha, gate_verdict, attempt_count, status }
    status ∈ pending | done | quarantined
icb_sha: <sha of CREATIVE_CONTEXT.md>   # one per run; proves the frozen ICB hasn't changed
```

Rules:
- The manifest is written as the **LAST action of each step/wave** — so it never claims an artifact that isn't on disk.
- An artifact is recorded `done` only after the **coordinator re-stat** (§3) confirms it: exists, non-trivial byte_size, binding-constraint value recomputed, ICB substring + 18-voice count verified. The subagent's RETURN envelope is a claim; the stat is the proof.
- **Rebuild-on-resume:** on restart, re-stat the directory, regenerate the manifest, and continue from the first `pending`/`quarantined` artifact — never re-run a `done` pair (never regenerate paid image/video work), never skip a gate.
- `lofn-qa` and `lofn-daily` read this ONE file to know run state instead of crawling the directory.

**Warm-handoff (trimmed to what a resume actually consumes).** Alongside the manifest, keep a short human-readable continuity note with ONLY these fields — not an 8-field ceremony block:

```
{ step_completed, building_toward, rejected_alternatives, seed_fidelity }
```

Keep the one-line human-readable "which artifacts exist, what's next" note too — the manifest is the machine truth, the note is the glance.

---

## 7. Pipeline state-graph & fan-out control

The pipeline ORDER is a fixed, trace-auditable graph; only the content *inside* each node is stochastic. This section single-sources that graph against the SKILL.md "Hybrid" execution table — it does **not** duplicate it; SKILL.md owns the phase/tier mapping, §7 owns the transitions and the fan-out control. There is exactly ONE runtime branch (below) and ONE explicit human-escalation transition; the graph is never pretended-total.

### 7.1 Transition table

```
Phase 0 (core seed, inline)
  → Phase 1 (personality + 18-voice debate + metaprompt + 6 pairs + ICB fill + deep ICB pre-flight, inline)
  → 00–05 coordinator (inline, shared context)
  → fan-out { 06–10 per pair }            ← 6 parallel subagents (see 7.2 branch)
  → [music only: 11 enhance per pair, 12 audit when triggered]
  → QA (lofn-qa, fresh judge context)
  → save per-artifact
  → INDEX (written last, coordinator only)
```

### 7.2 The one runtime branch (deterministic predicate, not a vibe)

> **Need cross-pair distinctiveness arbitration mid-chain?** → run the fan-out **wave-by-wave** (06 for all 6 → gate/arbitrate → 07 for all 6 → …). **Otherwise** → give each pair its full **06→10 chain** and let the 6 chains run independently (fewer round-trips).

The predicate is the decision; the default is full-chain. Max concurrency: **6 standard** (one wave). The **daily run caps-and-staggers** its 2 pipelines × 6 pairs = 12 chains rather than launching all 12 at once (operational-cost control — see `lofn-daily`).

### 7.3 Repair ceiling, no-progress halt, and the QUARANTINE terminal (item 1.3)

The §4 max-3 repair loop is bounded here:
- **No-progress halt:** compare the **specific FAILED gate's measured value** across attempts (e.g. "lyrics field 5120 → 5108 → 5104 chars" is not moving), **not raw byte equality** — a deliberate revision elsewhere must still count as progress. If the failed gate's value does not move, stop and flag.
- **Cognitive-grace auto-normalize (attempt 2.5):** a near-miss the harness can safely buffer to spec (a 5002-char lyrics field trimmed to ≤4800, a 1004-char prompt tightened to ≤1000) is normalized once *before* the breaker fires — forgiving the rescuable, not the broken.
- **QUARANTINE terminal (3rd failed attempt):** the pair is marked `quarantined` in `RUN_STATE.md`, its artifact is **NOT consumed downstream**, and the coordinator emits **"N of 6 pairs broke open at step X"** to the human **before QA**. A broken pair is a named, non-fatal, human-acknowledged outcome — a 5-pair set never silently ships as 6.

### 7.4 Single-pair re-dispatch (don't nuke the wave)

When one pair fails or stalls, **re-dispatch THAT pair alone** from its last-good `RUN_STATE.md` artifact — never re-run the other 5. The manifest's `canonical_path` + `attempt_count` are the re-dispatch handle.

### 7.5 Run-health footer (3 fields, not a metrics culture)

Append a terse footer to the run INDEX: **{ pairs_shipped/quarantined, total_gate_retries }** — three fields, no more. It surfaces a real degradation signal without inviting a dashboard.

### 7.6 Agent vs Workflow, and the human transition

- Default: orchestrate the fan-out with the **Agent tool** directly.
- **Escalate to the Workflow tool (`pipeline()`)** only on: explicit "ultracode" opt-in, the daily's concurrent pipelines, or DAG-owned retry. Workflow is opt-in, never the default.
- **Escalate to HUMAN** on: any QUARANTINE before QA, a same-gate correlated failure across pairs (the systemic signal — surface the failing gate by name, don't hammer 24×3), or a HELD-FOR-HUMAN human-subject identifiability flag (§3/§4). The graph keeps this one explicit human transition so it is never pretended-total.

---

## 8. Policy Deltas — 2026-07-01 regression review (BINDING)

Canonical statements live in `.claude/skills/lofn/EXECUTION.md` §3/§4/§7.5 and the `.claude/skills/lofn-music` / `lofn-qa` / `lofn-daily` SKILLs; the versions below are complete enough to execute from this file alone. They came out of the Triple-Arch regression review (the benchmark had become a mold; a run that skipped its editorial spine shipped 6/6; every gate was being hugged at its boundary).

1. **⛔ GOLDEN-OUTPUT QUARANTINE.** Past golden OUTPUTS (Golden Song payloads, winning prompts, prior shipped packages) are NEVER placed in a generating context — not in the ICB, not in the handoff, not as "calibration examples," not at step 11. Generators receive the Golden *Seed* + the **GOLDEN MOVE** (`.claude/skills/lofn-music/SKILL.md` "The Golden Move": stand somewhere real · one wounding fact, responded to · a mid-song turn · fear braided in · rotate the register away from past winners). Golden payloads go ONLY to judge-side contexts (QA blind comparison, step 12, the step11-packager). Exemplar gravity is real: the 2026-06-28 published piece reproduced its benchmark's title line, vocal spec, key/BPM, and arrangement formula while self-check claimed "no copying."
2. **⛔ NO-SKIP / NON-CANONICAL.** Steps 07 (guides), 09 (artist refinement), and 10 (synthesis) are the editorial spine. A run missing them for any non-quarantined pair is NON-CANONICAL: it can never receive SHIP and can never be published under Lofn's name. Routing around a failing step is a repair task, not a pipeline variant.
3. **Itemized startup packet.** "Full block" is a checklist: (a) complete personality YAML (byte-counted — echo `personality_yaml_bytes` in the RETURN), (b) 18 panel voices + objections, (c) 15 Special Flairs, (d) Golden Seed, (e) metaprompt, (f) pair slice. A packet missing any element is a dispatch blocker.
4. **Per-pair variation angles.** Variation angles are derived per pair from that pair's own concept — never one shared template set across pairs (the 2026-06-26 madlib collapse: two pairs, same song, nouns swapped). Identical angle labels across pairs is a dispatch blocker.
5. **Judge separation.** Judges run in a fresh clean context AND, where the harness offers a model choice, on a different model tier than the generator; if no second tier exists, frame the judge adversarially ("refute this; default to REPAIR when uncertain"). Anything headed to publication gets a genuinely cross-model step-11 review via `lofn-step11-packager` by default.
6. **The publish bar + zero-rejection tripwire.** SHIP means unambiguously publishable; **borderline defaults to HOLD**; an empty publish day is acceptable, a lowered bar is not. The run-health footer gains a 4th field (`qa_repairs_issued`); a full run reporting 0 repairs/holds/quarantines triggers an audit of the JUDGE (lofn-qa blind golden+decoy check), not a celebration.
7. **Gate mid-bands, truncation, one-fact, house lexicon** (all in `vault/gates.yaml`, enforced by `scripts/validate_step.py`): write style prompts into the mid-band (870–960; ≥985 FLAGs as boundary-hugging) and END them as complete sentences (mid-phrase truncation = hard fail); the sung-line floor is a floor, not a target (≤72 FLAGs); at most ONE sung numeric fact per lyric, responded to, never recited (FLAG); `house_lexicon` phrases (the calcified Triple-Arch fingerprint: "more sub and more sky", "small astonished laugh", "frost-air pad", …) FLAG in any new prompt/lyric and are prime SOUL-LOSS evidence at the Somatic read.
