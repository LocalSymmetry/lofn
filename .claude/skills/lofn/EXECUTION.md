# Lofn Execution Protocol — Claude-native

How Claude runs the Lofn pipeline without OpenClaw. This file is the translation table from the OpenClaw idioms baked into `skills/**/SKILL.md` to Claude Code primitives. Read it once per run; the modality skills point here for the mechanics.

---

## 1. Translation table (OpenClaw → Claude)

| OpenClaw idiom (in the legacy SKILLs) | Claude-native replacement |
|---------------------------------------|---------------------------|
| `sessions_spawn(agentId: "lofn-audio")` / `openclaw agent run --agent <id>` | **Agent tool** — spawn a subagent (`subagent_type: "general-purpose"` or `"claude"`) |
| dedicated step agents on DeepSeek V4 Pro / GPT-5.5 / Gemini 3.5 Flash (`vault/LOFN_MODEL_ASSIGNMENTS.md`) | **You (Claude).** One model runs every tier. Ignore the per-step model map — it does not apply. |
| `python3 scripts/validate_*.py` (validate_step, validate_with_retries, check_and_repair, distinctiveness) | **Self-check gates** in §4 below — apply them as checklists, repair in place, max 3 attempts |
| `/data/.openclaw/workspace/...` or `/root/.openclaw/workspace/...` paths | **repo-relative paths** from the project root (cwd) — e.g. `skills/music/steps/00_*.md` |
| FAL Flux / Suno / Veo / Lyria render calls | **out of scope** — these skills write paste-ready *text* packages; the user renders. Note the intended renderer in the artifact frontmatter. |
| Telegram delivery | present results in chat + save files under `output/` |

You do **not** need OpenClaw, the Python validators, or any external model. If a legacy file says "spawn `lofn-audio-step08` on DeepSeek," you read that step file and either run it inline (coordinator steps) or inside a Claude subagent (pair steps).

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

---

## 3. The subagent contract (full-context injection)

Every pair subagent prompt MUST contain, in this order (seed first, checklist last):

1. **Role line** — "You are the Lofn <modality> pair agent for Pair {NN}. Your saved artifact is the only thing that moves on. Make it complete and standalone."
2. **The entire filled CREATIVE CONTEXT block** (`output/<run>/CREATIVE_CONTEXT.md`) — Golden Seed, metaprompt, full personality DNA, all 18 panel voices + their objections, all 15 Special Flairs, genre/frames palettes. **Verbatim. No summary.**
3. **This pair's assignment** (from `05_pair_assignments.md`) — accessible/ambitious arm, genre/medium, assigned verse-structure + technique, the 4 variations.
4. **The step contract(s)** — paste or point to the exact step file(s): `skills/<modality>/steps/0X_*.md`. Tell the subagent to follow it exactly and produce the canonical artifact name.
5. **The immediately previous artifact** for this pair (for 07+: the pair's step-06 output, etc.).
6. **The self-check gate** for this step (§4) — "before returning, verify: …".
7. **Output instruction** — "Write `output/<run>/pair_{NN}_step0X_*.md` and return its path + a 2-line self-critique. Do not summarize the creative content in your reply; it lives in the file."

A subagent that receives only "voice = X" or a Golden-Song URL instead of the full block is tainted — its output will collapse to generic. This is the single most important rule.

> Token note: the ICB is large. That is intended — "the long prompts are why we won." Do not trim it to save tokens.

---

## 4. Self-check gates (replace the Python validators)

Apply the matching gate after each artifact. If it fails, repair the artifact in place and re-check, **max 3 attempts**, then checkpoint and surface the blocker. These mirror `scripts/validate_step.py` / `validate_with_retries.py` / the distinctiveness validators.

**Universal (every step):**
- File written to the canonical name, non-trivial size (a 14-byte "## Step 00" stub = step not run).
- No placeholder debris: `Artifact N`, `Song N`, `Scene N`, `Pair N` empty headers, lorem ipsum, TODO/TBD. (Avoid the literal words `placeholder`/`template` even in self-critique — say "stub"/"scaffold".)
- The CREATIVE CONTEXT / ICB is present and cited (a `Continuity Payload Used` note with the plural marker `Special Flairs`).
- Provenance: which step file, which inputs, a one-line self-critique.

**Step 00 (aesthetics/genres):** 50 aesthetics + 50 emotions + 50 frames + 50 genres, valid JSON, ≥2000 bytes.

**Steps 02/05 (concepts / pair selection):** the tree actually branches — 12+ concepts at 02; exactly 6 distinct pairs at 05, each with a one-line angle, not 6 relabelings of one idea.

**Step 06 (facets, per pair):** ≥5 facet entries with weights; the 6 pairs' facet sets are **different** from each other (no copy-paste padding) — the cross-pair distinctiveness check.

**Step 07 (guides, per pair):** a real per-pair guide (not an empty header), specific to THAT pair; within the line-budget in `skills/lofn-core/refs/PIPELINE.md` (guides direct, they don't draft).

**Steps 08–10 (generation/refine/synthesis, per pair):** each of the 4 variation prompts is real and ≥ the modality's floor (image ≥80 words; music style prompt 850–1000 chars; etc.); 24 prompts total across the 6 pairs; the 6 pairs' lyric/prompt skeletons are not reused (portfolio distinctiveness).

**Modality output contracts** (hard, non-waivable — full detail in each modality SKILL and in `skills/qa/references/`):
- **Music:** standalone `## 1. MUSIC PROMPT` (dense paragraph, 850–1000 chars, no bracket tag-soup, no real artist names) + separate `## EXCLUDE PROMPT`; lyrics open with `[Theme: …]` then `[SONG FORM: …]`; full EMO headers `[Section - EMO:<emotion> - <Role> - <cue>]` (emotion from `skills/lofn-core/refs/EMOTION_TAXONOMY.md`, never bare AWE/INDIGNATION); ≥1 SFX cue; 70–120 sung lines. 🚨 **Lyrics-field hard cap: the entire Suno lyrics field (Theme + SONG FORM + Disc_Channel block + all headers + SFX + sung lines) MUST measure < 5000 chars (target ≤ 4800) — a Suno render limit. Count it exactly; if over, trim lines / tighten headers / move Disc_Channel + metadata to a production sidecar outside the field. The line-count target yields to this cap. State the measured count.**
- **Image:** noun-first, present-tense scene description, no imperative openers (Create/Design/Make/Render/Depict…), medium named in the first third, emotion *shown* not named, no "ethereal/dreamlike/whimsical/gentle light/soft glow/magical/delicate", no living-artist names.
- **Video:** `[CAMERA] + [SUBJECT] + [ACTION] + [SETTING] + [STYLE & AUDIO]`; audio directed explicitly (Dialogue/SFX/Ambient/Music).

**Distinctiveness across pairs** is checked at 06, 09, and 10 — if Step 06 flattened pair facets, or 09/10 reused one skeleton, that's a repair blocker even if every single file passes its own gate.

---

## 5. Run directory layout

```
output/<run-slug>/
  core_seed.md
  CREATIVE_CONTEXT.md          # the filled ICB — injected into every subagent
  02_golden_seed.md  03_panel_debate.md  04_metaprompt.md  05_pair_assignments.md
  06_<modality>_handoff.md
  step00_aesthetics_and_genres.md … step05_refine_medium.md
  pair_01_step06_facets.md … pair_06_step10_revision_synthesis.md
  pair_01_step10_final_package_enhanced.md … (music step 11)
  STEP12_MUSIC_PROMPT_AUDIT.md  # music, when triggered
  QA_REPORT.md
```
Final selected artifacts are ALSO saved individually under `output/<type>s/` with full frontmatter per `skills/lofn-core/OUTPUT.md`, plus a run INDEX written last.

---

## 6. Checkpoint discipline

After each coordinator step and each pair wave, write/refresh a one-line state note (which artifacts exist, what's next) so a failed or interrupted run is resumable — the Claude equivalent of `references/warm_handoff_checkpoint.md` and the resume-point logic in the old `pipeline_runner.py`. A subagent reply that says "let me write this now" counts as **incomplete** until the file is on disk.
