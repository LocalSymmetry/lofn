# Streamlit-fossil study — chain order & slot-filling fidelity vs `llm_integration.py`

**WS0 deliverable (plan §11 done-criteria, §0 Heritage, §5/§6).** The Lofn Creative Studio
engine restores what `origin/streamlit` did three years ago: `{slot}`-template chains run
against provider APIs. This note records the FIDELITY findings from studying the vendored
fossil at `docs/reference/lofn-legacy-llm/llm_integration.py` so WS2 (resolver) and WS3
(engine retry/repair) port the proven behavior instead of re-deriving it. **Reference only —
never a dependency.**

## 1. How the fossil assembles a step prompt (→ resolver §6 fidelity)

Each modality's step prompt is a **string concatenation of three parts**, read from disk and
glued at import:

```
<modality>_<step>_prompt = concept_header + <step>_prompt_middle + prompt_ending
```

- `llm_integration.py:751` `refine_medium_prompt = concept_header + refine_medium_prompt_middle + prompt_ending`
- `:766,780,794,808` — the per-modality chain dicts (`video`/`music`/`story`/base) map a step
  name → its assembled prompt; e.g. music `:780` `'refine_medium': music_concept_header + read_prompt('music_refine_medium_prompt.txt') + prompt_ending`.
- The **header** carries the shared CREATIVE-CONTEXT framing; the **middle** is the step-specific
  instruction; the **ending** is the shared output-format/JSON-schema demand.

**Fidelity mapping to the Studio resolver:** canon has since fused header+middle+ending into
one self-contained `skills/<m>/steps/NN_*.md` file (the file the FlowSpec `prompt:` points at).
The resolver's job is therefore *simpler* than the fossil's: it reads the whole step file
(header already inlined), fills CREATIVE-CONTEXT slots, and appends the RUN PARAMETERS block.
**Do not** try to reconstruct the header/middle/ending split — canon already merged it. The
one behavior to preserve: the **shared ending** (output contract) must survive intact after
overlay hunks, because the fossil's ending is what made outputs parseable.

## 2. Slot filling (→ verbatim-injection rule)

The fossil uses LangChain `ChatPromptTemplate.from_messages([...])` (`:2199+`) with an
`args_dict` of `{slot}` values. Slots are filled by **value substitution**, and critically the
downstream steps receive the **prior artifacts by value**, threaded through `args_dict`:

- `generate_concept_mediums(:2160)` and the video/story twins thread `concept`, `medium`,
  `facets`, `image_gen_prompts` forward into each later step's args (`:2065,2108,2151,2156`
  `premessage=f'... for {concept} in {medium}:'`).
- This is the ancestor of the Studio's **verbatim full-context injection** (plan §6 step 3):
  the fossil already injected the *actual content* of the prior artifact, not a name reference.
  The anti-personality-collapse doctrine ("A name reference is INSUFFICIENT",
  `.claude/skills/lofn/SKILL.md:29`) is a hardening of what the fossil did by construction.

**Fidelity requirement for the resolver:** fill slots with the full artifact body (the ICB),
never a pointer. The FlowSpec `context.slots` list is exactly the fossil's `args_dict` keys,
and each generated `@1` flow copies `creative_context_slots` from `pipeline_map.yaml` verbatim
(11 slots: input, seed, meta_prompt, personality, concept_panel, medium_panel, marketing_panel,
flairs, genres_list, frames_list, image_context).

## 3. Chain ORDER (→ importer stage/step ordering)

The fossil's chain dicts are keyed by step name in the same 00→10 order the FlowSpec importer
emits, and the fan-out over concept×medium pairs happens in `generate_concept_mediums`
(`:2160`) / `generate_video_concept_mediums` (`:2404`) — one call per (concept, medium) pair,
which is exactly the FlowSpec `foreach: pair` stage. The coordinator steps (aesthetics →
essence → concepts → artist → medium → refine_medium) run once, single-threaded, before the
fan-out — matching the importer's `coordinator` stage (steps 00–05) vs `pairs` stage (06+).

**Verified:** the importer's coordinator/pairs split at step 06 reproduces the fossil's
"coordinator once, then per-pair" topology. No drift.

## 4. The retry-with-correction loop (→ engine `on_fail: retry`, WS3)

`run_chain_with_retries` (`:1604`) is the ancestor of the Studio's `on_fail: {policy: retry}`:

- On a parse/validation failure it sets `is_correction = True` (`:1646,1650`) and re-runs the
  SAME chain with a correction-mode prompt, up to `max_retries`.
- `run_any_chain(..., is_correction, retry_count, ...)` (`:1707`) selects the correction prompt
  variant when `is_correction` is true and passes the expected schema
  (`:1719` embeds `expected_schema` into the correction instruction).

**Fidelity mapping to WS3:** the Studio engine builds its repair prompt from the *validator's
stderr* (per `validate_with_retries.py` semantics + its `.repair_attempt_N.md` convention),
not from a JSON-schema echo — canon steps are prompt→text, not prompt→JSON. So port the
**loop shape** (retry same step with an appended correction, bounded by `max_attempts`), not
the JSON-schema correction payload. `retry_max_attempts` default 3 matches
`validate_with_retries.py:27`. Exhausted retries → **quarantine the pair** (the Andon Cord),
which the fossil did NOT have — that is a v1-gates hardening the Studio adds.

## 5. `max_tokens` heritage (→ FlowSpec `model_tiers.max_tokens`)

The fossil sets per-adapter `max_tokens`: concept/medium tier `32000` (`:1055`), prompt/utility
tiers `2000`–`4096` (`:462,1119,1220`). The importer's `model_tiers` follows this shape:
`coordinator: 16000`, `pair: 32000`, `enhance: 32000` — the heavy generation steps get the big
budget, matching the fossil's `32000` on the concept/medium generation path. (Modern Anthropic
budgets differ from the fossil's LangChain values; the numbers are re-derived for
`claude-opus-4-8`, the topology is preserved.)

## 6. Provider dialects the fossil proves (→ WS1, out of WS0 scope, noted for handoff)

`get_llm()` routes by model-name prefix: bare Anthropic/OpenAI, `OR-` OpenRouter (raw HTTP),
`Poe-` Poe (`fastapi_poe`, NOT OpenAI-compatible), `LOCAL-` custom base_url, native Gemini
(`google.genai`). This confirms plan §7's three-dialect model + the `Poe`/`OR`/`Gemini`
corrections in the vendored README. WS0 only records this; WS1 ports it.

---

**Bottom line for the fleet:** the fossil's chain is `header+middle+ending` string prompts,
`{slot}` value-filled with prior artifacts injected verbatim, run in 00→10 order with a
per-pair fan-out, guarded by a bounded retry-with-correction loop. Canon has since inlined the
header/ending into each step file, so the resolver is a *read-file → fill-slots → append
RUN-PARAMETERS* path, and the engine's retry mirrors the fossil loop but repairs from validator
stderr and quarantines on exhaustion. The FlowSpec importer's coordinator/pairs/portfolio
staging reproduces the fossil topology exactly (verified — zero manifest drift).
