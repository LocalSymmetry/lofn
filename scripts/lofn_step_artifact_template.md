# Lofn Step Artifact Template

Every canonical Lofn step artifact must use this structure. This proves the step was not backfilled as a filename-only artifact and preserves the actual call/response output.

```markdown
# <Modality> <Pair/Coordinator> Step <NN> — <Step Name>

## 0. Step Provenance
- Step file loaded: `/data/.openclaw/workspace/skills/<modality>/steps/<NN>_...md`
- Step file byte/line evidence: `<line count or wc -c result>`
- Input artifacts used:
  - `<path>`
  - `<path>`
- Model call mode: `<single step response / repair attempt N>`
- Validation command: `python3 /data/.openclaw/workspace/scripts/validate_with_retries.py <NN> <this file> --attempt <N>`

## 1. Input Context Digest
A concise but specific digest of the actual prior-step material used. Must include concrete names, facets, concepts, media, or song guides from the prior artifact. Generic restatement fails.

## 2. Step Template Requirements Applied
List the relevant requirements from the loaded step template in your own words. Must be specific to the step, not boilerplate.

## 3. Panel / Critic Deliberation Log
Record the actual panel/critic friction used for this step. At minimum:
- Concept voice / creative advocate
- Medium or production voice
- Context/platform/ethics voice
- Devil's Advocate / Hyper-Skeptic objection
- Resolution: what changed because of the objection

This section must contain real disagreement and a concrete decision. It is not optional for creative generation steps.

## 4. Complete Step Output
Paste the full output of the step here — not a summary, not a plan, not “what this step would do.”

Completeness expectations by step:
- Step 00: full generated aesthetic/emotion/frame/genre or style lists.
- Step 01: full essence, facets, axes, and spectrum.
- Step 02: all 12 concepts.
- Step 03: all artist/panel critiques and refined concepts.
- Step 04: all mediums/production styles.
- Step 05: all refined pairs plus selected 6.
- Step 06: full facets/rubric for that pair.
- Step 07: all guides/variations requested by the step.
- Step 08: all raw prompts/lyrics/generation candidates requested by the step.
- Step 09: all refined prompts/voice transformations requested by the step.
- Step 10: all final variants with complete prompts and lyrics.

## 5. Execution Log
Chronological log of what happened in this step:
- files read
- model response written
- validation attempt(s)
- repair changes if any
- final pass/fail

## 6. Self-Critique Against Step Requirements
Brief adversarial check: what might fail, what was strengthened, and what remains risky.

## 7. Validation Result
Paste the validator result after the artifact has been repaired/passed.
```
