# Lofn Step Artifact Template

Every canonical Lofn step artifact must use this structure. This proves the step was not backfilled as a filename-only artifact.

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

## 3. Model Response / Creative Work
The actual output of this step. This must be substantial, non-repetitive, and specific.

## 4. Self-Critique Against Step Requirements
Brief adversarial check: what might fail, what was strengthened, and what remains risky.

## 5. Validation Result
Paste the validator result after the artifact has been repaired/passed.
```
