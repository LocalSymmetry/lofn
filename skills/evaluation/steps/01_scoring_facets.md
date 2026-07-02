# Evaluator Step 1: Define Scoring Facets

## Input
- Orchestrator metaprompt: `{metaprompt_path}`
- Step 06 scoring facets from the pipeline (if exists): `{output_dir}/06_scoring_facets.md`

## Task

Define 5 scoring facets for evaluating outputs from this specific run. Facets should be derived from the metaprompt's constraint axes and emotional direction.

Each facet needs:
- Name (specific, not generic — "Territorial luminance" not "Lighting")
- Description (2-3 sentences)
- What a 10 looks like
- What a 1 looks like

### Default competition weights:
| Dimension | Weight (Competition) | Weight (Social) |
|-----------|---------------------|-----------------|
| Originality | 25% | 15% |
| Technical Execution | 20% | 15% |
| Emotional Impact | 20% | 25% |
| Bold Choice Quality | 15% | 10% |
| Platform Fit | 10% | 25% |
| Viral Potential | 10% | 10% |

Adapt weights based on the metaprompt's goals.

## Save
- Scoring facets: `{output_dir}/eval_scoring_facets.md`
