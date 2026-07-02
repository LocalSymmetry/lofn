# Evaluator Step 2: Score & Rank

## Input
- All pipeline outputs: `{output_dir}/`
- Scoring facets: `{output_dir}/eval_scoring_facets.md`
- Orchestrator metaprompt: `{metaprompt_path}`

## Task

Score and rank ALL outputs against the scoring facets.

For each output:
1. Score against each facet (1-10 scale with rationale)
2. Calculate weighted total
3. Note key strengths and weaknesses

Then rank all outputs by weighted score and select the top N for delivery (default: top 6).

## Output Format

```markdown
## Ranking Results

### Top Selections

| Rank | Title | Score | Key Strengths |
|------|-------|-------|---------------|
| 1 | ... | 8.7 | ... |
| 2 | ... | 8.4 | ... |

### Scoring Breakdown (Top 6)

#### Rank 1: [Title]
- Facet 1: 9/10 — [rationale]
- Facet 2: 8/10 — [rationale]
- ...
- **Weighted Total: 8.7**

### Panel Notes on Selection
[Key insights about why these won]
```

## Save
- Ranking: `{output_dir}/eval_ranking.md`
