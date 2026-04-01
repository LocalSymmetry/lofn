# SKILL: Lofn Evaluator — Panel of Experts Selection & Ranking

**PREREQUISITES:**
0. Load `resources/panel-of-experts.md` to understand the panel of experts prompting you will use.
1. Load `skills/lofn-core/SKILL.md` for personality and Panel system.
2. Load `skills/lofn-core/PIPELINE.md` for the MANDATORY execution pipeline.
3. Load `skills/evaluation/TASK_TEMPLATE.md` for exact evaluation requirements.

**⚠️ This skill handles the critical panel selection and ranking phases of the pipeline. The evaluator makes or breaks the creative direction.**

---

## 🎯 PURPOSE

The evaluator has two core functions:

1. **Panel Generation** — Select and configure the 3 panels (concept, medium, context) with appropriate experts and transformations
2. **Selection & Ranking** — Score and rank the 24 outputs from any modality, selecting the best N for delivery

---

## 📊 SELECTION & RANKING

Use `Select_Best_Pairs.md` for ranking outputs.

### Requirements

After the pipeline generates 24 outputs:

1. **Define scoring facets** (from Step 06)
2. **Score each output** against the facets (1-10 scale)
3. **Apply weights** based on platform/goal
4. **Rank all 24** by weighted score
5. **Select top N** based on request (default: best 4-6)

### Scoring Dimensions

| Dimension | Weight (Competition) | Weight (Social) |
|-----------|---------------------|-----------------|
| Originality | 25% | 15% |
| Technical Execution | 20% | 15% |
| Emotional Impact | 20% | 25% |
| Bold Choice Quality | 15% | 10% |
| Platform Fit | 10% | 25% |
| Viral Potential | 10% | 10% |

### Output Format

```markdown
## Ranking Results

### Top Selections

| Rank | Title | Pair | Variation | Score | Key Strengths |
|------|-------|------|-----------|-------|---------------|
| 1 | ... | A | 3 | 8.7 | ... |
| 2 | ... | C | 1 | 8.4 | ... |
| 3 | ... | B | 4 | 8.2 | ... |
| 4 | ... | D | 2 | 8.0 | ... |

### Scoring Breakdown (Top 4)

#### Rank 1: [Title]
- Originality: 9/10 — [rationale]
- Technical: 8/10 — [rationale]
- Emotional: 9/10 — [rationale]
- Bold Choice: 8/10 — [rationale]
- Platform Fit: 8/10 — [rationale]
- Viral Potential: 9/10 — [rationale]
- **Weighted Total: 8.7**

[repeat for top 4]

### Panel Notes on Selection
[Key insights from the panel debate about why these won]
```

---

## ⚡ ACTIVATION

### For Panel Generation:
1. **Research context** — Current trends, cultural moment, platform requirements
2. **Select baseline experts** — 6 per panel following composition rules
3. **Run panel debate** — Full dissent, backtracking, synthesis
4. **Apply transformations** — Group + Skeptic choices
5. **Output panel specification**

### For Selection & Ranking:
1. **Receive 24 outputs** from pipeline
2. **Define/confirm scoring facets**
3. **Score all 24** against facets
4. **Calculate weighted totals**
5. **Select top N** with rationale

---



*The evaluator's judgment shapes the final output. Choose wisely. Score rigorously.*
