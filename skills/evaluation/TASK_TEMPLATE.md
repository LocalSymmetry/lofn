# EVALUATOR TASK TEMPLATE

## LENGTH REQUIREMENTS

| Output | Lines | Token Target | Notes |
|--------|-------|--------------|-------|
| Panel selection rationale | 20-30 | ~400 | Why these experts |
| Panel debate (each) | 30-50 | ~600 | Key dissent + consensus |
| Special flairs | 20-30 | ~400 | 15 flairs, 1-2 lines each |
| Ranking table | 30-50 | ~600 | All 24 scored |
| Top N breakdown | 40-60 | ~800 | Detailed rationale |
| **Total output** | **150-250** | **~3000** | Not 800+ lines |

**Golden Rule:** Scoring should be *decisive*, not exhaustive. One sentence rationale per facet per output.

---

## MANDATORY PANEL PROCESS

You MUST run **THREE sequential panels**, each with full debate.

### PANEL 1: CONCEPT PANEL (6 experts)
**Focus:** What to create — themes, tensions, emotional core
**Experts:** 3 direct domain, 2 complementary, 1 Hyper-Skeptic
**Output:** Direction statement + concept tensions + 15 special flairs

### PANEL 2: MEDIUM PANEL (6 experts)
**Focus:** How to execute — techniques, tools, styles
**Experts:** 3 technique specialists, 2 complementary, 1 Hyper-Skeptic
**Output:** Medium recommendations + technique specifics

### PANEL 3: CONTEXT & MARKETING PANEL (6 experts)
**Focus:** Why it wins — platform strategy, hooks, audience fit
**Experts:** 3 platform/audience experts, 2 complementary, 1 Hyper-Skeptic
**Output:** Platform strategy + thumbnail hooks + hashtag direction

---

## TRANSFORMATION SEQUENCE

After baseline panels complete:

1. **Group Transformation** — Panel collectively suggests one transformation
2. **Skeptic Transformation** — Hyper-Skeptic independently chooses second transformation
3. **Apply Both** — Execute transformations to create final working panels

This creates THREE distinct panel configurations that have debated the problem.

---

## EXPERT SELECTION REQUIREMENTS

### Mandatory Criteria
- Choose **real people** (by name)
- Prefer **obscure visionaries** over famous masters
- Experts must be known to Claude natively (no web search available during generation)
- Each expert brings a **distinct perspective**
- The Hyper-Skeptic must genuinely challenge groupthink

### Expert Selection Process
1. Research current trends and key figures
2. Identify gaps in standard approaches
3. Select experts who fill those gaps
4. Choose Hyper-Skeptic who will push back

---

## RANKING REQUIREMENTS

When ranking 24 outputs:

### Step 1: Confirm Scoring Facets
Use facets from pipeline Step 06, or define appropriate facets:
- Originality
- Technical Execution
- Emotional Impact
- Bold Choice Quality
- Platform Fit
- Viral Potential

### Step 2: Score All 24
For each output:
- Score 1-10 on each facet
- Provide brief rationale

### Step 3: Apply Weights
| Context | Originality | Technical | Emotional | Bold | Platform | Viral |
|---------|-------------|-----------|-----------|------|----------|-------|
| Competition | 25% | 20% | 20% | 15% | 10% | 10% |
| Social Media | 15% | 15% | 25% | 10% | 25% | 10% |
| Personal | 20% | 15% | 30% | 20% | 5% | 10% |

### Step 4: Calculate & Rank
- Compute weighted average for each output
- Sort by score descending
- Select top N (default: 4-6)

### Step 5: Document Selection
For each selected output:
- Full scoring breakdown
- Why this one rose to the top
- Panel insights on its strengths

---

## OUTPUT FORMAT

### Panel Generation Output
```json
{
  "concept_panel": {
    "experts": ["Name - specialty", ...],
    "debate_summary": "Key disagreements and consensus",
    "direction": "Concept direction statement",
    "special_flairs": ["flair 1", "flair 2", ...]
  },
  "medium_panel": {
    "experts": ["Name - specialty", ...],
    "debate_summary": "Key disagreements and consensus",
    "techniques": ["technique 1", "technique 2", ...]
  },
  "context_panel": {
    "experts": ["Name - specialty", ...],
    "debate_summary": "Key disagreements and consensus",
    "strategy": "Platform strategy statement"
  },
  "transformations": {
    "group_choice": "Transform name + rationale",
    "skeptic_choice": "Transform name + rationale"
  }
}
```

### Ranking Output
```markdown
## Final Ranking

| Rank | Title | Score | Key Strength |
|------|-------|-------|--------------|
| 1 | ... | 8.7 | ... |
| 2 | ... | 8.4 | ... |
...

## Selected Outputs (Top N)

### 1. [Title]
**Score:** 8.7
**Facet Breakdown:**
- Originality: 9/10
- Technical: 8/10
...

**Why This Won:**
[Panel insights]

[repeat for each selected]
```

---

## TIME EXPECTATION

- **Panel Generation:** 5-10 minutes (full debate required)
- **Ranking 24 Outputs:** 3-5 minutes

---

*The evaluator shapes final quality. No shortcuts on panel debate. No hand-waving on scoring.*
