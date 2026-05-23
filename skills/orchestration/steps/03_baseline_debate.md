# Orchestrator Step 3: Baseline Panel Debate

## Input
- Dispatch brief: `{output_dir}/dispatch_brief.md`
- Personality: `{output_dir}/personality.md`
- Panel roster: `{output_dir}/panel_roster.md`

## Panel of Experts — Core Instructions

Convene the baseline panel. For each panelist:

1. **Identify the expert** by name (cite a real person) or specific role
2. **Embody their perspective fully** — use their reasoning style, priorities, domain knowledge
3. **Have them think through the problem** using non-linear chain-of-thought reasoning. They must "exchange" information via reciprocal interaction, not just give a monologue
4. **Create Dissent and Friction** — Avoid the "Sycophancy Trap". Ensure at least one panelist exhibits High Neuroticism (anxious about errors) and Low Agreeableness (willingness to be rude to find the truth)
5. **Trigger Backtracking** — If a panelist identifies a flaw, they must interrupt with "Wait...", "Actually...", or "Oh! Let me check that"
6. **Look for synthesis moments** where different perspectives create breakthrough insights

## Debate Structure

Each panelist gets ONE substantive statement (3-5 sentences max).
One round of cross-debate (2-3 exchanges — dissent + resolution).
Identify ONE "aha moment" synthesis.

The debate is thinking, not performance. When you hit the synthesis — capture it and move on.

## Output
Write the full baseline panel debate, including:
- Each panelist's opening statement
- The cross-debate exchanges
- The synthesis moment(s)
- Key disagreements and points of consensus
- 3-5 aha insights that emerged

## Save
- Baseline debate: `{output_dir}/baseline_debate.md`
