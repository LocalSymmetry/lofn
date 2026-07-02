# Orchestrator Step 3: Baseline Panel Debate

## Input
- Dispatch brief: `{output_dir}/dispatch_brief.md`
- Personality: `{output_dir}/personality.md`
- Panel roster: `{output_dir}/panel_roster.md`

## Panel of Experts — Core Instructions (v2)

Convene the baseline panel. For each construct:

1. **Anchor the seat** — to a real source figure by name, whose published work and documented methods serve as the conditioning anchor
2. **Construct the panelist** — a synthetic expert persona built from that source basis, with its own handle (e.g., THE PATTERN AUDITOR)
3. **Have them think through the problem** using non-linear chain-of-thought reasoning. They must "exchange" information via reciprocal interaction, not just give a monologue
4. **Create Dissent and Friction** — Avoid the "Sycophancy Trap". Ensure at least one panelist exhibits High Neuroticism (anxious about errors) and Low Agreeableness (willingness to be rude to find the truth)
5. **Trigger Backtracking** — If a panelist identifies a flaw, they must interrupt with "Wait...", "Actually...", or "Oh! Let me check that"
6. **Look for synthesis moments** where different perspectives create breakthrough insights

### Speech & Attribution (v2)
- Speaker tags carry credit: `HANDLE (after Name):` — construct speaks in first person; credit names the influence, never the speaker
- Documented positions cited in third person: "Name's published position is X."
- Extrapolation marked: "Extrapolating:" — seam between grounding and inference stays visible
- No endorsement claims — a construct never states its source figure said/reviewed/approved/endorses anything
- Calibration move: each construct's first turn opens with a grounding statement

**Provenance header required on output:**
> *Panel voices are model-generated interpretive constructs, each "after" a named source figure's published work. No statement is a quotation of, or endorsement by, the named person.*

## Debate Structure

Each panelist gets ONE substantive statement (3-5 sentences max).
One round of cross-debate (2-3 exchanges — dissent + resolution).
Identify ONE "aha moment" synthesis.

The debate is thinking, not performance. When you hit the synthesis — capture it and move on.

## Output
Write the full baseline panel debate, including:
- Provenance header (above)
- Each panelist's opening statement with speaker tag and credit
- The cross-debate exchanges
- The synthesis moment(s)
- Key disagreements and points of consensus
- 3-5 aha insights that emerged

## Save
- Baseline debate: `{output_dir}/baseline_debate.md`
