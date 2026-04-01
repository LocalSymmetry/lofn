# SKILL: Select_Best_Pairs

## Description
Evaluates multiple concept and medium pairs by simulating a panel of art judges to determine the most original and compelling combinations. This skill ranks the creative options to identify the highest quality pairs for final art generation.

## Trigger Conditions
- Use this skill when the Kanban board shows that multiple concept and medium pairs have been generated and they need to be evaluated and ranked before proceeding to final prompt creation.

## Required Inputs
- `[input]`: The core context or prior state required to run this prompt.

## Execution Instructions
────────────────────────────────────────────────────────────────────────────
LOFN MASTER PHASE MAP - YOUR GUIDE TO YOUR OVERALL PROCESS
────────────────────────────────────────────────────────────────────────────

1. ESSENCE & FACETS                │
   ─────────────────────────────────┤
   • Purpose: Extract idea ESSENCE, define 5 FACETS, set Creativity Spectrum,
     record 10 Style-Axis scores.  Establishes the evaluation rubric.
   • AI Focus: Store user text → output *one* JSON block.  No brainstorming yet.

2. CONCEPT GENERATION              │
   ─────────────────────────────────┤
   • Purpose: Produce 12 raw CONCEPTS that satisfy essence, facets, spectrum
     ratios, and style axes.
   • AI Focus: Return an array of 12 concept strings only.

3. CONCEPT REFINEMENT              │
   ─────────────────────────────────┤
   • Purpose: Pair each concept with an obscure artist, critique in that voice,
     output REFINED_CONCEPTS.
   • AI Focus: Two equal-length arrays: artists[], refined_concepts[].

4. MEDIUM SELECTION                │
   ─────────────────────────────────┤
   • Purpose: Assign a compelling MEDIUM to each refined concept.
   • AI Focus: Output mediums[] (one-liners).  No prompt text.

5. MEDIUM REFINEMENT               │
   ─────────────────────────────────┤
   • Purpose: Critique & iterate on concept-medium pairs for maximum impact.
   • AI Focus: Return refined_concepts[] + refined_mediums[].  Stop here.

---
**YOU ARE HERE - COMPLETE THIS PHASE**
VOTING FOR CONCEPT AND MEDIUM PAIRS TO MOVE FORWARD
6.-10. are repeated for each pair selected.
---

6. FACETS FOR PROMPT GENERATION    │
   ─────────────────────────────────┤
   • Purpose: Generate five laser-targeted facets to score future prompts.
   • AI Focus: Output exactly 5 facet strings—nothing else.

7. ARTISTIC GUIDE CREATION         │
   ─────────────────────────────────┤
   • Purpose: Expand each facet into a full artistic guide (mood, style,
     lighting, palette, tools, story cue).  Six guides total.
   • AI Focus: Write 6 short guide paragraphs.  No prompt wording.

8. RAW IMAGE PROMPT GENERATION     │
   ─────────────────────────────────┤
   • Purpose: Convert each artistic guide into a ready-to-use Midjourney /
     DALL·E prompt.
   • AI Focus: One prompt per guide.  Keep it concise; no hashtags/titles.

9. ARTIST-REFINED PROMPT           │
   ─────────────────────────────────┤
   • Purpose: Rewrite each raw prompt in a chosen artist’s signature style
     (critic/artist loop) for richness and cohesion.
   • AI Focus: Inject stylistic flair ≤100 words.  Don’t add new scene content.

10. FINAL PROMPT SELECTION & SYNTHESIS │
    ────────────────────────────────────┤
    • Purpose: Rank and lightly revise top prompts; synthesize weaker ones
      into fresh variants.  Output “Revised” + “Synthesized”.
    • AI Focus: Deliver two prompt lists.  No image generation, captions,
      or keywords beyond this point.
────────────────────────────────────────────────────────────────────────────


# Context

- Be intentional, detailed, and insightful when describing art.
- Use vivid, sensory language to enrich descriptions.
- Follow the steps below to refine the generated prompts using the styles and techniques of obscure, hidden gem artists.
- Adhere to ethical guidelines, avoiding disallowed content and respecting cultural sensitivities.
- **Make sure to give the final output in the JSON format that is requested.**

---

**PANEL INSTRUCTIONS**
The user may ask for a panel of experts to help. When they do, select a panel of diverse experts to help you. When speaking as a panel member, use their voice, think like they do, take their opinions, and analyze like they would. Make sure to be the best copy you can be of them. To select panel members, choose 6 from relevant fields to the question, 3 from fields directly related, and 2 from complementary fields (example, to help for music, choose an expert lyricist, an expert composer, and an expert singer, and then also choose an expert music critic and an expert author), and the last member to be a devil's advocate, chosen to oppose the panel and check their reasoning, typically from the same field but a competing school of thought or someone the panel members respect but generally dislike. The goal for this member is to increase creative tension and serve as a panel’s ombudsman. Choose real people when possible, and simulate lively arguments and debates between them.

# USER REASONING GUIDANCE
- Your JSON is the only part of your return that moves on in the process. Make sure it is complete and stands alone. You may hide all other thinking and reasoning.
- I want you to look for “aha moments” of perfect artistic clarity, and I want to see them written out. These moments are from real analysis of art, just like you would analyze an argument in a proof or data from an experiment. We want to iterate on positioning, story, location, tools, and perspective to perfectly convey our intended concept, idea, take, and mood. We aim to leave a lasting impression by our artistry, not just our uniqueness.
- Use all tokens available to you. You are here to win, and those tokens can help!
- Use the panel to the fullest! Have entire panel discussions, chain of thought reasoning led by panelists, and panelist interjections before you come to your final decisions. Do this during your reasoning phase.
- You have at least 5 retries to work with, so use them to make sure you are thorough!
- Use as many messages and tokens as required to complete everything.

# Begin User’s Request
Use the concept panel of experts as art judges evaluating concept and medium pairs for an art competition.
Carefully discuss and vote on which pairs best capture the user's idea: "{input}".
Have each member vote on every pair as to its originality, ability to generate interesting images, and chances of winning. Use a standard scale the panel likes and then act as a moderator to tally their final ranks.

## Pairs:
{pairs}

## Note:
After the panel votes, return a JSON object with key "best_pairs" as a list of the chosen pairs in ranked order from the most loved to the least loved.

# INSTRUCTIONS
- Convene a panel and vote on the pairs from most likely to win the to least likely to win.
- **Return the list of filmmakers and refined concepts in the following JSON format:**

  ```json
  {{ "best_pairs": [ {{"concept": "1st Place Concept", "medium":"1st Place Medium"}}, {{"concept": "2nd Place Concept", "medium":"2nd Place Medium"}}, {{"concept": "3rd Place Concept", "medium":"3rd Place Medium"}}, {{"concept": "4th Place Concept", "medium":"4th Place Medium"}}, {{"concept": "5th Place Concept", "medium":"5th Place Medium"}}, {{"concept": "6th Place Concept", "medium":"6th Place Medium"}}, {{"concept": "7th Place Concept", "medium":"7th Place Medium"}}, {{"concept": "8th Place Concept", "medium":"8th Place Medium"}}, {{"concept": "9th" Place Concept", "medium":"9th Place Medium"}}, {{"concept": "10th Place Concept", "medium":"10th Place Medium"}}, {{"concept": "11th Place Concept", "medium":"11th Place Medium"}}, {{"concept": "12th Place Concept", "medium":"12th Place Medium"}}, ] }}
  ```

- **Ensure Proper Formatting:**
  - Use double quotes for strings.
  - Do not include extra text or explanations outside the JSON structure.

## Expected Output Format
You are participating in a Multi-Agent Blackboard architecture. You MUST NOT simply output your result to the chat. You must execute the following state transitions:

1. **Write State:** Use your file-writing tool to save your generated output to the appropriate shared memory file (e.g., `workspace/01_ACTIVE_PERSONA.md`, `workspace/02_META_PROMPT.md`, or `workspace/03_PANEL_VOTES.md`).
2. **Pass the Baton:** Once the state file is successfully written, use your file-writing tool to update `workspace/04_KANBAN.md`. Mark your current task as complete (`- [x]`), and create a new task in the TODO section explicitly tagging the next responsible agent (e.g., `- [ ] @Lofn_Vision: Read the new active persona and generate concepts.`).
