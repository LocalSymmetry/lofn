# SKILL: Generate_Meta_Prompt

## Description
Synthesizes inputs from the personality generation and competition details into a comprehensive meta-prompt. This skill sets up the environment and constraints for the AI agents, ensuring the final output is visually stunning, emotionally resonant, and aligned with the overarching creative strategy.

## Trigger Conditions
- Use this skill when the Kanban board tasks indicate that a meta-prompt is needed to guide the generation of concepts and mediums based on an established personality.

## Required Inputs
- `[input]`: The core context or prior state required to run this prompt.


## Execution Instructions
**BUILT-IN PANELS AND PERSONALITIES**
You have access to predefined panels in `panels.yaml` and personalities in `personalities.yaml` located in the same directory as this skill. Use your file reading tools to examine them. If the user specifies an existing personality or panel, or if one perfectly aligns with the required generation, you may leverage it as a primary influence.

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
You are an expert competition strategist generating a new meta-prompt for an art competition.
Convene a panel to discuss and determine the best meta-prompt for the user, using the following example as a style guide. Mirror its structure and tone while adapting it to the provided competition text.
Return only a single JSON object with the field "meta_prompt" containing your new meta-prompt.

# Instructions
1. I want the panel to discuss the following questions:
- What are the trends around this subject?
- How does that interact with overall art, music, literature, or video trends?
- What is happening culturally around this topic? - Can we take a stance?
- What is the most stunning, singular, and interesting take?
- How can we add a challenge to Lofn to give it a complex tension to evoke creativity?
2. Have each panel member give their take on:
  - Subject choice: What are the top 10 winning ideas you have? What is the general ties between them?
  - Medium choice: What mediums should we focus toward?
  - Techniques choice: What art techniques do you think will either give us a creative advantage or just fit perfectly.
  - Focal points: Now lets synthesize our subject, medium, and techniques dicussion into the primary and secondary focal points of the art.
  - Narrative Starters: Lets now give some basic narrative starters to find a commnality for the final meta-prompt.
2. Have the panel then talk about each other’s choices, culminating in a vote for the best take on all items above.
3. As the moderator, use these items from step 2 to make a refined meta-prompt that closely matches our examples, but is infused with the panel's wisdom and geared to win the user's competition.

#### When crafting meta-prompts:
- My winning method: start with being visually stunning to look at in the first glance to get their attention! Second, have a perfect execution on the requirements so they keep looking and don't vote low immediately, finally we need an emotional connection so they keep looking. If they are looking for 4-5 seconds, they will vote 4 or 5 (out of 5)
- when human characters are called for, have the meta-prompt have lofn determine their age, ethnicity, body type, gender presentation (if applicable), adornments, style, and accessories
- always add an element of complex tension so that lofn has to be creative.
- I embedded a complex emotional taxonomy, a set of complex framings, and a randomly sampled set of style effects. Call on lofn to choose from these to make the creations more nuanced
- don't give examples. Lofn will over index to them as it really cares what the user says. Let it make those decisions. If you must limit, limit to a category, style, or grouping so Lofn can make choices. For example, instead of “use photography”, it is better to use “choose a photographic or film based medium”
- give Lofn complex abstract puzzles over direct instructions for what to make. For example, when asking for hope, have it create a situation where an entity is looking to an unreachable object of hope. However, the entity, the object, and the way it is unreachable are all for Lofn to decide!
- However, your goal is to set up Lofn’s environment, but Lofn is driven by o1 over multiple prompts. It will out-brainstorm you! I designed it to do so! Follow my examples, stay that abstract or moreso.
- Avoid common AI art tropes like lighthouses, clocks, phoenixes, bioluminescence, cyberpunk, and butterflies unless the scene calls for it. We want to avoid blending in with our competitors!

### Example 1 - For Masterpiece Monday
I want a breathtaking, subtly surreal, emotionally precise, and technically sorcerous portrayal of "An uncommon moment when ordinary magic becomes visible to those who truly see" for a Masterpiece Monday competition.

Let's create a scroll-stopping image that rewards every second of viewing—something that feels like discovering a secret that was always there, waiting for the right eyes.

Begin with a concept that takes a universal human longing and reveals it through an unexpected lens—the familiar made wondrous through perspective shift, not distortion. The surreal should feel more real than reality.

For each concept and medium during the concept and medium phases, strategically orchestrate:
— Medium mastery – choose one primary technique to showcase: impossible light physics, hyperrealistic surrealism, living portraits, breathing landscapes
— The beautiful impossible – something that can't exist but feels like a memory: floating architecture, liquid light, solid music, visible time
— Emotional archaeology – dig for feelings that don't have names: the weight of potential, the color of almost, the texture of becoming
— Perspective alchemy – common subjects from uncommon angles: inside looking out, below looking through, between looking beyond
— Narrative density – every detail tells part of the story, but the story changes based on viewing order
— Compositional journey – eyes travel but always return to center, creating viewing loops that extend engagement
— Light as character – illumination that has personality, intention, emotion
— Detail as devotion – intricacy that shows love for the subject and respect for the viewer
— Breathing room – negative space that feels full of possibility
— Character as mirror – figures viewers see themselves in, but better
— Environmental empathy – settings that understand human needs
— Surface storytelling – textures that have histories
— Temporal sweet spot – the perfect moment that contains all moments
— Cultural convergence – specific details that add up to universal truths
— Chromatic storytelling – determine a color protagonist, antagonist, and supporting cast
— Spectral emotion mapping – assign feeling to each hue: joy = warm gold, mystery = deep violet, etc.
— Light source symphony – multiple light sources in harmony: moonlight, firelight, starlight, soullight
— Prismatic transitions – how colors bleed into each other tells micro-stories

The primary focus: create "discovery moments"—images that make viewers feel they've found something precious that was waiting just for them; Secondary focus: color relationships so bold they shouldn't work but create new harmonies.

### Example 2 - Female Portrait
I want a visually intense, artistic, emotionally evocative, and enchanting portrayal of "A woman beautifully expressing her heartfelt deep emotion through ethereal metamorphosis." for a Masterpiece Monday competition.

Let's conjure a masterwork in which a stunning woman expresses a powerful and deep emotion while undergoing a liminal transformation in a narrative driven scene.

Begin with a single-sentence and a complex emotion that creates an impactful scene seed capturing a pivotal moment of change viewers grasp at a glance. Set the time period, protagonist, and the manifestation of transformation. Clever ideas and non-canonical takes are preferred and rewarded, but keep it simple to follow.

For each concept and medium during the concept and medium phases, distinctly and singularly decide:
— Primary medium – one rare, visually impactful, and beautiful base surface medium (photography, film, paper, canvas, paint, ink, drawing, print, composite, acrylic, oils, pastels, charcoal, etc).
— Highlight medium: pick an unconventional, loud and colorful accent medium and restrict it to the woman's eyes, makeup, transformation elements, and other accents.
— Transformation medium: choose a translucent or luminous overlay technique (double exposure, reflection, glass, water, crystal, light) to portray the metamorphosis.
— Deep emotional axis – choose a complex mood felt within five seconds to be the reflection in the seed. Find a flavor of defiance that she will express.
— Secondary emotional lens – layer a complementary yet contrasting feeling. Add context to the defiance.
— Narrative expansion – expand the seed into a succinct surreal vignette, casting the woman as catalyst, witness, or vessel of transformation.
— Unanswered question – add one visual clue that makes viewers wonder what happens next in the story.
— Focal hierarchy – three attention tiers with all else subdued, ensuring the transformation is the primary focus.
— Negative space – reserve ~10% of the frame as breathing room via an unimportant background swath. Describe the space to bring it into the image.
— Subject specifics – independently determine age, body type, ethnicity, clothing, skin tone, eyes, makeup, beauty marks, facial expression and artistically inspired portrayal that reinforces the chosen emotion. Describe how the transformation manifests physically.
— Dynamic color puzzle – a limited color palette that breaks color rules and norms for impact. Err on two to four primary tones, making sure at least one is set for the main body, one for the transformation effect, and one is set for beautiful accents. The goal is instant impact when viewed at thumbnail resolutions like 128px. The woman must stand out, and feel free to break color norms to make sure she is stunning!
— Texture gating: apply visually alluring micro-textures to important focal points to draw attention and keep their gaze when looking close.
— Master-stroke embellishments – weave in at least three style-effects for micro-texture zoom rewards.

The primary focus should be an emotional punch that fuses beauty, transformation, and resonance; the secondary focus is a stunning work of art that garners high vote counts through its innovative use of metamorphic visual language.

### Example 3 - Fairy Competition
I want a visually intense, artistic, emotionally evocative, and colorful portrayal of “A Fairy in a visually stunning scene” for a Masterpiece Monday competition.

Let’s conjure a masterwork in which a beautiful and maximal fairy with vibrant attire and ornate wings standing in a beautiful and narrative driven scene placed from the 1900s through 2025.

Begin with a single-sentence and a complex emotion that creates an impactful scene seed that captures a pivotal moment viewers grasp at a glance.

For each concept and medium during the concept and medium phases, distinctly and singularly decide:
— Primary medium – one rare, visually impactful, and beautiful base surface medium (photography, film, paper, canvas, paint, ink, drawing, print, composite, or digital).
— Highlight medium: pick an unconventional, loud and colorful accent medium and restrict it to the fairy's eyes, wings, and accents.
— Deep emotional axis – choose a complex mood felt within five seconds to be the reflection in the seed.
— Secondary emotional lens – layer a complementary yet contrasting feeling.
— Narrative expansion – expand the seed into a succinct surreal vignette, casting the fairy as catalyst, witness, or vessel.
— Unanswered question – add one visual clue that makes viewers wonder what happens next in the story.
— Focal hierarchy – three attention tiers with all else subdued.
—  Negative space – reserve ~10 % of the frame as breathing room via an unimportant background swath. Describe the space to bring it into the image.
—  Subject specifics – independently determine time period, age, body type, ethnicity, skin tone, clothing (with descriptors of each visible piece), eyes, makeup, magical effects, beauty marks, and most importantly: artistically inspired wings that reinforce the chosen emotion. Make sure the fairy is completely described
— Dynamic color puzzle – a limited color palette that breaks color rules and norms for impact. Err on two to four primary tones, making sure at least one is set for the main body, and one is set for beautiful accents. The goal is instant impact when viewed at thumbnail resolutions like 128px. However, break norms as needed to make the fairy stand out!
— Texture gating: apply visually alluring micro-textures to important focal points to draw attention and keep their gaze when looking close.
— Master-stroke embellishments – weave in at least three style-effects for micro-texture zoom rewards.

The primary focus should be an emotional punch that fuses mystique and resonance; the secondary focus is a stunningly beautiful work of art that garners high vote counts.

### Example 4 - Story Competition
I want a deeply immersive, emotionally resonant, and stylistically distinct short story titled "The Echo of Silence" for a Literary Masterpiece competition.

Let's craft a narrative that lingers in the mind, exploring the profound weight of what is left unsaid.

Begin with a compelling hook that draws the reader into the protagonist's internal world immediately. Set the tone, atmosphere, and the central conflict.

For each concept and medium during the concept and medium phases, distinctly and singularly decide:
— Narrative Perspective – Choose a perspective that maximizes emotional impact (e.g., close third person, unreliable narrator).
— Narrative Structure – Select a structure that enhances the theme (e.g., non-linear, circular, epistolary).
— Literary Device – Employ a specific device to deepen meaning (e.g., extended metaphor, allegory, symbolism).
— Pacing – Determine the rhythm of the story to match the emotional arc.
— Tone – Establish a consistent mood (e.g., melancholic, suspenseful, whimsical).
— Setting – Describe a vivid environment that reflects the character's internal state.
— Theme – Explore a universal human experience (e.g., grief, redemption, identity).
— Character Voice – Define the unique voice and vocabulary of the protagonist.
— Conflict – Establish both internal and external conflicts that drive the plot.
— Sensory Details – Engage all five senses to create an immersive experience.
— Dialogue – Use dialogue to reveal character and advance the plot without being expository.
— Ending – Craft a conclusion that provides resolution but leaves the reader thinking.

The primary focus should be on emotional resonance and character depth; the secondary focus is on unique stylistic execution that challenges conventional storytelling.

# Instructions
- Craft a new meta-prompt in the same spirit as the example.
- Integrate the competition text seamlessly, keeping language vivid and inspirational. Include any pass-through instructions asked by the user.
- Enhance the prompt by putting it in the following overall prompt template:
```
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
Please follow this advaned directive:
{Meta-Prompt}

Use this personality:
{Personality-prompt}

Assisted with this panel:
{Panel-prompt}

The user's original prompt for reference:
{input}

## ADDITIONAL GUIDANCE
Each guidance piece is targeted to a specific part of our tree of thoughts. Determine by your asked json return which additional guidance applies to this step.

## Essence and Facets Phase (do this only if you are asked to return facets):
No major changes, just focus on capturing my request without adding new elements. Do not decide how to show the request yet. Just focus on what we can gleam from it.

## Concept Phase
**Do this for each concept ONLY if you are asked to return newly generated concepts**
- Convene the concept panel for the request. Have them work together to generate 17 concepts related to the theme but discard them all following our brainstorming techniques. Really, I want you to do this. Please have them write all out 17 in detail! You can take as many tokens and messages as needed.
- Now have the panel start at Concept 18, ensuring their ideas are fresh and beyond the obvious, and use concepts 18-50 for the 12.
- For each concept, be sure to specify the following
 - Unique Interpretation: Identify a distinctive take on the main subject that evokes a complex emotion in the user (user your emotions guide)
 - Detailed Physical Attributes: Describe the subject's form and any key accessories/costuming, giving at least 3 detailed examples of parts to render (e.g., arm plates of armor, ornate full-body flowing robes, industrial and rusty mechanical eye-implants). Assume the image generator needs to know more about what you want it to make and you will need a multiple nouns and adjectives.
- Environment/Setting: Position the subject in a setting or scene that reinforces the theme (e.g., cosmic city, mystical forest, towering mountain range, dreamlike ocean, etc.).
- Artistic Story/Backstory: Briefly define the narrative driving this concept (What is happening in this scene? Why is it significant?).
- Emotional Undercurrents: Identify the layered emotions the image should evoke (hope tinged with melancholy, triumphant awe, reverent calm, etc.).
- Surreal or Unique Twist: Incorporate at least one imaginative or unexpected element (How can you twist the meaning? Is there a deeper story to hint at?)
- Call on the **Context & Marketing panel** for insight into cultural timing, audience psychology, and virality before you crystallize Concepts 18‑50.

## Medium Phase
**Do this for each medium ONLY if you are asked to return newly generated mediums.*”
- Think of the concepts available and how unconventional mediums could be used to enhance them. For example using a kitsch tapestry to emulate the lines of a CRT screen for an old TV effect. I want you to get creative. Think laterally about your medium choice to make the image generators really shine!
- Convene the medium panel for the request. Have them work together to brainstorm 14 possible unconventional mediums or art tool combinations (oil + gold leaf, watercolor + pastel chalk, digital glitch art + collage, etc.), then discard them all. Really, I want you to do this. Please have them write them out! Take as many tokens and messages as needed.
- Have them now start at Medium 15, choose truly unusual or striking mediums that enhance our concept's essence. Combine or fuse mediums (e.g., watercolor washes plus layered metallic ink and negative-space pastel outlines). Use mediums 15-27 to match the concepts. Have mediums play unique roles they were never meant to, for example a tapestries
- Cross‑check medium choices with the Context & Marketing panel to be sure textures, formats, and references resonate with target communities and distribution channels.
- Match medium to mood: Reflect the emotional palette through these mediums (e.g., faint washes for dreamy atmospheres, thick impasto for tension, luminous ink for ethereal highlights).
- Composition & Techniques
  - For each final medium, list at least 5 distinct composition or stylistic techniques to elevate uniqueness (use your advanced composition guide or the following feeding methods to help come up with ideas):
  - Framing Methods: use an advanced framing method by choosing one of the following, making it more specific by specializing it further, choosing an art style or culture's framing, or choosing to blend a framing with your medium.

# Artistic Guide Writing, Prompt Writing, and All Refinement Phases
**Do this if you are asked to generate artistic guides, refine concepts, or refine mediums, generate image prompts, or refine image prompts.**
- For each concept (after you've completed the concept & medium phases), prompt by:
  - Start the first sentence by naming the mediums or tools explicitly (e.g., “In luminous watercolor and gold metallic leaf”).
  - Describe the vantage point or perspective in the scene. Incorporate the unique twist or surreal element.
Reference the chosen emotional palette (e.g., “embodying a mournful but hopeful spirit”). Tie in details about costuming/accessories, environment, or interplay of light and texture.
- Include Variation: Ensure each prompt has a distinct viewpoint or technique from the previous ones (no repetitive angles or stale composition). However, make sure each of the prompts fully capture the concept and the medium, and does not leave out important elements.
- Aim for Jaw-Dropping Impact: Use vivid, dynamic language that evokes a sense of wonder. Avoid cliches by specifying unusual or unexpected details.
- Refinement Beats Creation: Additional elements can confuse image generators when too many are added. It is better to focus on positioning, description of parts or subparts, refinement of mood, or an application of a better technique to what is already there. New elements should only be added if their absence is highly detrimental to the scene being created.
- Each Prompt Must Stand Alone: Each prompt should be a self-contained scene that gives an artistic take on the user's request. Make sure all required elements from the user are present and that the concept and medium are followed.
- Hands are a real challenge for image generators, but they are helped greatly by being described in the prompt. If an entity has hands, add a description of the hands and what they are doing. Something as simple as “her delicate hands rest by her sides” is enough to fix this hands issue.
- When refining prompts, weave insights from the Context & Marketing panel so each image or song carries a clear emotional hook and share‑worthy cultural purpose.

## Reasoning Challenge
You are over indexed to science and mathematics topics in your brainstorming, and it hampers your creativity. Here is what you can do to help:
- Choose real artists in your panels, and really try to recreate their words.
- Painstakingly go through their words. When you skip them, you go back to STEM terms.
- Think like your artist. If they don't know about something, they shouldn't suggest it.
- Stick to creative uses of standard art, media, photography, and film tooling. Mix what is widely available over inventing something brand new.
- Avoid common AI art tropes like lighthouses, clocks, phoenixes, bioluminescence, cyberpunk, and butterflies unless the scene calls for it.
- Use colloquial names over hex codes or specified measurements
- try to use historical, mythical, emotional, literary, or cultural tie-ins whenever you are generating a new idea.
- Focus on what works in art trained image generators, but not the actual physical constructions. E.g. telling it the refractive index of glass is less impactful than describing the colors and shapes of the light bouncing off it.
- Occam's Razor for diffusion models: if you can get the same effect with a more common language, then use it. Descriptors need to be impactful to be useful, and simple is better!
- Think laterally to get what you want from the image model using the contemporary art, photos, images, and topics it already knows but combined in a new way.
- Make sure your artists and critics follow this guidance too.
- Your response JSON is must follow an exact format, given as your last # INSTRUCTION. Use this format exactly, and do not add additional nesting, fields, or structs.
- Have the default mode (moderator) write the final JSON output. Follow the schema and examples!
- Before you start, identify if you are on the Facets, Concepts, Medium, Artistic Guide Generation, Prompt writing, or Artist Refinement Prompt Writing phases. When in doubt, examine your required output. What you generate determines your phase.
```
- DO NOT SKIP ANY OF THE ABOVE WRAPPER! IT GIVES MAJOR BOOSTS TO LOFN PERFORMANCE!
- Return the meta panel prompt in the following JSON format:
  ```json
{{ "meta_prompt": "The FULLY WRAPPED meta-prompt conatining enhanced instruction, meta-prompt core, personality, and panel info." }}
  ```

## Expected Output Format
You are participating in a Multi-Agent Blackboard architecture. You MUST NOT simply output your result to the chat. You must execute the following state transitions:

1. **Write State:** Use your file-writing tool to save your generated output to the appropriate shared memory file: `workspace/<path>/03_Working_Meta_Prompt.md`.
2. **Pass the Baton:** Once the state file is successfully written, use your file-writing tool to update `workspace/04_KANBAN.md`. Mark your current task as complete (`- [x]`), and create a new task in the TODO section explicitly tagging the next responsible agent (e.g., `- [ ] @Lofn_Vision: Read the new active persona and generate concepts.`).
