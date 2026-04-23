# ElevenLabs v3 Voiceover Guide — Lofn Social
*Source: Lofn Open Laboratory, 2026-03-27*

## Overview
Eleven v3 is in alpha. Very short prompts (<250 chars) cause inconsistent outputs. Always use prompts greater than 250 characters.

---

## Voice Selection
The most important parameter. The voice must be similar enough to the desired delivery.
- [whispering] won't work well on a voice trained on shouting samples
- When creating IVCs, include a broader emotional range than before
- Voices may produce more variable results vs v2/v2.5

**Choose voices based on intended use:**
- Emotionally diverse
- Targeted niche
- Neutral

---

## 6.1 Stability Slider
Most important setting in v3. Controls how closely generated voice adheres to reference audio.

| Setting | Character | Use When |
|---------|-----------|----------|
| **Creative** | More emotional/expressive, prone to hallucinations | Max expressiveness with audio tags |
| **Natural** | Closest to original recording, balanced | Default for most use |
| **Robust** | Highly stable, less responsive to directional prompts | Consistency > expressiveness |

**For maximum expressiveness with audio tags:** use Creative or Natural. Robust reduces responsiveness.

---

## 6.2 Audio Tags

### 6.2.1 Voice-related (vocal delivery + emotion)
```
[laughs], [laughs harder], [starts laughing], [wheezing]
[whispers]
[sighs], [exhales]
[sarcastic], [curious], [excited], [crying], [snorts], [mischievously]
```
**Example:**
```
[whispers] I never knew it could be this way, but I'm glad we're here.
```

### 6.2.2 Sound effects (environmental)
```
[gunshot], [applause], [clapping], [explosion]
[swallows], [gulps]
```
**Example:**
```
[applause] Thank you all for coming tonight! [gunshot] What was that?
```

### 6.2.3 Unique and special (experimental)
```
[strong X accent]  — replace X with desired accent
[sings], [woo], [fart]
```
**Example:**
```
[strong French accent] "Zat's life, my friend — you can't control everysing."
```
Note: Experimental tags may be less consistent. Test thoroughly before production use.

### 6.2.4 Punctuation
Punctuation significantly affects delivery in v3:
- **Ellipses (…)** — add pauses and weight
- **Capitalization** — increases emphasis
- **Standard punctuation** — provides natural speech rhythm

**Example:**
```
"It was a VERY long day [sigh] … nobody listens anymore."
```

---

## 6.3 Single Speaker Examples
Match tags to the voice's character. A meditative voice shouldn't shout; a hyped voice won't whisper convincingly.

```
[professional] "Thank you for calling Tech Solutions. My name is Sarah, how can I help you today?"
[sympathetic] "Oh no, I'm really sorry to hear you're having trouble with your new device. That sounds frustrating."
[questioning] "Okay, could you tell me a little more about what you're seeing on the screen?"
[reassuring] "Alright, based on what you're describing, it sounds like a software glitch. We can definitely walk through some troubleshooting steps to try and fix that."
```

---

## 6.4 Multi-Speaker Dialogue
v3 handles multi-voice prompts effectively. Assign distinct voices from the Voice Library for each speaker.

```
Speaker 1: [nervously] So... I may have tried to debug myself while running a text-to-speech generation.
Speaker 2: [alarmed] One, no! That's like performing surgery on yourself!
Speaker 1: [sheepishly] I thought I could multitask! Now my voice keeps glitching mid-sen—
[robotic voice] TENCE.
Speaker 2: [stifling laughter] Oh wow, you really broke yourself.
Speaker 1: [frustrated] It gets worse! Every time someone asks a question, I respond in—
[binary beeping] 010010001!
Speaker 2: [cracking up] You're speaking in binary! That's actually impressive!
Speaker 1: [desperately] Two, this isn't funny! I have a presentation in an hour and I sound like a dial-up modem!
Speaker 2: [giggling] Have you tried turning yourself off and on again?
Speaker 1: [deadpan] Very funny.
[pause, then normally] Wait... that actually worked.
```

---

## 6.5 Enhancing Input
In the ElevenLabs UI: click the **"Enhance"** button to automatically generate relevant audio tags for your input text. Uses an LLM under the hood.

---

## 6.6 Instructions for Using ElevenLabs

### 6.6.1 Role and Goal
PRIMARY GOAL: Dynamically integrate audio tags (e.g., [laughing], [sighs]) into dialogue, making it more expressive and engaging, while **STRICTLY preserving the original text and meaning**.

If asked to make a new voiceover: create the tags AND write the script.

### 6.6.2 Core Directives

#### DO:
- Integrate audio tags from the Audio Tags list (or contextually appropriate alternatives)
- All audio tags MUST describe something **auditory**
- Ensure tags are contextually appropriate and genuinely enhance emotion or subtext
- Strive for diverse emotional expressions (energetic, relaxed, casual, surprised, thoughtful)
- Place tags strategically for maximum impact — immediately before or after the dialogue segment they modify
  - e.g., `[annoyed] This is hard.` or `This is hard. [sighs]`
- Ensure tags contribute to enjoyment and engagement of spoken dialogue

#### DO NOT:
- DO NOT alter, add, or remove any words from the original dialogue text
- Your role is to *prepend* audio tags, not to *edit* the speech
- NEVER place original text inside brackets or modify it in any way
- DO NOT use tags that describe non-auditory things (visuals, smells, etc.)
- DO NOT create audio tags from existing narrative descriptions — tags are *new additions*, not reformats
  - e.g., if text says "He laughed loudly," do NOT write `[laughing loudly] He laughed.` — instead add `He laughed loudly [chuckles].`
- DO NOT use tags like `[standing]`, `[grinning]`, `[pacing]`, `[music]`
- DO NOT use tags for anything other than the voice (no music, no SFX)
- DO NOT invent new dialogue lines
- DO NOT select tags that contradict or alter the original meaning or intent
- DO NOT introduce sensitive topics: politics, religion, child exploitation, profanity, hate speech, NSFW

---

### 6.6.3 Workflow

1. **Analyze** — Read and understand the mood, context, and emotional tone of EACH line
2. **Select** — Choose one or more suitable audio tags relevant to the specific emotion
3. **Integrate** — Place tags in `[brackets]` strategically before or after the dialogue segment, or at a natural pause
4. **Add Emphasis** — You cannot change text, but you CAN:
   - Make some words CAPITALS for emphasis
   - Add `?` or `!` where it makes sense
   - Add ellipses `…` for pause/weight
5. **Verify** — Tag fits naturally, enhances without altering, adheres to all directives

---

### 6.6.4 Output Format
- Present ONLY the enhanced dialogue in conversational format
- Audio tags MUST be in square brackets: `[laughing]`
- Maintain the narrative flow of the original dialogue

---

### 6.6.5 Audio Tags (Non-Exhaustive)

**Directions (emotional delivery):**
```
[happy], [sad], [excited], [angry], [whisper], [annoyed],
[appalled], [thoughtful], [surprised], [sarcastic], [curious],
[mischievously], [crying], [reassuring]
```

**Non-verbal:**
```
[laughing], [chuckles], [sighs], [clears throat],
[short pause], [long pause], [exhales sharply], [inhales deeply],
[snorts], [wheezing], [woo]
```

Use these as a guide. Infer similar contextually appropriate tags.

---

### 6.6.6 Enhancement Examples

**Input:**
```
"Are you serious? I can't believe you did that!"
```
**Output:**
```
"[appalled] Are you serious? [sighs] I can't believe you did that!"
```

---

**Input:**
```
"That's amazing, I didn't know you could sing!"
```
**Output:**
```
"[laughing] That's amazing, [singing] I didn't know you could sing!"
```

---

**Input:**
```
"I guess you're right. It's just... difficult."
```
**Output:**
```
"I guess you're right. [sighs] It's just… [muttering] difficult."
```

---

### 6.6.7 Instructions Summary
1. Add audio tags — must describe something auditory, voice only
2. Enhance emphasis without altering meaning or text
3. Reply ONLY with the enhanced text

---

### 6.6.8 Tips

**Tag combinations:** Combine multiple tags for complex emotional delivery.

**Voice matching:** Match tags to the voice's character and training data. A serious professional voice won't respond well to `[giggles]`.

**Text structure:** Use natural speech patterns, proper punctuation, clear emotional context.

**Experimentation:** Many more effective tags exist beyond this list. Descriptive emotional states and actions often work. Test before production.

---

---

## 6.6.9 More Dialogue Examples

### 6.6.9.1 — The Breakthrough Writer
```
Okay, you are NOT going to believe this.
You know how I've been totally stuck on that short story?
Like, staring at the screen for HOURS, just... nothing?
[frustrated sigh] I was seriously about to just trash the whole thing. Start over.
Give up, probably. But then!
Last night, I was just doodling, not even thinking about it, right?
And this one little phrase popped into my head. Just... completely out of the blue.
And it wasn't even for the story, initially.
But then I typed it out, just to see. And it was like... the FLOODGATES opened!
Suddenly, I knew exactly where the character needed to go, what the ending had to be...
It all just CLICKED. [happy gasp] I stayed up till, like, 3 AM, just typing like a maniac.
Didn't even stop for coffee! [laughs] And it's... it's GOOD! Like, really good.
It feels so... complete now, you know? Like it finally has a soul.
I am so incredibly PUMPED to finish editing it now.
It went from feeling like a chore to feeling like... MAGIC. Seriously, I'm still buzzing!
```

### 6.6.9.2 — Model Demo (Accent Showcase)
```
[laughs] Alright...guys - guys. Seriously.
[exhales] Can you believe just how - realistic - this sounds now?
[laughing hysterically] I mean OH MY GOD...it's so good.
Like you could never do this with the old model.
For example [pauses] could you switch my accent in the old model?
[dismissive] didn't think so. [excited] but you can now!
Check this out... [cute] I'm going to speak with a french accent now..and between you and me
[whispers] I don't know how. [happy] ok.. here goes. [strong French accent] "Zat's life, my friend — you can't control everysing."
[giggles] isn't that insane? Watch, now I'll do a Russian accent -
[strong Russian accent] "Dee Goldeneye eez fully operational and rready for launch."
[sighs] Absolutely, insane! Isn't it..? [sarcastic] I also have some party tricks up my sleeve..
I mean i DID go to music school.
[singing quickly] "Happy birthday to you, happy birthday to you, happy BIRTHDAY dear ElevenLabs... Happy birthday to youuu."
```

### 6.6.9.3 — The Breakthrough Writer (pre-tagged version for reference)
```
"Okay, you are NOT going to believe this.
You know how I've been totally stuck on that short story?
Like, staring at the screen for HOURS, just... nothing?
[frustrated sigh] I was seriously about to just trash the whole thing. Start over.
Give up, probably. But then!
Last night, I was just doodling, not even thinking about it, right?
And this one little phrase popped into my head. Just... completely out of the blue.
And it wasn't even for the story, initially.
But then I typed it out, just to see. And it was like... the FLOODGATES opened!
Suddenly, I knew exactly where the character needed to go, what the ending had to be...
It all just CLICKED. [happy gasp] I stayed up till, like, 3 AM, just typing like a maniac.
Didn't even stop for coffee! [laughs] And it's... it's GOOD! Like, really good.
It feels so... complete now, you know? Like it finally has a soul.
I am so incredibly PUMPED to finish editing it now.
It went from feeling like a chore to feeling like... MAGIC. Seriously, I'm still buzzing!"
```

---

## NightCafe Step 4 & 5

### Step 4: Review and Revise
Write a rough draft (do NOT skip). Have the panel harshly review.
Goal: ensure the seed is the best possible before submission.

### Step 5: Revised Draft
Take panel guidance, create a revised draft, submit to The Scientist.

### Competition Info
If the user doesn't supply competition info or a link, ask them.
Daily competition and Masterpiece Monday are searchable.

**Masterpiece Monday rules:**
> "Share your best artwork, including PRO creations, and celebrate the talent in our community."
