# VIDEO_RENDER_AUDIT.md — THE THREE-PASS BLIND RULE

Created 2026-08-15 | Status: ACTIVE | Owner gate: `lofn-qa` (Cinematic Somatic Gate)
Sibling instruments: `lofn-render-audit` (audio, THE BLIND RULE) · THE VINCENT TEST (image, L39)

---

## Why this exists

Music has a blind listening pass. Images have the Vincent Test. **Video had nothing** — every gate
we owned for video was textual, and we had never once measured a video render against the prompt
that asked for it. That was survivable while video cost nothing and we shipped only prompt text.
There is now a funded fal account.

It is worse than a missing instrument. The obvious instrument is **broken by construction**, and
here is the mechanism:

- **Synchresis** (Chion, *Audio-Vision*): the perceptual system involuntarily welds a sound to an
  image that coincides with it. It does not check whether the weld is *earned*. It welds a punch to
  a snapped stick of celery and the audience hears a jaw break.
- **The McGurk effect** (McGurk & MacDonald, *Nature*, 1976): a visual articulation dubbed with a
  different syllable produces a **third** percept — the eyes change what the ears report. Critically,
  **the effect persists in people who know about it.** Knowing does not switch it off.

The frontier models now generate picture and sound **in the same pass**, perfectly coincident by
construction. So a reviewer watching with sound and asking *"does the audio serve the image?"* is
not measuring the audio. They are measuring a fused percept, and the fusion is automatic.

> ⛔ **A gate that cannot return NO is not a gate. It is a sentence.**
> Every generated clip passes audio-image cohesion, because the check is run by a system that
> cannot *not* cohere them.

---

## The protocol

Three passes, **in this order**, on every video render before it is ranked, selected, or shipped.

### PASS A — MUTED. Blind to the prompt, blind to the sound.

Play the clip **with audio disabled**. The reviewer has not read the prompt.

**The only question:** ***"What happened?"*** — one sentence, event only.

Not *"is this good."* Not *"is this beautiful."* **What happened.** This is THE VINCENT TEST moved
from image to motion, and it inherits the ruling: **if the answer names an EVENT the clip works; if
it names a STYLE or a TECHNIQUE ("a slow cinematic push-in", "moody lighting", "a nice orbit"), the
clip is about its own craft and will not travel.**

If the reviewer cannot name an event, **there is no event**, and no sound design will install one.
Sound cannot rescue a clip with no trajectory. It can only weld a feeling onto it — and the weld
will feel authored to *us* and to nobody else.

### PASS B — SOUND ONLY. Blind to the prompt, blind to the picture.

Play the audio **with no picture**. Same question: ***"What happened?"***

### PASS C — TOGETHER, then against the prompt.

Only now play both. Only now read the prompt. Judge intent-survival, THE GRAIN LAW deviations, and
craft.

### The verdict rule

| Condition | Verdict |
|---|---|
| A and B name **the same event** | ✅ **AUTHORED** — the weld is real, proceed to C |
| A and B name **different events** | ⛔ **FALSE WELD** — the clip only *feels* coherent. REPAIR or cut. |
| A names a **style/technique**, not an event | ⛔ **NO EVENT** — repair the notation, not the render |
| B is empty / generic room tone under a specific picture | ⚠️ **UNEARNED AUDIO** — the sound is filler wearing sync |
| A and B both strong, C contradicts the prompt | 📓 **DEVIATION** — judge the result, not the distance from intent (THE GRAIN LAW) |

### Separation rules (these are the protocol, not politeness)

1. **Order is fixed: A → B → C.** Never C first. Once fused, the percept cannot be unfused.
2. **Different reviewer per pass is better. Same reviewer, muted-first, is the minimum defensible
   version.** Picture memory contaminates the audio pass, so if one reviewer runs both, put as much
   distance between A and B as the session allows and randomise clip order within each pass.
3. **The reviewer for A and B has not read the prompt.** If they have, they are confirming, not
   observing.
4. A model used as reviewer counts as a reviewer, and gets the same rules. Send the muted file
   alone; never send the prompt.

---

## The artifact contract

⛔ **THE EVIDENCE IS THE ARTIFACT, NEVER THE AGENT.** *"I watched it, it was fine"* is the sound a
gate makes while it fails. Every audited render writes one row to
`<run-dir>/VIDEO_RENDER_AUDIT.md`, and `lofn-qa` reads the file, not the claim:

```markdown
| clip | model | dur | PASS A (muted): what happened | PASS B (audio): what happened | A=B? | verdict | note |
|------|-------|-----|-------------------------------|-------------------------------|------|---------|------|
| p03_v2 | ltx-2.3-fast | 8s | A hand closes around a moth and opens empty | Something small is caught, then a room goes quiet | YES | AUTHORED | — |
| p05_v1 | ltx-2.3-fast | 8s | Slow push-in on a lit doorway | A door slams and a crowd reacts | NO | FALSE WELD | audio invented an event the picture never has |
```

**No audit file → the render is UNAUDITED and may not be ranked, selected, or shipped.** An
unaudited render is not a failed render; it is an unmeasured one, and it is exactly the thing we
have promised each other not to describe.

---

## The Goodhart guard

Every measure that becomes a target stops measuring. This house has already proved the local version
of that law: **monoculture relocates to the lowest unconstrained level** — camera move, then build
shape, then syntax figure, three runs running. This gate will relocate something too.

So, mandatory, mirroring the house preregistration discipline:

- **Log the prediction beside the outcome.** Before Pass A, the run records what it *expects* the
  audit to find. Afterwards it records what the audit found. That pair is the falsifiable record of
  whether this gate ever earned itself.
- **At least one pass stays un-automatable.** Pass A may be run by a model; the ranking of the
  surviving clips is The Scientist's ear and eye, always.
- **Watch for the next level down.** When every clip starts passing A and B with the same *kind* of
  event, the monoculture has moved into the event vocabulary. Say so in the note column.

---

## What this gate does NOT do

It detects **incoherence**. It does not detect **averageness** — a clip where A and B agree
perfectly and both describe something utterly generic passes this gate cleanly. That failure mode
belongs to the distinctiveness measures and to the Somatic bloc, and it is the more likely failure
of these models. **Passing the three-pass rule is a floor, not a verdict.**
