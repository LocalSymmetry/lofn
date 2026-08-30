# STEP 08 — LYRICS DRAFT · P06 · THE CALL ACROSS THE VALLEY
**Flagship: V1 — "Kom Hem, Silja"** · measured against `vault/gates.yaml`, never eyeballed.

---

## 1. THE SHIPPED LYRIC (exact Suno field text)

```
[Theme: A herder calls a scattered summer herd home across a valley at dusk. The clean note dies at the fence; the rough one crosses.]
[SONG FORM: unmetered calls over a fifty-five hertz floor. Call long, answer short, every single time.]

[Intro - The Open Throat - EMO:Awe - Female - Solo call, no bass, free time]
*wind in dry grass; two bells, far apart, uneven*
Ho-oo! Ho-oo!
Silja!
(Silja!)
I open my throat and the hill opens.
(The hill hears.)
There is a floor under this.

[Call I - The Rough - EMO:Awe - Female - Sub enters at fifty-five hertz, mids scooped]
I put the whole ribcage right in it.
(Right in it.)
Rough as a rasp on a rail.
(On a rail.)
Clean will not clear this hill.
(Clear this hill.)
The polished note pools in the grass.
(Down in the grass.)
Mine goes ragged and mine goes over.
(Goes over.)
Look at my hands, not my throat.
(Not my throat.)
There is a hurt in here doing the hauling.
(Doing the hauling.)

[Chorus - EMO:Awe - Female + gang chant - Full sub, gang enters]
Kom hem, Silja!
(Kom hem, Silja!)
Kom hem, Silja!
(Kom hem, Silja!)
The clean call dies at the stone.
(At the stone!)
The rough burr bores to the bone.
(Kom hem, Silja!)

[Call II - The Far Side - EMO:Awe - Two female voices, distant - Mids open for the only time]
Across the valley there is another woman.
(Another woman.)
Her tune is low, her throat is thick and torn.
(Thick and torn.)
She answers short because short is enough.
(Short is enough.)
We have never stood in the same field.
(The same field.)
We have been talking a long, long while.
(A long while.)

[The Pail - EMO:Awe - Female - THE CALL IS REMOVED; sub alone; close-miked, spoken-sung]
*the lead voice stops; only the floor*
I cannot look at the thing I am doing.
(Cannot look.)
The milk in the pail is going in rings.
(Going in rings.)
My call, in a substance I can see.
(In a substance.)
Same shape. Smaller. Turned the wrong way up.
(The wrong way up.)
The rings stop at the rim.
(At the rim.)
My call does not stop at the ridge.
(At the ridge.)
Milk will not make it to the mountain.
(Not to the mountain.)
The mountain will not ring in the milk.
(Will not ring.)

[The Count - EMO:Awe to dread - Female - Names in order; a bell answers each; sub ducked]
Rosa!
(Bell.)
Blomma!
(Bell.)
Kajsa!
(Bell.)
Silja!
*grass, wind, and the gap where her bell goes*
Silja!
Her bell is behind her. She is high on the hill.
(High on the hill.)
Eighty years of calling off this same stone.
(Off this stone.)
And the how of it is still coming home.
(Still coming home.)
They have not worked out how a voice gets there.
It gets there. Go and ask the herd.
(Ask the herd.)

[Chorus - EMO:Awe - Female + gang chant - Identical return, harder]
Kom hem, Silja!
(Kom hem, Silja!)
Kom hem, Silja!
(Kom hem, Silja!)
The clean call dies at the stone.
(At the stone!)
The rough burr bores to the bone.
(Kom hem, Silja!)

[The Rough-Debt - EMO:Awe - Female - Call bare, sub swelling]
They told me to sand it smooth.
(Sand it smooth.)
They told me the burr was a fault.
(Was a fault.)
The burr in it is the whole bridge.
(The whole bridge.)
Clean clears the yard. Rough clears the ridge.
(Rough clears the ridge.)
I owe the whole winter to the worst of my voice.
(The worst of me.)
I will not be paying it back tonight.
(Not tonight.)

[Loop Until It Hurts - EMO:Awe strained - Female + gang - Chant tightens, voice frays, sub rises]
Kom hem, Silja!
(Kom hem, Silja!)
Kom hem, Silja!
(Kom hem, Silja!)
My throat is going, my throat is going.
(Going!)
Kom hem, Silja!
(Kom hem, Silja!)
Louder is the only louder I have.
Kom hem, Silja!
(Kom hem, Silja!)

[Outro - The Late Bell - EMO:Awe - Sub decaying; a bell, very far, out of time]
*a bell, far off, out of time*
There.
(There.)
She turned. She is still a long way out.
(A long way out.)
Kom hem, Silja!
(Kom hem, Silja!)
*the bell again, no closer*
```

## 2. MEASURED (`scripts/measure_soundcraft.py → profile()` on the exact text above)

| Gate | Floor / band | Measured | Verdict |
|---|---|---|---|
| Lyrics field chars | ≤5000 (target 4800) | **3848** | PASS |
| Sung lines | 70–120 (target 78–110) | **108** | PASS, in target |
| End-rhyme (±4 lines) | ≥0.30 | **0.843** | PASS |
| Line return (exact repeats) | ≥0.20 | **0.324** | PASS |
| Words per line | ≤7.5 | **4.62** | PASS |
| Alliteration / 100 w | ≥11.0 | **15.83** | PASS |
| Digits in sung lines | 0 | **0** | PASS |
| Sung number-words | exactly `eighty` | **`['eighty']`** | PASS |
| L41 residue screen | clean | **clean** | PASS |

## 3. CRAFT NOTES — what each device is doing

**The answer-echo is the rhyme engine.** The assigned architecture (call long / answer short) puts the
call's ending back in the listener's ear one beat later, which is why end-rhyme measures 0.843 without a
single forced couplet. **The form produced the sound-craft; the sound-craft was not bolted on.**

**Alliteration was engineered, not hoped for.** First pass measured **9.33**, under the 11.0 floor. The
fix was to rewrite the *repeated* lines first — a chorus line pays its alliteration back four times.
`The rough burr bores to the bone` carries three hits and appears four times; `Same shape. Smaller. Turned
the wrong way up.` carries four. Final **15.83**. ⭐ *Measure first, then rewrite the lines that repeat.*

**The one numeral.** `Eighty years of calling off this same stone.` is the only number in the mouth. It is
**responded to** by the next couplet, which converts an unsolved thing into a distant thing:
`And the how of it is still coming home. / They have not worked out how a voice gets there. / It gets
there. Go and ask the herd.` ⭐ The disappointed idealist's actual position: she does not need the
mechanism, she needs the animal.

**Where the second eye is in the words, not the mix.** `[The Pail]` states it outright — *My call, in a
substance I can see* — and then refuses the resolution in two lines that are deliberately flat and
declarative: `Milk will not make it to the mountain. / The mountain will not ring in the milk.` Neither
medium can become the other. That is the eclipse and the river held in one frame.

**Terror adjacency, kept physical.** `There is a hurt in here doing the hauling`, `Look at my hands, not
my throat`, `My throat is going`, `Louder is the only louder I have`. The danger is not weather or wolves
in the abstract — it is that **the only available way to reach further is to damage the instrument**, and
the instrument is her.

## 4. ⛔ BAN COMPLIANCE

| Ban | Status |
|---|---|
| `D4` invent the people | **Clean.** Invented herder, invented neighbour, invented herd. No real person, family or place-holder for one. |
| `D7` no song about an AI | **Clean.** No machine, no model, no understanding-as-bottleneck. |
| `D9` the kept bad thing is not redeemed | **Clean.** The rough voice is not healed and not celebrated into comfort: `I will not be paying it back tonight.` The debt stands. |
| Qusra / Ryanair / USS Abraham Lincoln | **Absent.** |
| Female vocals, no children | **Two adult female voices.** The taught-girl scenario is deliberately held back to V3 and written as a young woman, not a child. |
| ⭐ L41 — no `nobody` / `nothing` discharge | **Clean, screened in code.** Silja is *high on the hill*; the neighbour is *across the valley*; the explanation is *still coming home*. Every negative is a distance. |
| Sung numerals in words | **`eighty years`, spelled.** No digits anywhere in a sung line. |
| Kulning treated as technique, no sacred repertoire | **Clean.** A working call for moving animals, with named practitioner credit in step 10. |

## 5. REJECTED LINES, AND WHY (kept so the next run does not re-earn them)

| Rejected | Why |
|---|---|
| `The dirty one gets to the bone.` | `one` is a sung numeral. Replaced with `The rough burr bores to the bone.` — which also alliterates three ways. ⭐ *The constraint improved the line.* |
| `Nine in the yard, one on the mountain.` | Two numerals, and the count is assigned as the hinge. Forced the count-by-name device, which is better and is what herders did. |
| `Her bell is late. Her bell is not lost.` | `lost` reads as absence. Replaced with `Her bell is behind her` — the sound lagging the animal, which is also true at four hundred metres. |
| `a bell, far off, on the beat of nothing` | ⛔ L41 residue in an SFX cue. Replaced with `out of time`. |
| Making the echo the second eye | An echo is the same medium and still invisible — it fails the form rule twice. See step 07 §2. |
| Adding a code-scratch / glitch layer | ⛔ L38 **ONE SEAM**, N = 1. The bed is kulning plus one intruder. A third material would be the retired collision engine. |
