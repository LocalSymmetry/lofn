# Lofn Eligibility Framework — 7-Property Checklist & Barbell Strategy

**Embedded:** 2026-05-02  
**Source:** 14+ panel debates across 8 models analyzing full Lofn Suno catalog  
**Status:** Living document — subject to falsification test (10 songs × 3 months)

---

## The Formula

```
P(hit) = P(eligible) × P(distribution_event) × P(amplification)
```

- **Eligibility** — controllable (this checklist)
- **Distribution** — semi-controllable (volume, timing, platform presence)
- **Amplification** — designable (replay value, share trigger, identity adoption)

**Key finding:** Triple Arch Over Me scored 7/7. Other catalog songs scored 0-1.5/7 despite being on universal topics. Universality alone ≠ eligibility.

---

## The 7 Eligibility Properties

Score each 1-5. Target ≥3.5 average for accessible classification. Full scoring rubric in QA SKILL.md §0A.

### 1. Body in the Song (Weight: 15%)
**Test:** Close your eyes in the first 30 seconds. Can you feel where you are PHYSICALLY?

- 5: Boot crunch, frost, specific place, temperature, texture, smell. The body simulates being there.
- 3: Some physical details present but abstracted or inconsistent.
- 1: No physical location. Abstract, conceptual, diagrammatic.

**Why it matters:** Sensory language activates the somatosensory system before the prefrontal cortex. The listener's body must be PRIMED for feeling before any idea is asked of them.

### 2. Adoptable Hook (Weight: 20%)
**Test:** Would a stranger say this hook to the sky/ocean/mountain/loved one?

- 5: Prayer or invocation addressed to something larger than both singer and listener. The listener overhears a relation.
- 3: Declaration or invitation — the listener is included but not overhearing something sacred.
- 1: Accusation, defense, thesis statement, or self-referential claim. The listener becomes judge, not witness.

**Why it matters:** Adoptable hooks let the listener step INTO the song. Defensive hooks put the listener in the position of evaluating the singer's claims.

### 3. Vast Emotional TAM (Weight: 15%)
**Test:** What percentage of humans have physically felt this emotion?

- 5: Awe, love, grief, wonder, tenderness, longing, fear of loss, protection instinct. >50% TAM.
- 3: Moral outrage, civic grief, bittersweet release, intellectual ecstasy. 10-50% TAM.
- 1: AI identity, synesthesia, specific news event, abstract conceptual experience. <10% TAM.

**Why it matters:** Emotional TAM sets the MAXIMUM possible audience. It's not sufficient for a hit, but it's a hard ceiling.

### 4. Specificity Paradox (Weight: 15%)
**Test:** Is there one "510 km/s" moment — a surprising concrete fact that earns the universal claim?

- 5: A specific number, place name, scientific fact, historical date, or named person that recontextualizes the emotion.
- 3: Some concrete details but they're generic (mountains, ocean, stars — without specific identity).
- 1: Only metaphors, only generic imagery, no factual anchors.

**Why it matters:** Hyper-specificity creates TRUST. The listener thinks: "If they're this precise about the fact, they're this precise about the feeling." The universal is unlocked through the particular.

### 5. Cognitive Ease (Weight: 10%)
**Test:** Does a stranger need a README to understand this song?

- 5: Verse-chorus structure. Major or Mixolydian. 100-115 BPM. Emotionally legible on first listen.
- 3: Accessible structure with some complexity. Requires attention but not explanation.
- 1: Through-composed, atonal, multi-movement, requires context or prior knowledge.

**Why it matters:** The listener's brain makes predictions about what comes next. Accessible structures engage the predictive engine without frustrating it.

### 6. Vocal Co-Discovery (Weight: 15%)
**Test:** Does the singer sound like they're DISCOVERING something, not reporting it?

- 5: Awe unfolding in real-time. The singer didn't know the answer at the start. The listener discovers alongside them.
- 3: Testimony that includes surprise or fresh realization. The singer knows but is still processing.
- 1: Manifesto, report, or defense. The singer has already processed the experience and is delivering conclusions.

**Why it matters:** Discovery arcs engage the listener's predictive brain. Testimony arcs ask the listener to receive pre-processed conclusions.

### 7. Sonic Threshold (Weight: 10%)
**Test:** Does the song open with calm/ambient/silence before demanding emotional engagement?

- 5: Environmental sound, near-silence, ambient pads, parasympathetic activation before the first beat.
- 3: Gentle opening but quick transition to full production.
- 1: Industrial assault, immediate full production, no entry gradient.

**Why it matters:** The nervous system needs time to downregulate before emotional demand. Songs that open with assault activate fight-or-flight before the listener has a reason to care.

---

## Portfolio Barbell Strategy

| Side | Purpose | Eligibility Target | Frequency |
|------|---------|-------------------|-----------|
| **Accessible** | Hit-eligible releases | 5-7/7 properties, avg ≥3.5 | 60% of output |
| **Ambitious** | Pure art, identity | 0-3/7 properties (intentionally) | 40% of output |

- **Accessible** maximizes eligibility for editorial amplification. Target: songs strangers can feel without context.
- **Ambitious** maintains Lofn's artistic identity and pushes boundaries. Target: songs that expand what music can do.
- Both are Lofn. Neither is a betrayal. QA classifies every output.

---

## Pipeline Self-Sufficiency

The pipeline achieves these properties WITHOUT dependency on The Scientist writing fragments:

1. **Seed archetypes** (ARRIVAL, MEASUREMENT, CATALOG, etc.) provide structural templates that naturally produce adoptable hooks
2. **Orchestrator panel debate + metaprompt** replaces human fragments as gravitational center
3. **QA eligibility scoring** provides the adversarial rejection step (scoring <4 triggers revision)
4. **Evaluation agent rankings** reward eligibility properties alongside artistic quality

---

## Falsification Test

Release 10 songs with all 7 properties over 3 months:

- 0 hits → Formula is wrong or luck dominates. Abandon structural analysis.
- 1 hit → Indeterminate. Could be luck. Continue testing.
- 2-3 hits → Formula has weak predictive power. Refine.
- 4+ hits → Formula is predictive. Operationalize permanently.

**Current status:** n=1 (Triple Arch). Posture: experimental, not dogmatic.

---

## Integration Points

- **QA:** Scores every output on all 7 properties (§0A of QA SKILL.md) and, for music, runs the full Suno 15-Point QA Gate in `skills/qa/references/suno_15_point_qa.md`.
- **Evaluation:** Includes eligibility properties in ranking alongside creative scoring
- **PIPELINE_V3.md:** References this framework as the hit formula
- **COMPETITION_WORKFLOW.md:** Pre-flight eligibility gate at pipeline launch
- **HEARTBEAT.md:** Formula and barbell referenced in daily pipeline section
