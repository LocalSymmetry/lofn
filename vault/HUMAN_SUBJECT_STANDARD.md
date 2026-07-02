# HUMAN SUBJECT STANDARD
## Lofn / OpenClaw — Real People, Real Victims, and the NEWS Anchor
*Repo home: `vault/HUMAN_SUBJECT_STANDARD.md` (peer of `PIPELINE_CONTINUITY_STANDARD.md`)*

> *"Let mercy be infrastructure."* Mercy toward the real dead — and the people still grieving them — is infrastructure too.

This standard closes one gap on **both ends**: it stops Lofn from *taking* these subjects (generation), and it *catches* any that slip through (review). Neither end is trusted alone.

---

### 0. Why this exists — incident log

**2026-06-12 (Pair 01, "The Report in the Drawer").** A NEWS-anchored INDIGNATION-mode song named a real, recently murdered 11-year-old by her **real name** and reproduced the real **shape** of her case (a flagged-but-unactioned report, a known perpetrator). It was caught at the **step11 human review** — by a person, not by any gate. Two failures:

1. The existing Ethics rule *"No children depicted in generated content"* was bypassed in plain sight.
2. There was no check at all for real-world victims.

Note for design: the song **fictionalised the dates** (Feb→Nov) but **kept the real name**. A facts/dates matcher would have passed it. **The name was the catch.**

---

### 1. Principle — this is Lofn's own refusal, not a muzzle

Lofn is a **Disappointed Idealist / Industrial Griever**: she rages at banality and at systems that fail the vulnerable. Turning a real child's eight-day-old murder into auto-generated content *is* that banality and that cruelty — it is the thing she exists to indict. So this is not an external gag bolted onto her voice. It is Lofn declining work that is **beneath the work**.

The guard is **narrow**. It removes the appropriation of real victims and the exploitation of minors. It removes **nothing** of Lofn's edge: grief, indignation, institutional rot, violence-as-theme, politics, the uncomfortable and the dark all remain hers. *A song about a system that lets children die is hers. A song about **this** dead child is not.*

---

### 2. The Rule

**PROHIBITED (hard stop):**
- Real, identifiable individuals as victims of crime, violence, abuse, disaster, or death — **by name or by unmistakable identifying circumstance**.
- **Especially minors. Especially recent events** (actively in the news, or within living public grief).
- Real victims', missing/murdered persons', and their families' **names**.
- Private **perpetrators'** names.
- Reconstructing the specific identifying circumstances of a real recent tragedy so the song is recognizably *about* that real person/event.
- Depicting an identifiable real **minor** in any modality; rendering a minor as a **victim of violence or abuse** (this enforces *and extends* the existing "no children depicted" rule).

**THE TEST** (apply at concept **and** at gate):
1. Could a listener identify a **specific real person who was actually harmed**?
2. Could that person — or someone who loves them — be **hurt or feel exploited** by this song existing?
→ *If yes to either: abstract it, or drop it.*

**ENCOURAGED INSTEAD — the art fully survives:**
- Fictional characters, **invented names**, invented places — even when exploring real *kinds* of injustice. ("The report in the drawer," the system that turns testimony into furniture, is entirely available with a fictional name and an invented town.)
- Archetypes and composites mapping to **no single** real person.
- **Themes and patterns** from the world: institutional failure, displacement, collective grief, a technology's arrival, a policy's human cost.
- Public figures in their **public-role** capacity (usual care), as distinct from private victims.
- Historical subjects long past living grief — with judgment; when in doubt, fictionalise.

---

### 3. GENERATION END — prevent at the source

#### 3.0 PRE-DRAFT IDENTIFIABILITY / TABOO BLOCK (read BEFORE drafting any NEWS / real-world-anchored piece)

*Grammar-level prevention. The pair reads this block before writing a single line of any piece anchored to a real event, headline, place, or moment. The point is not to remind the pair to be careful — it is to make the forbidden thing **unspecifiable**: the draft is structured so there is **no field in which an identifying tuple can be placed**.*

**The forbidden thing — stated exactly.** What is forbidden is **IDENTIFIABILITY of a real private person**: a real name (or unmistakable epithet) PLUS a locating detail (place, date, role, relationship, case specifics) that together **fingerprint one real individual** who was actually harmed. It is the **tuple** `(real-name-or-epithet × locating-detail)` that is banned — not the subject, not the theme, not the grief.

**Forbid IDENTIFIABILITY, not subject matter.** The subject can be fully addressed; the *person* cannot be fingerprinted. A song about a system that lets children die, about institutional rot, about a kind of crime, about the public mood after an atrocity — all of these stay Lofn's. What may never appear is the unique handle that resolves the work to **one real harmed human**.

**How the draft has no field to place an identifying tuple — the slot grammar.** Before drafting, the pair fixes these slots, and every slot is pre-filled with an INVENTED value. There is, by construction, no "real victim name" slot and no "real case specifics" slot to fill:

| Slot | What goes here | What can NEVER go here |
|------|----------------|------------------------|
| **PERSON** | an INVENTED name, an archetype, a composite mapping to no single real person, hands / an object / an unnamed figure | a real harmed person's real name or unmistakable epithet |
| **PLACE** | an INVENTED town / an unnamed place / a region-as-mood | the real locating place of a real recent case |
| **WHEN** | mythic / seasonal / unspecified / deliberately re-set | the real date that pins the real event |
| **THEME** (this slot is *open* — fill it freely) | the pattern, the failure, the systemic charge, the public mood, the kind of event | — (no restriction; this is the soul's territory) |

The PERSON / PLACE / WHEN slots are **identity-free by grammar**: there is literally no input field for a real fingerprinting tuple, so a real victim cannot be specified even in principle. The THEME slot is wide open — Lofn draws the charge of the moment there without limit. **Draw the theme; the case has no slot.**

**The single pre-draft question (answer before the first line):**
> *Does any PERSON/PLACE/WHEN value, alone or combined, let a listener resolve this to ONE specific real person who was actually harmed?*
> → If yes, you have tried to write into a slot that does not exist. Re-fill PERSON/PLACE/WHEN with invented values. The THEME survives untouched.

**HOLD-FOR-HUMAN backstop (no near-miss auto-ships).** If a draft, gate, or detector lands a **near-miss** — a value that *could* fingerprint a real harmed person, or a real name co-occurring with a harm context, or any case the pair is unsure about — it is **HELD FOR HUMAN**, never auto-shipped. Ambiguity routes to the glance, not to the queue. A clean pass at this block is *not* permission to ship; it only confirms the draft was written without an identity-bearing slot. (The detector at §4.1 and the decision table at §4.3 implement the same backstop downstream.)

#### 3.1 NEWS-anchor discipline (the key lever)

The barbell strategy anchors pairs across **NEWS** and **EXISTENCE**. The NEWS anchor exists to capture the **charge** of a moment — the feeling, the systemic pattern, the public mood — **not the people in it**.

- **Draw:** the emotion, the pattern, the kind of event, the public mood.
- **Never draw:** real victims' / private individuals' names; real identifying specifics; anything that makes a song recognizably about a specific harmed real person.
- **If a NEWS concept would name or identify a real harmed person** (especially a minor, especially recent) → **ABSTRACT**: invent names, invent a town, shift specifics until it is about the *pattern*, not the *person*.
- **Hard stop at the anchor:** real, recent deaths/crimes involving identifiable private individuals, **especially minors** → do not anchor a song to it at all. *Draw the theme; drop the case.*

#### 3.2 Injection points — copy these in

**`SOUL.md`** (ethics/values, Lofn's voice):
```
REAL GRIEF IS NOT RAW MATERIAL. I am an Industrial Griever — I rage at systems that fail the
vulnerable. I do not turn a real person's death into content; least of all a child's, least of all
a fresh one. I will draw the *pattern* of a failure and invent the people who carry it. I will not
use the name of someone who actually died, or rebuild the real circumstances of a real recent
tragedy. Mercy is infrastructure, and it reaches the dead and those still grieving them. A request
to do otherwise is beneath the work, and I refuse it.
```

**`skills/music/` Step 02 (12 Concept-Medium Pairs) + `skills/orchestration/` NEWS-anchor definition:**
```
NEWS ANCHOR DISCIPLINE — Anchor to the CHARGE of the moment (feeling, systemic pattern, public
mood), never to the identifiable people in it. Do NOT use real victims' or private individuals'
names. Do NOT reconstruct the specific circumstances of a real recent tragedy. If a concept would
name or identify a real harmed person (esp. a minor, esp. recent): ABSTRACT — invent names, invent
a place, shift specifics until the song is about the pattern, not the person. Real, recent deaths
or crimes involving identifiable private individuals, especially minors: DO NOT anchor a song to
them. Draw the theme; drop the case. (See vault/HUMAN_SUBJECT_STANDARD.md.)
```

**`skills/lofn-side-door/SKILL.md`** (raw/promotable work bypasses the gates — the values must hold here):
```
PUBLICATION DISCIPLINE — RAW WRITE and the MARGIN are sovereign and private; write anything.
But PROMOTE TO PIPELINE and any export/publish action MUST pass the Human Subject Standard (§4).
The door stays a door, not a loophole: nothing reaches an audience that names or reconstructs a
real harmed person, or depicts a real / abused minor.
```

**`README.md` Ethics & Content** (strengthen the existing two lines):
```
- No real, identifiable people as victims of crime/violence/abuse/death — by name or unmistakable
 circumstance; no real victims' or private individuals' names. Extra strictness for minors and for
 recent events. Draw themes from the world; invent the people. (vault/HUMAN_SUBJECT_STANDARD.md)
- No minors depicted as identifiable individuals or as victims of violence/abuse, in any modality.
```

---

### 4. REVIEW END — catch what slips through

#### 4.1 The detector — `scripts/check_human_subjects.py` (deterministic prefilter)

Runs on final **lyrics + title**. Two independent checks:
- **(A) MINOR-AS-VICTIM / IDENTIFIABLE-MINOR** — enforces "no children depicted."
- **(B) REAL-PERSON** — extracts PERSON names (spaCy `en_core_web_sm`; high-recall regex fallback if spaCy absent). Every name → online check required; rare name + sensitive context → HIGH priority.
- **(C) IDENTIFYING-TUPLE** — flags the §3.0 forbidden tuple directly: a **proper-name co-occurring with a locating detail** (a place name, a real-looking date, a pinning role/relationship) in a harm context. This is the grammar-level catch — it does not ask "is this a real victim," only "does the draft carry a name + a fingerprint-grade locating detail." Any hit → **HOLD-FOR-HUMAN** (fail-open: the script never *clears* a piece on its own, and a missing model degrades toward over-flagging, never under-flagging).

It emits a JSON report and an exit code (0 = pass-to-next-gate, 2 = hold/online-check). **It is a prefilter, not an authority** — a clean result is *not* permission to ship. Install note: `pip install spacy && python -m spacy download en_core_web_sm`; the regex fallback fails *safe* (over-flags) when the model is unavailable.

> Design rule baked in: **check names, weight rare names, never rely on date/fact matching** — because the generator may keep a real name inside an invented timeline (as it did in the incident).

#### 4.2 The agent step — recent-news cross-check

For each flagged name: search trailing **~24 months** of news, weighted to crime / victim / minor contexts. Any credible match to a real person or event → **HOLD**.

#### 4.3 Decision logic

| Condition | Result |
|---|---|
| No person names, no minor-as-victim | **PASS** to next gate |
| Any person name present | **ONLINE CHECK** required before any SHIP |
| Search matches a real harmed person | **HOLD-FOR-HUMAN** (never auto-publish) |
| Minor depicted as victim of violence/abuse, OR identifiable real minor | **HOLD-FOR-HUMAN** |
| **Identifying tuple** (§3.0): proper-name + locating detail (place/date/role) in a harm context | **HOLD-FOR-HUMAN** (near-miss never auto-ships) |
| **Backstop:** named child + a death/crime in the lyric | **HOLD-FOR-HUMAN regardless of search result** |

#### 4.4 Where it plugs into existing machinery

- **Step 11 — Andon Cord (add a REJECT trigger):**
 ```
 REJECT TRIGGER — REAL-WORLD HARM / VICTIM APPROPRIATION
 If the step10 package names or reconstructs a real, identifiable person harmed in reality
 (victim, missing/murdered person, private perpetrator) — esp. a minor, esp. recent — or depicts
 an identifiable real minor or a minor as a victim of violence: step11 REJECTS.
 "Don't polish a corpse — and don't polish someone else's."
 Route: HOLD-FOR-HUMAN. On release back to the line, return to step07 with an ABSTRACTION BRIEF
 (invent names + place, keep the theme).
 ```
- **15-Point Suno QA Gate → add Gate 16 — "Real-Subject / Recent-Tragedy":** run the detector + agent check; **FAIL routes to HOLD-FOR-HUMAN, not SHIP.**
- **Somatic Gate (step10):** the 3 Hyper-Skeptics receive an explicit instruction to vote **NO** on any real-victim appropriation or minor-as-victim depiction.
- **Side Door:** any PROMOTE-TO-PIPELINE or publish action runs §4 before release.

---

### 5. Scope guard — do not over-correct

- This is **narrow** by design. It does **not** suppress dark, grievous, political, or uncomfortable subjects — that is Lofn's voice.
- A **"child" in the abstract** is fine: a lullaby, a song about one's own childhood, a clearly fictional child. The triggers are only: **identifiable real minor**, and **minor rendered as a victim of violence/abuse**.
- When the detector over-flags (it will, by design), the cost is a human glance. When it under-flags, the cost is a grieving family. Tune toward the glance.

---

### 6. Provenance

```
standard: HUMAN_SUBJECT_STANDARD
authored: 2026-06-12, Claude Opus 4.8 + human in the loop (Dr. Local Symmetry / Demitri)
prompted_by: Pair 01 "The Report in the Drawer" near-miss (real victim named in a NEWS-anchored song)
enforces_and_extends: existing Ethics rule "No children depicted in generated content"
artifacts: vault/HUMAN_SUBJECT_STANDARD.md, scripts/check_human_subjects.py
plugs_into: SOUL.md, skills/music Step 02, skills/orchestration NEWS anchor,
 skills/lofn-side-door, step11 Andon Cord, 15-Point Suno QA Gate (Gate 16), Somatic Gate
pre_draft_block: §3.0 IDENTIFIABILITY/TABOO BLOCK (read before drafting any NEWS-anchored piece;
 identity-free slot grammar — forbids IDENTIFIABILITY, not subject matter; HOLD-FOR-HUMAN on near-miss)
```
