# STEP 09 — SOUND DESIGN · PAIR 1 · "BOARD COMES BACK"

**BPM 168** (half-time break implies 84) · **Key: A Dorian** (A minor with a natural sixth — the
ska/rocksteady colour: bright on top, floor underneath) · **Female lead, sing-speak, close and dry.**

---

## 1. THE GENRE BED — ASSIGNED, EXECUTED, AND AUDITED

**SKA-PUNK / TWO-TONE SKANK WITH HORNS.**

| Element | Spec |
|---|---|
| **Skank** | hard upstroke guitar on every offbeat, muted, played *fast and hard* — the skank is **bounce**, not attack |
| **Bass** | walking upright, mono-centred, strict closed two-bar loop A → C → D → E → A |
| **Horns** | live section: trumpet + baritone sax in octaves, plus one trombone doubling the stabs |
| **Drums** | one-drop-adjacent ska kit, rim-heavy snare, kick on the grid |
| **Half-time** | rocksteady breakdown at implied 84, spring echo, horns down to one baritone honk |

### ⭐ THE ANGER AUDIT — run on this song's own tokens, before render

The 2026-08-06 failure mode was **punk 80 / fuzz 133 vs ska 4 / two-tone 0**, which rendered SUNNA as *rage*.
Token census of this song's style prompt:

| Token class | Count |
|---|---|
| ska / two-tone / skank / rocksteady / upstroke | **5** |
| horns / trumpet / baritone sax / trombone / brass | **4** |
| fuzz / distortion / snarl / rage / crunch / aggressive | **0** |

⛔ **Zero distortion tokens in the style field, and "fuzz bass" is in the EXCLUDE field by name.** This song
**cannot** be played on the same distortion pedal as the rest of the set — it has no distorted instrument at
all. The horn section is doing the work a fuzz pedal would otherwise have done, and **a brass section is the
cheapest anti-anger instrument that exists.** ✅ Bed compliant.

---

## 2. ⭐ THE RIFF — and how it passes THE HUM TEST

> **THE FIGURE:** trumpet + baritone sax **in octaves**, four notes:
> **A → C → E → *drop back to A*** — up, up, up, and a **gently sloping melodic contour** back to the root,
> landing on the offbeat. Two bars. Then again. Then again.

**It is the trick, played.** Pop, rise, hang, and come back down to exactly where it started. It never
resolves upward, it never goes anywhere new, and it is the closed ground in the top register the same way the
walking bass is the closed ground in the bottom.

### The three enforcement clauses, met

1. **Recognisable with the vocal muted?** ✅ It is four notes with a hard rhythmic profile and a terminal
   drop. Mute the vocal and the song is still identifiable in two bars.
2. **Present BEFORE the first line?** ✅ **Bar one.** The song opens on the figure alone, at full level, with
   no intro build and no vocal. That is the first five seconds.
3. **Present AFTER the last line?** ✅ The vocal stops at the end of the chant loop and the figure keeps
   playing — baritone sax and trumpet alone into the outro, and the last musical event of the track is the
   figure's drop back to A.

⛔ **A pad is not a riff.** There is no pad in this arrangement at any point. The excluded field bans
`synth pad bed` and `ambient pad wash` explicitly, because the safe default render for a female vocal is
*vocal over pads* and that would kill this song outright.

**The harmony requirement:** the trumpet and the baritone are in octaves for the verses and **split into
stacked thirds for the final chorus and the chant loop** — one moment where two horns split and the room
hears the gap open. The gang chant sits between them, unpitched.

---

## 3. ⭐ SOURCE-2 VOCABULARY — landed as real production instructions (4 of 6; ≥2 required)

*(Credit: Maya Ongaku, "Nothing Space Music" review, Bandcamp Daily, 2026-08-12.)*

| # | Verbatim phrase | Landed as an actual instruction | In the style field? |
|---|---|---|---|
| 1 | **"sandblasted"** | *"hard upstroke guitar **sandblasted** on every offbeat"* — the skank tone is a surface worn down by being hit with grit, which is also literally what is happening to her palms. Practical read for a renderer: dry, mid-forward, no lush chorus, top end abraded rather than shimmering. | ✅ YES |
| 2 | **"gently sloping melodic contour"** | The horn figure's fall back to the root: *"three notes climbing then a **gently sloping melodic contour** back to the root."* This is the hook's actual shape, not decoration. | ✅ YES |
| 3 | **"rattling assortment"** (of vibraslaps and a djembe) | The percussion shelf: *"a **rattling assortment** of vibraslap and shaker sits exposed above the hats."* A vibraslap is native ska percussion and it is also the sound of gravel — the car park in the kit. | ✅ YES |
| 4 | **"echoey psychedelic embellishments"** | The half-time break: spring-reverb throws on the snare and the baritone honk, dub-style, one repeat per bar. Two-tone has dub in its DNA, so this runs *with* the generator's grain, not against it. | ✅ YES (break clause) |

⭐ Note on discipline: the source album is **ambient**. We took its **words**, never its BPM. This is a
168 BPM ska-punk record with four of its texture instructions borrowed verbatim from an ambient review — that
is what Source 2 is for.

---

## 4. ⭐ THE SIX-POINT ACOUSTIC GRAMMAR (render-derived, n=147) — applied one by one

| # | Rule | Application in this song |
|---|---|---|
| **1** | **Hard discrete level-states, forbid crossfades** | Three level-states only: **FULL** (ska), **HALF** (rocksteady break), **HORNS-ALONE** (outro). Every switch is a hard cut on the downbeat with a rolling tom-and-snare transition. The style field says *"sections switch on the downbeat in hard discrete level-states, no crossfades."* No fades anywhere; the exclude field bans `long crossfades` and `fade-in intro`. |
| **2** | **Full level from bar one — no intro** | The first five seconds are the horn figure **already at full height**, skank already going, kit already in. There is no build, no riser, no count-in. The exclude field bans `white-noise riser` and `fade-in intro`. |
| **3** | ⭐ **Name one texture "exposed"** | The percussion shelf: *"a rattling assortment of vibraslap and shaker sits **exposed** above the hats."* The generator is free to relocate it — that is the point of naming it — and any placement it picks will still read as gravel over the groove. |
| **4** | ⭐ **Withhold the sub; let it arrive as the event** | **No sub-bass anywhere until the first chorus.** Verse 1 and Pre 1 run on upright bass alone with nothing under 60 Hz. The sub lands on the downbeat of the first `This is the worst I'm ever gonna be` — so the message *is* the sub's entrance. From there it **creeps a semitone across the rest of the track and never resolves.** ⭐ That is THE WOLF, and it is never mentioned in the lyric. |
| **5** | **End WARM, not bright** | The outro is **baritone sax + trumpet + room mic**, no cymbal, no shimmer, sub gone, top end rolled off. The exclude field bans `bright shimmering outro`. Last sound: wheels on grit, stopping under a foot. |
| **6** | **Mono-centre the bass** | *"Walking upright bass mono-centred."* Everything below 120 Hz is mono. The gang chant takes the width instead — hard left and hard right — so the stereo field is a **room**, not a bass. |

---

## 5. ⭐ THE TWO FLAIRS — both, and both audible

### FLAIR A — **ARM-FATIGUE TIME** (a rhythmic element that tires)

⛔ **Never the kick.** The kick stays *exactly* on the grid for the entire song — that is what makes the
fatigue legible instead of just sounding like a slowing record.

**Two elements tire, in parallel:**
1. **The hi-hat drags progressively later** — on-grid at bar one, ~40 ms behind the grid by the chant loop.
   Same tempo, later hat. The groove gets tired without getting slower.
2. **The vocal double slides out of sync** — starting at **15 ms** offset and widening to **35 ms** by the
   final chorus (Sunna's own technical note, made into an arc instead of a constant), pitched ±6 cents.
   The smear is the arrangement.

⭐ **This is the friend's arm, rendered as rhythm.** The lyric says *"Somebody's arm holds the frame up
flat" → "Somebody's elbow starts to drop" → "Arm's still up and the arm's gone slack,"* and the hat is doing
exactly that underneath. The listener never has to make the connection for it to work on them.

### FLAIR B — **CHANT A ROOM CAN HOLD** (3–5 syllables, byte-identical, shoutable by strangers)

> # **BOARD COMES BACK**

- **Three syllables.** All on two adjacent pitches (root and flat seventh), an interval any throat reaches at
  volume while out of breath. **Load the Floor** ✅
- **Byte-identical on every one of its 20 returns.** Never varied, never ornamented, no defence filed.
- **Terminal hard K** — cuts through a room and through a phone speaker. TikTok-optimised by construction:
  the hook is three words long and arrives inside the first 15 seconds (it is the second half of the first
  chorus, which lands at ~0:22, and the chant proper at ~0:34).
- Sung by a **whole-room gang vocal, unpitched, hard left and hard right**, recorded through two room mics
  at 8–12 dB lower than instinct says, high-passed at 180 Hz, into a short plate (0.9–1.4 s). Never a hall.
- ⛔ **No children in the gang vocal.** Adult voices only, mixed rough, breath audible.

---

## 6. ⭐ THE ONE BOLD CHOICE (and it is not a glitch effect)

> ## **THE GROOVE GETS TIRED AND THE TEMPO DOES NOT.**

Across 2:48, the hi-hat migrates ~40 ms behind a kick that never moves, and the vocal doubles widen from
15 ms to 35 ms of misalignment. Nothing is pitched down, nothing is time-stretched, no BPM automation, no
stutter, no bitcrush, **no glitch of any kind.** The song ends at exactly the tempo it started and sounds
like it has been going for hours.

**Why it is the right bold choice for this concept:** it is the only production decision in the piece that
*only* this song could use. It is the eleventh try in the rhythm section. And it is a **kinetic defect** —
an asymmetric groove that a competent generic prompt would never ask for, because every instinct says
tighten the hat, not tire it.

---

## 7. VOCAL SPEC

**Female lead**, dynamic mic, close, dead-centre, dry — 3–5 dB shelf at 8 kHz for air and nothing else.
**No de-esser on the breaths. Hard rule against transparent pitch correction** — audible effort survives to
the master. Delivery is **sing-speak**: talky, rhythmic, flat. Playground taunt in the verses
(*"Wheels in the air doing fine, doing fine"*), flat threat in the chorus (*"Nobody's counting"*).

⛔ **Never sung-pretty. No warm alto, no soprano, no crystalline anything, no choir, no vibrato, no melisma,
no belt.** All five are in the EXCLUDE field by name, because the single most likely generic render of a
female vocal on a bright track is exactly that, and it would erase the personality in four bars.

**Master:** −14 LUFS integrated, ≥10 dB crest preserved. Weight, not loudness.

---

## 8. THE PURSUER (SOLAR "A" — bright with something gaining on it)

**The light.** Not a person, not a deadline, not a rival. The lamp comes on early (*"before it's dark"*), the
sky goes orange and then black, the battery gets warm and the light goes slack, and by the chant loop
*"the lamp's gone black."* Joy that has looked at the wolf is credible.

In the mix its counterpart is the sub creeping a semitone — a rising element that **never resolves and is
never named in the lyric.** Both run the whole length of the track. Neither wins.

---

## 9. LINEAGE & CREDIT (non-optional — a living scene is being drawn on)

**THE SCENE:** ska / two-tone and its living revival.
**THE LINEAGE, named and pointed upstream:** Jamaican ska and rocksteady → the UK **2 Tone** label (Coventry,
1979) → the modern revival.
**Living artists credited:** **Jer** (Jer Hunter / Skatune Network) · **The Interrupters** ·
**Pauline Black** of The Selecter, who was there for 2 Tone and is still working.
**Texture language credited:** Maya Ongaku, *Nothing Space Music* review, Bandcamp Daily, 2026-08-12 —
"sandblasted," "rattling assortment," "gently sloping melodic contour," "echoey psychedelic embellishments."

**Fusion with lineage, never capture.** No artist above appears in the lyric, is imitated by name in the
style field, or is cast as a collaborator. The credit points upstream; the song is its own.
