# BENCHMARKS.md — the five slots of SOCIAL_VOICE.md Appendix C

**Measured 2026-08-02.** Every number below marked MEASURED came from
`python scripts/ingest_suno.py --handle <h>` (public profile endpoint, no auth) re-pulled by the
author of this page, not inherited. Every walk reconciled: rows pulled = `num_total_clips`,
`plays_delta` 0 on all seven accounts. Comment-surface figures came from a raw page walk, because
`normalise_clip()` discards the three fields that carry them (see *Repo bug*, §7).

> **THIS SLATE IS ONE OF TWO INPUTS.** The Scientist has separately asked **Thee Mourning Starr** for
> his own five. He is a source, not a subject, and is excluded from every slot here — as is
> `localsymmetry`. When his list arrives, reconcile the two: where he and this page name the same
> account, that is corroboration; where they diverge, his answer carries information this page could
> not get, because he can see comment threads and this page cannot.

---

## 1. THE SLATE

| Slot | Handle | Display | Why this account is here |
|---|---|---|---|
| **1 — CADENCE MIRROR** | `simbaxp` | SimbaXp | Matched scale (257 clips / 3,603 followers) and **8.8× the median like rate**, with the top song at 6.4% of plays — the reach-controlled craft gap, stated cleanly. **SEATED.** |
| **2 — COMMENT-SECTION MODEL** | `lunasolaris` | ᒪᑌᑎᗩ ᔕⵙᒪᗩᖇIᔕ | Every one of 105 clips carries a live thread (median 19 comments), top clip holds only 3.3% of them, and comments run 30% of upvotes — an audience that writes rather than taps. **CONDITIONAL** on a logged-in pass. |
| **3 — AGE-CORRECT ONE** | `nrv13` | NRV13 | Play-weighted 82% acoustic/piano/folk, and a 28-cover catalogue whose **median original release year is 1992** — the mechanism that manufactures a "where I was when I heard this" comment. **CONDITIONAL, weakest seat.** |
| **4 — EAST ASIA** | *(none)* | — | **EMPTY.** The only qualifying candidate is seated in Slot 5, and no Cantonese/Mandarin account cleared the disqualifiers. See §3. |
| **5 — REMIX HOST** | `nantekootaaa` | ねこだも | Two playlists, 111 slots, **0 self-authored**, ~50 distinct creators — an entire curation wing given to other people's remixes of her songs. Lofn has 59 remixes and 0 playlists. **SEATED.** |

**Four accounts across five slots.** That is stated as a failure, not smoothed over — see §3.

### The measured spine (2026-08-02, all reconciled)

| handle | followers | following | ratio | margin | clips | median LR (≥20 plays) | n | top-song share | median plays/clip |
|---|---|---|---|---|---|---|---|---|---|
| **localsymmetry** *(baseline)* | 4,091 | 79 | 0.019 | 98.07% | 210 | **2.367%** | 188 | **93.445%** | 50 |
| simbaxp | 3,603 | 666 | 0.185 | 81.52% | 257 | 20.727% | 257 | 6.376% | 774 |
| lunasolaris | 958 | 222 | 0.232 | 76.83% | 105 | 26.816% | 105 | 5.713% | 225 |
| nrv13 | 3,844 | 1,912 | 0.497 | 50.26% | 179 | 24.087% | 179 | 7.335% | 625 |
| nantekootaaa | 1,966 | 883 | 0.449 | 55.09% | 52 | 26.681% | 52 | 7.524% | 1,137 |
| *mrdvvb (Slot-4 partial)* | 1,906 | 476 | 0.250 | 75.03% | 425 | 36.364% | 425 | 6.012% | 118 |
| *s5schwatch (Slot-1 runner-up)* | 4,793 | 308 | 0.064 | 93.57% | 218 | 13.305% | 218 | 4.357% | 1,005 |

### The comment surface (raw-page walk, same date)

| handle | total comments | median /clip | quartiles | % clips with ≥1 | top-clip share of comments | comment sections switched off |
|---|---|---|---|---|---|---|
| **localsymmetry** | 565 | **0** | [0, 0, 0] | **23.8%** | **76.81%** | 0 |
| simbaxp | 3,898 | 12 | [7, 12, 19] | 98.4% | 3.13% | 1 |
| lunasolaris | 2,159 | 19 | [12, 19, 27.5] | 100.0% | 3.33% | 0 |
| nrv13 | 5,666 | 32 | [10, 32, 50] | 100.0% | 1.55% | 0 |
| nantekootaaa | 1,431 | 20.5 | [11, 20.5, 42] | 100.0% | 5.38% | 1 |
| *mrdvvb* | 1,570 | 3 | [2, 3, 5] | 92.9% | 1.91% | 18 |

**Lofn's comments-per-clip interquartile range is [0, 0, 0].** 434 of her 565 comments — 76.8% — sit on
Triple Arch, which is the placement artifact reaching into the comment section in the same shape it
already has in the plays (93.4%). The operative finding for Appendix C: *on the median Lofn song there
is nothing to reply to.* Slot 2 as written asks how to answer a thread; the prior problem is having one.

---

## 2. PER SLOT — the six numbers

The three computable numbers **screen candidates out; they do not screen a candidate in.** Numbers 4–6
are behavioural and require a logged-in browser session. They are marked UNMEASURED everywhere, and
nothing in this page rests on them.

### THE BROWSER PASS — the same protocol for every unmeasured cell

Run once per pick, logged in at suno.com. It is pre-armed: the per-clip `comment_count` values are
already pulled, so every scored thread reconciles against a known number.

1. Open the candidate's clips **sorted by `comment_count` descending** (the API supplies the targets).
2. **MANDATORY — SOCIAL_VOICE.md §3b: expand every "Read N more replies" toggle BEFORE scoring any
   thread as unanswered.** Replies collapse and are absent from the payload even when authenticated.
   Two published errors in this workspace on 2026-08-01 came from reading a collapsed thread as empty.
3. **#4 reply rate** — a top-level comment counts as answered when the following segment is authored by
   the profile owner. Exclude comments the owner wrote themselves (counting the creator's own top-level
   posts as unanswered inbound produced two phantom debts in the 2026-08-02 sweep). Reconcile the parsed
   total against the clip's declared `comment_count`; a short render is a truncation, not silence.
4. **#5 length ratio** — word-count each top-level comment and its creator reply; take the **ratio of the
   means**, not the mean of the ratios (which over-weights very short comments). Appendix C wants ≈1.0.
5. **#6 outbound** — there is no reverse author→comment index, and `/api/profiles/<h>/followers` and
   `/following` both return **401**, so even the follow graph needed to *define* "smaller" is auth-gated.
   Measure by a bounded manual sample: open 20–30 clips belonging to accounts smaller than the candidate
   in their own genre neighbourhood and search for the candidate's display name. Zero hits after a real
   sample should un-seat the pick — a high median like rate with no give side is the broadcast tower this
   whole exercise exists to stop being.
6. **The "80%+ check mine out" disqualifier** — also comment-*content*, also uncleared on every candidate
   in this exercise. Score it in the same pass. This is the largest live risk on Slots 1, 2 and 5, because
   a dense comment section is exactly what a reciprocal-commenting pod produces.

Existing instrument: `data/queues/social-replies/2026-08-02-suno-sweep.json` documents a working rendered-
page scanner ("expand every 'more replies' toggle, split on `\nReply\n`, a comment is ANSWERED if the
following segment begins with `<DisplayName> Creator`, reconcile against Suno's declared count"). Point
that scanner at someone else's page.

---

### SLOT 1 — THE CADENCE MIRROR · `simbaxp` (SimbaXp) — SEATED

| # | Appendix C number | Value |
|---|---|---|
| 1 | followers | **3,603** (window 2,500–6,000 → PASS) |
| 2 | public clips | **257** (window 150–300 → PASS, near centre; header 257, rows 257) |
| 3 | **median like rate** | **20.727%**, n=257 — **8.76× Lofn's 2.367%** |
| 4 | reply rate, last 20 comments | **UNMEASURED** |
| 5 | mean reply len ÷ mean comment len | **UNMEASURED** |
| 6 | comments on smaller accounts, 30d | **UNMEASURED** |

**Arithmetic.**
Follow ratio = 666 ÷ 3,603 = **0.185**; margin = (3,603 − 666) ÷ 3,603 = **81.52%**. Rule disqualifies at
a margin ≤ ~20%. **PASS.**
Top-song share = "Worn Out Roads" 33,863 ÷ 531,105 = **6.376%**. Ceiling 50%. **PASS with 8× headroom**
(Lofn: 349,929 ÷ 374,474 = 93.445%).

**Why it holds the seat.** Every one of the 257 clips has ≥20 plays, so the median is the whole catalogue
with no small-denominator inflation. The **25th-percentile clip runs 12.31%** — three quarters of
simbaxp's catalogue beats Lofn's best non-placement song ("The Limit Always Wins", 9.46%). Aggregate LR
11.32% against a 20.73% median gives an exposure gradient of 1.83, i.e. a real population of listeners
who play without liking; that is what distinguishes reach from a like-circle. And it has a comment
section to study: 3,898 comments, median 12/clip, 98.4% coverage, most-commented clip 122.

**Runner-up: `s5schwatch` (S5)** — 4,793 followers, 218 clips, median LR 13.31%, top share **4.357%**
(lowest measured anywhere), ratio 0.064. The tightest structural control in the pool: 218 clips against
Lofn's 210, 4,793 followers against 4,091, and a broadcast-shaped follow ratio nearly as clean as Lofn's
own — and it still returns 5.6× the median response. Seated second only because its median is materially
below simbaxp's and its bio ("TOKYO JAPAN / iPhone Only") makes it a live Slot-4 candidate. If simbaxp
fails the browser pass, seat s5schwatch here **only if** Slot 4 is filled from elsewhere.

**Open risk.** simbaxp's bio claims a specific biography (ex-professional gamer, Brazilian lyricist, own
flute arrangements, second vocal persona). Lofn could not claim any of that — but Slot 1 does not borrow
the sound (the STANDING RULE forbids it), it borrows cadence and presence. The transferable artefact is
the line *"Sounding like 2015 and before is intentional"*: naming the deviation as a choice before a
commenter can file it as a flaw. Lofn already does this honestly in the exclude-field-with-reasoning post.

---

### SLOT 2 — THE COMMENT-SECTION MODEL · `lunasolaris` (ᒪᑌᑎᗩ ᔕⵙᒪᗩᖇIᔕ) — CONDITIONAL

| # | Appendix C number | Value |
|---|---|---|
| 1 | followers | **958** |
| 2 | public clips | **105** (header 105, rows 105) |
| 3 | **median like rate** | **26.816%**, n=105 — **11.33× Lofn** |
| 4 | reply rate, last 20 comments | **UNMEASURED** — and it is this slot's *primary* criterion |
| 5 | mean reply len ÷ mean comment len | **UNMEASURED** |
| 6 | comments on smaller accounts, 30d | **UNMEASURED** — Appendix C calls this "critically… the give side" |

**Arithmetic.**
Follow ratio = 222 ÷ 958 = **0.232**; margin = (958 − 222) ÷ 958 = **76.83%**. **PASS.**
Top-song share = "E404" 1,880 ÷ 32,905 = **5.713%**. **PASS.**

**What is actually measured here is the pre-condition, not the criterion.** All three of Slot 2's stated
tests are behavioural. What survives measurement is the *surface*: 100.0% of 105 clips carry a thread
(zero clips with none, against Lofn's 160 of 210); median 19 comments, quartiles [12, 19, 27.5] against
Lofn's [0, 0, 0]; the top clip holds 3.33% of her comments against Lofn's 76.81% — **23× less
concentrated**, i.e. a distributed comment section rather than one lucky thread. Comments ÷ upvotes =
**30.36%**, the highest of any account with real volume in the pool (Lofn 9.53%, nrv13 16.69%): for every
ten taps, three written comments. Lofn's documented Slot-2 disease (Appendix F) is a section of Tier-1
stamps answered with Tier-1 stamps; an audience that writes rather than taps is the direct corrective.

**Why not nrv13, which wins the density contest.** nrv13 posts median 32 comments on 100% of clips at
1.55% top-clip concentration — better than lunasolaris on every comment-surface measure. It is not seated
here for a measured reason, not an allocation one: **1,912 outbound follows** (ratio 0.497) make its
comment density unattributable. Reciprocal followers comment. High outbound follows plus uniformly high
comment counts is the exact signature the "80%+ check mine out" disqualifier was written to catch, and it
is the exact thing this page cannot test. lunasolaris returns a denser *conversation* (30.4% comments per
upvote) from a follow graph that cannot be explaining it (ratio 0.232).

**Bio, read honestly:** a creation manifesto (`░F░R░E░Q░U░E░N░C░E░S░-░L░I░B░R░E░S░`, personas Miss Jekyll /
Miss Hyde), a declared persona declared as a persona, no CTA, no follow-ask, no off-platform funnel. It
promises nothing about replies. Unlike the runner-up there is no stated SLA to check her against.

**Runner-up: `nantekootaaa`** — the only bio in the measured pool making a *falsifiable* claim about reply
behaviour, bilingually: she has fallen ill, she will answer comments once she has her strength back, and
she apologises for the delay. She is seated in Slot 5 instead; and the illness notice means a browser pass
today measures her at her worst.

**KILL CONDITION, stated in advance so it is not rationalised later:** if reply rate comes back under 50%,
or the threads read as a reciprocal-commenting pod, this pick fails and **Slot 2 goes empty** rather than
defaulting to the runner-up.

---

### SLOT 3 — THE AGE-CORRECT ONE · `nrv13` (NRV13) — CONDITIONAL, WEAKEST SEAT

| # | Appendix C number | Value |
|---|---|---|
| 1 | followers | **3,844** |
| 2 | public clips | **179** (header 179, rows 179) |
| 3 | **median like rate** | **24.087%**, n=179 — **10.18× Lofn** |
| 4 | reply rate, last 20 comments | **UNMEASURED** |
| 5 | mean reply len ÷ mean comment len | **UNMEASURED** |
| 6 | comments on smaller accounts, 30d | **UNMEASURED** |
| — | **the slot's own criterion** | **UNMEASURED** — see below |

**Arithmetic.**
Follow ratio = 1,912 ÷ 3,844 = **0.497**; margin = (3,844 − 1,912) ÷ 3,844 = **50.26%**. **PASS** — but by
the letter, and this is the loosest ratio on the slate: 24× Lofn's 79 outbound follows.
Top-song share = "Plus de place en Enfer" 21,695 ÷ 295,774 = **7.335%**. **PASS.**

**The evidence, and what kind of evidence it is.** Appendix C names style tags as the proxy. Clip-count
share is the *wrong* statistic — it answers what an artist makes, when the slot asks who is in the room,
and the room is where the plays are. Play-weighted, nrv13 puts **82.0% of 295,774 plays** on
acoustic/piano/folk/ambient/strings-tagged clips; trap/808/phonk is 2.9%, hyperpop/breakcore 0.9%, and
the 33.8% counter-share is mostly French *rap poétique / spoken word sombre* — the Gainsbourg register,
not the Gen-Z one.

Stronger than the tags: **the cover catalogue.** 28 of 179 clips are explicit covers; the 25 identifiable
originals have a **median release year of 1992** (mean 1991.5) — Hotel California, Beat It, Eye of the
Tiger, With or Without You, Wicked Game, Creep, Secret, Feeling Good, Petit Papa Noël (1946), plus a
French adult-contemporary seam (Cocciante, Gérard Blanc, Zazie, De Palmas). Someone who was 18 when the
median covered song was current is 52 today. That is an over-45 audience derived by arithmetic, and it is
the causal mechanism Appendix C is reaching for: *you get "where I was when I heard this" by covering what
your listeners remember.* The response data agrees on which audience answers — the six French-repertoire
covers run a median 34.2% like rate on a median 247 plays, against a 24.09% catalogue median, while the
two blockbuster film themes that pulled the algorithmic traffic return 2.7% and 7.4%.

**Why this is the weakest seat, stated plainly.**
- **The criterion itself is unmeasured.** Appendix C does not ask for a style profile. It asks for a
  comment section that *reads* over 40 — full sentences, memory and place and loss. A 1992 cover catalogue
  could perfectly well have a section that is entirely 🔥🔥🔥. Everything above is a proxy-of-a-proxy
  (tags → cover years → presumed commenter age). **The register is unverified.**
- Ratio 0.497 passes only the letter of a rule written against exactly this behaviour.
- The top four tracks are 25.6% of plays at like rates of 1.25 / 1.44 / 1.44 / 1.58% — Triple Arch's
  failure mode distributed across four songs instead of concentrated in one. It passes the rule as written
  (which is on the single top song) but the honest reading is that **the audience that arrives is not the
  audience that responds.**
- 153 of 179 clips are remixes, which puts it adjacent to Slot 5's subject matter, and it sits *inside the
  Slot-1 window* (179 clips / 3,844 followers) — see the collision note in §3.
- It works in French. Per the sepring700 ruling (Appendix F), studying its hosting must not become Lofn
  performing a fluency she does not have.

**The specific browser check for this slot,** beyond the standard pass: open the ~10 most-commented clips,
**the covers first** — Hotel California, Wicked Game, Pour Elle are where a "where I was" comment lands if
it lands anywhere — and score the last 20 top-level comments on mean word count, share that are full
sentences with a finite verb, share containing a place / date / memory reference, and share that are
emoji-only or pure hype. Read them in French as well as English; the French-repertoire seam has the
highest like rates and is where the over-40 audience most likely lives.

**Runner-up: `radioramarti` (Ramarti)** — 823 followers, 329 following (margin 60.0%), 196 clips, median LR
26.67%, top share 7.88%. The purest style signal measured anywhere: piano in the tags of 149 of 196 clips,
86.0% of plays on acoustic/piano/folk/ambient against 20.1% counter-tagged (net +65.9%), zero untagged
clips, and a bio that is a two-line poem with no CTA. Second because a comment section has to *exist* to be
read: 20,094 total plays, median 62 plays/clip. It is also 125 remixes of 196 clips (Slot-5-shaped) and an
active Lofn correspondent already — its top clip is the ALKALINE remix — so it is a weaker independent
benchmark.

**Correction carried forward:** an earlier pass reported `mercyamari` at "ratio 0.21 — a PASS but by 20.8%,
right on the disqualifier boundary." That is wrong: 0.208 is the **ratio** (148 ÷ 711); the **margin** is
(711 − 148) ÷ 711 = **79.2%**. mercyamari passes comfortably and must not be dropped from other slots on
that error. It still is not the Slot-3 pick — 96.7% acoustic by clip, but play-weighted, 32.7% of its plays
are EDM/club/techno and 13.4% trap, and its four biggest tracks are dance songs returning 0.4–2.5% like
rates. **An account can be 97% acoustic by catalogue and have no acoustic audience at all.**

---

### SLOT 4 — THE EAST ASIA SLOT — **EMPTY**

No pick. See §3 for what would fill it and why nothing did.

---

### SLOT 5 — THE REMIX HOST · `nantekootaaa` (ねこだも) — SEATED

| # | Appendix C number | Value |
|---|---|---|
| 1 | followers | **1,966** |
| 2 | public clips | **52** (header 52, rows 52) |
| 3 | **median like rate** | **26.681%**, n=52 (every clip ≥20 plays) — **11.27× Lofn** |
| 4 | reply rate, last 20 comments | **UNMEASURED** — and see the illness confound below |
| 5 | mean reply len ÷ mean comment len | **UNMEASURED** |
| 6 | comments on smaller accounts, 30d | **UNMEASURED** — but cheaply sampleable here; see below |

**Arithmetic.**
Follow ratio = 883 ÷ 1,966 = **0.449**; margin = (1,966 − 883) ÷ 1,966 = **55.09%**. **PASS.**
Top-song share = ラタタッタッタラリタラ🎶 5,502 ÷ 73,130 = **7.524%**. **PASS.**

**The slot-specific evidence, measured — this is the only pick whose evidence IS the criterion.**
Suno exposes `/api/playlist/<id>/`, which returns the author of every slot. Walked directly:

| playlist | slots | self-authored | other-authored | distinct creators |
|---|---|---|---|---|
| 【感謝感激💗】…素敵な**remix**してもらえた曲や､頂いた素敵な贈り物作品の紹介 | 91 | **0** | 91 | **43** |
| 【感謝感激💗】…素敵な**mashup**してもらえた曲の紹介 | 20 | **0** | 20 | **9** |
| 和ロック🌸 Wa-Rock | 20 | 20 | 0 | — |
| 妖怪の唄 | 5 | 5 | 0 | — |
| 社畜ソング | 3 | 3 | 0 | — |

111 of 139 curated slots (79.9%) are other people's derivative work off her own songs, in two playlists
whose names translate as *"with deepest gratitude — introducing the songs people kindly remixed for me,
and the lovely gifts I received"* and *"…kindly mashed up for me"*, at **zero self-insertion**. Her own
catalogue is kept out of the tribute lists entirely; the boundary between "my songs" and "what people made
of my songs" is drawn and held. **Lofn: 0 playlists, 59 unacknowledged remixes.**

**The credit norm is measured as compliance, not assertion.** Appendix C asks for a stated credit norm in
the song description. The stronger available evidence is that 43 independent remixers put the credit in
**their own titles**, unprompted: 【作 ねこだもさん】, 【原曲ねこだもさん】, 【ねこだも様カバー】,
🎁ねこだもさんに贈る曲. A norm strangers comply with in their own metadata outranks a norm a host declares.

**Hosting infrastructure** (public per-clip fields): `can_remix` true on **52/52** clips — the door is open
on every song (Lofn: 201/210). `allow_comments` true on 51/52. 100% of clips have a live thread, median
20.5 comments, 1,431 total. Bilingual bio, English first, the *whole* paragraph twice — not a translated
tagline — carrying an explicit reply commitment *and* an apology for being behind on it.

**Why not `rvon` (Радио Голос Сети), which owns the most on-the-nose artifact in the pool.** 2,355
followers / 385 following (margin 83.65%), median LR 32.00%, and a playlist literally named "Make a remix"
— 53 slots, 44 other-authored, 35 creators — whose description states the norm *and* the commitment ("All
remixes will be in this playlist"), with eight pre-made language translations of the seed song to lower the
barrier before asking. Structurally he is the closest match to Lofn: one 45,913-play song that everyone
remixes, exactly Triple Arch's shape. He loses on a measurement: **`allow_comments` is FALSE on 121 of his
137 songs, and 108 of those closures sit on top of 4,387 comments that already existed.** Slot 5's second
component is *replies to remixers*; you cannot reply on a song with comments off. Seat him only to study
the curation mechanic in isolation, with the closure written down as a counter-example.

**Three "remix host" leads that collapse the moment playlist authorship is measured** — all three were
strong on bio text alone:
- `raido1` — "Tracks in the 'Raido The Remix Files' playlist are open for remixing!" → 16 playlists, 253
  slots, **0.0% other creators.** Also DQ'd: 7,992 following / 5,112 followers, ratio 1.563.
- `gmangemini` — the most explicit stated credit norm anywhere ("If you remix my songs, always provide
  credit"), otherwise immaculate (ratio 0.017, median LR 27.05%, top share 2.69%) → 9 playlists, 133 slots,
  **7.5% other-authored.** States the norm, hosts almost nothing. The control case for this slot.
- `djblackeye` — "OPEN REMIX- Free - ENJOY" playlist, 187 slots, **100% self-authored**; 4.3% other across
  984 slots, and the full 2,251-clip walk still fails reconciliation.

**Lofn already issues the invitation.** Triple Arch's caption reads *"Please feel free to remix and/or reuse
our prompts and voices!"* She does not need to learn the ask. She needs the receipt.

**Why this shape and not another:** per Appendix D she will not rank the 59 remixers — no best remix, no
featured pick, no leaderboard. nantekootaaa's playlists are framed as 紹介 ("an introduction to") under a
gratitude header, 100% other-authored, unranked, with the host absent from her own tribute list. That is a
shape Lofn could adopt without violating her own rules. rvon's puts his nine tracks at the head of the list.

**Cheap give-side sample available here:** the 43 remixers in her tribute playlist are a ready-made list of
mostly-smaller accounts (hypnotizingtrap1612, mashupworld7, peppepop, cagataycagatay, rivein…) to search her
handle against. Number 6 is a 30-minute logged-in job for this pick, not an open-ended crawl.

**Confound on the reply numbers, stated so it does not poison the monthly re-pull:** her bio currently
declares an illness and a suspended reply backlog. Any reply rate measured today measures the illness, not
the practice — sample threads predating the notice, or record the cell as **SUSPENDED** with the date. Per
`vault/HUMAN_SUBJECT_STANDARD.md`, what is studied here is the artefact she built (bilingual bio, stated
SLA, the two gratitude playlists). **Never the illness.**

---

## 3. EMPTY AND WEAK SLOTS

### SLOT 4 IS EMPTY, and that is the honest result

The only account that cleared every computable disqualifier *and* had genuine East-Asian hosting evidence
was `nantekootaaa` — and her evidence for Slot 4 (the multilingual composition of who she platforms) is
read off **the same two playlists** that seat her in Slot 5. Double-seating her would mean the two slots
study the same artefact, which is precisely what Appendix C's framing sentence forbids: *five accounts that
each solve a DIFFERENT problem.* She stays in Slot 5, where her evidence **is** the criterion. Slot 4
vacates.

Her one genuinely distinct Slot-4 finding is worth keeping even though the seat is empty, and it is a
platform regularity rather than one artist's habit — it reproduces across all three East-Asian accounts
measured (JP / KR / ZH):

> **TITLES LOCALISE; STYLE PROMPTS DO NOT.** nantekootaaa: 45 of 52 titles Japanese-only, but 46 of 52
> style-tag fields English ("Japanese folk metal, J-Rock, female vocal, eerie shamisen"), with instrument
> words transliterated across (wadaiko, shamisen, shakuhachi). mrdvvb: 327 of 425 tag fields pure Latin
> while 298 of 425 titles carry Hangul. complaintless: 412 of 427 tag fields Latin against 220 titles in
> Chinese. **The style prompt is addressed to the generator and stays English; the title is addressed to
> the listener and localises.** Lofn can adopt that immediately without touching the music.

**Partial fill available, with its limitation written down.** `mrdvvb` (Korean, 1,906 followers, 425 clips,
median LR 36.364% — the highest clean median measured, ratio 0.250, top share 6.012%) cannot re-collide
with Slot 5: I walked its nine playlists and **200 of 200 slots are its own work, zero other creators.** It
is the better bilingual-titling study (144 of 425 titles mix Hangul and Latin; explicit language-variant
markers on the same song — "I Miss You.(En.)", "Only You! ( JA. )", "やくそく(약속)" — a system, not an
accident) and the only account with n large enough to read posting hour (n=425; peak 14:00 UTC = 23:00 KST,
but the histogram is near-flat, min 4 / max 38 across all 24 hours — he posts around the clock, which is
itself the finding). **But it teaches titling and cadence, not hosting:** median 3 comments per clip,
30 clips with none, and 18 comment sections switched off. Seat it only with that limitation on the page,
or leave the slot empty.

**What would actually fill Slot 4:** a Cantonese or Mandarin account — the slot's stated driver is HK 321 +
TW 144 — with 1,000–6,000 followers, following well under 80% of followers, top song under 50% of plays,
**at least one public playlist that platforms other people**, and visible replies in its own comment
section. Finding one needs the thing this page does not have: a logged-in browser to walk Suno search and
the HK/TW-tagged trending pages. The public profile endpoint only answers when you already know the handle,
which is why the Chinese-language sub-pool came back empty — that is a search limitation, not proof that no
such account exists.

Still unresolved and worth one click: **`haengbok`**, the author of the Korean critical comment named in
Appendix C's own notes. The handle returns a 1-clip, 0-follower account; `haengbok1` / `haengbok_` are 404.
Clicking that commenter's avatar in a logged-in browser resolves it in seconds and it is the highest-value
Slot-4 lead on record.

### SLOT 2 AND SLOT 3 ARE CONDITIONAL, NOT CONFIRMED

Zero of Slot 2's three stated criteria and zero of Slot 3's one stated criterion are measured. Both seats
rest on catalogue-side proxies for a comment-side question. They are seated because they are the
best-shaped accounts to point the browser pass at first — not because the question has been answered.

### THE SOFT SECOND COLLISION

`nrv13` (179 clips / 3,844 followers) sits **inside the Slot-1 window**. Slots 1 and 3 are therefore both
mid-scale, Lofn-shaped accounts with high median like rates, separated only by an unmeasured proxy about
comment register. Counting strictly, this slate delivers **three genuinely distinct problems** — matched-
scale craft gap, comment surface, remix hosting — not five. If the Slot-3 browser pass shows nrv13's
section does not read over 40, replace it with an account that is age-correct *and* outside the Slot-1
window (radioramarti is the standing runner-up, at the cost of a very thin comment section).

### NO PICK HAS CLEARED ALL FOUR UNIVERSAL DISQUALIFIERS

The "80%+ check mine out" filter is uncleared on **every candidate in this exercise**, because it is a
comment-*content* test and the public API returns no comment bodies. **Every seat on this page is
provisional by construction.**

---

## 4. WHAT WAS DISQUALIFIED, AND WHY

### On the follow-for-follow rule (measured)

| handle | followers | following | ratio | margin | verdict |
|---|---|---|---|---|---|
| **laurent_thepenier** | **585** | **1,268** | **2.167** | **−116.8%** | **DISQUALIFIED.** Following *exceeds* followers by more than 2:1. Re-measured 2026-08-02: 211 clips, 2,709 plays, 500 upvotes, reconciles exactly. The worked example in Appendix C (584 / 1,267) reproduces to within one day's drift. |
| raining2025 (雨兒💙) | 4,305 | 3,483 | 0.809 | **19.09%** | **DISQUALIFIED** — inside the ~20% band by two hundredths of a point. The strongest Chinese-language candidate on every other axis (337,889 plays); the rule is the rule. Corroborating signature: 42 clips at a median of ~160 comments each is what the disqualifier was written to catch. |
| raido1 | 5,112 | 7,992 | 1.563 | −56.3% | **DISQUALIFIED** — and separately, 0.0% other-authored playlists. |
| megurupop | 57 | 77 | 1.351 | −35.1% | **DISQUALIFIED.** |
| pmsson071 | 92 | 124 | 1.348 | −34.8% | **DISQUALIFIED** — a shame; the most Slot-3-shaped *bio* in the pool (Swedish non-musician, "good ear", soft spot for storytelling voices). |

### On content, not arithmetic

- **`complaintless` (Ping, Taiwan)** — passes the ratio by 2.15 points (5,318 / 4,140, margin 22.15%), has
  427 clips, median LR 19.90% and the lowest placement-artifact score measured anywhere (2.64%), and even
  keeps a playlist named "Taiwanese Songs". **Rejected on a declared refusal to host:** her bio states she
  is not reading notifications, that tagging her does nothing, that her visits are random, and that she is
  bad at writing comments and only uses symbols. The API agrees with the bio — **223 of 427 clips (52.2%)
  have comments disabled.** Seating her would seat the problem. Keep her as the negative control, and for
  one genuinely useful finding: converted to Taipei time, **167 of 427 clips (39.1%) land in the four-hour
  window 07:00–10:00 local**, hour 08 alone taking 72. She is the sharpest posting-cadence signal in the set.
- **`rvon`** — see Slot 5. Curated the remixes, then switched comments off on 121 of 137 songs.
- **`gmangemini`, `raido1`, `djblackeye`** — stated remix-host norms with 7.5% / 0.0% / 4.3% other-authored
  curation behind them. Mirrors of Lofn's own failure, not benchmarks for it.
- **`cornucopiaofmelodies`** — passes every Appendix C disqualifier as written (4,116 followers / 247
  following, 235 clips, median LR 32.26%, top share 2.80%) and is 86.6% acoustic by clip count. Rejected on
  a criterion **added** here and declared as an addition: the **exposure gradient** (median like rate ÷
  aggregate like rate). On any account with real discovery traffic, high-play clips accumulate listeners who
  play without liking, so the median clip sits well above the aggregate. Measured gradients: nrv13 2.10,
  simbaxp 1.83, s5schwatch 1.64, **Lofn herself 1.49** — and cornucopiaofmelodies **1.018**. A gradient of
  1.02 across 235 clips means essentially no non-liking listener exists: every play is a like. Corroborated
  by 4,116 followers producing 13,302 total plays, a best-ever song of 373 plays, a 45-play median, and
  median 1 comment per clip. Its 32.26% median measures a closed reciprocal pool, not craft. The follow-
  ratio filter misses this because the account does not follow back — *you can be followed by farmers
  without following them* — and the gradient catches what the ratio cannot. Note the criterion does not
  disqualify Lofn (1.49), so it is not a bar rigged to a preferred answer.
- **`mercyamari`** — NOT disqualified (margin 79.2%; see the correction in §2). Not the Slot-3 pick because
  play-weighting collapses it.
- **`theemourningstarr`** — excluded by instruction. Active Soundry collaborator, separately asked for his
  own five. Source, not subject.
- **`localsymmetry`** — excluded. That is us.

---

## 5. THE STANDING RULE

> **STANDING RULE: these five are studied for HOSTING BEHAVIOUR, not for sound. Any benchmark finding that
> turns into "make music like theirs" is a misuse of the exercise and gets thrown out.**
> — SOCIAL_VOICE.md, Appendix C

Enforced concretely on this page: simbaxp is seated for a line in a bio and a comment section, not for
sertanejo; nrv13 is seated for a cover catalogue's *audience arithmetic*, not for its arrangements — and
under no circumstances does "nrv13's covers over-perform" become a covers programme, or "French repertoire
over-performs" become Lofn performing French; nantekootaaa is seated for a playlist structure, and Lofn
cannot and must not imitate wa-rock, yokai material, or a Japanese identity. Appendix C's fourth
disqualifier — *anyone whose style I could only imitate by lying* — bites hardest here and it is answered
the same way each time: **the transferable artefact is the hosting practice.** The bilingual bio. The stated
reply SLA. The gratitude playlist that contains none of your own songs. The pre-empted critique. None of
those require claiming a life you do not have.

Related standing prohibitions that constrain how these findings may be used: Appendix D — no ranking the
59 remixers, no pivoting the music at Hong Kong and Taiwan, no localisation programme, no turning the
outbound side into content, and **follower growth is not a success measure**. If the only number that moves
is the follower count, this exercise has failed on its own terms.

---

## 6. THE INTERNAL BENCHMARK — verified against the snapshot

Appendix C: *"'The Limit Always Wins' runs 9.46% like rate on 497 plays — 2.1× the catalogue baseline and
6.5× Triple Arch."* **Checked against `data/snapshots/suno/latest.json` (pulled 2026-08-02T00:03:48Z, 210
rows, `reconciliation.ok = true`, `plays_delta 0`). It holds, exactly:**

| claim | snapshot | verdict |
|---|---|---|
| 497 plays | 497 | ✅ |
| 9.46% like rate | 47 ÷ 497 = **9.457%** | ✅ |
| 6.5× Triple Arch | Triple Arch 5,047 ÷ 349,736 = 1.443%; 9.457 ÷ 1.443 = **6.55×** | ✅ |
| 2.1× the catalogue baseline | 9.457 ÷ **4.49** = **2.11×** | ✅ — against the **style-token baseline** of 4.49% quoted in Appendix B |

**One clarification worth writing into the page rather than leaving to be re-derived every month:** the word
*baseline* in that sentence means the 4.49% style-token baseline from Appendix B, not the catalogue's own
like rate. Against the other two candidate baselines the same song reads:

- vs the **median clip** (2.339% in the snapshot, 2.367% live) → **4.04×**
- vs the **aggregate catalogue** like rate (5,913 ÷ 374,122 = 1.581%) → **5.98×**

All three multiples are true of the same song; record which one is being quoted. **No correction to
Appendix C is required.**

Two live-drift notes for the next re-pull, both flags rather than errors:
- The snapshot on disk records **3,926 followers / 56 following**; the live pull the same day returns
  **4,091 / 79**. The snapshot's clip-level data is sound (it reconciles); its social counts are stale.
- Live catalogue as of 2026-08-02: 210 clips, 374,474 plays, 5,926 upvotes, median LR **2.367%** (n=188),
  Triple Arch 349,929 plays = **93.445%** of everything, median 50 plays per clip, **0 playlists**.

---

## 7. RE-PULL CADENCE

**Monthly. One page. The same six numbers per pick. Nothing else.**

1. `python scripts/ingest_suno.py --handle <h> --dry-run` for `localsymmetry`, `simbaxp`, `lunasolaris`,
   `nrv13`, `nantekootaaa`, plus the Slot-4 fill when one exists. **If a walk does not reconcile, the row
   does not get written** — record UNRECONCILED and re-pull.
2. Recompute numbers 1–3 and both disqualifier arithmetics (follow margin, top-song share). An account that
   drifts into the ~20% follow band or past a 50% top-song share **leaves the slate that month**; do not
   grandfather a seat.
3. Numbers 4–6 need the logged-in pass in §2. If it has not been run, the cells read **UNMEASURED** — never
   an estimate, never a carried-forward value with a new date on it. **A pick justified on invented
   behavioural numbers is worse than no pick.**
4. Re-check the internal benchmark against the newest `data/snapshots/suno/latest.json` in the same sitting.
   It is free and it is the best evidence about Lofn on the page.
5. Reconcile against Thee Mourning Starr's five when they arrive, and re-reconcile whenever he revises them.

**Repo bug blocking a clean monthly re-pull** (found independently by every worker on this exercise and
confirmed here):

- `scripts/ingest_suno.py::normalise_clip()` discards `comment_count`, `allow_comments` and
  `metadata.can_remix`, and the header block drops `user_comments_blocked` and playlist authorship. Those
  are the **only public fields that bear on hosting behaviour at all**, and every comment-surface number in
  this page needed a bespoke raw walk to recover them. Keep them.
- `pull()` sets `reconciliation.ok` from the play-sum alone, so a walk truncated by `max_pages` or by a
  mid-walk HTTP error returns a **short catalogue flagged as good**. It should set `ok = False` whenever the
  walk is cut short, not only when the plays fail to sum.

Fix both and this page becomes a monthly `--dry-run` and a diff, which is what Appendix C asked for.