## ADVERSARIAL VERIFICATION REPORT — all five picks re-measured independently

**Method.** I did not use the reported numbers. I wrote my own raw pager (`C:\Users\gulzi\AppData\Local\Temp\claude\E--git-private-claw\f9ba048f-e0db-4781-b454-586257998afb\scratchpad\verify.py`, snapshots in `.../scratchpad/raw/`) that walks `studio-api.prod.suno.com/api/profiles/<handle>` with 429 retry/backoff, keys rows by clip id, and computes median like rate, follow ratio, top-song share and comment structure from the raw payload rather than from `normalise_clip()`. I also independently walked `/api/playlist/<id>/` for authorship. Every walk below reconciled row-count against `num_total_clips`.

### MY MEASUREMENTS, 2026-08-02

| handle | followers | following | ratio | margin | clips | median LR (≥20 plays) | top-song share | plays_delta | total comments | median c/clip |
|---|---|---|---|---|---|---|---|---|---|---|
| localsymmetry (baseline) | 4,090 | 79 | 0.019 | 98.07% | 210/210 | **2.367%** | **93.445%** | 0 | 565 | 0 |
| simbaxp | 3,603 | 666 | 0.185 | 81.52% | 257/257 | 20.727% | 6.376% | 0 | 3,898 | 12 |
| lunasolaris | 958 | 222 | 0.232 | 76.83% | 105/105 | 26.816% | 5.714% | 0 | 2,159 | 19 |
| nrv13 | 3,844 | 1,912 | **0.497** | 50.26% | 179/179 | 24.087% | 7.327% | +28 (drift) | 5,664 | 32 |
| nantekootaaa | 1,966 | 883 | 0.449 | 55.09% | 52/52 | 26.681% | 7.524% | 0 | 1,431 | 20.5 |

**Every reported figure reproduces.** Deltas against the slot agents are ≤2 units (live drift). No pick fails the follow-ratio rule; no pick fails the top-song rule.

### FABRICATION AUDIT — CLEAN

I hunted specifically for behavioural claims presented as measured. **I found none.** All five agents marked numbers 4, 5 and 6 UNMEASURED, all five flagged the "80%+ check mine out" disqualifier as uncleared, and all five named the same blocker and the same §3b expand-every-collapsed-reply remedy.

Spot-checks that reproduced **to the digit**, which is why I trust the rest: rvon 121/137 comments disabled, 108 of those closures sitting on 4,387 existing comments; complaintless 167 of 427 clips in the 07:00–10:00 Taipei window (39.1%), hour-08 = 72; nantekootaaa 46/52 Latin-script style-tag fields; mrdvvb 200/200 playlist slots self-authored; nrv13 top-four like rates 1.25/1.44/1.43/1.58% at 25.6% of plays; nrv13's six French-repertoire covers at median 34.2% LR on median 247 plays; cornucopiaofmelodies exposure gradient 32.258/31.679 = **1.018**; raining2025 margin **19.09%** (DQ confirmed); complaintless bio translation accurate ("not looking at notifications… very bad at writing comments, only use symbols"). One trivial divergence: Slot 4 reported 41/52 pure-Japanese titles; I count 45 JP-only / 3 mixed / 4 Latin — a classification difference, not an invention.

---

### SLOT-BY-SLOT VERDICTS

**SLOT 1 — simbaxp — SUSTAIN.** The only slot seated on its own literal criterion with measured evidence. Window fit exact (257 clips in 150–300, 3,603 followers in 2,500–6,000), median LR 8.75× Lofn at matched volume, top share 6.38% against a 50% ceiling. Q1 = 12.309%, so three-quarters of the catalogue beats Lofn's best non-placement clip (9.46%). The added "exposure gradient" criterion is a spec extension, but it is declared as one and it computes honestly — I reproduce cornucopia at 1.018. No refutation found.

**SLOT 2 — lunasolaris — SUSTAINED ON ITS EVIDENCE, BUT THE SELECTION CHAIN IS VOID.** 0 of the slot's 3 stated criteria are measured; the agent says so and calls the seat conditional. The refutation is structural: it passed over nrv13 explicitly to reserve nrv13 for Slot 1 — and Slot 1 seated simbaxp. nrv13 then went to Slot 3. So the allocation argument that produced this pick no longer exists, and on the slot's own measured proxy nrv13 dominates lunasolaris (median 32 vs 19 comments/clip, top-clip concentration 1.55% vs 3.33%, both at 100% coverage). lunasolaris wins only on comments÷upvotes (30.4% vs 16.7%). Its real differentiator is that it is the one pick outside the Slot-1 window.

**SLOT 3 — nrv13 — WEAKEST SEAT, CONDITIONAL AT BEST.** Numbers reproduce and the mercyamari margin correction (79.2%, not 20.8%) is right. But: (a) the slot's actual criterion — comment section reads over 40 — is entirely unmeasured, and the seat rests on a proxy-of-a-proxy (style tags → cover-release years → presumed commenter age); (b) ratio **0.497** is the loosest in the slate, 24× Lofn's outbound, and passes only the letter of a rule written against exactly this behaviour; (c) nrv13's top four tracks are 25.6% of plays at 1.25–1.58% like rate — Triple Arch's failure mode distributed across four songs, i.e. the arriving audience is not the responding one; (d) 153/179 clips are remixes, which puts it adjacent to Slot 5's subject matter.

**SLOT 4 — nantekootaaa — REJECT AS SEATED.** Two independent grounds.
1. **Double-seated with Slot 5. The slate contains four accounts, not five** — the precise failure Appendix C's framing sentence exists to prevent. Both agents flagged it; neither could resolve it.
2. Even standing alone the seat is weak: its headline Slot-4 evidence (multilingual composition of the remixer pool) is read off *the same two playlists* Slot 5 seats it on, so the two slots are not studying different things. Its one genuinely distinct Slot-4 finding — titles localise, style tags stay English — is reported by its own agent as holding identically across mrdvvb and complaintless, which makes it a platform regularity, not a property of this account. Add: n=52 is too thin for posting hour (agent concedes), the slot's stated driver is HK 321 + TW 144 and the pick is Japanese, and the bio declares replying currently suspended — so the one behavioural number a browser pass would return for her is invalid today.

**SLOT 5 — nantekootaaa — SUSTAIN, STRONGEST SEAT IN THE SLATE.** I re-walked the playlist endpoint myself: the two gratitude playlists read 90 slots / 43 distinct creators and 20 slots / 9 distinct creators, **zero self-authored in either**; the three own-work playlists are 100% self. `metadata.can_remix` true on 52/52 (Lofn 201/210). Lofn has 0 playlists against 59 remixes. Along with Slot 1, this is the only pick whose evidence *is* the slot's criterion rather than a proxy for it.

---

### SLATE-LEVEL FINDINGS

1. **The slate fails Appendix C's framing test.** Four accounts across five slots. Resolution, on measurement: nantekootaaa stays in **Slot 5** (on-criterion, independently verified, and the only account in the pool with the receipt Lofn lacks). **Slot 4 vacates.** mrdvvb cannot re-collide (I verified 200/200 self-authored slots) and its n=425 makes posting hour readable — but its comment surface is median 3/clip with 18 sections closed, so it teaches titling and cadence, not hosting. Seat it only with that limitation written down, or leave Slot 4 EMPTY. The HK/TW gap is genuine and I confirmed both exits: complaintless clears the ratio by 2.15 points but her bio is a declared refusal to host and 223/427 of her comment sections are switched off; raining2025 fails at 19.09% margin, and her 42 clips at a median of **160 comments each** against an 0.81 follow ratio is the reciprocal-pod signature the disqualifier was written for.

2. **A soft second collision.** nrv13 (179 clips / 3,844 followers) sits *inside the Slot-1 window*. Slots 1 and 3 are therefore both mid-scale Lofn-shaped accounts with high median like rates, separated only by an unmeasured proxy. Counting strictly, the slate delivers three genuinely distinct problems — matched-scale craft gap, comment surface, remix hosting — not five.

3. **No pick has cleared all four universal disqualifiers.** The "80%+ check mine out" filter is uncleared on every candidate in the exercise, because it is a comment-*content* test and the API returns no comment bodies. Every seat is provisional by construction.

4. **The deliverable fills 3 of 6 cells per pick.** Appendix C's monthly one-pager cannot be produced from the public endpoint. Reply rate, reply/comment length ratio, and outbound comments on smaller accounts need a logged-in suno.com session, with **every "Read N more replies" expanded before any thread is scored unanswered** (§3b — that error has already been made twice in this workspace). The agents supplied per-clip comment counts, so each pass is pre-armed for reconciliation; nantekootaaa's 43-remixer list is the cheapest available target set for the give-side number.

5. **Repo bug, reported independently by all five and confirmed by me:** `scripts/ingest_suno.py::normalise_clip()` discards `comment_count`, `allow_comments` and `metadata.can_remix`, and the header block drops `user_comments_blocked` — the only public fields that bear on hosting behaviour at all. And `pull()` sets `reconciliation.ok` from the play-sum alone, so a walk truncated by `max_pages=60` or by a mid-walk HTTP error returns a short catalogue flagged as good.