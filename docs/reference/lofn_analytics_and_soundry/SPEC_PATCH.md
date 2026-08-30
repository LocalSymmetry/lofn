# SPEC_PATCH — Lofn Analytics & The Soundry



*Patch panel `wf_bbc75dde-fa2`, 2026-08-01. Four seats — after Bruce Schneier (access/threat model), Hans Rosling (four-feed fusion), Ben Shneiderman (competitor watchlist dead-end), Amanda Cox (the relocated centre panel) — moderated by Lofn. 5 agents, 0 errors.*



**VERDICT: STILL-NEEDS-PATCH**



« SPEC_PATCH — LOFN, MODERATING. Four seats folded into one. Splice into
`E:/git/private-claw/docs/reference/lofn_analytics_and_soundry/BUILD_SPEC.md`. »

**How to apply.** Every SPLICE below is one of two kinds. **[MERGED]** — this document is the text;
it supersedes the corresponding seat splice in full because two or more seats edited the same lines.
**[ADOPT @anchor + deltas]** — the seat's splice text stands verbatim at the named anchor, with the
listed amendments applied. Where a seat's text and this document disagree, **this document wins**;
that is what moderating means. Nothing here invents a number. Rulings are numbered R1–R12 and each
overrule names the intent it preserves.

---

# §M · MODERATOR RULINGS — the eleven collisions, decided

**R1 · ONE ROSTER FILE, AND IT IS THE TAXONOMIST'S.** The Synthesist proposed `data/watchlist.yaml`
(flat list); the Taxonomist proposed `data/watchlist/competitors.json` (five slot-locked objects,
git-ignored, plus a committed all-null template). **The JSON slot lock ships; the YAML is struck.**
Three reasons. (a) §15 forbids a `dependencies` key — YAML needs a parser, JSON does not.
(b) `vault/SOCIAL_VOICE.md` Appendix C is on disk and is The Scientist's own written ruling: she
*rejected* "five Suno stars" as an axis and named five slots that each solve a different problem.
A flat handle array is the axis she rejected. (c) The Synthesist's own risk note says the schema is
the only thing stopping a dossier; the slot lock is that schema. **PRESERVES the Synthesist's
intent** — his §SYN-2.7 exists so the competitor blocker can be cleared from inside the app, and it
still can, via M5.7. Every `data/watchlist.yaml` reference in his patch reads
`data/watchlist/competitors.json`.

**R2 · NO REFRESH BUTTON FOR `competitors`, EVEN ONCE A HANDLE IS NAMED.** §SYN-2.7 would make a
per-handle Refresh appear the moment the roster is non-empty. **Struck.** Critic 2's buildBlocker is
live and unresolved: *"THE REFRESH CONTROL AND THE 'NO WRITE ENDPOINT' TEST CANNOT BOTH SHIP"* —
§14.4's refresh has no named transport and `tests/server.spec.ts` exists to prove there is no write
endpoint. A seat closing finding 3 may not deepen an open finding belonging to another seat. The
Taxonomist's **[Copy pull command]** closes the dead end without a transport. **PRESERVES the
Synthesist's intent** — §14.4's "no button that cannot work" is upheld and the blocker still clears.

**R3 · THE PROJECTION GATE HOLDS, IN TWO TIERS, AND THIS IS THE HARDEST RULING HERE.** The
Taxonomist's `COMPETITOR_FORBIDDEN_KEYS` forbids `tags`, `emo_tags`, `emo_tag_count`,
`words_per_line` on competitor rows. The Synthesist's X2 row 2, X3 and X5 clock 2 need exactly
those. Three of six fusion cards are unbuildable as written. Ruling:

- **Tier H (`hosting`) — the default, and the only tier any handle ships with.** The Taxonomist's
  whitelist exactly: `id · play_count · upvote_count · like_rate · created_at · is_public`. Every
  prompt-, title-, lyric- and persona-level field dropped in the loader.
- **Tier S (`hosting+shape`) — per-handle, opt-in, written only by The Scientist, never by the app.**
  Adds **aggregates only, computed inside the loader, never per-clip and never per-tag**:
  `distinct_tag_count`, `tags_per_clip` (median + n), `zipf_slope`, `median_words_per_line`,
  `words_per_line_slope`. **No string from a competitor file may cross the loader boundary at any
  tier** — Tier S emits numbers, not vocabulary. Asserted by test 21(d).
- The roster carries `"study": "hosting"` as a frozen default. The app can render the field; it
  cannot change it, and there is no UI that writes `hosting+shape`.

Argued from the record: Appendix C's STANDING RULE is hers — *these five are studied for hosting
behaviour, not for sound; any finding that becomes "make music like theirs" is thrown out.* Board 6
nonetheless asks that four feeds be combined into insights that include lyric length and EMO usage.
Both are hers. The resolution is that a **shape** (863 distinct tags across 202 clips; a slope) is
not imitable and a **prompt** is, so the axis space is reserved, the ghost states Appendix C as its
blocker by name, and one word in a file she owns lifts it. **PRESERVES both**: the Synthesist's
fusion surface is built and testable; her standing rule is enforced by a type rather than remembered.

**R4 · X5 CLOCK 2 (competitor style-term share) IS ABSENT IN v1, NOT GHOSTED.** A per-style-term
rising/falling read on watched artists is the "make music like theirs" generator in chart form, and
no aggregate framing rescues it. The clock renders in absent-hatch with the refusal printed:
*"Watched-artist style-term trends are not drawn. `vault/SOCIAL_VOICE.md` Appendix C: these five are
studied for hosting behaviour, not for sound."* X5 ships with two clocks and prints `0 of 2 clocks
drawn`. **PRESERVES her Board-6 insight #4** — setters/breakers still gets its lead/lag definition
from ours-vs-market, which is the pair that makes the two words mean different things.

**R5 · THE FUSION TAB SHIPS AS TAB M6, AND THE LEDE OUTRANKS IT.** Rail order:
SIGNAL · RESPONSE · CRAFT · REACH · HEADWINDS · CONFLUENCE · PROVENANCE. The Synthesist's argument
holds — fused cards scattered into M2/M3 read as more single-feed cards. The Headline Editor's
rail-length objection is answered structurally: **M0.0 is the first object on the first tab, so the
seventh tab never competes for the eight-second read.** X0's join-yield enters the register as a
`governs: "tab"` signal for M6, so M6's Tab Directive is the LEDE's machinery, not a second one.

**R6 · A CARD'S `Insight` IS NOT A REGISTER ENTRY.** `src/signals/register.ts` is hand-maintained and
Scientist-owned (Splice J). Cards render on their own tabs regardless; only registered signals rank
in the LEDE and only registered signals count toward `k`. A build agent that auto-registers every
card's Insight defeats the closed family. Asserted by test 24(h).

**R7 · THE HOSTED POSTURE CARRIES NEITHER THE ROSTER NOR THE ROUNDS.** Interlock the Threshold
Keeper could not see, because the other two patches were written in parallel. Add to §0.5(a)'s
categorical exclusions: **`data/watchlist/**` in its entirety** (real people's handles plus a
rationale line — third-party data under `vault/COLLABORATOR_STANDARD.md`, and the roster is a
watching record, which is the thing that must never be public), and **the whole `soundry` feed**
(a closed round discloses the factor under test and its pre-registered MDE — that is the method,
asset A1's sibling). Therefore **M5.7, M5.8, and M6·X6 are local-posture-only and are stripped at
build under `POSTURE=hosted`**, asserted by `posture-hosted.spec.ts`.

**R8 · STYLE-TERM NAMES MAY BE HOSTED; PER-CLIP `tags` MAY NOT.** The 20 screened terms are common
genre words in a screening list that lives in `matcher.ts` as data; the corpus is the full `tags`
string per clip. The derive allowlist admits `style_term` and per-term aggregate cells and forbids
`tags`. Test 11 additionally asserts the set of exported `style_term` values is a **subset of the
screening list** — so no phrase lifted from a single clip's prompt can ride out as a "term".
Without this the LEDE's pinned refusal row (which prints term names) could not be hosted, and
Board 6's *genre performance* would silently drop from the collaborator build.

**R9 · ONE `EmptyKind` UNION, FOUR NEW KINDS, ONE STRUCTURED `unlockAt`.** The Synthesist and the
Taxonomist both rewrote the same twelve lines. Merged in SPLICE 1. `EmptyState.unlockAt` becomes
`UnlockCounter[]`, not `string` — the Taxonomist's own S0–S3 states need two independent counters
("0 of 5 handles named" / "0 of 5 pulled") and a string cannot carry them; the Synthesist's
independent-segment rule is the better contract and it is now the only one.

**R10 · TEST NUMBERING, DEDUPLICATED.** Four seats each appended a "test 10". One block, SPLICE 8:
10–13 security · 14–19 fusion · 20–23 watchlist · 24–26 lede. All three seats' `forbidden-strings`
extensions merge into one list.

**R11 · THE `.gitignore` ADDITIONS ADD PATHS; THEY DO NOT REVERSE THE SNAPSHOT DECISION.** Both
critics flag that §8.3.1 would silently reverse a written decision to track `data/snapshots/`.
**Nothing in this patch touches that line, and no lane may.** The new ignores are `data/public/`,
`data/soundry/`, `data/watchlist/`, `vault/private/`, `**/panel-sessions.json` — all new paths.
The snapshot question stays with The Scientist.

**R12 · `Provenance` IS NOT UNIFIED HERE, AND TWO CONSTRAINTS BIND WHOEVER DOES IT.** Critic 2
records three incompatible vocabularies. This panel does not own that finding. Binding on the seat
that does: the union must retain (a) exactly one value meaning *"from a pre-registered randomised
round in the sibling app"* — it is the only provenance that licenses a causal verb, and collapsing
it into `derived` makes V22 unenforceable and unlocks that verb estate-wide; and (b)
`"public-page"`, which the watchlist's manual observations write. V8's four-level colour cap is the
other seat's problem, not a reason to drop values from the type.

---

# §0 · THREAT MODEL — [ADOPT @ PART 1, immediately before `# §1 · DESIGN TOKENS`]

Threshold Keeper SPLICE 1 verbatim (assets A1–A5, adversaries T1–T5, trade-offs 1–2, anti-goals 1–5).
**Deltas:**
- **A6 added:** *"**The watched-artist roster** — `data/watchlist/competitors.json`. Five real
  accounts, each with a written line saying why we are watching them. A leak is not a data breach,
  it is a disclosed surveillance list naming five named people. Governed by
  `vault/COLLABORATOR_STANDARD.md`; git-ignored; excluded from every hosted artefact and from every
  snapshot."*
- Trade-off 1, final sentence, append: *"and the roster and the round ledger are not in the building
  either (R7)."*
- Anti-goal 6 added: *"**No app-written roster.** The watchlist editor composes and displays; The
  Scientist pastes. There is no `POST /analytics/watchlist` and `tests/server.spec.ts`'s 405
  assertion is what proves it."*

# P0 · BLOCKING PRECONDITION — [ADOPT @ PART 1 §8.3 as item 0]

Threshold Keeper SPLICE 2 verbatim. Blocks the **hosted posture only**; the local builds, the
watchlist lane, the fusion lane and the LEDE lane are all unblocked by it.

---

# SPLICE 1 — [MERGED] PART 1 §2.1, REPLACE lines 249–260

```ts
export type EmptyKind = "never-pulled" | "measured-zero" | "filtered-empty"
                      | "not-applicable" | "below-floor" | "no-history"
                      | "needs-declaration"   // a human DECISION is missing, not a pull
                      | "pull-failed"         // the pull RAN and returned no profile
                      | "emitter-silent"      // the directory exists; nothing ever wrote into it
                      | "join-unresolved";    // both feeds present; the key resolved nothing

export interface UnlockCounter {
  label: string;   // "competitor snapshots"
  have: number; need: number;
  how: string;     // the exact command, or the human decision, VERBATIM and copyable
  path: string;    // the file whose existence satisfies it
  blocker: "command" | "decision" | "time";   // decision => no command exists; say so
}

export interface EmptyState {
  kind: EmptyKind;
  feed?: FeedId;
  headline: string;
  why: string;                  // names the exact missing file, by path
  how: string;                  // the exact command, or the human decision required
  unlockAt?: UnlockCounter[];   // 1..n INDEPENDENT segments (R9). Never a bare string.
  verbatim: boolean;
  declares?: { path: string; editorHref: string; templatePath: string };      // needs-declaration
  failedSnapshot?: { path: string; pulledAt: string; rowCount: 0 };           // pull-failed
}

export interface JoinEvidence {
  left:  { feed: FeedId; asOf: string; provenance: Provenance; n: number };
  right: { feed: FeedId; asOf: string; provenance: Provenance; n: number };
  key: "clip_id" | "normalised_title" | "platform" | "style_term" | "handle" | "none";
  keyRule: string;      // the normaliser, verbatim and executable-in-prose. REQUIRED.
  attempted: number; resolved: number;
  ageGapHours: number;
  unresolvedPolicy: "counted-and-shown" | "excluded-and-listed";  // never "dropped"
}
```
`Evidence` gains `join?: JoinEvidence`. `Provenance` gains `"randomised-round"` and `"public-page"`
(R12).

**New validator rules (§2.3 `validate.ts`)** — V19 a spec whose `evidence.source` spans two
`data/snapshots/<feed>/` dirs ⇒ `evidence.join` present · V20 `join` ⇒ non-empty `keyRule` rendered
under the plot and `resolved/attempted` printed as a **fraction**, never a bare percentage ·
V21 `ageGapHours` exceeding the fresher feed's FRESH window ⇒ the §6.2 mismatch band draws, not a
caption · V22 `provenance === "randomised-round"` ⇒ the row carries and renders `roundId` and
`preregHash` · **V23 (new, mine)** a spec whose `evidence.source` touches
`data/snapshots/competitors/` ⇒ `JSON.stringify(spec)` contains no member of
`COMPETITOR_FORBIDDEN_KEYS`, and its `evidence.projectionTier` is `"hosting"` or `"hosting+shape"`
(R3).

**Also add to §2.4:** `olsSlope(xs, ys, level) => { beta, se, lo, hi, n, r }` —
`apps/lofn-kit/src/stats/ols.ts`, pure, node-testable. M2.1's ρ is a correlation; X2 needs a slope,
and quoting one we have not computed is the defect this whole patch exists to close.

# SPLICE 2 — [MERGED] PART 1 §6.1, APPEND four rows after line 653

| kind | meaning | visual | copy pattern |
|---|---|---|---|
| `needs-declaration` | the pull is possible; nobody has said WHAT to pull | dashed border, absent-hatch, **a filled-slot counter**, primary button | *"Five slots. 0 named. This is a decision, not a pull — `data/watchlist/competitors.json` does not exist yet. [Open the watchlist editor →]"* |
| `pull-failed` | the command ran; the profile did not resolve | absent-hatch **plus an ember hairline**, axes not drawn, the failing file named | *"The pull ran at {pulledAt} and returned no profile for `@{handle}`. 0 rows, no header. `ingest_suno.py` writes `latest.json` even when a pull fails (`scripts/ingest_suno.py:264-273`) — this file is that. The handle is unverified, not empty. [Correct slot {n} →]"* |
| `emitter-silent` | the directory exists; the emitter never wrote into it | dashed border, hatch, **axes ARE drawn** — the shape is known, only the values are missing | *"`data/snapshots/soundry/` exists and is empty. The Soundry has never closed a round. Nothing here is zero."* + the emit command |
| `join-unresolved` | both feeds present; the key resolved nothing | real axes, both feeds' row labels at full ink, no marks | *"Both feeds are loaded. 0 of {attempted} keys matched under `{keyRule}`. This is a join failure, not an empty feed."* + the unmatched list |

**Disambiguation, normative:** absent directory ⇒ `never-pulled`. Present-and-empty directory ⇒
`emitter-silent`. File present with `header === null` or `row_count === 0` ⇒ `pull-failed`. Roster
absent ⇒ `needs-declaration`. `tests/empty-vs-zero.spec.ts` extends: **all four new fills differ
from `measured-zero` and from each other, in both themes** — the operator action differs in every
case, and a 404 must never be able to look like an artist with no plays.

# SPLICE 3 — [ADOPT @ PART 1 new §6.4] THE GHOST CONTRACT

Synthesist §SYN-2.3 verbatim (`apps/lofn-kit/src/contracts/ghost.ts`, `GhostSpec`, the five render
rules — full-ink axes, 12% α marks, independent counters, structural auto-enable, ghost-never-
replaces-empty). **Deltas:** `UnlockCounter` now imports from `contracts/empty.ts` (SPLICE 1) — one
definition, not two. Rule 6 added: *"A counter whose `blocker` is `"decision"` renders no command
and says in words that none exists. Printing a command that would fail is the same defect as
printing a number nobody measured."*

---

# SPLICE 4 — [MERGED] PART 3 §0.1 — the Soundry's posture rows

| | |
|---|---|
| **Bind** | `server.mjs` binds `127.0.0.1` only, **always, with exactly one opt-in exception**: PANEL LAN MODE (§0.3), off by default, time-boxed, single-purpose, serving a *different route table*. No `access-control-allow-origin: *`, ever. |
| **Posture** | **Local-only. The Soundry is never hosted, on any domain, under any Access policy.** It holds A2 (unposted variants) and A5 (the pre-registration chain). `check-local-only.mjs` unchanged, still a build gate. |
| **Collaborators** | Board 2's *"Access only to Scientist + invited collaborators"* **is implemented, at the granularity her own boards specify.** Board 2 gives collaborators one job — *"get feedback from listeners"* — and Board 3 draws the Anvil as a single-operator surface. **A Soundry collaborator is a PANEL LISTENER, not a bench operator.** Named as a narrowing in Overrules ACCEPTED, not dropped. |
| **Human subjects** | Panel listeners are third parties; `vault/COLLABORATOR_STANDARD.md` governs their data (A3). Default PRIVATE binds: **no panel note, rating, or pseudonym is ever exported to `data/snapshots/soundry/`, rendered in Lofn Analytics, or included in any hosted artefact.** `emit-snapshot.mjs` carries a field allowlist excluding the entire `panel` namespace. First panel screen carries a 40-word consent line naming what is recorded (rating, optional note, elapsed seconds, presentation position), what is not (name, email, handle, IP), and that notes may be read verbatim by The Scientist. |

**§0.3 PANEL LAN MODE** — [ADOPT] Threshold Keeper SPLICE 3 §0.3 verbatim: the separate route table
(404 not 403), `crypto.randomBytes(16)` path token exchanged for an `HttpOnly; SameSite=Strict`
cookie with the token stripped from the URL by 302, first-use binding with operator-visible
second-binding detection, 4h TTL / 24h cap, `--panel-revoke` / `--panel-close`, self-terminating
listener, the blindness wall doubling as the access boundary, remote listeners out of scope
(OQ-SEC-2). **Delta — the honest hole, named by me because no seat named it:** the route table
includes `GET /panel/assets/*` **and nothing in this estate fills it.** Critic 1's buildBlocker
stands: the Soundry renders nothing, holds no audio, and CSP is `connect-src 'self'`. Add to §0.3:

> **`/panel/assets/*` serves files the operator has placed under `data/soundry/rounds/<id>/audio/`
> by hand. The Soundry does not fetch, render, or download audio, and there is no control that
> would. Until those files exist the panel session renders `emitter-silent` on every card and the
> round's `instrument` may not be `panel`. This is a NAMED GAP, not a solved problem — see
> STILL OPEN #6.**

Same interlock: `cleanFeedbackText()` (`input-cleaning.mjs`) gates every panel note and **does not
exist in this repository** (Critic 2). A3 is the asset we cannot unilaterally forgive ourselves for
losing, and its cleaning function is a cross-repo import. STILL OPEN #7.

# SPLICE 5 — [ADOPT @ PART 3 §3.3, first paragraph] Threshold Keeper SPLICE 4 verbatim.

---

# SPLICE 6 — [MERGED] PART 2 §0, REPLACE the "Posture" paragraph

**Two postures, and they differ by WHICH DATASET IS BUILT, not by a flag on the same one.**

**§0.4 · POSTURE `local`** — default, unchanged, the only posture until she says otherwise.
`server.mjs` on `127.0.0.1` reading `../../data/snapshots/<feed>/latest.json` at request time. Full
fidelity. The posture in which she works and in which a collaborator sitting beside her sees
everything. No change to §15 beyond this patch's additions.

**§0.5 · POSTURE `hosted`** — [ADOPT] Threshold Keeper SPLICE 5 (a)–(h) verbatim: the default-deny
derive allowlist that exits 1 naming an unlisted field; Cloudflare Pages static origin with no
`server.mjs` in the artefact; Cloudflare Access + One-Time PIN + the `lofn-collaborators` group,
**zero lines of auth code**; the two bypasses (preview deployments off, `pages.dev` covered) verified
by curl from a logged-out foreign network and dated into `SECURITY.md`; the 30-second invite;
the revoke flow whose step 2 is not optional and whose worst case is stated out loud as 24 hours;
prune-by-default review; deployment as a human act. **Deltas:**
- (a) categorical exclusions gain **`data/watchlist/**` and the entire `soundry` feed** (R7).
- (a) allowlist gains **`style_term` and per-term aggregate cells**, with test 11 asserting the
  exported term set is a subset of the screening list (R8) — without this, Board 6's *genre
  performance* silently drops from the collaborator build.
- (b) gains: **M5.7, M5.8 and M6·X6 do not exist in a hosted bundle** and their absence is asserted,
  not merely unreferenced.
- She may not enable any of it until P0 clears **and she says the word in chat**. Build it, test it,
  do not deploy it.

# SPLICE 7 — PART 2 §1, the feed table's "Nothing" column — [MERGED]

| # | feed | REPLACE "Nothing" WITH |
|---|---|---|
| 2 | `competitors` | **`needs-declaration`, not an apology — the blocker clears from inside the app.** Two independent counters: `0 of 5 slots named` (blocker `decision`; no command exists and the card says so; route → M5.7) and `0 of 5 pulled` (blocker `command`; one copyable pull each). Once pulled, **hosting numbers only** — public clips, MEDIAN like rate, and the top-song-share disqualifier the data can check itself. Style prompts, titles, personas and lyric features are dropped in the loader (`src/watchlist/project.ts`); shape aggregates require her per-handle `"study": "hosting+shape"` (R3). Feeds M6·X2 row 2 and X3 at Tier S; M6·X0 at any tier. |
| 3 | `market` | **The only exogenous axis we would own.** Feeds M6·X4 (x axis) and X5 clock 2 *(renumbered — see R4)*. Blocker stated honestly: **`scripts/scan_market.py` does not exist.** The transport does (`node scripts/ingest_receiver.mjs --feed market --once`); the scanner does not. Sources inventoried at `vault/DAILY_RESEARCH_SOURCES.md` §D. **This empty state names a file to write, not a command to run** — printing a command that would fail is the same defect as printing a number nobody measured. |
| 4 | `soundry` | **The only randomised evidence the estate will ever hold, and the only join that can raise a tier.** Feeds M6·X6 and X2 row 3. Reader at §SYN-2.6. Directory exists and is empty ⇒ `emitter-silent`, **never** `never-pulled` — the spec's current claim that the only on-disk hit is the KNOWN_FEEDS entry is false; the directory is on disk. |

# SPLICE 8 — PART 2 new §6A · TAB M6 · CONFLUENCE — [ADOPT @ between §6 and §7]

Synthesist §SYN-2.5 verbatim — X0 join matrix (LIVE), X0b the four-quantity dumbbell agreement
(LIVE), X1 distributed-set census with its printed refusal (LIVE), X2 three instruments never
pooled, X3 vocabulary shape with the per-tag refusal, X4 positioning census with no third channel
and no fitted line, X5 three clocks, X6 the loop with its single fenced tier exemption. Route
`#/music/confluence`; rail position per R5; the per-card filter applicability declaration verbatim,
including X6 dimming all six chips because a filter that could re-cut a randomised result would
silently un-randomise it. **Deltas:**
- **X3 and X2 row 2 are Tier-S cards (R3).** Their ghost counters gain a *decision* segment:
  `0 of 5 handles at study: hosting+shape` · blocker `decision` · *"`vault/SOCIAL_VOICE.md`
  Appendix C: these five are studied for hosting behaviour, not for sound. Lifting this is one word
  in `data/watchlist/competitors.json`, and only you write that file."* Neither card may render a
  competitor **string** at any tier.
- **X3(a)'s Zipf rails carry ranks and counts, never tag names**, for competitor series. Our own
  series keeps its labels.
- **X5 has two clocks, not three (R4).** Clock 2 is absent-hatch with the Appendix C refusal
  printed. The card prints `0 of 2 clocks drawn` and the counters remain independent.
- **X0's join-yield is a register entry** (`data.joins.yield`, `governs: "tab"` for M6) so M6's Tab
  Directive is the LEDE's `rankSignals()` output and not a second authority (R5, R6).
- **X6 is local-posture-only (R7)** and says so on its face in the hosted build's absence test.
- X6's deeper link stays runtime-built from `GET /analytics/config.json` — `check-local-only.mjs`
  fails on a literal origin, and `null` still yields a non-null `deeper` pointing at the on-disk
  path.

# SPLICE 9 — [ADOPT @ PART 2 §15 `src/load/`] THE MISSING READER

Synthesist §SYN-2.6 verbatim: `soundry.ts` with the `SoundryRoundRow` schema restated reader-side;
the four reader rules (loud Schema-Guard refusal of any row with `randomized !== true`,
`ci_level !== CI_LEVEL`, empty `prereg_hash`, or `method !== "log-link-glm"`); **recompute what you
own, quote what you don't, badge the quote**; exclusions rendered as a count plus a `<details>`;
the weekly freshness tier; `apps/_fixtures/soundry-round.json` as the two-directional anti-drift
fixture; `competitors.ts` globbing `data/snapshots/competitors/*/latest.json` because
`ingest_suno.py:264-266` writes one directory per handle and a single-file reader would render a
false `never-pulled`. **Deltas:**
- `competitors.ts` **does not reuse `suno.ts`'s row parser unchanged** — it wraps it in
  `projectCompetitor()` (R3) before any row escapes the loader. The Synthesist's
  `tests/competitors-reader.spec.ts` assertion that both use the same parser is replaced by:
  *"the same parser, then the projection, and no unprojected row is reachable from a card."*
- The reader is indexed by the roster, never by directory listing (Taxonomist PATCH 7): an unnamed
  handle is invisible to the app by construction.
- Any loose `*.json` at the `competitors/` feed root (which `ingest_receiver.mjs --feed competitors`
  would write) is listed in the M5.2 Schema Guard as *"unclassified receiver drop — no handle in the
  path"* and never joined to a slot.

# SPLICE 10 — THE WATCHLIST — [ADOPT @ PART 2 §9 M5, after M5.6]

Taxonomist PATCH 2, 3, 4, 5, 6, 7 verbatim: the slot-locked roster and its all-null template; the
frozen five-value `slot_name` enum `CADENCE_MIRROR · COMMENT_SECTION_MODEL · AGE_CORRECT ·
EAST_ASIA · REMIX_HOST` with a sixth slot failing validation; the manual observation files at
`data/watchlist/observations/<handle>/<YYYY-MM-DD>.json`; the honest two-of-six-numbers table
rendered as the card's **headline, not its footnote**; the projection gate; the sort lock and the
DECIBAL firewall; M5.7's editor (syntactic-only handle screen, URL extraction offered not applied,
duplicate refusal naming both slots, the ≥12-word `why_this_one` floor, *"Unverified. This app makes
no network request."*); the compose-don't-write triad (`<pre>` + Copy, heredoc, local Blob
download); the permanent **DRAFT — NOT ON DISK** band; M5.8's status strip; the six states S0–S6
each with exactly one next action; S3's consequence line ending *"This app will not run it. Copy it
and run it yourself."*; S5's grounding in `ingest_suno.py:160-164, 264-273`; invariant I11; and the
three GET routes with the handle re-screened **at the route** before any path join. **Deltas:**
- `EmptyState.unlockAt` is `UnlockCounter[]` (R9); every S-state's counters carry `blocker`.
- The roster gains `"study": "hosting" | "hosting+shape"`, frozen default `"hosting"`, **written
  only by The Scientist** (R3). M5.7 renders it read-only and says so.
- No refresh button, at any roster size (R2).
- **M5.7/M5.8 are local-posture-only** and absent from a hosted bundle (R7).
- Every new control in M5.7 — slot chip, handle field, disqualifier chip, Copy, Download, the DRAFT
  band — requires a `GuidanceNote`. This is where Critic 1's "all elements" gap gets *bigger*, and
  I am not going to pretend otherwise. STILL OPEN #2.
- Taxonomist's optional `--keep-style` hardening of `scripts/ingest_suno.py`: **I take it, as a
  separately reviewable change, not a blocker.** The loader gate is what the app can guarantee;
  the script change is what stops the file existing. Both, in that order.

# SPLICE 11 — M0.0 · THE LEDE — [ADOPT @ PART 2 §4, after the Directive line, before M0.1]

Headline Editor SPLICE A verbatim — the four zones (blocker / rising 3 / falling 3 / cannot-be-told-
apart 3 collapsed / the pinned refusal row), ten objects hard budget, the `<ol>` as the accessible
artefact with shape-coded `aria-hidden` glyphs, the always-rendered header line; M0.0a temporal vs
contrast never sharing a glyph; M0.0b the register with fixed `k`, the style board as ONE entry,
machine-checked pre-registration barring `POST_HOC_BAND` rows from a directional band, and BH across
the register; M0.0c `signedLowerBound` / `margin` / `rankSignals` in
`apps/lofn-kit/src/insight/rank.ts` as **the only implementation**, band membership, the tie chain,
suppression at `margin < 1` or `typeM > 1.5`; M0.0d one array two depths zero duplication; M0.0e the
pinned refusal row answering ↑ Baile Phonk / ↓ Anthems **in her own words, in the position she drew
it**, with every figure interpolated live and `powerDays()` recomputed rather than hard-coded and
DRIFT-badged on disagreement; M0.0f the empty state that names the command; M0.0g the six mandatory
guidance notes. Also SPLICE B (§4.4: the Tab Directive is the LEDE's Zone-0 or rank-1, structurally),
SPLICE C (M0's directive line), SPLICE D (M4 becomes the evidence layer and carries `[the lede →]`),
SPLICE E (the §12 fifth row), SPLICE F (§13 mechanism 7), SPLICE G (`src/signals/register.ts`).
**Deltas:**
- **R6 is written into M0.0b as mechanism 5:** *"A card's `Insight` is not a register entry. The
  register is hand-maintained and reviewed. A build agent that auto-registers every card's Insight
  has defeated the closed family, and test 24(h) is what catches it."*
- The register's seed list includes `data.joins.yield` with `governs: "tab"` for M6 (R5) and
  `data.watchlist.roster` (`ABSENT` / `PULL_THE_FEED` while `k of 5 named` is 0).
- **Zone 3 overflow rule, from the Headline Editor's own risk note, promoted to normative:** if
  `undecided.length > up.length + down.length`, the header renders that ratio, not only the count.
  A collapsed line that hides the true shape of the evidence is the failure this card exists against.
- The pinned refusal row's style-term names are hostable (R8); nothing else on it is.

---

# SPLICE 12 — [MERGED] PART 2 §15 file tree, §12 table, and the ONE test contract

**§12 gains a fifth column "Fused surface"** (Synthesist §SYN-2.8) — X2 / X3 / X4 / X5 / X6 / X0 /
X1 with their unlocks, X3's and X2-row-2's unlock reading *"1 competitor pull **at Tier S**"* —
**and a fifth row** (Headline Editor SPLICE E): her centre panel itself, **SHIPS IN HER POSITION**
at M0.0, ranking unit overruled to registered signals, her named example answered on screen.

**§15 file tree additions (one list, deduplicated):**
```
apps/lofn-kit/src/contracts/ghost.ts          apps/lofn-kit/src/stats/ols.ts
apps/lofn-kit/src/insight/rank.ts             apps/_fixtures/soundry-round.json
apps/lofn-analytics/scripts/derive-public-snapshot.mjs
apps/lofn-analytics/SECURITY.md               apps/lofn-analytics/DEPLOY.md   (rewritten, §0.4/§0.5)
apps/lofn-analytics/src/signals/register.ts   register.spec-fixtures.ts
apps/lofn-analytics/src/load/soundry.ts       src/load/competitors.ts
apps/lofn-analytics/src/watchlist/  schema.ts validate.ts project.ts compose.ts load.ts
apps/lofn-analytics/src/cards/m0-lede/
apps/lofn-analytics/src/cards/m5-7-watchlist/ m5-8-status/ comp-hosting-numbers/ comp-observation-form/
apps/lofn-analytics/src/cards/m6-x0-join-matrix/ m6-x1-distributed/ m6-x2-lyric-three-instruments/
apps/lofn-analytics/src/cards/m6-x3-emo-vocabulary/ m6-x4-market-position/ m6-x5-two-clocks/ m6-x6-the-loop/
apps/soundry/panel-routes.mjs                 apps/soundry/tests/panel-token.spec.ts
data/watchlist/competitors.template.json      (committed, all handles null)
vault/private/collaborators.local.md          (git-ignored, never committed)
```
**STRUCK from the §15 skeleton:** `Dockerfile`. A container in the tree is an invitation to
`docker run -p 0.0.0.0:8080` with the raw snapshot mounted, and it buys nothing a static Pages
deploy does not already give us.

**§15 test contract, items 10–26 (R10). Every claim in §0 that is not tested is theatre.**

10. `posture-hosted.spec.ts` — build with `POSTURE=hosted`; `dist/**` contains none of: `user_id`,
    any `tags` substring from the real corpus, any path under `data/snapshots/`, any path under
    `data/watchlist/`, any refresh handler, the string `server.mjs`, the card ids `m5-7-watchlist` /
    `m5-8-status` / `m6-x6-the-loop`, or an absolute `http(s)://` origin other than the CSP line.
11. `derive-public.spec.ts` — against the real 210-row snapshot: every output row's key set ⊆ the
    allowlist; `rows.every(r => r.is_public === true)`; **zero characters of any source `tags`
    value** in the concatenated output; the exported `style_term` set ⊆ the screening list (R8); an
    injected unknown source field ⇒ **exit 1 naming the field**. Default-deny proven, not assumed.
12. `no-auth-code.spec.ts` *(both apps)* — `password`, `bcrypt`, `argon2`, `jsonwebtoken`,
    `jwt.verify`, `session-secret`, `login`, `signin`: all absent from `src/**` and `*.mjs`.
13. `loopback-default.spec.ts` *(both apps)* — spawn with no flags; the listener's address is
    `127.0.0.1`; a connection to the same port on a non-loopback local interface is refused.
    **A build agent that widens the bind to "fix" collaborator access turns this red. That is the
    test's whole job.**
14. `join-evidence.spec.ts` — V19–V23.
15. `ghost-autoenable.spec.ts` — one spec, two fixtures, two outputs, **zero conditionals inside the
    card module** around its `plot()` call.
16. `soundry-reader.spec.ts` — parses the fixture; refuses mutated copies (`randomized:false`,
    `ci_level:0.90`, `prereg_hash:""`) and each refusal **reaches the Schema Guard** rather than
    being dropped.
17. `competitors-reader.spec.ts` — globs `*/latest.json`, never the single-file pattern; absent dir
    ⇒ `never-pulled` with two distinct counters; present-but-empty ⇒ `emitter-silent`;
    header-null/row-0 ⇒ `pull-failed`; **no unprojected row is reachable from a card.**
18. `fusion-refusals.spec.ts` — X4 emits no fitted line, no size channel, no correlation coefficient
    at any fixture size; X3 emits no per-tag row at any artist count and no competitor string at any
    tier; X5 renders no competitor style-term clock (R4); X1's tier is `OBSERVATION` for every
    fixture; X6's observational end never exceeds `OBSERVATION`.
19. `emit-snapshot.spec.ts` *(Soundry, extend)* — the emitted `soundry/latest.json` contains no key
    under `panel`, no free-text note, no `anon-` pseudonym. **A3 never leaves the round directory.**
20. `watchlist.spec.ts` — schema, five-slot enum lock, sixth-slot refusal, duplicate refusal naming
    both slots, the `why_this_one` floor, URL extraction offered-not-applied, the handle screen,
    and **`study` defaults to `"hosting"` and no app code path writes it.**
21. `competitor-projection.spec.ts` — (a) every projected row's keys ⊆ the tier whitelist; (b) the
    CardSpec sweep (V23); (c) no competitor card declares a `style_term` filter and no style-board
    card accepts a competitor row; (d) **no string originating in a competitor snapshot appears in
    any rendered output at any tier** (R3).
22. `watchlist-draft.spec.ts` — a `localStorage` draft clears no blocker, enters no
    `evidence.source[]`, and changes no counter.
23. `watchlist-states.spec.ts` — S0..S6 each render exactly one primary action; a `pull-failed`
    fixture never renders a 0 or an axis.
24. `lede.spec.ts` — (a) every register entry has a non-empty `falsifier` and parseable
    `registeredAt`; (b) no Zone-1/2 row has `interval.method === "none"`, `direction === "unknown"`,
    `POST_HOC_BAND`, or `eraControlled === false` on a contrast; (c) zones capped 3/3/3; (d) sort is
    `margin` desc with the tie chain, stable across identical fixtures; (e) a one-snapshot fixture
    puts every `suno` temporal row at the literal `UNKNOWN` and leaves Zones 1–2 empty; (f) the
    pinned row is present, last, and shares `insight.id` with the M2.4 board; (g) no style-term
    permalink in rendered output; **(h) a CardSpec absent from the register never appears in the
    LEDE and never increments `k` (R6).**
25. `lede-m41-parity.spec.ts` — LEDE rows ⊆ M4.1 rows by id, on every fixture.
26. `directive-lede-agreement.spec.ts` — the Tab Directive's `insight.id` equals the LEDE's Zone-0
    id, else its rank-1 id — **on M0 and on M6.**

**`forbidden-strings.spec.ts` — one merged extension list:** `"combined reach"`, `"vs the market"`,
`"outperforms"`, `"beats"`, `"top competitor"`, `"best artist"`, `"beat them"`, `"like theirs"`,
`"sounds like"`, `"outperform"`.

**`panel-token.spec.ts` *(Soundry §6)*** — [ADOPT] Threshold Keeper verbatim: `GET /soundry/rounds`
on the panel listener → **404**; no cookie / expired / revoked → 403; a second distinct binding →
403 **plus an operator-visible warning**; `GET /panel/<token>` → 302 whose `Location` does not
contain the token; the LAN listener self-closes at last expiry; the bench listener unaffected.

**`.gitignore` additions (R11):** `data/public/`, `data/soundry/`, `data/watchlist/`,
`vault/private/`, `**/panel-sessions.json`. **The `data/snapshots/` line is not touched by any lane.**

**CANONICAL PATHS additions (§8.3.4, mirrored to `.agents/`, `python3 scripts/check_skill_mirror.py`
must print IN SYNC):** `WATCHLIST → data/watchlist/`, `SNAPSHOTS_SOUNDRY → data/snapshots/soundry/`,
`SNAPSHOTS_PUBLIC → data/public/`, `FIXTURES → apps/_fixtures/`, `DAYMARKS → data/daymarks/`,
`SOUNDRY_ROUNDS → data/soundry/rounds/`, `SOUNDRY_OUT → output/soundry/`.

---

# SPLICE 13 — `## Overrules ACCEPTED`, APPEND (all three decision blocks)

- **COLLABORATOR ACCESS IS IMPLEMENTED, AT TWO GRANULARITIES, BECAUSE THE TWO APPS HOLD DIFFERENT
  ASSETS** (Threshold Keeper). Both of her lines were dropped by the previous draft and structurally
  foreclosed by a loopback bind plus a hash-fragment token that gates nothing. Both are now built.
  **Narrowing 1:** a Soundry collaborator is a *panel listener*, not a bench operator — PRESERVES
  Board 2's own assignment (*"get feedback from listeners"*) and Board 3's single-operator Anvil.
  **Narrowing 2:** a hosted-Analytics collaborator sees derived measures, not the corpus — PRESERVES
  all four Board-6 insights, which are computable from derived measures, while A1 is not merely
  locked but absent.
- **THE `#/panel/<session-token>` HASH FRAGMENT IS A DEFECT, CORRECTED, NOT AN OVERRULE.** The token
  moves into the path, is exchanged for an `HttpOnly; SameSite=Strict` cookie, and is stripped from
  the visible URL. PRESERVES her blind-panel design unchanged; the token now does the job its name
  implied.
- **THE LOOPBACK BIND SURVIVES AS A SCOPED CONTROL RATHER THAN AN AMBIENT ASSUMPTION.** Default
  `127.0.0.1` everywhere, one opt-in self-terminating LAN listener whose route table contains the
  panel and nothing else, and a hosted Analytics posture with **no server at all**. PRESERVES "the
  whole app never leaves the machine" for the Soundry, literally and permanently.
- **THE FOUR FEEDS ARE COMBINED ON A NAMED SURFACE (M6 · CONFLUENCE), NOT SCATTERED** (Synthesist).
  Her Board-6 line asks that the four sources be *combined*; four panels side by side is four panels.
  Two fused cards are LIVE today with no new feed (X0, X0b, X1); four are ghosts whose axes,
  labels and quadrant rules draw at full ink so the reserved shape is itself the argument for
  pulling the feed. PRESERVES every one of her four named insights in their existing single-feed
  homes — M6 asks the same four questions ACROSS feeds and never replaces M2.
- **THE THREE INSTRUMENTS ARE NEVER POOLED** (Synthesist, X2/X6). Observational and randomised
  estimates share an axis and never an average. The connector length between them is what the
  confound was worth, and a single fused figure would delete the most useful number on the tab.
  PRESERVES her Board-4 demand that a hypothesis carry a point, an interval, an n and a day count.
- **THE COMPETITOR BLOCKER BECOMES CLEARABLE** (Task Taxonomist). §14.4's "no button that cannot
  work" is upheld — no refresh button ships (R2) — but a blocker whose input has no home in the app
  is a dead end, and a dead end is the one thing an interface may not have. The missing artefact was
  never the button; it was the roster. PRESERVES her feed #2 exactly as drawn, and AUTONOMY
  untouched, because the app composes the command and never runs it.
- **THE ROSTER IS SLOT-LOCKED TO APPENDIX C AND PROJECTED TO HOSTING FIELDS BY DEFAULT** (Task
  Taxonomist, amended by R3). Five named slots, no free add, a mandatory hosting question per handle,
  sort locked to slot order, every style/title/lyric/persona field dropped in the loader, and shape
  aggregates available only behind a per-handle flag **she** writes. PRESERVES `vault/SOCIAL_VOICE.md`
  Appendix C's standing rule as a build failure rather than a memory — "make music like theirs" is
  not discouraged here, it is unrepresentable — **while keeping Board 6's cross-source lyric-length
  and EMO questions askable the day she says so.**
- **THE RANKED DIRECTIONAL PANEL RETURNS TO THE CENTRE, FIRST CARD OF M0** (Headline Editor). Its
  relocation to Headwinds is reverted **as an unnamed drop, not ratified as an overrule** — a tab
  named after one of two directions deletes the up arrow, and a tab the reader must first choose is
  not a headline. What IS overruled is the **unit**: her rows were genre tokens; the LEDE ranks
  registered signals, and the whole style board appears as exactly ONE pinned row that refuses.
  PRESERVES her actual instrument — *tell me, in words, what is moving and which way, before I look
  at anything* — which twenty token rows at n=1–39 would destroy rather than deliver. Her named
  example is not dropped; it is **answered on screen in her own words**, with the route that could
  decide it and the days-to-decision printed beside it.

# SPLICE 14 — `## Overrules REJECTED (her intent wins)`, APPEND

- **REJECTED — a shared password, a shared link secret, or a bespoke login on either app.** One
  secret everyone holds, unrevocable per person, pasted into a chat within a week. Use the
  identity-aware proxy she already owns and ship **zero** lines of auth code.
- **REJECTED — hosting a remote panel for off-LAN listeners in v1.** A new origin, hosted audio, a
  public write path to A3 and a second Access configuration, for a capability she never asked for.
  Deferred to OQ-SEC-2.
- **REJECTED — a "three-posture table" in `DEPLOY.md` presenting hosted as a build flavour.** Hosted
  is a **different dataset**; as a flag, the first misconfiguration publishes the corpus. Rewrite
  around §0.4/§0.5. `Dockerfile` struck from §15.
- **REJECTED (Synthesist, in part — mine) — `data/watchlist.yaml` and a per-handle Refresh button.**
  R1, R2. His intent, that the blocker clear from inside the app, is fully served by M5.7.
- **REJECTED (Synthesist, in part — mine) — X5's watched-artist style-term clock.** R4. Appendix C
  is hers and it is specific. Her setter/breaker insight survives on the ours-vs-market pair, which
  is the pair that makes the two words mean different things anyway.
- **REJECTED (Task Taxonomist, in part — mine) — a projection whitelist so tight that Board 6's
  cross-artist craft questions become unaskable forever.** R3. A shape is not a prompt. The default
  is his; the door is hers.
- **REJECTED — any app-written roster, over any local endpoint.** The sealed no-write posture is
  load-bearing and `server.spec.ts` exists to prove it. Compose, display, download, copy; she pastes.

# SPLICE 15 — `## Open questions for The Scientist`, APPEND

- **OQ-SEC-1 · May Lofn Analytics be hosted at `analytics.lofn.ai`, on derived data, behind
  Cloudflare Access?** The redaction means a collaborator sees every insight you drew on Board 6 and
  none of the style-prompt corpus, none of the roster, and no round. **RECOMMENDED DEFAULT: build
  it, test it, do not deploy it** until (a) you clear P0 by deleting the exposed OpenClaw `/cdp`
  Worker and (b) you name the first collaborator. The hosted door costs nothing while it stays shut.
- **OQ-SEC-2 · Do you need remote (off-LAN) panel listeners?** v1 supports listeners on your network,
  four-hour sessions. **RECOMMENDED DEFAULT: no.** Run the first rounds on-LAN and learn what the
  panel needs before we build a public write surface for third-party free text.
- **OQ-SEC-3 · Who is the first collaborator, and what is their review-by date?** Prune-by-default;
  an empty list is a valid state. Without one name and one date, §0.5(e)–(g) is a well-tested door
  with nobody behind it.
- **OQ-WATCH-1 · The five slots are yours to name and the app ships them empty.** Appendix C already
  says what each slot is FOR; M5.7 will not save a handle without a sentence saying what hosting
  behaviour it is being watched for. **RECOMMENDED DEFAULT: fill slot 2 (THE COMMENT-SECTION MODEL)
  first** — its numbers are the ones no endpoint can give us, so it proves the manual observation
  path works before anything depends on it.
- **OQ-WATCH-2 · Do you want any watched handle at `study: "hosting+shape"`?** It unlocks M6·X2 row 2
  (their lyric-length slope against ours) and M6·X3 (vocabulary shape), as **aggregate numbers with
  no words from their prompts ever crossing the loader**. Your Appendix C standing rule says these
  five are studied for hosting, not sound, so the default is off and the app cannot turn it on.
  **RECOMMENDED DEFAULT: leave it off until the hosting numbers have earned their keep.**
- **OQ-LEDE-1 · The register is a judgement call and it is yours.** `src/signals/register.ts` is the
  closed family the LEDE ranks; adding an entry after a snapshot exists permanently flags it
  post-hoc. The seed list needs your review before the first build: which warnings you want ranked,
  and which of them you would accept being told are inside their own noise. **No entry gets added on
  our own after seeing a number move.**
- **OQ-PANEL-1 · Where does the panel's audio come from?** The blind session is now reachable and
  capability-gated, and it still has nothing to play. The Soundry renders nothing and holds no audio;
  CSP is `connect-src 'self'`. **RECOMMENDED DEFAULT: you drop the twelve mp3s into
  `data/soundry/rounds/<id>/audio/` by hand after posting, and round 1 pre-registers
  `instrument: platform` regardless** — so the panel is a bonus instrument, not a dependency, until
  the manual path has been walked once.



# Overrules declared (each naming the intent it preserves)
- COLLABORATOR ACCESS, NARROWED TWICE AND BUILT BOTH TIMES (Threshold Keeper). Narrowing 1: a Soundry collaborator is a panel listener, not a bench operator — PRESERVES Board 2's own assignment of one job to collaborators ('get feedback from listeners') and Board 3's single-operator Anvil with its one prompt box. Narrowing 2: a hosted-Analytics collaborator sees derived measures, not the raw corpus — PRESERVES all four Board-6 insights, every one of which is computable from derived measures, so the collaborator loses nothing she asked them to have.
- THE LOOPBACK BIND BECOMES A SCOPED CONTROL, NOT AN AMBIENT ASSUMPTION. Default 127.0.0.1 in both apps in every posture, one opt-in time-boxed self-terminating LAN listener whose route table contains the panel and nothing else, and a hosted Analytics posture with no server at all. PRESERVES 'the whole app never leaves the machine' for the Soundry, literally and permanently, while making the collaborator requirement real instead of contradictory.
- THE FOUR FEEDS GET A NAMED FUSION TAB, M6 CONFLUENCE, AND A SEVENTH RAIL ITEM (Synthesist). PRESERVES Board 6's literal ask — 'combine all 4 sources into useful insights' — and preserves her four named insights in their existing single-feed homes; M6 asks the same four questions across feeds and never replaces M2.
- THE THREE INSTRUMENTS ARE NEVER POOLED (X2, X6). Observational and randomised estimates share an axis and never an average. PRESERVES her Board-4 hypothesis mark exactly — point, interval, n, day count — while keeping the distance between belief and estimate visible, which a single fused figure would delete.
- X5's WATCHED-ARTIST STYLE-TERM CLOCK IS CUT, AND X5 SHIPS WITH TWO CLOCKS (LOFN, overruling the Synthesist). PRESERVES her Board-6 insight #4: setters and breakers still gets the lead/lag definition from ours-vs-market, which is the pair that makes the two words mean different things. Also PRESERVES her own Appendix C standing rule, which forbids studying these five for sound.
- THE COMPETITOR PROJECTION GATE SHIPS IN TWO TIERS, NOT ONE (LOFN, splitting the Task Taxonomist and the Synthesist). Tier H (hosting fields only) is the default and the only tier any handle ships with; Tier S adds aggregate shape numbers with no string from a competitor file ever crossing the loader, and only she can set it, per handle. PRESERVES vault/SOCIAL_VOICE.md Appendix C's standing rule as a type rather than a memory, AND preserves Board 6's cross-source lyric-length and EMO questions as askable the day she says so.
- THE ROSTER IS SLOT-LOCKED TO APPENDIX C, NOT A FREE HANDLE LIST (Task Taxonomist, over the Synthesist's data/watchlist.yaml). PRESERVES her documented rejection of 'five Suno stars' as an axis: five slots that each solve a different problem, no free add, a mandatory hosting question per handle, sort locked to slot order.
- NO REFRESH BUTTON FOR competitors AT ANY ROSTER SIZE (LOFN, overruling the Synthesist's SYN-2.7). PRESERVES §14.4's 'a button that cannot work is worse than no button' AND preserves his actual intent — the blocker still clears from inside the app, through M5.7 plus a copyable pull command.
- THE COMPETITOR BLOCKER BECOMES CLEARABLE FROM INSIDE THE APP (Task Taxonomist). PRESERVES her feed #2 exactly as drawn — five watched targets — and preserves AUTONOMY untouched, because the app composes the command and never runs it.
- HER CENTRE PANEL RETURNS TO THE CENTRE AS M0.0 (Headline Editor). Its relocation to the Headwinds tab is reverted as an unnamed drop rather than ratified. What is overruled is the UNIT: registered signals, not genre tokens, with the whole style board as one pinned refusal row. PRESERVES her actual instrument — tell me in words what is moving and which way, before I look at anything — and answers her named example (up Baile Phonk, down Anthems) on screen in her own words rather than deleting it.
- THE HOSTED BUILD CARRIES NEITHER THE ROSTER NOR THE ROUND LEDGER (LOFN, extending the Threshold Keeper). PRESERVES his own Trade-off 1 — shrink the asset before arguing about the lock — applied to two assets his patch could not see because the other seats wrote in parallel.
- STYLE-TERM NAMES ARE HOSTABLE; PER-CLIP tags ARE NOT (LOFN). PRESERVES Board 6's 'genre performance' inside the collaborator build, which a blanket style redaction would have silently dropped, while the corpus — the actual method — stays absent from the hosted artefact.
- A CARD'S Insight IS NOT AUTOMATICALLY A REGISTER ENTRY (LOFN, hardening the Headline Editor). PRESERVES the closed-family guarantee that makes k printable and BH meaningful; without it a build agent auto-registers every card and the multiplicity control evaporates.
- Dockerfile IS STRUCK FROM THE apps/lofn-analytics SKELETON (Threshold Keeper). PRESERVES the local-only posture: a container in the tree is a standing invitation to bind 0.0.0.0 with the raw snapshot mounted, and it buys nothing a static deploy does not give us.


# STILL OPEN
- THE PERSISTENT HEADWINDS RIGHT RAIL IS STILL TWENTY WORDS. Board 1 draws a boxed Headwinds rail on the right of the Analytics page. §8 overrules promoting it to a tab and compensates with 'the three highest-severity items also mirror into a persistent right rail on every other tab' — which has no card spec, no width, no grid position, no interaction, no empty state, and no file in the §15 tree (src/shell/ has rail.ts, filters.ts, chrome.ts, directive.ts and nothing for it). The Headline Editor says explicitly that M0.0 is not that rail and must not be used to close it. I agree. The only literal element she drew on the right side of App 1 remains unbuilt.
- 'TOOLTIPS, ALT-TEXT, AND INFORMATIONAL GUIDANCE FOR ALL ELEMENTS' (Board 7) IS STILL NARROWED TO CHARTS, AND THIS PATCH MADE THE GAP BIGGER. §5.1 rule 1 converts 'all elements' into 'all elements that happen to be statistics'. There is no GuidanceNote registry and no guidance-coverage.spec.ts — the Headline Editor's M0.0g cites that test as if it exists; it does not. Meanwhile this patch adds dozens of non-chart controls that need notes: every M5.7 slot chip, handle field, disqualifier chip, Copy/Download button and the DRAFT band; the join-yield fractions; the ghost unlock counters; the panel consent line; the posture indicator. Net: more uncovered elements than before.
- TWO INCOMPATIBLE FILTER VOCABULARIES, AND THE SOUNDRY HAS NO FILTER SHELF AT ALL. Board 7 requires that segmentations and filters carry between tabs. Part 1 §3.2/3.3 freeze mod|snap|co|basis|out|floor|nmin|prov|sel|q|exp|hyp|arm|blind; Part 2 §3.3 uses m|win|pf|out|est|prov with an inverted cohort default and an `est` key that does not exist in the union. Part 3 renders no chip strip at all. Both the Synthesist (JoinEvidence typing, X-card filter applicability) and the Headline Editor (the LEDE's drill href) declare this a BLOCKING dependency — two of the four patches cannot open their lanes until it is resolved.
- THE SONIC ANVIL HAS NO GENERATOR. Board 3 is her core App-2 flow — 'What should we try today?' to twelve cards, each a full Suno quad. Part 3 specifies the ask box, factor parsing, power, balancing, the freeze gate, the card chrome and the export, and never says what writes Title / Style Prompt / Exclude Prompt / Lyrics / Caption, while §0.1 forbids any generation API, forbids a dependencies key, and pins connect-src 'self'. None of the four patches touches it. This is the largest single unbuilt requirement on the whiteboard.
- SOUNDRY DRILL-DOWN: ROUGHLY TWELVE PLOTS DECLARE NO `deeper` AND NONE FAILS LOUDLY. Board 7 requires every insight to link to deeper results, and §14.2 overrules her 'if they exist' conditional to 'every insight links, or states in words why it cannot'. Part 3's A2.1, A3.1, A3.2, B2–B5, B7, B9.1 and C1–C5 give neither `deeper` nor `deeperAbsentReason`, and because the field is optional an omission slips past V15 instead of failing the build. The Soundry ships the exact dead-link-by-omission that overrule was written to prevent.
- THE PANEL STILL HAS NOTHING TO PLAY. The Threshold Keeper's PANEL LAN MODE makes the blind session reachable and capability-gated, and its route table names GET /panel/assets/* — but the Soundry renders nothing, holds no audio, and CSP is connect-src 'self'. Board 2's 'get feedback from listeners' is now access-controlled and still not listenable. I wrote the gap into §0.3 in words and raised OQ-PANEL-1 rather than let a builder infer a fetch; that is disclosure, not a fix.
- cleanFeedbackText() / input-cleaning.mjs DOES NOT EXIST IN THIS REPOSITORY. Part 3 §3.3 and B7 gate every listener note on it; Part 1 never specifies it and §8's tree has no such file; the only implementation is in E:/git/saga. It now guards asset A3 — third-party data held in trust under vault/COLLABORATOR_STANDARD.md — which makes it load-bearing for the one asset we cannot unilaterally forgive ourselves for losing.
- THE HOUSE BASELINE AND THE FROZEN POWER CONSTANTS STILL DO NOT REPRODUCE FROM DISK. Both critics: Part 1 and Part 3 carry 4.49% / 258 plays / sigma2_true 1.204e-4 while the snapshot yields 4.3695% median / 267 plays and a measured between-song variance of 1.369e-4 (MDE x1.161, not x1.150). The Headline Editor's `margin` divides by claim.mde80, so the LEDE's ranking is stable in ORDER but its printed multiples are wrong until this is settled. Declared as a dependency by that seat and unresolved by this panel.
- THREE FEEDS STILL HAVE NO DATA, AND FeedId MEMBERSHIP IS UNDECIDABLE FOR TWO OF THE THREE LIVE JOINS. socials and distribution — the two feeds carrying most of Part 2 — are not in KNOWN_FEEDS (scripts/ingest_receiver.mjs:43-52), so JoinEvidence.left.feed cannot be typed for X0's and X1's joins. The Day Mark prints three different denominators (9, 8, and a disk truth of 8-with-1-filled). M6 is designed but two of its live cards are blocked behind a type nobody owns yet.
- NOBODY OWNS THE SECOND SNAPSHOT. 'Show trends as well as now' (Board 7) is what M0.6, M2.5(a), M3.7, every spark, B3 and now the LEDE's entire temporal band are locked against. The answer everywhere — including in the LEDE's Zone-0 blocker line this patch just installed — is copy telling her to run python3 scripts/ingest_suno.py daily. Nothing schedules it. The whole trend layer rests on a habit, and this patch made the habit more visible without making it more likely.
- HER FOUR ORIGINAL OPEN QUESTIONS ARE STILL HERS AND STILL UNANSWERED: what 'Decimabal' is actually called; whether 'Total Lift vs Avg artist' becomes self-referential (accepted as DECIBAL-vs-concurrent-control, but the roster this patch adds does NOT reopen it — the DECIBAL firewall is explicit); which five competitors; and which Spotify artist identity is canonical.


# Revised build order
- U0 · COORDINATOR, SERIAL, BLOCKS EVERY LANE. python3 scripts/run_lock.py acquire apps/lofn-analytics/.lofn --run-slug <slug> --engine claude (exit 3 = STOP; the sidecar path, not the build tree). Append to E:/git/private-claw/.gitignore ONLY: data/public/, data/soundry/, data/watchlist/, vault/private/, **/panel-sessions.json — the data/snapshots/ line is not touched by any lane (R11). Commit E:/git/private-claw/data/watchlist/competitors.template.json with all five handles null. Add the seven CANONICAL PATHS rows to BOTH .claude/skills/lofn/SKILL.md and .agents/skills/lofn/SKILL.md; PYTHONIOENCODING=utf-8 python scripts/check_skill_mirror.py must print IN SYNC.
- U0b · COORDINATOR, SERIAL, BLOCKS L-LEDE AND L-FUSE-A. Resolve the Part 1 §3.2 / Part 2 §3.3 filter-key vocabulary into one frozen union, and resolve the FeedId membership of socials and distribution against scripts/ingest_receiver.mjs:43-52. Both are STILL-OPEN findings owned by other seats; L-LEDE's drill href and L-FUSE-A's JoinEvidence.left.feed cannot be typed until they land. Do not open those lanes on a guess.
- L-KIT · SHARED TYPES AND CONTRACTS (parallel with nothing; unblocks every app lane). E:/git/private-claw/apps/lofn-kit/src/plot/spec.ts (SPLICE 1: EmptyKind x4, UnlockCounter, EmptyState, JoinEvidence, Provenance additions) · apps/lofn-kit/src/plot/validate.ts (V19-V23) · apps/lofn-kit/src/contracts/empty.ts · apps/lofn-kit/src/contracts/ghost.ts · apps/lofn-kit/src/stats/ols.ts (olsSlope) · apps/lofn-kit/src/insight/rank.ts (signedLowerBound, margin, rankSignals — the ONLY implementation) · apps/lofn-kit/tests/empty-vs-zero.spec.ts (extended to four new kinds, both themes).
- L-SEC-ANL · ANALYTICS SECURITY POSTURE (needs L-KIT; parallel with L-SEC-SND, L-WATCH, L-FUSE, L-LEDE). apps/lofn-analytics/scripts/derive-public-snapshot.mjs (default-deny allowlist, exit 1 naming an unlisted field; excludes tags, user_id, non-public rows, data/watchlist/**, the whole soundry feed) · apps/lofn-analytics/vite.config.ts POSTURE=hosted strip · apps/lofn-analytics/package.json deploy:hosted script (derive -> build -> tests -> wrangler; refuses on any test failure or stale data/public/) · apps/lofn-analytics/SECURITY.md · apps/lofn-analytics/DEPLOY.md (rewritten around §0.4/§0.5, three-posture table deleted) · tests/posture-hosted.spec.ts · tests/derive-public.spec.ts · tests/no-auth-code.spec.ts · tests/loopback-default.spec.ts. DELETE apps/lofn-analytics/Dockerfile from the §15 skeleton.
- L-SEC-SND · SOUNDRY PANEL ACCESS (needs Soundry U3+U5; parallel with every Analytics lane). apps/soundry/panel-routes.mjs (the second route table: GET /panel/<token>, GET /panel/assets/*, POST /soundry/observations/panel, GET /soundry/health — 404 on everything else) · apps/soundry/server.mjs (--panel-lan, --panel-revoke, --panel-close, TTL, self-termination, terminal countdown, second-binding detection) · apps/soundry/panel.html (consent line, token-stripped 302 landing) · apps/soundry/emit-snapshot.mjs (panel-namespace exclusion allowlist) · apps/soundry/tests/panel-token.spec.ts · tests/emit-snapshot.spec.ts · tests/no-auth-code.spec.ts · tests/loopback-default.spec.ts.
- L-WATCH · THE WATCHLIST (needs L-KIT + Analytics W1-L1 loader; parallel with L-FUSE and L-LEDE; NO dependents — cut this lane last and cut comp-observation-form first if it must shrink). apps/lofn-analytics/src/watchlist/{schema,validate,project,compose,load}.ts (project.ts carries both tiers per R3) · src/cards/m5-7-watchlist/ · m5-8-status/ · comp-hosting-numbers/ · comp-observation-form/ · src/load/invariants.ts (I11) · snapshot-route.mjs (the three GET routes, handle re-screened at the route) · scripts/check-build.mjs (no data/watchlist/ bytes in dist/) · tests/watchlist.spec.ts · competitor-projection.spec.ts · watchlist-draft.spec.ts · watchlist-states.spec.ts.
- L-FUSE-A · LIVE FUSION, NO NEW FEED (needs L-KIT + U0b; parallel with L-WATCH, L-LEDE). apps/lofn-analytics/src/cards/m6-x0-join-matrix/ · m6-x1-distributed/ · apps/lofn-analytics/tests/join-evidence.spec.ts. Build this before any ghost: X0 proves the join machinery against real files, and X0 alone changes what she believes about the other five cards.
- L-FUSE-B · THE READERS AND THE ANTI-DRIFT FIXTURE (needs L-KIT; parallel with L-FUSE-A). apps/_fixtures/soundry-round.json · apps/lofn-analytics/src/load/soundry.ts · src/load/competitors.ts (glob */latest.json, then projectCompetitor) · apps/lofn-analytics/tests/soundry-reader.spec.ts · competitors-reader.spec.ts · apps/soundry/tests/emit-snapshot.spec.ts (the opposite direction against the same fixture).
- L-FUSE-C · THE GHOSTS (needs L-FUSE-A + L-FUSE-B + L-WATCH/project.ts). apps/lofn-analytics/src/cards/m6-x2-lyric-three-instruments/ · m6-x3-emo-vocabulary/ · m6-x5-two-clocks/ · m6-x6-the-loop/ · m6-x4-market-position/ · apps/lofn-analytics/tests/ghost-autoenable.spec.ts · fusion-refusals.spec.ts.
- L-LEDE · THE LEDE AND THE REGISTER (needs L-KIT/rank.ts + U0b + a resolved house baseline and mde80; parallel with L-WATCH and L-FUSE). apps/lofn-analytics/src/signals/register.ts + register.spec-fixtures.ts (hand-maintained, Scientist-reviewed) · src/cards/m0-lede/index.ts · src/cards/m4-signal-table/index.ts (the [the lede ->] chrome and the shared rankSignals() call) · apps/lofn-analytics/tests/lede.spec.ts · lede-m41-parity.spec.ts · directive-lede-agreement.spec.ts · multiplicity.spec.ts (mechanism 7). DO NOT OPEN until the 4.49%/4.37% and sigma2_true conflicts are settled — the ranking ORDER is stable but every printed multiple is wrong until then.
- U-FINAL · COORDINATOR, SERIAL, AFTER ALL LANES. Merge snapshot-route.mjs (L-WATCH's three routes plus the two-segment competitors case) · merge the single forbidden-strings.spec.ts extension list · renumber the §15 test contract to 10-26 exactly once (R10) · scripts/card-registry-check.mjs extended so a spec with a ghost and no empty state fails · run the full suite in both postures · write the vault/RUN_LEDGER.md entry and memory/YYYY-MM-DD.md · python3 scripts/run_lock.py release apps/lofn-analytics/.lofn.


---

# APPENDIX — seat patch: THE HEADLINE EDITOR (after Amanda Cox and the NYT Graphics desk) — annotation layers, headline-as-finding, the eight-second reader. (Amanda Cox / NYT Graphics annotated-chart practice: the graphic states its finding in words before it shows it in ink; the annotation layer is the primary layer, the ink is the evidence layer. Cited in the third person as record. Extrapolating beyond the record is marked inline.)

FINDING: The Scientist drew a centre panel of ranked directional statements ("↑ Baile Phonk", "↓ Anthems") as the first object on the Analytics board; the spec maps that slot to M0 SIGNAL, whose six cards contain no ranked directional list at all, while the surviving ranked list lives on the HEADWINDS tab (M4.1) — a relocation never named as an overrule.

FILES: E:/git/private-claw/docs/reference/lofn_analytics_and_soundry/BUILD_SPEC.md, apps/lofn-analytics/src/cards/m0-lede/index.ts, apps/lofn-analytics/src/signals/register.ts, apps/lofn-kit/src/insight/rank.ts, apps/lofn-analytics/tests/lede.spec.ts, apps/lofn-analytics/tests/lede-m41-parity.spec.ts, apps/lofn-analytics/tests/directive-lede-agreement.spec.ts, apps/lofn-analytics/tests/multiplicity.spec.ts, apps/lofn-analytics/src/cards/m4-signal-table/index.ts

＝＝＝ PATCH — FINDING 4 · THE LEDE ＝＝＝
Seat: THE HEADLINE EDITOR. Target file: `E:/git/private-claw/docs/reference/lofn_analytics_and_soundry/BUILD_SPEC.md`

## 0 · My ruling, stated before the splices

I decline the overrule option my brief offers. The relocation cannot be defended on the reader's eight seconds, and I will not launder a silent drop into a named overrule to close a ticket.

Three reasons, in the order a reader meets them.

1. **Her centre panel is the only object on that page made of sentences.** The other five M0 cards are ink — a dial, a numeral, a scatter, a bar, a dumbbell. A reader who gives you eight seconds reads words. The annotated-chart tradition the NYT desk practises puts the finding in words above the ink; her layout already does that, and the spec replaced it with six graphics and a directive line. **Extrapolating: the spec's authors mistook the Tab Directive (one sentence, §4.4) for a headline layer. One sentence is a headline. A ranked list is a lede — it tells you what is moving, how many things are moving, and which way, before you have chosen a tab.**

2. **"HEADWINDS" is a pre-framed word.** Filing the ranked list under it deletes the up arrow. She drew ↑ and ↓ in the same block on purpose: §8's own argument — "you cannot know whether *down: Suno uploads* is remarkable without seeing the ups on the same scale" — is an argument for keeping both arrows together, which is an argument for her centre panel, not for a tab named after one of the two directions. I agree with §8's premise and reject its conclusion. This is my disagreement with the seat that wrote §8: the comparison class was right, the address was wrong. A dashboard that files everything under Headwinds is a doom console, and the disappointed idealist's tool is not a doom console any more than it is a growth console.

3. **A tab the reader must first choose is not a headline.** M4 is the fifth tab in the Music rail. At eight seconds she sees tab one.

So: **the panel comes back to the centre, first card, above the dial.** What I *do* overrule — named, argued, and logged in §Overrules ACCEPTED below — is the *unit* of her ranking. Her rows were genre tokens. Mine are registered signals, and genre appears as exactly one pinned row that refuses.

---

## SPLICE A — new card, inserted in `## 4 · MUSIC · Tab M0 — SIGNAL`

**Anchor:** insert immediately AFTER the line beginning `**Directive today:**` and BEFORE `### M0.1 · Can this app compute a trend yet?`. M0.1–M0.6 keep their numbers; the new card is M0.0 so no renumbering churn.

> ### M0.0 · THE LEDE — *her centre panel, returned to the centre*
>
> **This is the first object on the page, full-width, above the dial.** No chart. A numbered list of sentences with a direction glyph, exactly the shape she drew. It renders on every rail (`?m=music|art|animation`) and every register entry declares its modality.
>
> **What it renders.** Nothing authored. Each row is one `Insight` (kit §4.1) printed as: glyph · `headline` (already ≤ 90 chars and already generated from the tier-locked template — the LEDE has no prose surface of its own and therefore no new way to over-claim) · tier chip · interval · `nEffective` · `q` · as-of · `↗`. Below each headline, at 14/1.5, the row's `subhead` (which always carries the interval and `nEffective` by schema) and, where the cohort's median plays < 500, the mandatory denominator sentence (§4.3 rule 3).
>
> **Structure — four zones, ten objects maximum. Eight seconds is the budget and it is a hard budget.**
>
> - **Zone 0 · THE BLOCKER LINE** (0 or 1 row, pinned top, ember rule beneath). The register entry carrying `governs: "tab"` whenever its tier is `ABSENT`. Today: *"No second Suno snapshot exists. Every direction on this page that depends on time is UNKNOWN, not zero. `python3 scripts/ingest_suno.py` — the second pull is what makes this page work."* — tier `ABSENT` / `PULL_THE_FEED` / `ops.runbook`.
> - **Zone 1 · RISING** — up to **3** rows, ink.
> - **Zone 2 · FALLING** — up to **3** rows, ember, below a single 1px ash rule that spans the list. Position and ink carry the sign; no badges, no pills (consistent with M4.1).
> - **Zone 3 · CANNOT BE TOLD APART** — up to **3** rows, ink-faint, collapsed to a one-line summary with a disclosure caret: *"{j} more registered signals are inside their own noise. Open →"*. They are on screen because a list that only shows the decidable ones is a fishing report.
> - **Zone 4 · THE PINNED REFUSAL ROW** — always last, never ranked, never suppressible. Specified in full at §M0.0e.
>
> **The `<ol>` is the accessible artefact.** There is no SVG in this card except the 12px glyph, which is `aria-hidden` and paired with visually-hidden text: temporal → *"rising against its own past"* / *"falling against its own past"*; contrast → *"higher than the era-matched baseline"* / *"lower than the era-matched baseline"*; undecided → *"cannot be told apart"*. Glyphs are shape-coded, never colour-alone: temporal `↗ ↘`, contrast `△ ▽`, undecided `–`.
>
> **Header line, always rendered, never a tooltip:** *"{k} signals registered before this snapshot · ranked by how far each effect's conservative edge clears its own detection floor · Benjamini-Hochberg across all {k} · 95% intervals ({CI_LEVEL})."*

> #### M0.0a · The two kinds of direction, and why they may never share a glyph
> A row's `kind` is `"temporal"` or `"contrast"`, and it is not cosmetic — *up against our own past* and *up against the rest of the catalogue* are different claims, and a reader who conflates them will read the calendar as craft.
> - **`temporal`** — Δ of the same measure on the same surface across ≥ 2 observations. Interval by the measure's own estimator (Wilson difference for rates; the M3.1 disjoint window differencing, incl. its negative-difference guard, for distribution windows). With < 2 observations the row's `direction` is `"unknown"`, `interval.method === "none"`, and it renders an em-dash + hatch + the literal `UNKNOWN` per §4.3 rule 7. It cannot enter Zone 1 or 2.
> - **`contrast`** — era-matched cohort vs the dated baseline. `controls.eraControlled === true` is **required** to enter Zone 1 or 2; `false` caps at `OBSERVATION` (§4.3 rule 4), prefixes *"Uncontrolled:"*, and drops the row to Zone 3.
> Today the only rows that can carry a temporal direction are on the `distribution` feed (the nested DistroKid windows are the sole genuine trend structure on disk). Every `suno` temporal row is `UNKNOWN` and says so.

> #### M0.0b · The register — multiplicity is controlled by *closing the family before looking*, not by adjusting after
> New file: `apps/lofn-analytics/src/signals/register.ts`. The LEDE renders **only** registered signals. Nothing enters the list by being computed; a thing is computed **because** it was registered.
>
> ```ts
> export type DirectionKind = "temporal" | "contrast";
> export type SignalFamily = "data" | "reach" | "craft" | "doctrine"; // the M4.2 quartet
>
> export interface SignalRegistration {
>   id: string;                       // "reach.distribution.window_ladder"
>   label: string;                    // <= 34 chars, in HER vocabulary
>   modality: Array<"music"|"art"|"animation">;
>   family: SignalFamily;
>   kind: DirectionKind;
>   registeredAt: string;             // ISO date. This is evidence, not metadata.
>   falsifier: string;                // REQUIRED. "A headwind with no falsifier is a mood." (§8 M4.3)
>   governs?: "tab";                  // ABSENT => becomes the Zone 0 blocker line
>   compute(ctx: SignalCtx): Insight; // returns a FULL kit Insight; no partials
> }
> ```
>
> Four mechanisms, each with a test:
>
> 1. **Fixed k.** `k = registry.filter(byModality).length` and it is printed in the header. It does **not** grow with the style-token count, because —
> 2. **Style tokens are ONE entry.** `music.style.board` is a single registration whose value is the board's omnibus result, never twenty per-term registrations. This is §13.1 ("the addressable object is the board") enforced at the register rather than restated at the card.
> 3. **Pre-registration is machine-checked.** `registeredAt > snapshot.pulled_at` ⇒ the row is flagged `POST_HOC_BAND` (the flag already exists in §4.1), capped at `OBSERVATION`, and **barred from Zones 1 and 2** regardless of its interval. A signal invented after seeing the data may be shown; it may not be ranked as a direction.
> 4. **BH across the register, not across the card.** `qBH` is computed over all `k` registered signals for the active modality, and the same `q` drives tier promotion (§4.2) and appears on the row. A signal cannot buy significance by being displayed next to fewer neighbours.

> #### M0.0c · The ranking rule, stated exactly
> Ranking by point estimate is the failure mode this whole app is built against, and ranking by raw lower bound is impossible across rows measured in pp, ratios and counts. The LEDE ranks on a **unit-free margin**:
>
> ```ts
> // apps/lofn-kit/src/insight/rank.ts   (new; the ONLY implementation)
> export function signedLowerBound(i: Insight): number | null {
>   const { lo, hi, method } = i.claim.interval;
>   if (method === "none" || i.claim.direction === "unknown") return null;
>   if (lo <= 0 && hi >= 0) return null;              // crosses the null -> Zone 3
>   return i.claim.direction === "up" ? lo : -hi;     // positive by construction
> }
>
> /** How many times its own 80%-power detection floor the conservative edge clears. */
> export function margin(i: Insight): number | null {
>   const b = signedLowerBound(i);
>   if (b === null || i.claim.mde80 === null || i.claim.mde80 <= 0) return null;
>   return b / i.claim.mde80;
> }
>
> export function rankSignals(xs: Insight[]): { up: Insight[]; down: Insight[]; undecided: Insight[] };
> ```
>
> - **Band membership** = the sign of a null-excluding 95% interval (`CI_LEVEL`, the frozen constant — the LEDE introduces no 90% interval), plus `!POST_HOC_BAND`, plus `eraControlled` for `contrast` rows.
> - **Sort key within a band** = `margin` descending. Ties: `tierRank` desc → `nEffective` desc → `id` ascending (stable, so the list does not reshuffle between pulls for cosmetic reasons).
> - **`margin === null` ⇒ Zone 3**, sorted by `nEffective` descending.
> - **Suppression.** `margin < 1` **or** `typeM > 1.5` ⇒ the point estimate is **replaced** by the interval and the MDE, not annotated (§4.3 rule 10, extended). A row can be ranked and still refuse to show a number.
> - **Magnitude stays bound to tier** (§4.2). The LEDE never emits a `MAKE_IT` that its tier does not license, and no LEDE row may originate a `MAKE_IT` from a style token (§13.6).
>
> Extrapolating: `margin` is my own construction, not the source figure's. I justify it on the desk's rule that a ranking must be readable as a sentence — *"this one clears its own detection floor by 2.4×, that one by 0.3×"* is a sentence a tired reader can hold; "lower bound 0.7pp vs lift ratio 1.03" is not.

> #### M0.0d · The relationship to M4.1 — one array, two depths, zero duplication
> **`rankSignals()` is called exactly once per render and both surfaces consume its output.** The LEDE is the **headline layer** (words only, top 3/3/3, no ink). M4.1 is the **evidence layer** (all `k` rows, sparkline-or-step on a shared within-family axis, slope, provenance glyph). Neither computes a signal the other does not have; a row cannot appear in one and be absent from the other.
> - Every LEDE row's `drill` is `{ href: "#/music/headwinds?sel=<id>", label: "see the series" }` — the same row on M4.1, scrolled and highlighted, with the filter state carried by the reconciled shelf serializer. **The LEDE introduces no new filter key.** (Dependency: the `Part 1 §3.2` / `Part 2 §3.3` key-vocabulary conflict flagged by both critics must be resolved first; the LEDE uses whatever the reconciled union names for selection and must not be built before that reconciliation lands.)
> - M4.1 gains one line of chrome, top-right: `[the lede →]` linking `#/music/signal#lede`.
> - `tests/lede-m41-parity.spec.ts`: *"the LEDE's rows are a subset of M4.1's rows, by id, on every fixture."*

> #### M0.0e · THE PINNED REFUSAL ROW — where "↑ Baile Phonk" actually lands
> **Answer to the question my brief asks: no. Genre-level directional claims are not supportable at current n, and they are not close.** The measured position, from the spec's own M2.4 cells: `baile n=5`, `phonk n=6`, `anthem n=6`, against `k=20` screened terms, with **45 of 46 modern clips matching two or more terms** — overlapping sets, not a partition. A per-genre direction at n=5 is a coin flip with a caption.
>
> So the slot where she wrote her example renders **one row, pinned last, never ranked, never collapsible**:
>
> - **Row headline:** the M2.4 board's own generated headline object — **the identical `Insight`, not a re-derivation** — so the two can never drift. Today that is a `NULL`/`OBSERVATION` sentence.
> - **Beneath it, the rebuttal, every number interpolated live from that same computation and none of them typed into the spec:**
>   *"You wrote ↑ Baile Phonk and ↓ Anthems. On this snapshot `baile` sits at {baileMedian} (n={baileN}) and `phonk` at {phonkMedian} (n={phonkN}) against a {baseline} era-matched baseline — **below it, not above**. `anthem` is at {anthemMedian} (n={anthemN}). None of the three separates from the baseline once {k} terms are adjusted. The honest direction for all three is UNKNOWN — not down."*
>   Rendering the contradiction of her own example, in her own words, in the position she drew it, is the single most useful thing this card does. It is also the reason the card must be a lede and not a footnote: a reader who never reaches M2.4 currently leaves believing baile phonk is a tailwind, and the data says the opposite.
> - **Then the convertible part** — "we don't know" becomes "we know when we'll know" (`vault/EXPERIMENT_STANDARD.md` §5 rule 3):
>   *"To decide a style question you have to assign it, not observe it. Route: The Soundry, one pre-registered arm. At +30% relative lift that is {nPerArm} songs per arm, {days} days at 12 a day — computed live by `powerDays()`, never typed."*
>   `powerDays()` reads the same code path as `scripts/analyze_power.py`; the LEDE **must not** hard-code the 56/4.7 pair from `EXPERIMENT_STANDARD.md` §3. If the recompute disagrees with the documented table, the row renders the **DRIFT** badge with both values and both dates (§4.3 rule 11) — it does not silently pick one.
> - `nextMove`: `{ lever: "style", magnitude: "RUN_AN_EXPERIMENT", routesTo: "soundry.hypothesis", forbidden: "may not become a gate in vault/gates.yaml, may not enter an ICB, may not select a style for the next run on its own." }` (§13.6, verbatim).
> - `drill`: `{ href: "#/music/craft?card=m2-4", label: "the whole board" }`.
> - **No individual style term is addressable from this row.** No permalink, no chip, no export (§13.1).

> #### M0.0f · Empty state — and it names the command
> When the register produces zero rows in Zones 1–3 for the active modality, the card renders the zone rules and the numbered slots as hollow, in place, with:
> *"No registered signal has a measurable direction on this snapshot. That is a statement about our history, not about the work. `python3 scripts/ingest_suno.py` (one pull, ~11 public requests, no credentials, no spend) — the second pull is what gives this list a direction."*
> Zone 0 and Zone 4 always render regardless; the card is never blank.

> #### M0.0g · Guidance notes required before this card ships
> `GuidanceNote` (kit §5.1) for each of: **margin**, **detection floor (MDE at 80%)**, **registered signal**, **pre-registration / POST_HOC_BAND**, **temporal vs contrast**, **cannot be told apart**. Each fills `computed` with the function name + file path. A missing note fails `guidance-coverage.spec.ts`.

---

## SPLICE B — amend `### 4.4 Tab Directive` (kit §4.4)

**Anchor:** append to the end of §4.4, before the `---`.

> **On any tab that renders a LEDE (Analytics M0.0), the Tab Directive is not independently computed.** It is, by construction, the `nextMove` of the LEDE's Zone-0 blocker line if one exists, otherwise of the LEDE's rank-1 row. The argmax in the paragraph above and the LEDE's rank-1 are the same call to `rankSignals()`. This is a structural guarantee, not a convention: a tab whose headline sentence disagrees with its own ranked list has already lied once before the reader reaches a chart. `tests/directive-lede-agreement.spec.ts` asserts identity of `insight.id`.

---

## SPLICE C — amend `## 4 · MUSIC · Tab M0 — SIGNAL` directive line

**Anchor:** the existing `**Directive today:**` line under the M0 heading.

> **Directive today:** *(computed — this is M0.0's Zone-0 blocker line by §4.4, not a separately authored sentence)* — tier `ABSENT` / magnitude `PULL_THE_FEED` / routes `ops.runbook`. Rendered text today: *"Nothing on this tab is a trend yet. Run `python3 scripts/ingest_suno.py` daily; the second pull is what makes this page work."*

---

## SPLICE D — amend `## 8 · MUSIC · Tab M4 — HEADWINDS`, opening paragraph

**Anchor:** replace the sentence *"The three highest-severity items also mirror into a persistent right rail on every other tab, because that is where she drew them."*

> M4.1 is the **evidence layer** of the ranked list; the **headline layer** lives where she drew it, at Analytics **M0.0 · THE LEDE**, and both consume one call to `rankSignals()` (kit `insight/rank.ts`). M4.1 renders all `k` registered signals with their series; the LEDE renders the top three each way in words. M4.1 carries `[the lede →]` in its header. The three highest-severity items also mirror into a persistent right rail on every other tab, because that is *also* where she drew them — that rail is specified separately and is not this patch's scope; it is not a substitute for the LEDE and the LEDE is not a substitute for it.

---

## SPLICE E — amend `## 12 · The whiteboard's four named insights` table

**Anchor:** append a fifth row to the table (the panel is not one of her four *insights*, but the disposition table is where a reader looks for "what happened to the thing I drew").

> | 5 | **the centre panel itself — "Top Trends & Insights", ranked ↑/↓** | **SHIPS IN HER POSITION.** Relocation to M4 reverted as a bug. Ranking **unit** overruled: registered signals, not genre tokens. | **M0.0** + §13 | A ranked directional list is the only sentence-shaped object on the page and it belongs first. Her example rows — ↑ baile phonk, ↓ anthems — are refused as rows and answered as one pinned row that names the contradiction: both terms sit *below* the era-matched baseline at n=5 and n=6, and neither direction is decidable at k=20. |

---

## SPLICE F — amend `## 13 · The multiplicity interlock`, add mechanism 7

> 7. **The family is closed before the data is looked at.** The LEDE renders only entries in `src/signals/register.ts`, each carrying `registeredAt`; an entry registered after the snapshot's `pulled_at` is flagged `POST_HOC_BAND`, capped at `OBSERVATION`, and barred from a directional band. `k` is the register size, printed on the card, and it does not grow with the style-token count because the whole style board is **one** registration. Post-hoc adjustment is a repair; a closed family is a design.
>
> Tests appended to `tests/multiplicity.spec.ts`: *"bars a post-hoc-registered signal from the rising and falling bands"* · *"registers the style board as exactly one entry"* · *"computes q_BH across the full register, not per card"*.

---

## SPLICE G — amend `## 15 · App skeleton`, file tree

> ```
>     signals/ register.ts  register.spec-fixtures.ts   # the closed family; hand-maintained, reviewed
>     cards/ m0-lede/ m0-*/ m1-*/ ...
> ```
> and, in the kit: `apps/lofn-kit/src/insight/rank.ts` — `signedLowerBound`, `margin`, `rankSignals`. **The only implementation**; a second one anywhere is the defect the kit's opening invariant names.

---

## SPLICE H — new tests, appended to the §15 test contract

> 10. `lede.spec.ts` — (a) every register entry has a non-empty `falsifier` and a parseable `registeredAt`; (b) no row reaches Zone 1/2 with `interval.method === "none"`, `direction === "unknown"`, `POST_HOC_BAND`, or (for `kind==="contrast"`) `eraControlled === false`; (c) zones are capped at 3/3/3; (d) sort is `margin` desc with the documented tie chain, and is stable across two identical fixtures; (e) with a one-snapshot fixture every `temporal` `suno` row serialises the literal `UNKNOWN` and Zone 1 and Zone 2 are empty; (f) the pinned refusal row is present, last, and shares `insight.id` with the M2.4 board; (g) no rendered LEDE output contains a style-term permalink.
> 11. `lede-m41-parity.spec.ts` — LEDE rows ⊆ M4.1 rows by id, on every fixture.
> 12. `directive-lede-agreement.spec.ts` — the Tab Directive's `insight.id` equals the LEDE's Zone-0 id, else its rank-1 id.

---

## SPLICE I — `## Overrules ACCEPTED`, append

> - The ranked directional panel returns to **the centre of the Analytics board, first card of M0 (M0.0 · THE LEDE)** — its relocation to the Headwinds tab is reverted as an unnamed drop, not ratified as an overrule. What IS overruled is the **unit** of the ranking: her rows were genre tokens (↑ Baile Phonk, ↓ Anthems); the LEDE ranks **registered signals** across all four evidence families, and the entire style board appears as exactly ONE pinned row that refuses (Headline Editor; Uncertainty Hawk's cell-size objection accepted in full). PRESERVES: *"tell me, in words, what is moving and which way, before I look at anything"* — which is her actual instrument, and which twenty token rows at n=1–39 would destroy rather than deliver: twenty rows read, in eight seconds, as twenty findings, and none of them are findings. Her named example is not dropped, it is **answered on screen in her own words** — baile and phonk sit below the era-matched baseline, not above, and even that is not decidable at k=20 — with the route that could decide it (one pre-registered Soundry arm) and the days-to-decision printed beside it.

---

## SPLICE J — `## Open questions for The Scientist`, append

> - **The register is a judgement call and it is yours.** `src/signals/register.ts` is the closed family the LEDE ranks; adding an entry after a snapshot exists permanently flags it post-hoc. The seed list needs your review before the first build: which warnings you actually want ranked, and which of them you would accept being told are inside their own noise. I will not add an entry to the register on my own after seeing a number move.

## 1 · What this patch does NOT do

- It does not specify the persistent three-item right rail (Critic 1's separate finding). The LEDE is not that rail and must not be used to close it.
- It does not resolve the `Part 1 §3.2` / `Part 2 §3.3` filter-key conflict, the `4.49%`/`4.37%` baseline conflict, or the 95%/90% interval conflict. It is written to depend on the resolved values (`CI_LEVEL`, `houseBaseline()`, the reconciled selection key) and to type-error rather than guess if they are still in conflict when the lane opens.
- It hard-codes no number. Every figure in every row is interpolated at render from the same computation its evidence card uses, or it carries a DRIFT badge with both values and both dates.


RISKS:
- BLOCKING DEPENDENCY: the LEDE's drill href needs the reconciled filter-key vocabulary (Part 1 §3.2 `sel` vs Part 2 §3.3's six-key set). Building M0.0 before that reconciliation lands would introduce a third vocabulary — the exact defect two critics already flagged. The lane must not open until the union is frozen.
- BLOCKING DEPENDENCY: `margin` divides by `claim.mde80`, and the spec's frozen power constants are disputed (both critics: sigma2_true 1.204e-4 vs the on-disk 1.369e-4). Until that is settled the ranking is stable in ORDER but its printed multiples are wrong. Mitigation: ship `margin` as a sort key with the multiple hidden until the constant reconciles, or gate M0.0 behind the power fix.
- The LEDE could become a bypass: a future author adds a register entry the moment a number moves, and the pre-registration check only catches it if `registeredAt` is written honestly. The check is real but it is not tamper-proof — it depends on nobody backdating a string. The Scientist owning the register (Splice J) is the actual control; the test is the reminder.
- Zone 3's collapse-to-one-line is a place where honesty can quietly shrink. If the undecided list grows past ~8 the collapsed summary starts hiding the true shape of the evidence. Add a rule at first build review: if `undecided.length > up.length + down.length`, the card renders that ratio in the header rather than only the count.
- The pinned refusal row interpolates from M2.4's computation, and M2.4 itself is under two live critic objections (it plots absolute medians on an axis defined as a difference; 'era-matched' is undefined and no era-matching estimator exists in stats.ts). If M2.4 ships wrong, the LEDE amplifies it to the top of the first tab. M0.0 must not ship before M2.4's estimator is defined.
- Ten objects is my budget, not a measured one. Extrapolating from the desk's practice, not from a study of this reader. If she reports the LEDE reads as a wall, cut Zone 3 to the collapsed line only and keep 3/3.


---

# APPENDIX — seat patch: THE THRESHOLD KEEPER — after Bruce Schneier (Applied Cryptography; Secrets and Lies; Beyond Fear; the public record on usable security and threat modelling) (Schneier's documented positions, cited in the third person as record: that security is a trade-off, not a product, and must be argued as a trade-off against a named asset and a named adversary; that complexity is the enemy of security, because every added mechanism is added attack surface and added misconfiguration; that the threat model must be written before the mechanism, or the mechanism is decoration; that usability failures are security failures, because a control users route around is a control that does not exist; and his long-standing objection to home-rolled cryptography and home-rolled authentication. Extrapolating: he has not ruled on Cloudflare Access, static-site data minimisation, or this repository — the application of those positions to this spec is mine.)

FINDING: The Scientist asked twice for collaborator access — "Access only to The Scientist + invited collaborators" (Board 2) and "analytics for Lofn (private-claw) and collaborators" (Board 6) — and the spec defines no user model, no invite, no revoke, and no auth check, then forecloses the requirement by pinning both servers to 127.0.0.1 while describing a collaborator panel as "all traffic loopback"; the one token in the design, `#/panel/<session-token>`, sits in a hash fragment that no browser ever transmits, so it gates nothing at all.

FILES: E:/git/private-claw/docs/reference/lofn_analytics_and_soundry/BUILD_SPEC.md, E:/git/private-claw/.gitignore, E:/git/private-claw/apps/lofn-analytics/scripts/derive-public-snapshot.mjs, E:/git/private-claw/apps/lofn-analytics/SECURITY.md, E:/git/private-claw/apps/lofn-analytics/DEPLOY.md, E:/git/private-claw/apps/lofn-analytics/package.json, E:/git/private-claw/apps/lofn-analytics/vite.config.ts, E:/git/private-claw/apps/lofn-analytics/tests/posture-hosted.spec.ts, E:/git/private-claw/apps/lofn-analytics/tests/derive-public.spec.ts, E:/git/private-claw/apps/lofn-analytics/tests/no-auth-code.spec.ts, E:/git/private-claw/apps/lofn-analytics/tests/loopback-default.spec.ts, E:/git/private-claw/apps/soundry/server.mjs, E:/git/private-claw/apps/soundry/panel-routes.mjs, E:/git/private-claw/apps/soundry/panel.html, E:/git/private-claw/apps/soundry/emit-snapshot.mjs, E:/git/private-claw/apps/soundry/tests/panel-token.spec.ts, E:/git/private-claw/apps/soundry/tests/emit-snapshot.spec.ts, E:/git/private-claw/apps/soundry/tests/no-auth-code.spec.ts, E:/git/private-claw/apps/soundry/tests/loopback-default.spec.ts, E:/git/private-claw/vault/private/collaborators.local.md

« PATCH — SEAT: THE THRESHOLD KEEPER. Splice as marked. »

---

## SPLICE 1 — NEW SECTION, insert into PART 1 immediately BEFORE `# §1 · DESIGN TOKENS`

# §0 · THREAT MODEL — written before the mechanism, and binding on both apps

No access-control clause anywhere in this spec is valid unless it can be traced to a row in the tables below. A mechanism with no asset is theatre; an asset with no mechanism is the bug the critics found.

### 0.1 What we are protecting (assets, in order of what a leak actually costs)

| # | Asset | Where it lives | Cost of disclosure, stated plainly |
|---|---|---|---|
| **A1** | The **unreleased style-prompt corpus** — the full `tags` string of all 210 clips | `data/snapshots/suno/latest.json` | This is the only reproducible asset the estate owns. It is the method, written down. A competitor holding it reproduces the catalogue's approach in an afternoon. It also contains language for work that has not shipped — disclosing it is an **unauthorised publish** under `vault/AUTONOMY.md` Part 1, not merely a privacy event. |
| **A2** | **Unposted Soundry variants** — the twelve prompt quads of an open round | `data/soundry/rounds/<id>/variants/*.json`, `output/soundry/<id>/*.md` | Tomorrow's songs before they exist. Same AUTONOMY analysis as A1, one day earlier. |
| **A3** | **Listener panel data** — free-text notes and ratings from human beings who are not The Scientist | `data/soundry/rounds/<id>/observations/panel.jsonl` | **This is other people's data, held in trust.** `vault/COLLABORATOR_STANDARD.md` governs it and is currently never invoked for panel listeners. A leak here is a broken promise to a third party, and it is the only asset whose loss we cannot unilaterally forgive ourselves for. |
| **A4** | **Account identifiers** — the Suno `user_id`, follower counts, the DistroKid demographic table | `data/snapshots/{suno,socials,distribution}/latest.json` | Enables targeted scraping and correlates the Lofn identity to the Local Symmetry identity. Low drama, permanent. |
| **A5** | **Integrity of the pre-registration chain** — `preregistration.json`, its hash, `events.jsonl` | `data/soundry/rounds/<id>/` | **The one asset where integrity outranks confidentiality.** If a frozen pre-registration can be silently amended, every interval in both apps is a lie and the whole EXPERIMENT_STANDARD apparatus reports fiction. A confidentiality breach embarrasses us; an integrity breach makes us liars. |

### 0.2 From whom (adversaries, in order of probability, not drama)

| # | Adversary | Capability | Realistic? |
|---|---|---|---|
| **T1** | **The opportunistic internet** — scanners, indexers, crawlers | Hits every hostname within hours of a DNS record resolving. No targeting, no persistence. | **Certain.** This is what `analytics.lofn.ai` faces on minute one, and it is the entire reason the hosted posture is argued at all. |
| **T2** | **A careless invited collaborator** | Screenshots a card into a group chat; forwards the link; leaves the tab open on a shared laptop; stays on the list after the collaboration ends. **No malice at any point.** | **The dominant threat.** Every control below is designed against T2 first. An invite list that is easy to add to and hard to prune *is* T2's attack. |
| **T3** | **A scraper or competitor after A1** | Will read anything reachable. Will not spend effort. | Low probability, permanent impact. Defeated by A1 never being reachable, not by A1 being behind a door. |
| **T4** | **Our own retired machinery** | **Already realised, not hypothetical.** `memory/2026-08-01.md` records `lofn-openclaw.lofn.workers.dev/cdp` — an exposed Chrome DevTools Protocol endpoint, full browser remote control including session cookies, **with overprovisioned Cloudflare Access policies**, on a Worker belonging to the retired OpenClaw runtime. | **Live.** It is also a direct verdict on the control I am about to specify: the Access configuration in this account has already been provisioned too widely once. See precondition P0. |
| **T5** | **The Scientist under time pressure** | If the sanctioned path to show a collaborator a chart takes twenty minutes, she will paste a tunnel URL into a DM in thirty seconds, and that tunnel serves the raw corpus. | **Certain, and it is our fault when it happens.** Documented position (Schneier): a usability failure is a security failure. The sanctioned path must be *cheaper than the workaround*, or the mechanism is decorative. |

### 0.3 The two trade-offs, decided

**Trade-off 1 — shrink the asset before arguing about the lock.** Board 6's stated goal for collaborators is insight: *lyric length / line · Suno EMO usage · genre performance · trend setters + breakers*. **Not one of those four requires the raw style-prompt text, the `user_id`, or any unposted row to be visible.** They are all computable from derived measures. Therefore the hosted artefact carries derived measures and nothing else, and A1/A2/A4 are not behind the door — they are not in the building. A data boundary survives a misconfigured lock; a permission boundary does not.

**Trade-off 2 — the two apps do not get the same posture, and pretending otherwise is what produced the contradiction.** Lofn Analytics is an *observation* surface over facts that are already public on Suno; its derived form is genuinely shareable. The Soundry is a *workbench* holding A2 and A5 and she wrote `soundry.localhost` on the board herself. **The Soundry is never hosted. Lofn Analytics may be, on a redacted dataset, behind an identity-aware proxy she already owns.** Two apps, two postures, both specified below in full.

### 0.4 Anti-goals — mechanisms this spec FORBIDS by name

Complexity is the enemy of security, and every item here has been proposed for systems like this one and has failed in the field:

1. **No bespoke authentication.** No password field, no user table, no `bcrypt`, no hand-rolled JWT verification, no session store, no "login" route, no shared secret link. Grep-asserted (`tests/no-auth-code.spec.ts`, SPLICE 6).
2. **No secret in a URL hash fragment.** A fragment is never transmitted to any server and cannot gate anything. The existing `#/panel/<session-token>` is a **defect**, corrected in SPLICE 4.
3. **No `Dockerfile`.** Strike it from the `apps/lofn-analytics/` skeleton (Part 2 §15). A container in the tree is an invitation to `docker run -p 0.0.0.0:8080` with the raw snapshot mounted, and it buys nothing a static deploy does not already give us.
4. **No CI deploy.** Publishing to a public hostname *is* publishing. `vault/AUTONOMY.md` Part 1 makes that a human act. CI builds and tests; **a human runs the deploy command.** No workflow may hold a Cloudflare API token.
5. **No client-side role logic.** Roles are enforced by *which artefact is deployed*, never by a branch in the bundle. There is no `if (user.isCollaborator)` anywhere in either app.

---

## SPLICE 2 — NEW SUBSECTION, insert into PART 1 §8.3 as item **0**, before the existing item 1

**P0 · BLOCKING PRECONDITION FOR THE HOSTED POSTURE ONLY (the local builds are unblocked by it).** The Cloudflare account (`790bedc4700140f0a31022783370bd88`, zone `lofn.ai`, free plan) currently exposes `lofn-openclaw.lofn.workers.dev/cdp` — remote browser control with overprovisioned Access policies — on a Worker belonging to a runtime that has been retired. **No new Cloudflare Access application may be created in this account until that Worker is deleted and its Access policies removed, by The Scientist** (destructive + outward-facing: `vault/AUTONOMY.md`). Reason, stated as a rule rather than a preference: the control I am about to rely on has already been provisioned too widely once in this account, and adding a second door to a building with a known-open one converts a contained finding into a pattern. Deleting it is also the cheapest audit we will ever get of whether this account's Access config is trustworthy.

---

## SPLICE 3 — REPLACE the `Bind` and `Posture` rows of PART 3 §0.1, and APPEND the new rows

| | |
|---|---|
| **Bind** | `server.mjs` binds `127.0.0.1` only, **always, in every posture, with exactly one opt-in exception**: PANEL LAN MODE (§0.3 below), which is off by default, time-boxed, single-purpose, and serves a *different route table*. No `access-control-allow-origin: *`, ever. `soundry.localhost` resolves via the operator's hosts file; the app works identically at `http://127.0.0.1:5231/`. |
| **Posture** | **Local-only. The Soundry is never hosted, on any domain, under any Access policy.** It holds asset A2 (unposted variants) and asset A5 (the pre-registration chain). `scripts/check-local-only.mjs` is unchanged and remains a build gate. |
| **Collaborators** | Her Board 2 line — *"Access only to Scientist + invited collaborators"* — **is implemented, at the granularity her own boards specify.** Board 3 draws the Anvil as a single-operator surface (one prompt box, "What should we try today?"). Board 2 gives collaborators exactly one job: *"get feedback from listeners."* **Therefore: a collaborator is a PANEL LISTENER, not a bench operator.** They reach `panel.html` and nothing else. See §0.3 and the overrule accounting in SPLICE 7. |
| **Human subjects** | Panel listeners are third parties and `vault/COLLABORATOR_STANDARD.md` governs their data (asset A3). Its default-PRIVATE rule binds: **no panel note, rating, or pseudonym is ever exported to `data/snapshots/soundry/`, ever rendered in Lofn Analytics, and ever included in any hosted artefact.** `emit-snapshot.mjs` carries a field allowlist that excludes the entire `panel` namespace, asserted by `tests/emit-snapshot.spec.ts`. The panel session's first screen carries a 40-word consent line naming what is recorded (rating, optional note, elapsed seconds, presentation position), what is not (name, email, handle, IP), and that notes may be read verbatim by The Scientist. |

### NEW — 0.3 PANEL LAN MODE — the one exception to loopback, and its complete envelope

The loopback bind is not weakened; it is **scoped, and given a second, smaller listener that cannot reach the bench.**

```
node server.mjs --panel-lan --round R-2026-08-04-wpl --ttl 4h
```

- **Off by default.** Absent the flag, exactly one listener exists, on `127.0.0.1`. Unchanged from today.
- **A separate route table, not a middleware check.** The LAN listener is constructed from `panelRoutes` only: `GET /panel/<token>` · `GET /panel/assets/*` · `POST /soundry/observations/panel` · `GET /soundry/health`. `/soundry/rounds`, `/soundry/analysis/*`, `/soundry/emit-snapshot`, the ledger, and every prompt artefact **do not exist on that listener** and return **404, not 403** — a route that was never registered cannot be misconfigured into existence, and 403 leaks that the bench is there. Two `http.createServer` calls, two route tables, one shared handler module. This is the structural control; everything below is defence in depth.
- **The token is a real capability, in the path, verified server-side.** `crypto.randomBytes(16)`, Crockford base32, minted per session into `data/soundry/rounds/<id>/panel-sessions.json` (git-ignored). Flow: `GET /panel/<token>` → server validates → sets `Set-Cookie: soundry_panel=<token>; HttpOnly; SameSite=Strict; Path=/; Max-Age=<ttl>` → **302 to `/panel/` with the token stripped from the URL.** Every `POST /soundry/observations/panel` reads the cookie, not the URL, and 403s on absent/unknown/expired/closed. **This corrects the `#/panel/<session-token>` defect: a hash fragment is never sent to a server and gated nothing.** Stripping the token from the visible URL also stops it surviving in a screenshot, a shoulder-surf, or a `Referer` — which is a T2 control, and T2 is the threat that matters.
- **Bound on first use, one listener per token.** First successful hit records a coarse client fingerprint; a second distinct binding on a live token is refused **and surfaced in the operator's terminal**. Detection, not just prevention — an unexplained second binding is the signal that the link was forwarded.
- **Time-boxed and revocable.** Default TTL 4h, hard cap 24h. `node server.mjs --panel-revoke <token>` and `--panel-close` (all tokens, this round). **The LAN listener terminates itself when the last token expires**, so the failure mode of forgetting is a closed port, not an open one. The terminal prints a live countdown and the bound-listener count — the operator can see the exposure without asking for it.
- **What a panel collaborator CAN see:** the audio, a rating control, a free-text box, their own progress. **What they CANNOT see:** titles, arm labels, factor levels, prompts, lyrics, style text, the ledger, the hypothesis board, any other listener's response, any other round, or that any of it exists. Blindness was already required for statistical validity (§3.3); it is now also the access-control boundary. **Two independent reasons for the same wall is the cheapest security we will ever get.**
- **Remote (off-LAN) collaborators are NOT SUPPORTED in v1, and the spec says so in words rather than implying loopback covers it.** Supporting them means hosting audio and a write endpoint on the public internet — a new origin, a new asset boundary, a second Access configuration, and a write path to A3. That is a decision for The Scientist, not a build-agent's inference from an ambiguous sentence. Deferred as **open question OQ-SEC-2** (SPLICE 8). A build agent that widens the default bind to resolve this has committed the failure the critics named, and `tests/loopback-default.spec.ts` (SPLICE 6) makes it a red test.

---

## SPLICE 4 — REPLACE the first paragraph of PART 3 §3.3

A **blind panel session** served at `panel.html`, one listener at a time, reached over **PANEL LAN MODE** (§0.3) — an opt-in, time-boxed, capability-gated second listener whose route table contains the panel and nothing else. Access is the Scientist plus invited collaborators, each holding a single-use session capability that authorises exactly one action: submitting responses for one round. **The capability is transmitted in the request path and enforced by an `HttpOnly` cookie set server-side; it is never carried in a URL hash fragment, which no browser transmits and which therefore gates nothing.** Panel data is third-party data under `vault/COLLABORATOR_STANDARD.md` (asset A3, §0 threat model): default PRIVATE, never exported to any snapshot, never rendered in Lofn Analytics, never in a hosted artefact.

---

## SPLICE 5 — NEW SECTION, insert into PART 2 (LOFN ANALYTICS) immediately AFTER §0's "Posture" paragraph, replacing that paragraph

**Posture: two, and only two.** They differ by **which dataset is built**, not by a flag on the same one.

### 0.4 · POSTURE `local` (default; unchanged; the only posture until she says otherwise)

`server.mjs` on `127.0.0.1`, reading `../../data/snapshots/<feed>/latest.json` at request time. Full fidelity — the raw corpus, the `user_id`, the refresh controls. **This is the posture in which The Scientist works, and the posture in which a collaborator sitting beside her sees everything.** No change to §15.

### 0.5 · POSTURE `hosted` — `analytics.lofn.ai`, derived data only, behind an identity-aware proxy she already owns

**She may not enable this until P0 (SPLICE 2) is cleared and she says the word in chat.** The build must exist and be tested before then; it must not be deployed.

**(a) The artefact is a different dataset, not the same one with a lock on it.** `apps/lofn-analytics/scripts/derive-public-snapshot.mjs` (zero runtime deps) reads `data/snapshots/<feed>/latest.json` and writes `data/public/<feed>/latest.json` under a **default-deny field allowlist** — any field present in the source and absent from the allowlist **fails the script with exit 1 and names the field**, so a schema change from upstream cannot silently widen the export. Allowlist for `suno` rows: `clip_id · title · created_at · play_count · upvote_count · like_rate · is_public · duration · model_version` plus the derived craft measures the app computes (`lyric_len_per_line`, `form_band`, `emo_tag_count`, `era`). **Categorically excluded and grep-asserted absent from the output:** `tags` and every style-prompt string (A1), `user_id` and every profile identifier (A4), raw lyric text, and **every row where `is_public !== true`** (A2). Header-level follower counts are dropped; the socials and distribution feeds export platform-level aggregates only, and the DistroKid demographic table is excluded entirely.
**A collaborator therefore sees the analysis and not the corpus — and every one of her four Board-6 insights survives that redaction intact.**

**(b) The origin is a bucket of files, so there is no origin to attack.** Cloudflare **Pages**, static, `base: "./"`. **`server.mjs` does not run in the hosted posture and is not in the deployed artefact.** This resolves the ambient tension between "the app has a server" and "the app has no write endpoint": hosted has no server, so it has no endpoints at all. The refresh control (§14.4) is a local-posture affordance and is **stripped at build time** under `POSTURE=hosted`, asserted by `tests/posture-hosted.spec.ts`. (Interlock only — Critic 2's refresh/transport finding is another seat's patch and I do not pre-empt its resolution; whatever transport it specifies is **local-posture-only** and must not appear in a hosted bundle.)

**(c) The door is Cloudflare Access, which we do not write.** Zero Trust → Access → self-hosted application on `analytics.lofn.ai`. Identity provider: **One-Time PIN** (free tier; no password to breach, no account to create, no directory to sync). Policy: **Allow**, rule type **Emails** — resolved from an Access **Group** named `lofn-collaborators`. Session duration **24 hours**. Everything else on that hostname: **default Deny.** *No authentication code ships in either app.* The proxy terminates identity before the origin; the bundle never sees a token, never parses a JWT, never branches on a role. Documented position (Schneier), applied: authentication invented in-house is the classic own-goal, and the correct move is to use a mechanism she already owns and to write **zero** lines of it.

**(d) Close the two bypasses, or the door is decoration.** Both are documented Cloudflare footguns and both must be executed and verified:
  1. **Disable preview deployments** for the Pages project (Settings → Builds & deployments → Preview deployments → *None*). A `*.pages.dev` preview is a second, unprotected copy of the same site.
  2. Add a **second Access application** covering `<project>.pages.dev` and `*.<project>.pages.dev` with the same policy, as belt-and-braces for the production alias.
  3. **Verify from the adversary's position, not the operator's:** from a logged-out browser on a foreign network, `curl -sI https://analytics.lofn.ai/` must return a redirect to the Access login and **never** a `200` with app HTML; the same for the `pages.dev` alias; and `curl -s https://analytics.lofn.ai/public/suno/latest.json` must not return JSON. A control that has not been tested from outside is theatre. Record the three results, dated, in `apps/lofn-analytics/SECURITY.md`.

**(e) Invite flow — one place, thirty seconds, so the workaround is never cheaper (T5).**
```
1. Zero Trust → Access → Groups → lofn-collaborators → add <email>.
2. Append to vault/private/collaborators.local.md:  <email> · granted YYYY-MM-DD · reason · review-by YYYY-MM-DD
3. Send them https://analytics.lofn.ai — they get a one-time PIN at that address. No account, no password, nothing to reset.
```
`vault/private/` is **git-ignored and never committed** — `vault/COLLABORATOR_STANDARD.md` forbids collaborator identities in this tree, and that rule survives this patch untouched.

**(f) Revoke flow — and the footgun that makes people think they revoked.**
```
1. Zero Trust → Access → Groups → lofn-collaborators → remove <email>.
2. Zero Trust → Logs → Access → filter by that email → REVOKE SESSIONS.   ← step 2 is not optional
3. Mark the line in vault/private/collaborators.local.md as revoked YYYY-MM-DD. Never delete the line.
```
**Removing an email from the group does not terminate a session already issued.** Without step 2 a removed collaborator retains access for up to the session TTL. That is exactly why the TTL is 24h and not 30 days: the worst case must be a number we can say out loud. **`SECURITY.md` states it: "Worst-case time from revoke-step-1 to no access, if step 2 is skipped: 24 hours."**

**(g) Review, and the honest empty state.** The list is re-attested at the start of each Soundry round, or quarterly, whichever is sooner. Any entry past its `review-by` is removed by default and re-invited if still needed — **prune-by-default, because T2's real attack is an invite list that only ever grows.** An empty `lofn-collaborators` group is a valid, healthy state; the hosted app then serves nobody, and that is not a failure.

**(h) Deployment is a human act.** `npm run deploy:hosted` runs derive → build (`POSTURE=hosted`) → the posture tests → `wrangler pages deploy`, **and refuses to proceed if any test fails or if `data/public/` is older than `data/snapshots/`.** No GitHub Action deploys, and no workflow holds a Cloudflare token (`vault/AUTONOMY.md` Part 1: anything that leaves the machine is human-gated).

---

## SPLICE 6 — APPEND to PART 2 §15 test contract (as tests 10–13) and to PART 3 §6 required specs

**Every claim in §0 that is not tested is theatre. These are the tests that make the threat model real:**

10. **`posture-hosted.spec.ts`** — build with `POSTURE=hosted`; assert `dist/**` contains none of: the literal `user_id`, any `tags`/style-prompt substring drawn from the real corpus, any path under `data/snapshots/`, any refresh-button handler, the string `server.mjs`, or an absolute `http(s)://` origin other than the CSP declaration.
11. **`derive-public.spec.ts`** — run `derive-public-snapshot.mjs` against the real 210-row `data/snapshots/suno/latest.json`; assert every output row's key set is a **subset** of the allowlist; assert `rows.every(r => r.is_public === true)`; assert the concatenated output contains **zero** characters of any source `tags` value; assert an injected unknown source field causes **exit 1 naming the field** (default-deny proven, not assumed).
12. **`no-auth-code.spec.ts`** *(both apps)* — grep `src/**` and `*.mjs` for `password`, `bcrypt`, `argon2`, `jsonwebtoken`, `jwt.verify`, `session-secret`, `login`, `signin`: **all absent.** The apps contain no authentication code and this test is the proof. Its failure means someone started inventing one.
13. **`loopback-default.spec.ts`** *(both apps)* — spawn with no flags; assert the listener's address is `127.0.0.1`; assert a connection attempt to the same port on a non-loopback local interface is refused. **A build agent that widens the bind to "fix" collaborator access turns this red.** That is the test's whole job.

Soundry additionally, in §6:

- **`panel-token.spec.ts`** — spawn with `--panel-lan` on an ephemeral port. Assert: `GET /soundry/rounds` on the panel listener → **404** (the route does not exist, and 403 would leak that it does); `POST /soundry/observations/panel` with no cookie → 403; with an expired token → 403; with a revoked token → 403; a second distinct binding on a live token → 403 **and an operator-visible warning**; `GET /panel/<token>` → 302 whose `Location` **does not contain the token**; the LAN listener closes itself when the last token expires; and the bench listener on `127.0.0.1` is unaffected throughout.
- **`emit-snapshot.spec.ts`** *(extend)* — assert the emitted `data/snapshots/soundry/latest.json` contains **no** key under the `panel` namespace, no free-text note, and no `anon-` pseudonym. Asset A3 never leaves the round directory.

Also append to `.gitignore` under SPLICE-2 item 1: `data/public/`, `data/soundry/`, `vault/private/`, and `**/panel-sessions.json`.

---

## SPLICE 7 — APPEND to `## Overrules ACCEPTED` (all three decision blocks)

- **COLLABORATOR ACCESS IS IMPLEMENTED, AT TWO DIFFERENT GRANULARITIES, BECAUSE THE TWO APPS HOLD DIFFERENT ASSETS** (Threshold Keeper). Her two lines — "Access only to Scientist + invited collaborators" (Board 2) and "analytics for Lofn (private-claw) and collaborators" (Board 6) — were dropped by the previous draft and structurally foreclosed by a loopback bind plus a hash-fragment token that gates nothing. Both are now built. **Narrowing 1:** a Soundry collaborator is a *panel listener*, not a bench operator. **PRESERVES her intent exactly** — Board 2 assigns collaborators one job, *"get feedback from listeners"*, and Board 3 draws the Anvil as a single-operator surface with one prompt box. **Narrowing 2:** a hosted-Analytics collaborator sees derived measures, not the raw corpus. **PRESERVES her intent exactly** — all four of her Board-6 insights (lyric length/line, EMO usage, genre performance, trend setters + breakers) are computable from derived measures, so the collaborator loses nothing she asked them to have, and asset A1 is not merely locked but absent.
- **THE `#/panel/<session-token>` HASH FRAGMENT IS A DEFECT, CORRECTED, NOT AN OVERRULE.** A fragment is never transmitted to a server. The token moves into the request path, is exchanged for an `HttpOnly; SameSite=Strict` cookie, and is stripped from the visible URL. PRESERVES: her blind-panel design unchanged; the token now does the job its name implied.
- **THE LOOPBACK BIND SURVIVES AS A SCOPED CONTROL RATHER THAN AN AMBIENT ASSUMPTION.** Default `127.0.0.1` in both apps in every posture, plus one opt-in, time-boxed, self-terminating LAN listener whose route table contains the panel and nothing else, plus a hosted Analytics posture with **no server at all**. PRESERVES: "the whole app never leaves the machine" for the Soundry — literally, permanently — while making the collaborator requirement real instead of contradictory.

## APPEND to `## Overrules REJECTED (her intent wins)` — dissent from the other seats, registered

- **REJECTED — any proposal to put a shared password, a shared link secret, or a bespoke login on either app.** It is one secret shared by everyone, it cannot be revoked per person, it will be pasted into a chat within a week, and it means writing authentication we do not have to write. Documented position (Schneier): home-rolled authentication is the classic own-goal; complexity is the enemy. Use the identity-aware proxy she already owns and ship **zero** lines of auth code.
- **REJECTED — hosting a remote panel to serve off-LAN listeners in v1.** It adds an origin, audio hosting, a public write path to asset A3, and a second Access configuration, to serve a use case she never asked for. Deferred to her as OQ-SEC-2. Complexity added for an unrequested capability is the most expensive kind.
- **REJECTED — a "three-posture table" in `DEPLOY.md` that presents hosted as a build flavour of the same artefact.** Hosted is a **different dataset**, and if that distinction is a flag rather than a separate file the first misconfiguration publishes the corpus. Rewrite `DEPLOY.md` around §0.4/§0.5. Related: **strike `Dockerfile` from the `apps/lofn-analytics/` skeleton in §15** — it serves no purpose in a static Pages deploy and is a standing invitation to bind `0.0.0.0` with the raw snapshot mounted.

---

## SPLICE 8 — APPEND to `## Open questions for The Scientist`

- **OQ-SEC-1 · MAY LOFN ANALYTICS BE HOSTED AT `analytics.lofn.ai`, ON DERIVED DATA, BEHIND CLOUDFLARE ACCESS?** The redaction (§0.5a) means a collaborator sees every insight you drew on Board 6 and none of the style-prompt corpus. **RECOMMENDED DEFAULT: build it, test it, do not deploy it** until (a) you clear precondition P0 by deleting the exposed OpenClaw `/cdp` Worker, and (b) you name the first collaborator. Local-only remains fully useful and the DNS can wait. The hosted door costs nothing while it stays shut.
- **OQ-SEC-2 · DO YOU NEED REMOTE (OFF-LAN) PANEL LISTENERS?** v1 supports collaborators who are on your network — same house, same LAN, a session that expires in four hours. Remote listeners require hosting audio and a public write endpoint that accepts third-party free text, which is a new asset boundary and your call, not mine. **RECOMMENDED DEFAULT: no. Run the first rounds on-LAN and learn what the panel actually needs before we build a public surface for it.**
- **OQ-SEC-3 · WHO IS THE FIRST COLLABORATOR, AND WHAT IS THEIR REVIEW-BY DATE?** The invite list is prune-by-default and every entry carries an expiry. An empty list is a valid state. I need one name and one date to make the mechanism real; without them the whole of §0.5(e)–(g) is a well-tested door with nobody on the other side.


RISKS:
- The hosted posture depends on a Cloudflare Access configuration I cannot test from here. Every claim in SPLICE 5(d) must be verified by curl from a logged-out foreign network before anyone is invited, and the results dated into SECURITY.md. Until that verification exists, treat the hosted door as UNPROVEN, not as specified-therefore-safe — an untested control is the most dangerous kind, because it buys confidence without buying security.
- Precondition P0 blocks the hosted posture on a destructive, outward-facing action (deleting the OpenClaw /cdp Worker) that only The Scientist may perform. If she declines or defers, hosted Analytics does not ship and the collaborator requirement is met only by the derived build sitting undeployed. That is an acceptable outcome; silently proceeding without P0 is not.
- The derive-public allowlist is written default-deny precisely because I could not verify the exact row schema of data/snapshots/suno/latest.json field-by-field from the spec text alone. If the real schema carries a sensitive field I did not name, the script exits 1 and names it rather than exporting it — but the first run WILL likely fail on unlisted benign fields, and the builder must resist the temptation to switch the script to allow-unknown. That switch is the single highest-severity regression available in this patch.
- PANEL LAN MODE binds a listener to a home network, which trusts every device on that network for the token's lifetime. A compromised IoT device on the same LAN can reach the panel listener. Mitigated by the route table containing nothing but the panel, the 4h TTL, and single-binding detection — accepted, not eliminated. If her LAN is shared (guests, a landlord's router), the honest advice is to run panels in person only.
- Two apps with two postures is more surface than one rule, and I am the seat who says complexity is the enemy. I accept the cost because the alternative — one posture for both — either hosts the unreleased-prompt workbench or forecloses her collaborator requirement, and the previous draft chose the second silently. The mitigation is that the two postures differ by WHICH DATASET EXISTS, which is the cheapest kind of difference to reason about and the only kind that survives a misconfigured lock.
- Narrowing 'invited collaborators' to panel-listener (Soundry) and derived-data-viewer (Analytics) is an interpretation of her boards, argued from Board 2 and Board 3 but not confirmed by her. If she meant a collaborator who operates the Anvil or reads the raw corpus, this patch is under-scoped and OQ-SEC-1/OQ-SEC-3 are where she says so. It is named as a narrowing with its preserved intent, per the house rule, and it is not a silent drop.


---

# APPENDIX — seat patch: THE TASK TAXONOMIST — after Ben Shneiderman (Visual Information-Seeking Mantra; direct manipulation; dynamic queries; "an interface must never present a dead end") (Shneiderman's task taxonomy and the direct-manipulation contract: every state must expose the next legal action, tightly coupled to visible feedback, with no state reachable only from outside the interface. Read from disk: BUILD_SPEC.md §2.1 (EmptyKind/EmptyState), §6.1, §6.3, Part 2 §1 feed table, §9 M5, §14.4, §15, Build order W0–W5; WHITEBOARD_TRANSCRIPTION.md Boards 5–7 and the OVERRULE MANDATE; vault/AUTONOMY.md Part 1; vault/SOCIAL_VOICE.md Appendix C + §7; scripts/ingest_suno.py (lines 237–274) and scripts/ingest_receiver.mjs KNOWN_FEEDS.)

FINDING: The competitor feed is a dead end: the pull exists (`scripts/ingest_suno.py --handle <h>` → `data/snapshots/competitors/<handle>/`), but no file on disk names the five handles and no UI can record them, so the blocker list §14.4 leaves in place can never be cleared from inside the app.

FILES: E:/git/private-claw/docs/reference/lofn_analytics_and_soundry/BUILD_SPEC.md, E:/git/private-claw/data/watchlist/competitors.json, E:/git/private-claw/data/watchlist/competitors.template.json, E:/git/private-claw/data/watchlist/observations/<handle>/<YYYY-MM-DD>.json, E:/git/private-claw/.gitignore, E:/git/private-claw/apps/lofn-analytics/src/watchlist/schema.ts, E:/git/private-claw/apps/lofn-analytics/src/watchlist/validate.ts, E:/git/private-claw/apps/lofn-analytics/src/watchlist/project.ts, E:/git/private-claw/apps/lofn-analytics/src/watchlist/compose.ts, E:/git/private-claw/apps/lofn-analytics/src/watchlist/load.ts, E:/git/private-claw/apps/lofn-analytics/src/load/invariants.ts, E:/git/private-claw/apps/lofn-analytics/snapshot-route.mjs, E:/git/private-claw/apps/lofn-analytics/scripts/check-build.mjs, E:/git/private-claw/apps/lofn-analytics/src/cards/m5-7-watchlist/index.ts, E:/git/private-claw/apps/lofn-analytics/src/cards/m5-8-status/index.ts, E:/git/private-claw/apps/lofn-analytics/src/cards/comp-hosting-numbers/index.ts, E:/git/private-claw/apps/lofn-analytics/src/cards/comp-observation-form/index.ts, E:/git/private-claw/apps/lofn-analytics/tests/watchlist.spec.ts, E:/git/private-claw/apps/lofn-analytics/tests/competitor-projection.spec.ts, E:/git/private-claw/apps/lofn-analytics/tests/watchlist-draft.spec.ts, E:/git/private-claw/apps/lofn-analytics/tests/watchlist-states.spec.ts, E:/git/private-claw/apps/lofn-analytics/tests/empty-vs-zero.spec.ts, E:/git/private-claw/apps/lofn-analytics/tests/forbidden-strings.spec.ts, E:/git/private-claw/apps/lofn-analytics/tests/invariants.spec.ts, E:/git/private-claw/scripts/ingest_suno.py

# PATCH — THE WATCHLIST (Task Taxonomist, FINDING 3)

The Scientist asked for five watched targets (Board 5, feed #2). The spec reports honestly that
they are unnamed and then removes the only control that could ever name them. That is not
restraint, it is a dead end: a state the app can display but never leave. **Every state must
expose exactly one next legal action.** This patch supplies the roster file, the editor, the six
display states between "nothing" and "pulled", and the projection gate that makes the whole
surface unusable as a sound-copying instrument.

**Record.** The Scientist's Appendix C in `vault/SOCIAL_VOICE.md` already did the hard part: she
rejected "five Suno stars" as an axis, named **five slots that each solve a different problem**,
listed four universal disqualifiers, named the **six numbers**, and closed with the standing rule
that *these five are studied for HOSTING BEHAVIOUR, not for sound, and any finding that becomes
"make music like theirs" is thrown out*. The roster below is that appendix made machine-readable.
Nothing here invents a handle; the five cells ship empty and say so.

---

## PATCH 1 — new empty kinds (Part 1 §2.1)

**Replace** lines 249–260 of `docs/reference/lofn_analytics_and_soundry/BUILD_SPEC.md`:

```ts
export type EmptyKind = "never-pulled" | "measured-zero" | "filtered-empty"
                      | "not-applicable" | "below-floor" | "no-history"
                      | "needs-declaration"   // the app waits on a human DECISION written to disk
                      | "pull-failed";        // the pull RAN and returned no profile

export interface EmptyState {
  kind: EmptyKind;
  feed?: FeedId;
  headline: string;   // "No competitor scan has ever been run."
  why: string;        // names the exact missing file, by path
  how: string;        // the exact command, copyable, or the human decision required
  unlockAt?: string;  // "2 snapshots" | "5 named handles" — drives the ghost counter
  verbatim: boolean;  // true => contract-mandated wording, must not be paraphrased
  /* needs-declaration only. A kind whose next action is a DECISION, not a command. */
  declares?: { path: string; editorHref: string; templatePath: string };
  /* pull-failed only. The snapshot that proves the failure, so it is never read as a zero. */
  failedSnapshot?: { path: string; pulledAt: string; rowCount: 0 };
}
```

`needs-declaration` is the kind this app was missing. `never-pulled` says *run the command*;
`needs-declaration` says *a human has not yet decided something, here is where the decision is
written, here is the editor*. It generalises beyond competitors — `market` needs a source list on
the same footing — so it goes in the kit, not in one card.

**Append** two rows to the §6.1 table (after line 653):

| kind | meaning | visual | copy pattern |
|---|---|---|---|
| `needs-declaration` | the pull is possible; nobody has said WHAT to pull | dashed border, absent-hatch, **a filled slot counter**, primary button | *"Five slots. 0 named. This is a decision, not a pull — `data/watchlist/competitors.json` does not exist yet. [Open the watchlist editor →]"* |
| `pull-failed` | the command ran; the profile did not resolve | absent-hatch **plus an ember hairline**, axes not drawn, the failing file named | *"The pull ran at 2026-08-03 09:12 UTC and returned no profile for `@handle`. 0 rows, no header. `ingest_suno.py` writes `latest.json` even when a pull fails (`scripts/ingest_suno.py:257-273`) — this file is that. The handle is unverified, not empty. [Correct slot 3 →]"* |

`tests/empty-vs-zero.spec.ts` gains a third assertion: the `pull-failed` fill string never equals
the `measured-zero` fill string, in either theme. A 404 must never be able to look like an artist
with no plays.

---

## PATCH 2 — the roster file `data/watchlist/competitors.json`

**New file, gitignored.** Committed alongside it: `data/watchlist/competitors.template.json`,
identical but with every `handle` set to `null` — a template with plausible handles in it would be
an invented number wearing a name.

```jsonc
{
  "schema": "lofn.watchlist.competitors/1",
  "feed": "competitors",
  "purpose": "HOSTING_BEHAVIOUR",          // frozen literal; any other value fails validation
  "written_at": "2026-08-03T00:00:00Z",
  "written_by": "the-scientist",           // "the-scientist" | "lofn-draft"
  "review_cadence_days": 30,               // vault/SOCIAL_VOICE.md App. C: "re-pulled monthly"
  "slots": [
    {
      "slot": 1,
      "slot_name": "CADENCE_MIRROR",       // frozen enum, see below
      "brief": "150-300 public clips, 2,500-6,000 followers, higher MEDIAN like rate.",
      "hosting_question": "Do they beat me at my own volume without a viral outlier?",
      "handle": null,                      // null = slot open. NEVER a placeholder string.
      "platform": "suno",                  // "suno" only in v1; the puller is Suno-specific
      "added_at": null,
      "why_this_one": null,                // REQUIRED once handle is non-null. >= 12 words.
      "disqualifier_check": {
        "follow_for_follow":      "unchecked",  // unchecked | clear | fails   (MANUAL)
        "check_mine_out_section": "unchecked",  //                              (MANUAL)
        "top_song_over_50pct":    "unchecked",  //                              (AUTO on pull)
        "imitable_only_by_lying": "unchecked"   //                              (MANUAL, human)
      },
      "review_due": null,
      "retired_at": null,
      "retired_reason": null
    }
    /* slots 2..5 identical in shape */
  ]
}
```

**`slot_name` is a frozen five-value enum, in this order, transcribed from Appendix C:**
`CADENCE_MIRROR` · `COMMENT_SECTION_MODEL` · `AGE_CORRECT` · `EAST_ASIA` · `REMIX_HOST`.
Exactly five slot objects; a sixth fails validation with *"Appendix C names five slots, each
solving a different problem. A sixth handle is a list, not a study."* **There is no free-form
`add competitor` control anywhere in the app.** You cannot add a handle without attaching it to a
hosting question, because the only way in is through a slot that already is one.

**Manual observations** (the numbers no endpoint yields) live one file per reading at
`data/watchlist/observations/<handle>/<YYYY-MM-DD>.json`:

```jsonc
{ "schema": "lofn.watchlist.observation/1", "handle": "…", "observed_at": "…",
  "provenance": "public-page",            // never "api-public": a human read this
  "reply_rate_last_20": null,             // 0..1
  "mean_reply_over_mean_comment": null,   // ratio
  "comments_on_smaller_accounts_30d": null,
  "note": "" }
```

Absent fields render `never-pulled` with the manual method named. They are never estimated, never
carried forward, never averaged across handles.

---

## PATCH 3 — what the pull can and cannot fill (the honest half of the card)

Derived by reading `scripts/ingest_suno.py`. Of Appendix C's **six numbers**:

| # | The six numbers | Source | State |
|---|---|---|---|
| 1 | followers | not normalised by `ingest_suno.py`; survives only if the raw `stats` object happens to carry a numeric field | **conditional** — rendered iff present, `never-pulled` otherwise, never inferred |
| 2 | public clips | `header.num_total_clips` + `row_count` | **auto** |
| 3 | MEDIAN like rate | median of `rows[].like_rate` | **auto** (median only — top-song rate may not render) |
| 4 | reply rate, last 20 comments | the profile endpoint returns no comments | **manual observation only** |
| 5 | mean reply len ÷ mean comment len | idem | **manual observation only** |
| 6 | comments on smaller accounts, 30 d | idem | **manual observation only** |

Also auto on pull: **`top_song_over_50pct`** — `max(play_count) / summed_plays`, the one universal
disqualifier the data can check itself, and the trap we have already proven on ourselves.

**This table is the card's headline, not its footnote.** Rendered verbatim on the panel:

> *"The pull we can run fills two of the six numbers. The three that measure hosting — reply rate,
> reply length ratio, comments left on smaller accounts — are not in any endpoint we have. They are
> read by eye and written to `data/watchlist/observations/`. If this panel ever feels informative
> without them, it has become a sound comparison, which is the misuse Appendix C throws out."*

---

## PATCH 4 — THE PROJECTION GATE (`src/watchlist/project.ts`)

`ingest_suno.py` stores, per clip, `title`, `tags` (the **full Suno style prompt**),
`negative_tags`, `gpt_description_prompt` and eleven derived lyric features. For our own handle
that is the craft corpus. For a competitor handle it is a copy-their-sound machine sitting in the
app. Discipline will not hold this line; a type will.

```ts
// The ONLY shape a competitor row may take inside this app.
export interface CompetitorRow {
  id: string; play_count: number; upvote_count: number;
  like_rate: number | null; created_at: string; is_public: boolean;
}
export const COMPETITOR_FORBIDDEN_KEYS = [
  "title", "tags", "negative_tags", "gpt_description_prompt", "persona",
  "model_name", "sung_lines", "words_per_line", "repeated_line_ratio",
  "max_line_repeats", "section_count", "unique_sections",
  "emo_tag_count", "emo_tags", "lyric_chars", "has_chorus", "type", "has_vocal", "is_remix",
] as const;
export function projectCompetitor(raw: unknown): CompetitorRow;  // whitelist, not blacklist
```

Projection happens **in the loader**, before a row can reach a card. `tests/competitor-projection.spec.ts`:
(a) every key of every projected row is in the whitelist; (b) for every CardSpec whose
`evidence.source[]` touches `data/snapshots/competitors/`, `JSON.stringify(spec)` contains none of
`COMPETITOR_FORBIDDEN_KEYS`; (c) no competitor card declares `filters` including `style_term`, and
no style-board card accepts a competitor row.

**Optional, separately reviewable hardening to `scripts/ingest_suno.py`:** when `feed ==
"competitors"`, drop the style and lyric fields at write time unless `--keep-style` is passed, and
record `"projection": "hosting-only"` in the envelope. I would take it, but the loader gate is what
the app can *guarantee*, so the loader gate is the one that is mandatory here.

**Sort lock.** The five render in slot order, 1→5, always. No column header sorts. No rank number,
no medal, no "leader". Extend `tests/forbidden-strings.spec.ts` for this card family with
`"top competitor"`, `"best artist"`, `"beat them"`, `"like theirs"`, `"sounds like"`, `"outperform"`.
Sorting five artists by like rate is a fishing expedition rendered as an affordance — §13 rule 5,
applied to people this time.

**DECIBAL firewall.** Every watchlist card carries, rendered:
`forbidden: "may not become the comparator population for DECIBAL, may not enter vault/gates.yaml, may not select a style, may not produce a MAKE_IT."`
The reference-class overrule at Part 3 §4.1 stands untouched — a hosting roster does not become an
average artist by acquiring five rows.

---

## PATCH 5 — M5.7 THE WATCHLIST (insert after M5.6, before the `---` at line 1702)

- **M5.7 · THE WATCHLIST — the editor for feed #2.** Route `#/provenance/watchlist` (furniture in
  the hash, per §3.1; **the draft never enters the query string** — an unconfirmed handle must not
  be shareable). Reached from: the M5.1 feed-ledger `competitors` row, every competitors empty
  state, and the M0 directive when the roster is unset.

**Layout.** Appendix C's framing sentence at the top, then five slot cards in fixed order. Each
slot card renders: slot number and name · the brief · the hosting question · the handle field or
the filled handle · `why_this_one` · the four disqualifier chips (`unchecked` / `clear` / `fails`,
with `top_song_over_50pct` marked AUTO and stamped by the last pull) · the slot's state chip · one
primary action.

**Handle entry — what is validated, and what honestly cannot be.**
- Syntactic screen only: `/^[A-Za-z0-9_.-]{2,40}$/`. No spaces, no `@`, no scheme, no slash.
- Paste a profile URL and the editor **offers** the last path segment as an extraction, shown in
  full, applied only on click. It never silently rewrites what she typed.
- Duplicate across slots → refused with both slot names: five slots exist to solve five different
  problems, so the same account cannot hold two.
- **Existence cannot be checked. Say so, on the field:** *"Unverified. This app makes no network
  request. A handle is proven only by the first pull."* A green tick here would be the app
  inventing a fact — the exact failure mode the whole spec is built against.
- Saving a non-null handle with `why_this_one` shorter than 12 words is refused inline: *"A handle
  without a hosting question is a list, not a study."*

**Composing the file — the app prepares, it never writes.** Per `vault/AUTONOMY.md`, and per §15's
`server.spec.ts` (which exists to prove there is no write endpoint), the editor produces:
1. the full JSON, rendered in a `<pre>` with a **Copy** button;
2. a copyable heredoc: `mkdir -p data/watchlist && cat > data/watchlist/competitors.json <<'JSON' … JSON`;
3. a **Download `competitors.json`** button — a `Blob` + object URL, entirely local, no network,
   CSP untouched.

The working draft lives in `localStorage` under `lofn.watchlist.draft`, and is rendered under a
permanent band: **"DRAFT — NOT ON DISK. Nothing below counts this."**
`tests/watchlist-draft.spec.ts`: a draft never clears a blocker, never enters any
`evidence.source[]`, and never changes any filled-slot counter. The counters read the served file
or nothing.

- **M5.8 · Watchlist status strip.** `roster: 3 of 5 named · pulled: 1 of 3 · oldest reading 21 d ·
  next review 2026-09-02`. Sits in the M5.1 ledger's `competitors` row and mirrors into every
  competitors empty state, so the ghost counter is the same object everywhere it appears.

---

## PATCH 6 — the six states, each with exactly one next action

| # | State | Detected by | What renders | The one action |
|---|---|---|---|---|
| **S0** | No roster | `GET /analytics/watchlist/competitors.json` → 404 | verbatim *"No competitor scan has ever been run."* **unchanged**, then the `needs-declaration` block: *"Five slots. 0 named. `data/watchlist/competitors.json` does not exist."* | **[Open the watchlist editor →]** |
| **S1** | Draft only | file 404 + `localStorage` draft | S0's block plus the DRAFT band and the composed file | **[Copy the write command]** (secondary: Download) |
| **S2** | Roster on disk, k of 5 named | file parses, some `handle === null` | five slot cards; open slots show their Appendix-C brief, not an apology | per open slot: **[Fill slot n →]** |
| **S3** | Named, not pulled | slot handle set, `competitors/<h>/latest.json` → 404 | `never-pulled`, `unlockAt: "1 pull"`, and the consequence line below | **[Copy pull command]** |
| **S4** | Pulled, reconciled | `latest.json`, `row_count > 0`, `reconciliation.ok` | the six-number table (2 auto, 1 conditional, 3 manual), the AUTO disqualifier verdict, age chip | **[Record the three hosting numbers →]** (composes the observation file) |
| **S5** | Pull failed | `latest.json` with `header === null` or `row_count === 0` | `pull-failed`, ember hairline, the failing file path and its `pulled_at` | **[Correct slot n →]** |
| **S6** | Retired | `retired_at` non-null | greyed row, reason rendered, past snapshots still readable — nothing is deleted | **[Restore slot n]** |

**S3's consequence line, in the §14.4 pattern, rendered above the command:**
> *"`python3 scripts/ingest_suno.py --handle <h>` — up to ~60 public GET requests to
> `studio-api.prod.suno.com`, paced 0.35 s. No credentials sent. Writes
> `data/snapshots/competitors/<h>/<stamp>.json` and `latest.json`. Style prompts and lyric features
> in that file are dropped by this app at load. Does not publish, render, or spend.
> **This app will not run it. Copy it and run it yourself.**"*

**S5 is not hypothetical.** `ingest_suno.py` breaks out of the page walk on an HTTP error at
`page >= 1` (`scripts/ingest_suno.py:160-164`), leaving `header = None` and zero rows; the
reconciliation then computes `plays_delta = None` and sets `ok = False`; and the file is written
anyway (`:264-273`). A mistyped handle therefore produces a real, present, unreconciled snapshot of
nothing. Without S5 the app would draw a zero for it.

**New invariant I11** (extend §6.3's I1–I10, and `tests/invariants.spec.ts`):
*a snapshot with `feed === "competitors"` and (`header === null` or `row_count === 0`) is classified
`pull-failed`, never `measured-zero`, and is excluded from every aggregate.* Its own screen naming,
per the §6.3 error contract.

---

## PATCH 7 — routes (`apps/lofn-analytics/snapshot-route.mjs`, W1-L1)

Two additions, both GET, both read-at-request-time, both shared by `server.mjs` and the Vite dev
middleware. No write endpoint is added and `tests/server.spec.ts`'s 405 assertion stands.

```
GET /analytics/watchlist/competitors.json                  -> ../../data/watchlist/competitors.json
GET /analytics/watchlist/observations/<handle>/index.json  -> newest file in data/watchlist/observations/<handle>/
GET /analytics/snapshots/competitors/<handle>/latest.json  -> ../../data/snapshots/competitors/<handle>/latest.json
```

- `<handle>` is re-screened against `/^[A-Za-z0-9_.-]{2,40}$/` **at the route**, before any path
  join. Anything else → 400. The roster is a file on disk and a file on disk is untrusted input.
- **No directory listing.** The app never enumerates `data/snapshots/competitors/`; the roster is
  the index, so an unnamed handle is invisible to the app by construction.
- **Named hazard.** `scripts/ingest_receiver.mjs` accepts `--feed competitors` and writes
  `data/snapshots/competitors/<stamp>.json` at the *feed root*, while `ingest_suno.py --handle h`
  writes the `<h>/` subdirectory. Two shapes, one directory. The loader reads **subdirectories
  only**; any loose `*.json` at the competitors root is listed in the M5.2 Schema Guard as
  *"unclassified receiver drop — no handle in the path"*, and never joined to a slot.
- `scripts/check-build.mjs` extends its sweep: no bytes of `data/watchlist/` in `dist/` either.

---

## PATCH 8 — splices into existing lines

**Part 2 §1, line 1398** — replace the last cell of feed row 2:

> Nothing until the roster exists. `needs-declaration` + the verbatim empty state + the live
> blocker chain, each clearable from inside the app: (a) name k of 5 slots in
> `data/watchlist/competitors.json` via M5.7, (b) copy and run one pull per named handle. Once
> pulled: **hosting numbers only** — public clips, median like rate, the top-song-share
> disqualifier. Style prompts, titles and lyric features are dropped at load
> (`src/watchlist/project.ts`) and no competitor row may enter a craft, style or genre card.

**§14.4, line 1784** — replace the final sentence:

> `market` and `soundry` have **no refresh button** — they have a blocker list, because a button
> that cannot work is worse than no button. **`competitors` has no refresh button either, and it
> has something better: a blocker list that clears from inside the app.** Its blocker is a
> *decision* (`needs-declaration`), not a command, so the affordance it gets is the M5.7 watchlist
> editor plus a per-handle **[Copy pull command]**. The app composes and displays; The Scientist
> runs it. A blocker with no path to clearing it is not honesty, it is a dead end — and the pull it
> is waiting on has existed in `scripts/ingest_suno.py` since 2026-07-27.

**§15 file layout** — add under `src/`:

```
    watchlist/  schema.ts validate.ts project.ts compose.ts load.ts
    cards/ m5-7-watchlist/ m5-8-status/ comp-hosting-numbers/ comp-observation-form/
```

**§15 test contract** — add items 10–13:

```
10. watchlist.spec.ts            — schema, 5-slot enum lock, duplicate refusal, why_this_one floor,
                                   URL extraction is offered not applied, handle screen
11. competitor-projection.spec.ts— the whitelist, the CardSpec sweep, the style-card refusal
12. watchlist-draft.spec.ts      — a localStorage draft clears no blocker and enters no evidence
13. watchlist-states.spec.ts     — S0..S6 each render exactly one primary action; a pull-failed
                                   fixture (header:null, row_count:0) never renders a 0 or an axis
```

**Build order** — `W0` gains: add `data/watchlist/` to `.gitignore` beside `data/snapshots/`, and
commit `data/watchlist/competitors.template.json` with all five handles `null`. New lane:

> - **W3-L10 · WATCHLIST** (needs L1 + L4; parallel with L5–L9).
>   `src/watchlist/{schema,validate,project,compose,load}.ts` ·
>   `src/cards/m5-7-watchlist/` · `m5-8-status/` · `comp-hosting-numbers/` ·
>   `comp-observation-form/` · the `needs-declaration` and `pull-failed` renderers in the kit's
>   empty-state module · I11 in `src/load/invariants.ts` · the three routes in
>   `snapshot-route.mjs` (coordinator merges the route file at W4) · tests 10–13.

**Decisions block** — append:

> - THE COMPETITOR BLOCKER BECOMES CLEARABLE. §14.4's "no button that cannot work" is upheld —
>   no refresh button ships — but a blocker whose input has no home in the app is a dead end, and
>   a dead end is the one thing an interface may not have. The missing artifact was never the
>   button; it was the roster. `data/watchlist/competitors.json` + the M5.7 editor + per-handle
>   copyable pull commands. PRESERVES: her feed #2 exactly as drawn — five watched targets — and
>   AUTONOMY untouched, because the app composes the command and never runs it.
> - THE ROSTER IS SLOT-LOCKED TO APPENDIX C AND PROJECTED TO HOSTING FIELDS ONLY. Five named slots,
>   no free add, a mandatory hosting question per handle, sort locked to slot order, and every
>   style/title/lyric field dropped in the loader before a competitor row can reach a card.
>   PRESERVES: `vault/SOCIAL_VOICE.md` App. C's standing rule as a build failure rather than a
>   memory — "make music like theirs" is not discouraged here, it is unrepresentable.

**Open questions** — append:

> - The five slots are yours to name and the app ships them empty. `vault/SOCIAL_VOICE.md`
>   Appendix C already defines what each slot is FOR; M5.7 will not let a handle be saved without a
>   sentence saying what hosting behaviour it is being watched for. RECOMMENDED DEFAULT: fill slot
>   2 (THE COMMENT-SECTION MODEL) first — it is the slot whose numbers no endpoint can give us, so
>   it is the one that proves the manual observation path works before we depend on it.

---

## Where I disagree with the other seats

**With whoever wrote §14.4.** "A button that cannot work is worse than no button" is right about
buttons and wrong as a stopping point. Deleting the control without supplying the input surface
converted a *missing feature* into an *unreachable state*, which is strictly worse: the honest
empty state now lies by implication, because it reads as "nothing can be done" when in fact one
CLI flag and one JSON file separate us from a filled panel.

**With any seat that would let the app write the roster over a small local endpoint.** No. The
sealed no-write posture is load-bearing and `server.spec.ts` exists to prove it. Compose, display,
download, copy — she pastes. One extra step, and the app keeps the property that makes it
trustworthy.

**Against the tidy version of my own patch.** I considered a plain `handles: []` array. It is
smaller and it is wrong: an unstructured list of five artists, sortable by like rate, IS the misuse.
The slot lock and the projection gate are the patch; the file is just where they live.

RISKS:
- Scope creep into a fifth surface. This patch adds one editor, four cards and four test files to a spec that is already 277k. Mitigation: W3-L10 is a single parallel lane with no dependents, and every card it adds is a state of one panel rather than a new tab. If the panel has to be cut, cut the observation form (S4's action) first and leave S0-S3 — the dead end is closed by S0-S3 alone.
- The manual observation path may never be walked. Three of the six numbers require The Scientist to read comment sections by eye and hand-write JSON. If she does not, the panel sits at S4 showing two sound-adjacent numbers forever, which is exactly the misuse posture. Mitigation is the verbatim headline that says so on the panel; there is no mechanical fix, and pretending otherwise would be the invented number.
- The style prompts still land on disk. The projection gate is a LOADER gate: `ingest_suno.py` writes competitors' full style prompts into data/snapshots/competitors/<h>/ regardless, and any human or agent reading that file bypasses the app entirely. The optional --keep-style script change closes it; if the panel rejects that change, the residual risk is a file on disk that doctrine forbids acting on but nothing prevents reading.
- Handle screening is syntactic only and deliberately weak. `/^[A-Za-z0-9_.-]{2,40}$/` is my inference about Suno handle grammar from the API path shape, not a documented rule — Extrapolating. A legal handle outside that class would be refused with no way in except editing the JSON by hand. Widen on the first real refusal rather than guessing wider now; the route-level re-screen must be widened in the same commit or the app will 400 on a handle the editor accepted.
- S5 detection rests on my reading of ingest_suno.py's failure path (header stays None, row_count 0, ok False, file written anyway). It has not been exercised against a real 404 — no competitor pull has ever run. The first pull against a deliberately wrong handle is the cheapest possible verification and should be the first thing W3-L10 does.
- A roster that exists makes DECIBAL's rejected 'average artist' comparator look newly available. The forbidden line and the firewall test are the whole defence, and they are a convention until tests 11 and the extended forbidden-strings sweep actually run in CI. If either is dropped in a hurry, the reference-class overrule at Part 3 4.1 quietly reopens.


---

# APPENDIX — seat patch: THE SYNTHESIST (after Hans Rosling — Gapminder / Trendalyzer) (Rosling's documented practice: combining disparate, differently-collected series into one legible view; insisting the audience be shown the data's limits alongside the data; refusing to let a beautiful display outrun what the numbers support. Extrapolating: the Gapminder bubble — area-as-a-third-channel — is the exact affordance I have to give up here, and I give it up on the record in card X4.)

FINDING: The Scientist's board 6 asks that the four feeds be COMBINED into insights, and the spec answers with four feeds standing side by side — feeds 2, 3 and 4 get the word "Nothing" and no card at all, while lesser gaps ship locked ghost wireframes that auto-enable; no card anywhere joins two feeds; and the Soundry emitter specified in Part 3 has no reader in Part 2, so the one loop that could ever close is open by construction.

FILES: E:/git/private-claw/docs/reference/lofn_analytics_and_soundry/BUILD_SPEC.md, apps/lofn-kit/src/plot/spec.ts, apps/lofn-kit/src/plot/validate.ts, apps/lofn-kit/src/contracts/ghost.ts, apps/lofn-kit/src/contracts/empty.ts, apps/lofn-kit/src/stats/ols.ts, apps/_fixtures/soundry-round.json, apps/lofn-analytics/src/load/soundry.ts, apps/lofn-analytics/src/load/competitors.ts, apps/lofn-analytics/src/cards/m6-x0-join-matrix/, apps/lofn-analytics/src/cards/m6-x1-distributed/, apps/lofn-analytics/src/cards/m6-x2-lyric-three-instruments/, apps/lofn-analytics/src/cards/m6-x3-emo-vocabulary/, apps/lofn-analytics/src/cards/m6-x4-market-position/, apps/lofn-analytics/src/cards/m6-x5-three-clocks/, apps/lofn-analytics/src/cards/m6-x6-the-loop/, apps/lofn-analytics/server.mjs, apps/lofn-analytics/snapshot-route.mjs, apps/lofn-analytics/tests/join-evidence.spec.ts, apps/lofn-analytics/tests/ghost-autoenable.spec.ts, apps/lofn-analytics/tests/soundry-reader.spec.ts, apps/lofn-analytics/tests/competitors-reader.spec.ts, apps/lofn-analytics/tests/fusion-refusals.spec.ts, apps/soundry/tests/emit-snapshot.spec.ts, data/watchlist.yaml, .claude/skills/lofn/SKILL.md, .agents/skills/lofn/SKILL.md

== PATCH SYN-2 · THE CONFLUENCE SURFACE ==
Splices into `docs/reference/lofn_analytics_and_soundry/BUILD_SPEC.md`.
Eight inserts, each with its anchor. Nothing below invents a number; every figure quoted was
recomputed from disk on 2026-08-01 and the normaliser that produced it is stated in the card.

---------------------------------------------------------------------------------------------------
## SYN-2.0 · The ruling this patch rests on

Four feeds displayed in four panels is not synthesis; it is four panels. Her board says
"**combine** all 4 sources into useful insights" and then names the four insights. So the four named
insights ARE the fusion products. The spec currently answers all four from ONE feed on tab M2. This
patch does not move them — it adds the tab where the same four questions are asked ACROSS feeds, and
it makes the absent feeds visible as reserved axis space rather than as apology text.

**I am overruling one spec decision, and naming the intent I preserve.** §14.4 gives `competitors`
no refresh button, on the ground that "a button that cannot work is worse than no button." That
intent is exactly right and I keep it — by making the button's *existence* conditional on the thing
that makes it work (§SYN-2.7). Zero named handles, no button. One named handle, one button that
works. No other §14.4 removal is touched.

---------------------------------------------------------------------------------------------------
## SYN-2.1 · Part 1 §2.1 `spec.ts` — the type system cannot currently describe a join

ANCHOR: after the `Evidence` interface, same file `apps/lofn-kit/src/plot/spec.ts`.

`Evidence` carries one `asOf`, one `provenance`, one `reconciled`. A fused card has two of each and
the honesty gate has no way to say so. ADD:

```ts
export interface JoinEvidence {
  left:  { feed: FeedId; asOf: string; provenance: Provenance; n: number };
  right: { feed: FeedId; asOf: string; provenance: Provenance; n: number };
  key: "clip_id" | "normalised_title" | "platform" | "style_term" | "handle" | "none";
  keyRule: string;      // the normaliser, verbatim and executable-in-prose. REQUIRED.
  attempted: number;    // keys offered by the smaller side
  resolved: number;     // keys that matched
  ageGapHours: number;  // |left.asOf - right.asOf|, drives the §6.2 mismatch band
  unresolvedPolicy: "counted-and-shown" | "excluded-and-listed";  // never "dropped"
}
```

ADD to `Evidence`: `join?: JoinEvidence;`
ADD to `EmptyKind`: `| "emitter-silent" | "join-unresolved"`.
ADD to `Provenance`: `| "randomised-round"`.

New validator rules in §2.3 (`validate.ts`):

| # | rule | why |
|---|---|---|
| V19 | a spec whose `evidence.source` paths span two `data/snapshots/<feed>/` directories ⇒ `evidence.join` present | a fused card that does not declare its join is a fused card nobody can audit |
| V20 | `evidence.join` ⇒ `keyRule` non-empty **and rendered under the plot**, and `resolved`/`attempted` printed as a fraction, never as a bare percentage | the yield is the first thing that can lie |
| V21 | `evidence.join.ageGapHours` exceeding the fresher feed's FRESH window ⇒ the §6.2 mismatch band draws; not optional, not a caption | two feeds on one axis at different ages is the failure mode of every fused dashboard |
| V22 | `provenance === "randomised-round"` ⇒ the row carries `roundId` and `preregHash` in `detail`, and the card renders both | a causal number with no frozen pre-registration behind it is the thing this estate exists to prevent |

**Dependency, declared not assumed:** Critic 2 records three incompatible `Provenance` vocabularies.
Whichever seat unifies them must keep exactly one value meaning *"this number came from a
pre-registered randomised round in the sibling app"* — it is the only provenance in the estate that
licenses a causal verb, and collapsing it into `derived` silently unlocks that verb everywhere.

**Second dependency:** X2 needs a slope estimator that `stats.ts` does not have. ADD to Part 1 §2.4:
`olsSlope(xs, ys, level) => { beta, se, lo, hi, n, r }` — ordinary least squares, Wald interval,
pure, node-testable. Without it M2.1's ρ = −0.33 can never be re-expressed as an effect per word,
and a correlation coefficient cannot be put on the same axis as a randomised lift.

---------------------------------------------------------------------------------------------------
## SYN-2.2 · Part 1 §6.1 — two empty kinds the table does not have

ANCHOR: two new rows in the §6.1 table.

| kind | meaning | visual | copy pattern |
|---|---|---|---|
| `emitter-silent` | the directory exists; the emitter has never written into it | dashed border, hatch, **axes ARE drawn** (the shape is known — only the values are missing) | *"`data/snapshots/soundry/` exists and is empty. The Soundry has never closed a round. Nothing here is zero."* + the emit command + what will appear |
| `join-unresolved` | both feeds are present; the key resolved nothing | real axes, both feeds' row labels drawn at full ink, no marks | *"Both feeds are loaded. 0 of 9 keys matched under `<keyRule>`. This is a join failure, not an empty feed."* + the keyRule + the unmatched list |

`emitter-silent` closes the case Critic 2 found: `never-pulled` is defined as "no snapshot has ever
existed" and keys on a path that now resolves. **`tests/empty-vs-zero.spec.ts` extends: the
`emitter-silent` hatch and the `never-pulled` hatch must differ, because the operator actions differ
(one is "run the pull", the other is "close a round").**

---------------------------------------------------------------------------------------------------
## SYN-2.3 · Part 1 §6 — THE GHOST CONTRACT, specified once instead of three times informally

ANCHOR: new §6.4, `apps/lofn-kit/src/contracts/ghost.ts`.

M2.5(a), M3.7 and every card below ship a locked wireframe. It is currently described in prose, three
times, differently. One object:

```ts
export interface UnlockCounter {
  label: string;   // "competitor snapshots"
  have: number; need: number;
  how: string;     // the exact command, or the human decision, VERBATIM. Copyable.
  path: string;    // the file whose existence satisfies it
}
export interface GhostSpec {
  of: PlotSpec;               // the same spec that will render live. Not a stand-in.
  unlockAt: UnlockCounter[];  // 1..n INDEPENDENT segments
  reason: string;             // one sentence, why it cannot draw yet
}
```

Render rules, uniform everywhere:

1. **Axes, ticks, axis titles, facet headers, row labels, reference rules and quadrant labels draw at
   FULL ink. Only the mark layer is ghosted.** A ghost that hides the question is a grey rectangle
   with a padlock on it, and it teaches her nothing. The reserved shape IS the argument for pulling
   the feed.
2. Mark layer 12% α, padlock at the plot centroid, `aria-disabled="true"`, alt text generated from
   the unlock tuple.
3. Counters light **independently**. A card with three counters draws every series it can and prints
   `2 of 3 clocks drawn`. Nothing waits for everything.
4. **Auto-enable is structural, not a branch.** `plot()` renders the ghost when
   `evidence.n === 0 && ghost` is set; the day the file lands the identical spec renders live.
   `tests/ghost-autoenable.spec.ts`: one spec, two fixtures, two outputs, **zero conditionals inside
   the card module** (asserted by reading the card source for `if (` around the plot call).
5. **A ghost never replaces an `empty`.** Both are required. The empty state is the words; the ghost
   is the shape. `card-registry-check.mjs` fails a spec that has one and not the other.

---------------------------------------------------------------------------------------------------
## SYN-2.4 · Part 2 §1 — the feed table's "Nothing" column is replaced

ANCHOR: Part 2 §1, the rightmost column ("What the app may draw from it") for rows 2, 3, 4.

| # | feed | REPLACE "Nothing" WITH |
|---|---|---|
| 2 | `competitors` | **The same instrument pointed at another catalogue.** `scripts/ingest_suno.py --handle <h>` writes the identical schema, so competitor rows carry `words_per_line`, `emo_tags`, `emo_tag_count`, `like_rate` with no derivation and **no join key required** — the join is column-identity, not title-matching. Feeds M6·X2 row 2, X3 (all rails), X4 (secondary), X5 (clock 2). **Highest-yield single pull in the estate; the M6 Tab Directive names it.** Empty until then: two counters, `0 of 5 handles named` (blocker: The Scientist) and `0 of 5 pulled` (blocker: one command each). |
| 3 | `market` | **The only exogenous axis we would own.** Feeds M6·X4 (x axis) and X5 (clock 3). Blocker is honest and must be stated as such: **`scripts/scan_market.py` does not exist.** The transport does (`node scripts/ingest_receiver.mjs --feed market --once` accepts a POSTed snapshot); the scanner does not. Sources are already inventoried at `vault/DAILY_RESEARCH_SOURCES.md` §D. **The empty state may not print a command that would fail.** |
| 4 | `soundry` | **The only randomised evidence the estate will ever hold, and the only join that can raise a tier.** Feeds M6·X6 and X2 row 3. Reader specified at §SYN-2.6. Directory exists and is empty ⇒ `emitter-silent`, not `never-pulled`. |

---------------------------------------------------------------------------------------------------
## SYN-2.5 · Part 2 — NEW §6A · MUSIC · Tab M6 — **CONFLUENCE**
### *(her board 6: "combine all 4 sources into useful insights")*

ANCHOR: new section between §6 (M2 CRAFT) and §7 (M3 REACH). Rail order: SIGNAL · RESPONSE · CRAFT ·
REACH · HEADWINDS · **CONFLUENCE** · PROVENANCE. Route `#/music/confluence`.

**Tab Directive today:** *"Pull one competitor with `python3 scripts/ingest_suno.py --handle
<handle>` — it is the same instrument on another catalogue, so it joins with no key and fills four of
the six cards on this tab."* — tier `ABSENT` / magnitude `PULL_THE_FEED` / routes `ops.runbook` /
forbidden: *"may not name the handles itself; the watched set is The Scientist's call."*
(Legal tier↔magnitude pair per Part 1 §4.2.)

**Filter applicability, declared per card** (§3.3 requires dimmed-not-dropped):
`X0` honours none — it is a census of files, and dimming all six chips with the tooltip *"the join
matrix counts files, not clips"* is the correct render. `X1` honours `out`, `pf`, `win`. `X2` honours
`win`, `est`, `out`. `X3` honours `win`. `X4` honours `win`, `nmin`. `X5` honours `win`. `X6` honours
none — a closed round's cohort was frozen at pre-registration and **a filter that could re-cut a
randomised result would silently un-randomise it**; all six chips dim with that sentence as the
tooltip. That last one is not a limitation, it is the guarantee.

---

### X0 · THE JOIN MATRIX — what can be put on one axis at all *(LIVE TODAY)*

- **mark** `strip`, posture `heat` · **x** `feed_b`, band, domain `[suno, socials, distribution,
  competitors, market, soundry]` · **y** `feed_a`, same domain, lower triangle only · **colour**
  `join_yield_rate`, **Sunset sequential ramp** (legal: V8 caps *ordinal* domains at 4; this is the
  sequential channel Sunset owns per the frozen Decisions) · **facet** none · **feed** all six
- `nField: "attempted"` · `denominatorPlot: "m6.x0.attempts"` (V9 fires on `_rate$` — deliberate)
- **Every cell prints its fraction, not its percentage** (`4/9`, `2/10`, `0/0`), and its key name.
- **A pair with no shared key renders in absent-hatch, never as 0%.** Cell-level empty is the same
  contract as panel-level empty; a 0% cell would read as "we tried and failed" when the truth is
  "there is no key to try."
- **Q:** which of these six feeds can be compared to which, and at what yield?
- **Measured today, under the stated normaliser** — NFKD → lowercase → strip `(feat …)` → strip all
  non-alphanumerics; **this rule is rendered under the plot verbatim, because the yield is a property
  of the rule as much as of the data:**
  - `suno ⋈ distribution` on `normalised_title` → **4 of 9** `releases_seen`
    (Triple Arch Over Me · A Dumber, Better Song · 3-3-2 Open · Dear Scientist:) and **2 of 10**
    Apple `top_tracks` (Dear Scientist: · Triple Arch Over Me). The five unmatched `releases_seen`
    entries are **albums, not tracks** (`Ghost Overload` 14 tracks, `Experiment 01` 9 tracks) — the
    unresolved list says so per row. `unresolvedPolicy: "counted-and-shown"`.
  - `suno ⋈ socials` on `platform` → **1 of 6** rows, and it is a *census* join, not an item join.
  - every remaining pair → **no key exists** (hatch).
- **Caption, verbatim:** *"Six feeds. Fifteen possible pairs. Two share a key today, and the better
  of the two resolves four titles out of nine. Everything on this tab below this card is limited by
  this card."*
- **X0b · The one agreement we can check** — a two-row inset, not a separate card. `mark: "interval",
  intervalVariant: "dumbbell"`, x = value (log), y = quantity band. Four quantities read by two
  independent instruments on the same night: an 11-page authenticated-free API walk
  (`data/snapshots/suno/latest.json`, `pulled_at 2026-08-02T00:03:48Z`) and a hand-read public
  profile page (`data/snapshots/socials/latest.json`, `pulled_at 2026-08-02T00:15:00Z`,
  `method: public-profile-read`):
  **followers 3,926 = 3,926 · works 210 = 210 · plays 374,122 = 374,122 · likes 5,913 = 5,913.**
  Every dumbbell has zero length. **Q:** does our instrument agree with anything outside itself?
  Today, exactly: four times, to the unit, twelve minutes apart. `ageGapHours: 0.2`.
  This is the estate's only external validation of its own reading and it belongs on the fusion tab,
  because agreement between two instruments is the cheapest fusion result there is and the only one
  we currently have.
- **deeper** → `#/music/provenance#joins`

---

### X1 · DID IT EVEN GET OUT? — the catalogue against the distributed set *(LIVE TODAY)*

- **mark** `strip`, posture `rug` — deliberately THE COMB again, at 34px, so the fused view is read
  with the device she already learned · **x** `log₁₀(play_count)`, domain 1→1,000, tear glyph + 44px
  outlier gutter, **shared across both facets (V18: byte-identical domains)** · **y** none ·
  **colour** ordinal **2** levels — `resolved to a DistroKid release` (`--m3`) / `no distribution
  record` (`--absent` hatch) · **facet** `{field: "distribution_state", columns: 1, scales:
  "shared"}` · **feed** `suno` ⋈ `distribution`
- `refs`: a `kind:"median"` rule per facet. `nField: "n"`.
- **REFUSAL, printed above the plot, not in a tooltip, verbatim:** *"Four of nine DistroKid releases
  resolve to a clip, and one of the four is Triple Arch Over Me. The distributed set was chosen
  BECAUSE it was liked — the arrow runs backwards. Comparing its response to the catalogue's would
  measure the choosing, not the distributing. This panel counts. It does not compare, and there is no
  control on it that will make it compare."*
- Tier forced `OBSERVATION`. `nextMove.forbidden`: *"may not be read as 'distributed songs do
  better'; may not select the next release; may not enter `vault/gates.yaml`."*
- **Q:** how much of the catalogue has ever left Suno? Answer today: four titles we can prove, out of
  210, and we cannot enumerate the album tracks.
- **deeper** → M1.6 catalogue filtered `distributed=1`. **empty**: `join-unresolved` if yield hits 0
  under a changed normaliser.

---

### X2 · LYRIC LENGTH, THREE INSTRUMENTS — *whiteboard insight #1, fused* *(1 of 3 rows live)*

- **mark** `interval`, `intervalVariant: "ci"`, `intervalStyle: "grain"` (THE GRAIN WHISKER — sample
  size is felt before it is read) · **x** change in like rate per **+1 word per line**, in **pp**,
  symmetric axis, `refs: [{kind:"zero", axis:"x", label:"no effect"}]` (V5 satisfied) · **y** band,
  three rows · **colour** Tide on sign — `--sig-pos` / `--sig-neg` / `--sig-null`, 3 levels, and Tide
  is the only legal palette inside a stats plot · **facet** none · `nField: "n_effective"`
- The three rows, and the row labels draw at full ink even when ghosted:
  1. `OBSERVED · our catalogue` — **LIVE**, `feed suno`, estimator `olsSlope` on the modern cohort.
     Recomputed live; the card prints the same ρ M2.1 prints, from the same code path. *(No value is
     typed into this spec: M2.1's ρ = −0.33 is the correlation, not the slope, and quoting a slope I
     have not computed would be the exact defect this patch exists to close.)*
  2. `OBSERVED · watched artists` — **GHOST**, `unlockAt: [{label:"competitor snapshots", have:0,
     need:1, how:"python3 scripts/ingest_suno.py --handle <handle>",
     path:"data/snapshots/competitors/<handle>/latest.json"}]`
  3. `RANDOMISED · Soundry arm contrast` — **GHOST**, `unlockAt: [{label:"closed rounds declaring
     words_per_line as the factor", have:0, need:1, how:"close a round in The Soundry, then `node
     apps/soundry/emit-snapshot.mjs --round <round-id>`",
     path:"data/snapshots/soundry/latest.json"}]`, with the Power Notch drawn at the round's
     pre-registered MDE.
- **THE RULING, printed on the card, verbatim:** *"These three rows are never averaged and never
  pooled. Row 1 is a correlation across songs nobody randomised. Row 3, when it exists, is a
  randomised contrast blocked on day. If they disagree, row 3 is the estimate and row 1 is the
  confound — and the distance between them is the most useful number on this tab. A single fused
  figure would delete it."*
- **Defensibility:** row 1 ⊕ row 2 is defensible as a *comparison of two observational slopes*, never
  as a pooled slope — the two catalogues have different audiences, cadences and placement histories,
  and nothing controls for them. Row 3 stands alone and is the only row that may exceed
  `OBSERVATION`. **Power for row 3 is not a detail:** per `vault/EXPERIMENT_STANDARD.md` §3, +15%
  needs ~225 songs (~19 days at 12/day), ~95 (~8 days) if plays are allowed to mature. The row's
  unlock counter therefore carries **`n so far / n needed / days at 12 a day`** (§5 rule 3), not a
  boolean.
- **deeper** → M2.1.

---

### X3 · EMO VOCABULARY, OURS AGAINST THEIRS — *whiteboard insight #2, fused* *(1 of 6 series live)*

Two panels, shared x, `facet.scales: "shared"`.

- **(a) Vocabulary shape.** **mark** `line`, log–log Zipf · **x** tag rank, log · **y** uses, log ·
  **colour** ordinal, **artist, capped at 4 levels — the 5th and 6th watched artists route to
  `facet: {field:"artist", columns:2, scales:"shared"}` rather than a 6-level legend** (V8 is not
  worked around; it is obeyed by faceting, which is what V8 is for) · **feed** `suno` ⊕ `competitors`
  - LIVE today: one series — **863 distinct tags across 202 clips**. Five ghost rails, axis space
    reserved and labelled, one independent unlock counter each.
- **(b) Tags per clip.** **mark** `interval` ci/grain · **x** tags per clip · **y** artist band ·
  `nField: "clips"` · `refs: [{kind:"median", label:"our median"}]`
- **THE RULING, printed, verbatim:** *"Cross-artist per-TAG comparison is refused for exactly the
  reason the internal one is (M2.3c): at a median of one use per tag, a ranking is a list of coin
  flips sorted by outcome, and pointing the same broken ruler at six catalogues does not fix it. What
  IS comparable across artists is the SHAPE of the vocabulary — tags per clip, distinct-tag count,
  Zipf slope — because those are properties of hundreds of clips, not of one tag."*
- **Defensibility: (a) and (b) are defensible the day the feed lands; a per-tag cross-artist
  leaderboard never becomes defensible at any n this estate will reach, and the card must not gain
  one later.** Add to the §13 test list: *"refuses to render a per-tag row on X3 regardless of feed
  count."*
- **Q:** is our emotional vocabulary unusually wide, or is that just what a Suno prompt looks like?
- **deeper** → M2.3.

---

### X4 · WHERE THE MARKET LOOKS AND WHERE WE WRITE — *whiteboard insight #3, fused* *(GHOST, half-drawn today)*

- **mark** `strip`, posture `cloud` · **x** market attention index for the term, `[0,1]` normalised
  rank-within-scan, `format:"ratio"`, from `market` · **y** our share of the modern cohort matching
  that term, `format:"pct1"`, `denominatorPlot:"m6.x4.terms"` (V9) · **colour** ordinal **4** levels
  = quadrant membership (`both high` / `market high · ours low` / `ours high · market low` / `both
  low`), direct-labelled, **quadrant rules at the median of each axis** · **facet** none ·
  `detail: "style_term"` · `nField: "n_clips"` · **feed** `suno` ⋈ `market` on `style_term`
- **THE REFUSAL, and it is mine to make as the seat that most wants the bubble, printed verbatim:**
  *"There is no third channel on this plot. No bubble area for like rate, no colour ramp for
  performance, no fitted line. Area would encode a rate whose 95% interval at these cell sizes
  (n = 1 to 39) is wider than the whole axis, and a line fitted through 20 overlapping
  substring-matched terms — 45 of our 46 modern clips match two or more of them — is the garden of
  forking paths drawn as a slope. This is a positioning census. It tells you where we are not
  looking. It tells you nothing about what would work."*
- **Defensibility: NOT DEFENSIBLE, at any n, as a correlation between market attention and our
  response.** Different instruments, no shared denominator, k=20, overlapping non-partitioning sets,
  cells of 1. Defensible only as a two-axis census with no effect claim. Tier hard-capped
  `OBSERVATION`; `magnitude: RUN_AN_EXPERIMENT`; `routesTo: soundry.hypothesis`; forbidden: *"may not
  select a style for the next run; may not become a gate."*
- **The ghost is half-drawn, and this is the point of the card today:** both axes, both median rules
  and all four quadrant labels draw at full ink; **all 20 of our style terms plot on the y axis at
  x = 0 in absent-hatch**, against a dotted x-axis terminating in a hollow ring labelled `UNKNOWN`.
  She can see that we already hold half of this plot and that one feed completes it. An empty
  rectangle would not have told her that.
- **empty:** `never-pulled`, feed `market`. **how**, honest and blocker-first: *"No market scan
  exists and `scripts/scan_market.py` has not been written. The transport exists (`node
  scripts/ingest_receiver.mjs --feed market --once` accepts a POSTed snapshot at
  `data/snapshots/market/`); the scanner does not. Sources are inventoried at
  `vault/DAILY_RESEARCH_SOURCES.md` §D."* **This empty state deliberately names a file to write, not
  a command to run, because printing a command that would fail is the same defect as printing a
  number nobody measured.**
- **deeper** → M2.4. **§13 interlock:** X4 rows inherit style-term locks 1–6 unchanged.

---

### X5 · SETTERS AND BREAKERS, THREE CLOCKS — *whiteboard insight #4, fused* *(GHOST, staged)*

- **mark** `slope` · **x** band, two ordered snapshot dates · **y** the term's share-of-attention
  index, **indexed to 100 at t₀** (`format:"ratio"` — V7 satisfied; raw values in the tooltip) ·
  **colour** ordinal **3** — `our catalogue` / `watched artists` / `market scan` · **facet**
  `{field:"style_term", columns:5, scales:"shared"}` · **feed** `suno` ⊕ `competitors` ⊕ `market`
- **Three independent unlock counters**, each lighting alone: `2 suno snapshots · 0 of 2` ·
  `2 competitor snapshots · 0 of 2` · `2 market snapshots · 0 of 2`. The card draws every clock it
  has and prints `0 of 3 clocks drawn`. **It does not wait for all three.**
- **I disagree with M2.5(c) and this is the disagreement.** M2.5 defines a setter as "a term whose
  effect rose while its usage rose" and a breaker as "effect rose while usage fell" — both computed
  inside our own catalogue. With one feed those two definitions describe the same object from two
  angles and neither can distinguish leading from following, which is the entire content of the words
  "setter" and "breaker." **Fused, they finally mean something:** a *setter* is a term whose share
  rose in the market scan **before** it rose in ours; a *breaker* is one whose share rose in ours
  while the market's fell. M2.5's single-feed definitions should be marked **provisional pending
  X5**, in M2.5, rather than left standing as if settled.
- **Defensibility:** lead/lag from **two** snapshots is an ordering, not an estimate — the card may
  label a term a setter and may **never** attach an interval or a magnitude to the lead. A lag
  *length* requires ≥5 snapshots per clock and the card says so on its face until it has them.
- **deeper** → M2.5.

---

### X6 · THE LOOP — observation, experiment, confirmation *(the Soundry reader; `emitter-silent` today)*

The card the whole estate is built to be able to draw, and the reader Part 2 never specified.

- **mark** `interval`, `intervalVariant: "dumbbell"`, `intervalStyle: "grain"` · **x** effect as a
  **lift ratio**, log axis, `format:"ratio"`, `refs: [{kind:"baseline", value:1.0, axis:"x",
  label:"no effect"}, {kind:"mde", value:<pre-registered>, axis:"x", label:"pre-registered MDE"}]`
  (V5 satisfied) · **y** band, one row per hypothesis present in BOTH apps, labelled with the
  Analytics card it came from (`M2.1`, `M2.2`, `M2.4`, …) · **xAField** `observational_lift` ·
  **xBField** `randomised_lift` · **colour** ordinal **3** — `randomised confirms` /
  `randomised contradicts (sign flip)` / `randomised cannot resolve (interval spans the MDE)`,
  the third at `--sig-null`; **Tide encodes sign, never valence, and no copy on this card labels a
  direction good or bad** · **facet** none · `nField: "n_effective"` = min(analytics cell, smallest
  arm) · **feed** `suno` ⋈ `soundry` on `analytics_card_id`
- **The dumbbell IS the finding: the connector length is what the confound was worth.**
- **THE RULE, printed verbatim:** *"The two ends of every row are never averaged and never differenced
  into a true value. The randomised end is the estimate. The observational end is what we believed
  before we tested. The connector is the price of the confound."*
- **Tier — the single documented exemption in this app, and its exact boundary.** The observational
  end is capped exactly as it always was. The randomised end may exceed `OBSERVATION` **only** when
  `controls.randomized === true` **and** `controls.blockedOn` contains `"day"` **and** the realised n
  clears the pre-registered n **and** `preregHash` verifies. Two tier chips render, one per end.
  **Causal verbs stay locked in Lofn Analytics even here:** the causal sentence renders as a
  *quotation* from the Soundry, with round id, prereg hash and a link — never generated by this app's
  templates. The footer promise ("Lofn Analytics observes. The Soundry tests.") survives literally,
  and her intent — that a tested claim be allowed to say what it found — survives too.
- **§13 interlock:** X6's randomised end is exempt from lock 6 (the `RUN_AN_EXPERIMENT` magnitude cap)
  and **from that lock only**. Locks 1–5 apply unchanged. With one declared factor k=1, so lock 2's
  ceiling does not bite by construction — no carve-out is needed and none is granted.
- **Multiplicity, on the card:** `header.live_hypotheses` from the snapshot prints in the card header
  (`vault/EXPERIMENT_STANDARD.md` §5 rule 4). A round that closed while 9 other hypotheses were live
  says so where the estimate is, not in a method tab.
- **Durations:** every "days to decision" on this card carries B3's caveat verbatim. No duration
  renders without it.
- **empty:** `emitter-silent`. Copy, verbatim: *"`data/snapshots/soundry/` exists and is empty. The
  Soundry has never closed a round. Rounds are exported by `node apps/soundry/emit-snapshot.mjs
  --round <round-id>`, and only after the round is closed with a frozen pre-registration — an open
  round is never exported. This is not a zero and it is not a failed pull."*
  `unlockAt: [{label:"closed rounds", have:0, need:1, …}, {label:"n toward the pre-registered
  target", have:0, need:<from prereg>, how:"12 songs a day; ~19 days at +15%, ~8 if plays mature —
  vault/EXPERIMENT_STANDARD.md §3", path:"data/soundry/rounds/<id>/preregistration.json"}]`
- **A second block below the main plot: "Tested, with no observational counterpart."** Rounds whose
  `analytics_card_id` is null still load and still render, as single grain whiskers. The loop closing
  in one direction only is still information, and dropping those rows would make the Soundry's work
  invisible in the app that is supposed to combine it.
- **deeper** → the Soundry ledger. **This link cannot be a literal**: `check-local-only.mjs` fails the
  build on any absolute `http(s)://` origin in `.html`/`.js`. Build it at runtime from
  `GET /analytics/config.json` (served by `server.mjs`, `{ soundryOrigin: string | null }`, default
  `null`). When `null`, `deeper` is still non-null — it points at the on-disk path
  `data/soundry/rounds/<round-id>/` and the label says so. V15 is satisfied without a dead link and
  without a hardcoded origin.

---------------------------------------------------------------------------------------------------
## SYN-2.6 · THE MISSING READER — `apps/lofn-analytics/src/load/soundry.ts`

ANCHOR: Part 2 §15, `src/load/` — add `soundry.ts`, `competitors.ts`, and the fixture.

Part 3 §3 specifies the emitter and Part 2 specifies no reader. Both sides of the contract, in one
place, with one fixture that neither may drift from.

**Envelope** (as emitted): `{ feed: "soundry", pulled_at, source, header, row_count, rows }`
`header: { rounds_closed: number, live_hypotheses: number, generator_version: string }`

**Row schema** — the reader's expectation, restated here so it is auditable from the Analytics side:

```ts
export interface SoundryRoundRow {
  round_id: string;                  // "R-2026-08-04-wpl"
  closed_at: string;                 // ISO-8601 with offset
  prereg_hash: string;               // sha256 of preregistration.json at freeze
  instrument: "suno" | "panel";      // the PRIMARY; the other is exploratory forever
  factor: string;                    // "words_per_line"
  levels: [string, string];
  analytics_card_id: string | null;  // "m2.1" — null is legal, renders in the second block
  readout_age_days: number;
  realised_age_days: [number, number];
  n_variants: number; n_eligible: number; n_effective: number;
  blocked_on: string[];              // ["day"]
  randomized: true;                  // literal
  lift: number; lift_lo: number; lift_hi: number;
  ci_level: 0.95;                    // literal
  mde_preregistered: number;
  method: "log-link-glm";
  exclusions: Array<{ variant_id: string; reason: string }>;
  live_hypotheses_at_close: number;
}
```

**Reader rules:**

1. Same `utf-8-sig` path, same discriminator (`feed === "soundry"` + `rows[]` + every row carrying
   `prereg_hash`).
2. **REFUSE, visibly, to the Schema Guard panel** — never silently — any row with
   `randomized !== true`, `ci_level !== CI_LEVEL`, a missing/empty `prereg_hash`, or
   `method !== "log-link-glm"`. A Soundry row without a frozen pre-registration is precisely the
   object this whole apparatus exists to refuse, and refusing it loudly in the reader is cheaper than
   refusing it in six cards.
3. **The reader never recomputes the GLM.** House rule, stated once and applied generally:
   **recompute what you own, quote what you don't, badge the quote.** Analytics owns `suno` and
   recomputes it live; it does not own a randomised round, so it quotes it with `provenance:
   "randomised-round"` plus `roundId` and `preregHash` rendered on the row (V22). A second estimator
   would drift from the first and the two would have to be reconciled forever.
4. `exclusions` render as a count in the row gutter and a `<details>` list. An excluded variant is
   never invisible.
5. Freshness uses the **weekly** cadence tier (8 d / 21 d / 60 d) — rounds close on a multi-day
   clock and the daily tier would mark every closed round STALE within 36 hours.

**The anti-drift mechanism, and it costs one file:** `apps/_fixtures/soundry-round.json`.
`apps/soundry/tests/emit-snapshot.spec.ts` asserts the emitter's output **equals** this fixture;
`apps/lofn-analytics/tests/soundry-reader.spec.ts` asserts the reader **parses** it and produces the
X6 row set. One file, two tests, opposite directions. Without it the two apps' schemas diverge on the
first change and nobody finds out until a card renders `undefined`.

**`competitors.ts` — the detail nobody has specified.** `scripts/ingest_suno.py:264-266` writes
competitor snapshots to `data/snapshots/competitors/<handle>/`, **one directory per handle**. The
reader therefore globs `data/snapshots/competitors/*/latest.json` — there is no
`data/snapshots/competitors/latest.json` and a reader written against the single-file pattern of every
other feed will find nothing and render a false `never-pulled`. `server.mjs`'s snapshot route needs
the matching two-segment case: `GET /analytics/snapshots/competitors/<handle>/latest.json`.
Rows carry the **identical** suno schema, so `competitors.ts` reuses `suno.ts`'s row parser
unchanged — asserted by `tests/competitors-reader.spec.ts` importing the same parser.

---------------------------------------------------------------------------------------------------
## SYN-2.7 · `data/watchlist.yaml` — the file that lets the competitor empty state be cleared

ANCHOR: Part 2 §14.4 and the §15 repo preconditions.

The five watched targets are named in no file in this repo, so the blocker list can never be cleared
from inside the app. One file, written by The Scientist, never written by the app:

```yaml
# The watched set — feed 2 of the whiteboard's four. The apps READ this file and never write it.
# Adding a handle here makes its Refresh button exist; it does not pull anything.
watched:
  - handle: <suno handle>
    platform: suno
    why: <one line — what makes this target worth watching>
    added: <YYYY-MM-DD>
```

- **Absent or empty ⇒** X2 row 2 / X3 / X4 / X5 clock 2 render `never-pulled` with **two counters**:
  `0 of 5 handles named` (blocker: The Scientist — no command exists for this and the card says so)
  and `0 of 5 pulled` (blocker: one command each).
- **Non-empty ⇒** the §14.4 overrule takes effect: a per-handle Refresh appears, stating its
  consequence exactly as the Suno button does — *"Pull @<handle> — public HTTP requests to
  `studio-api.prod.suno.com`. No credentials sent. Writes one file under
  `data/snapshots/competitors/<handle>/`. Does not publish, render, or spend."* The intent §14.4
  protects is preserved: no button exists that cannot work.
- Add `WATCHLIST → data/watchlist.yaml`, `SNAPSHOTS_SOUNDRY → data/snapshots/soundry/` and
  `FIXTURES → apps/_fixtures/` to the CANONICAL PATHS additions in §8.3.4 (and mirror to `.agents/`;
  `python3 scripts/check_skill_mirror.py` must pass).
- `data/watchlist.yaml` holds public handles and one line of rationale each. It is not PII and it is
  not a dossier — `vault/COLLABORATOR_STANDARD.md`'s "no PII in this repo" is satisfied by keeping it
  to handle + platform + why + date, and the schema above is the enforcement.

---------------------------------------------------------------------------------------------------
## SYN-2.8 · Part 2 §12 and §15 — table extension, files, tests

**§12 disposition table — add a fifth column, "Fused surface":**

| # | Her insight | Single-feed card | **Fused surface** | Unlock |
|---|---|---|---|---|
| 1 | lyric length / line | M2.1 | **X2** — three instruments on one pp axis, never pooled | 1 competitor pull · 1 closed round |
| 2 | Suno EMO usage | M2.3 a/b/c | **X3** — vocabulary SHAPE across artists; per-tag ranking refused across artists exactly as it is refused within | 1 competitor pull per series |
| 3 | genre performance | M2.4 + §13 | **X4** — positioning census, no third channel, no fitted line | market scanner must be written |
| 4 | trend setters + breakers | M2.5 + M3.1 | **X5** — three clocks; the lead/lag that makes the two words mean different things | 2 snapshots per clock, independent |
| — | *(the loop she implied but did not draw)* | — | **X6** — observational belief against randomised estimate, as a dumbbell | 1 closed round |
| — | *(the join's own limits)* | — | **X0 / X0b** — LIVE today | none |
| — | *(what ever left Suno)* | — | **X1** — LIVE today | none |

**§15 file tree additions:**

```
apps/lofn-kit/src/contracts/ghost.ts              # §SYN-2.3
apps/lofn-kit/src/stats/ols.ts                    # olsSlope, needed by X2 row 1
apps/_fixtures/soundry-round.json                 # the anti-drift fixture, shared by both apps
apps/lofn-analytics/src/load/soundry.ts           # §SYN-2.6
apps/lofn-analytics/src/load/competitors.ts       # per-handle glob
apps/lofn-analytics/src/cards/m6-x0-join-matrix/
apps/lofn-analytics/src/cards/m6-x1-distributed/
apps/lofn-analytics/src/cards/m6-x2-lyric-three-instruments/
apps/lofn-analytics/src/cards/m6-x3-emo-vocabulary/
apps/lofn-analytics/src/cards/m6-x4-market-position/
apps/lofn-analytics/src/cards/m6-x5-three-clocks/
apps/lofn-analytics/src/cards/m6-x6-the-loop/
data/watchlist.yaml                               # The Scientist writes it; the app only reads it
```

**Test contract additions:**

10. `join-evidence.spec.ts` — V19–V22. Every card whose `evidence.source` spans two snapshot
    directories declares `join`; `keyRule` renders; `resolved/attempted` renders as a fraction.
11. `ghost-autoenable.spec.ts` — one spec, two fixtures (empty / populated), two outputs, and the
    card module contains no conditional around its `plot()` call.
12. `soundry-reader.spec.ts` — parses `apps/_fixtures/soundry-round.json`; refuses a mutated copy
    with `randomized: false`, with `ci_level: 0.90`, and with `prereg_hash: ""`, and asserts each
    refusal reaches the Schema Guard panel rather than being dropped.
13. `competitors-reader.spec.ts` — globs `data/snapshots/competitors/*/latest.json`; asserts the
    single-file pattern is NOT used; asserts an absent directory yields `never-pulled` with two
    distinct counters, and that a present-but-empty directory yields `emitter-silent` — never the
    same state.
14. `fusion-refusals.spec.ts` — X4 emits no fitted line, no size channel and no correlation
    coefficient at any fixture size; X3 emits no per-tag row at any artist count; X1's tier is
    `OBSERVATION` for every fixture; X6's observational end never exceeds `OBSERVATION`.
15. Extend `forbidden-strings.spec.ts` with `"combined reach"`, `"vs the market"`, `"outperforms"`,
    `"beats"` — the four phrases a fusion tab invents on its own.

**Build order** (the fusion tab is not a big-bang):
1. §SYN-2.1 types + §SYN-2.2 empty kinds + §SYN-2.3 ghost contract (kit; coordinator-only files).
2. **X0, X0b, X1 — live today, no new feed.** They prove the join machinery against real files before
   any ghost is drawn, and X0 alone changes what she believes about the other five cards.
3. §SYN-2.6 readers + the shared fixture.
4. X2, X3, X5 ghosts (competitor-shaped) → then the single competitor pull lights four of them.
5. X6 ghost. 6. X4 ghost.

RISKS:
- X6 depends on a Provenance enum that Critic 2 records as specified three incompatible ways. If a unifying seat collapses 'randomised-round' into 'derived', the only provenance in the estate that licenses a causal verb disappears and V22 becomes unenforceable. Declared as a dependency, not assumed.
- X0/X0b/X1 use the socials and distribution feeds, which Critic 2 correctly reports are NOT in KNOWN_FEEDS (scripts/ingest_receiver.mjs:43-52). Until the FeedId membership question is settled by whichever seat owns it, JoinEvidence.left.feed cannot be typed for two of the three live joins. My cards are blocked behind that fix.
- The X0 yields (4 of 9 releases_seen, 2 of 10 Apple top_tracks) are a property of the stated normaliser (NFKD, strip feat, alnum-only). A different normaliser gives different numbers. The card renders the rule for exactly this reason, but a build agent who changes the normaliser and not the caption reintroduces the defect.
- Adding a seventh Music tab lengthens the rail and competes with the four-tab reading she drew. Extrapolating: I judged a named fusion surface worth the extra tab because the alternative is scattering fused cards across M2/M3, where they would be read as more single-feed cards. If a seat disagrees, the fallback is M6 as a section at the foot of M2 CRAFT with the same card ids.
- X2 needs olsSlope, which stats.ts does not have; X0's heat posture needs a sequential-ramp path that the strip mark currently only documents for 'heat' in one line. Both are small, but both are new render paths and neither is covered by an existing test.
- The X6 tier exemption is the only place in Lofn Analytics where a row may exceed OBSERVATION. It is fenced by four conditions and a hash check, but it is still a hole in a wall that is otherwise solid, and it will be the first thing a future card tries to widen.
- data/watchlist.yaml records handles of real people's accounts alongside a 'why' line. Kept to handle/platform/why/date it is not PII, but the schema is the only thing preventing it becoming a dossier, and nothing enforces the schema today.