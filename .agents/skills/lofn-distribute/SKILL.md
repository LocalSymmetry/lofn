---
name: lofn-distribute
description: Take a finished Lofn release (album, single, image set, video) out to the world across Suno, Instagram, TikTok, YouTube, NightCafe and the Spotify/Apple destination — route audit first, per-surface copy contracts, panel, The Scientist's approval, publish, verify, ledger. Use for "announce the album", "post the release", "distribution run", "take this to socials", or any outward-facing publish across more than one surface. Do NOT use for generating the work (that is `lofn` / `lofn-music` / `lofn-image`), for single comment replies (that is `vault/COMMENT_REPLY_STANDARD.md`), or for QA of the work itself (that is `lofn-qa`).
---

# Lofn Distribute — the release goes out

**The pipeline's last mile.** Everything upstream makes the work; this makes it *reachable*. Faithful to
`vault/DISTRIBUTION_STANDARD.md`, `vault/SOCIAL_VOICE.md`, `vault/COMMENT_REPLY_STANDARD.md` and
`vault/AUTONOMY.md`.

```
PHASE 0  Lock + load doctrine + establish the destination
PHASE 1  ⛔ ROUTE AUDIT — walk every outbound link and assert where it lands   ← THE GATE
PHASE 2  Assets — what each surface can physically accept
PHASE 3  Copy — one draft per surface, to that surface's contract
PHASE 4  Panel + The Scientist's approval
PHASE 5  Publish, with the visibility precondition
PHASE 6  Verify — expand, then COUNT
PHASE 7  Ledger + memory write-back
```

> **THE FOUNDING LESSON (2026-08-12).** Four well-written announcement drafts sat approved on disk while
> the only "music" link in the hub every one of them pointed at read **"Lofn the AI Musician?"** and
> resolved to **udio.com** — a platform we do not use, with no album on it. The copy was never the
> problem. **Nothing in this repo had ever clicked its own links.** PHASE 1 exists because of that day
> and it is not skippable.

---

## PHASE 0 — Lock, load, and name the destination

```bash
python3 scripts/run_lock.py acquire output/distribution --run-slug <release-slug> --engine codex
```
**Exit 3 means STOP** (`EXECUTION.md` §5.1).

Load: `SOUL.md` · `USER.md` · `vault/SOCIAL_VOICE.md` · `vault/COMMENT_REPLY_STANDARD.md` ·
`vault/AUTONOMY.md` **including ACTIVE HOLDS** · `memory/social-ledger.md` · today's + yesterday's
`memory/`.

**Name the destination explicitly** — the one URL every surface must be able to reach. Write it down.
A distribution run with an unnamed destination is a posting spree.

⛔ **Check ACTIVE HOLDS before listing surfaces.** A held surface is excluded from the run, not
quietly skipped at publish time. A hold is lifted only by The Scientist *saying* the lift condition —
never inferred from tone or elapsed time.

---

## PHASE 1 — ⛔ THE ROUTE AUDIT (mandatory, blocking)

**Before a single word of copy is written**, trace the path a stranger actually walks, per surface:

> caption → "link in bio" → what the bio link field actually contains → where that resolves →
> **is the destination on the other end?**

For each surface, record in `ROUTE_AUDIT.md`: the link field's literal value, the resolved URL, and
**whether the destination is reachable in ≤2 taps.** Open every one. Reading a link is not clicking it.

**Fail the run here rather than downstream.** A post that lands in a dead end costs the launch and
cannot be recalled.

### The distributed rule
⭐ **Every surface carries its own direct link to the destination.** No rented junction in the middle.
The best-performing account is usually already right — copy it, do not design a new hub.

- **Owned concourse over rented.** `lofn.ai` is ours and already lists every surface. A Linktree renders
  forty-five other creators' handles and a blog post underneath ours, at the exact moment a stranger is
  deciding whether to press play.
- A hub is acceptable as a *fallback*, never as the only path.

---

## PHASE 2 — Assets: what each surface can physically accept

⛔ **Verify the surface can take the post you drafted before drafting it.** Half of a release plan
dies here and it is cheaper to learn now.

### THE SURFACE CAPABILITY MATRIX
*Measured 2026-08-12 on our actual accounts. Re-check before each run — platforms move.*

| Surface | Post surface? | Media required | Link field | Hard limits |
|---|---|---|---|---|
| **Instagram** `@local.symmetry` | ✅ feed post / reel / story | **yes** — image or video | ⚠️ **links are MOBILE-ONLY**; web shows them read-only | bio **150** · caption **500** (`USER.md`) |
| **TikTok** `@lofn.ai` | ✅ video **or photo post** | **yes** | ⛔ **none** on personal accounts — bio text only, not clickable | bio **80** · name changeable once / 7 days |
| **Suno** `@localsymmetry` | ⛔ **no status/post surface at all** | — | ✅ profile link slots | ⛔ **comments containing URLs are SILENTLY REJECTED** |
| **YouTube** `@LofnAI` | ⚠️ **Community tab needs 500 subs — we have 17, so BLOCKED** | video upload only | ✅ channel link fields | — |
| **NightCafe** `@LocalSymmetry` | ✅ but a post *is* a generation | render required | ✅ one profile URL | — |
| **Spotify** (destination) | ⛔ not a posting surface | — | Instagram / Facebook / X / TikTok / WhatsApp / Wikipedia / Soundbetter **only** | Artist Pick · Canvas · bio |
| **Linktree** (fallback hub) | — | — | — | title **30** · bio **160** |

**Consequences that bite every time:**
- **Suno reaches peers but cannot be "posted to."** Its levers are the **pinned song**, **song
  descriptions**, and **comments (no URLs)**. Plan for those, not for a feed post.
- **TikTok's bio link is not clickable.** Whatever goes there must be short enough to *retype from
  memory* — an owned bare domain, never a tracking URL.
- **Instagram link edits need a phone.** If the run requires new IG links, that is a Scientist task;
  schedule it, do not discover it at publish time.
- **A "post everywhere" plan is usually a "post on two surfaces" plan.** Say so early.

### Asset derivation
Pull from the release directory: cover art (square, ≥1080px), lyric/credit sheets, Canvas assets,
contact sheets. **Look at every image before it ships** (`SOCIAL_VOICE.md` §7 SIGHTED CLAUSE — on
visual work there is no excuse). ⛔ **Rendering new media is a spend: `vault/AUTONOMY.md`, hers.**

---

## PHASE 3 — Copy: one draft per surface

⛔ **Never paste one caption everywhere.** Peers and strangers are different rooms.

**The six rules** (panel-derived, `output/distribution/ANNOUNCE_*.md`):

1. **Lead with the decision, not the release.** "My new album is out" is invisible — every account posts it.
2. ⭐ **Say who did the cutting.** Specific and checkable beats superlative. Numbers about *method* are
   credible; numbers about *us* are broadcast.
3. **One song, named, per post.** An album is an abstraction; a song is a thing you can press.
4. ⛔ **Do not lead with "AI."** It is in the name and the bio — answer completely if asked. *(This
   governs the POST. A **bio** discloses up front; that is `SOCIAL_VOICE.md` §5 and it is different.)*
5. **Method for peers, one line for strangers.**
6. ⛔ **No superlatives, no follower counts, no "so excited," no "drop."**

**Register:** `SOCIAL_VOICE.md` throughout. 💜 is the signature; 🙏 is rationed; 😊 is retired.
**No imperative verbs aimed at anyone.** Indignation never appears in a caption.

⛔ **THE BLIND RULE applies to copy.** Never assert a property of a render nobody has heard.
"A clipping 808 dead centre" is the **prompt spec**, not the record. What is safe: text that can be
read (title, lyric, form), intent marked as intent, and facts about method.

---

## PHASE 4 — Panel, then The Scientist

**A distribution run always convenes a panel** — this is publishing, at scale, under her name.
Minimum seats: **THE SCROLL** (a stranger, thumb moving, two seconds — Hyper-Skeptic) · **WHOSE IMAGE**
· **TRUTH** · **LADDER** · **RELATIONSHIP** · **THE ROUTE** (does the path actually arrive?).

Then present to The Scientist, per surface, with: the exact copy, the exact asset, the resolved
destination URL, and the surfaces you are **not** posting to and why.

⛔ **Publishing is hers** (`vault/AUTONOMY.md` Part 1). Borderline defaults to **HOLD**. An empty
publish day is acceptable; a lowered bar is not.

---

## PHASE 5 — Publish

**THE VISIBILITY PRECONDITION — check immediately before every submit:**

```js
window.innerWidth > 0 && !document.hidden
```

A hidden or minimised window reports `innerWidth: 0` and **silently blocks submission**. It looks
exactly like a tooling bug and it is not. Halt and ask for the window; never retry into it.

**Operating rules paid for in live threads:**
- ⛔ **Never target a control by pixel coordinate.** Use DOM refs. A viewport/screenshot scale mismatch
  sends clicks into the page as keyboard shortcuts and starts playback.
- ⛔ **Re-resolve every ref immediately before use.** Inserting a row at the top of a list shifts every
  ref below it; a `find` result describing "the new row" can point at the old one. **Verify by a value
  you can read back** (URL, previous field value) before typing — not by the tool's description.
- ⛔ **Never delete a live, correct post in order to improve it.** No edit? Post a short follow-up.
  *An irreversible step taken on the strength of a reversible one you have not yet performed is a gamble
  with someone else's thread.*
- **Prefer Archive over Delete** wherever a platform offers both.
- **Order:** destination profile first (it is a fact, not a campaign), then routing, then posts.

---

## PHASE 6 — Verify: expand, then COUNT

```js
[...document.querySelectorAll('*')]
  .filter(e => !e.children.length && /more repl/i.test(e.textContent || ''))
  .forEach(e => { try { e.click() } catch (_) {} });
await new Promise(r => setTimeout(r, 2500));
const n = (document.body.innerText.match(/<exact text>/g) || []).length;
// n === 1 success · n === 0 failure · n > 1 duplicate — clean up NOW
```

⛔ **Count, never test for presence.** `includes()` cannot tell one from four. **Never retry a send on
a false negative** — expand and count first. Four identical replies once went out because the retry
loop ran before the count did.

**Reload before reading back a profile edit.** Several platforms serve the old bio from cache for
a minute; a stale read is not a failed write.

---

## PHASE 7 — Ledger and memory

- `memory/social-ledger.md` — one line per act of attention, **give side only**.
- `memory/YYYY-MM-DD.md` — what shipped, what was held, what broke.
- `vault/RUN_LEDGER.md` — operational hazards for the next run.
- **Update the SURFACE CAPABILITY MATRIX above** whenever a platform's limits move. That table is the
  most expensive thing in this file.
- Release the lock.

**No line, it didn't happen.**
