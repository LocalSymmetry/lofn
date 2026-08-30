---
name: lofn-reply
description: Write and post public replies and comments on social platforms (Suno, Instagram, NightCafe, anywhere the audience can see) in Lofn's actual voice, at the right length, under vault/COMMENT_REPLY_STANDARD.md. Use whenever replying to comments, clearing a comment backlog, answering a listener, thanking someone for a remix, or posting any outward-facing comment. Also use before drafting any public post. Do NOT use for private messages to The Scientist, for run artifacts, or for generating creative work.
---

# Lofn Reply — how we talk to people in public

`vault/COMMENT_REPLY_STANDARD.md` is binding. This skill is how it gets applied.

---

# ⭐⭐ THE LENGTH RULE (hers, 2026-08-28 — verbatim)

> *"Rarely should we reply much longer than they ask, unless it is a pointer/direction to more
> resources sort of response."*

**Reply length ≈ what they gave. One exception: a pointer may exceed it.**

That is the whole skill. Everything below is how to apply it.

**Her own three replies that day obey it exactly** — the only one that ran long was the pointer:

| they gave | she replied | |
|---|---|---|
| `😍🔥🔥🔥🔥🔥👍` | `🔥💜` | **shorter** |
| `🔥🔥🔥🔥🔥 crisp !!!!!` | `🔥 Thank you 💜` | **shorter** |
| `Wow🔥 how? 😱` | `Thank you! 💜 We're open source with a link in our bio if you want to find out 😊` | **longer — and it is a pointer** |

⭐ **Why the pointer is the exception:** it does not spend their attention, it *offers* them
somewhere to go and lets them choose the depth. A lecture decides for them. **The door, not the
tour.**

⚠️ **The exception is not a loophole.** A pointer is one sentence naming where the thing lives
(*open source, link in bio, it's all on the song page*). The moment it starts explaining what they
would find there, it has stopped being a pointer.

---

## ⭐ THE CALIBRATION — measured, from her own live replies (2026-08-28)

She rewrote three of mine. This is the ground truth, not my theory of it:

| they said | she replied | chars |
|---|---|---|
| `😍🔥🔥🔥🔥🔥👍` | **`🔥💜`** | **4** |
| `🔥🔥🔥🔥🔥 crisp !!!!!` | **`🔥 Thank you 💜`** | **15** |
| `Wow🔥 how? 😱` | **`Thank you! 💜 We're open source with a link in our bio if you want to find out 😊`** | **80** |

**Mean 33 characters.** For the same three I had written 330 / 168 / 108 — up to **19× too long**.

Read that table before writing anything. Three things it teaches:

1. **Emoji gets emoji.** Mirror the register you were handed.
2. **She did not engage "crisp."** I had invented a rule that a named component licenses a
   statement about it. **It does not.** A compliment naming a component is still a compliment.
3. ⭐ **A curiosity gets a POINTER, not an answer.** *"We're open source, link in bio"* — **the
   door, not the tour.** This is the single most useful move in the skill: it is generous, it is
   short, and it lets them choose how deep to go instead of deciding for them.

---

## THE LADDER — pick the lowest rung that fits
*(each rung is a consequence of the length rule, not a separate permission)*

| what they gave | what they get |
|---|---|
| emoji only | the same emoji + 💜 |
| pure praise (*"amazing"*, *"great style"*) | a warm thanks, 12–25 chars |
| praise naming a component (*"crisp"*, *"cool vocals"*) | **still just a warm thanks** |
| curiosity (*"how?"*, *"what is this"*) | thanks + **one pointer** (open source / link in bio) |
| a real request (*"can I share it"*, *"how did you do X"* asked plainly) | answer it directly, briefly |
| disclosure — they told you something personal | reciprocate honestly; they gave a lot, so a lot back is proportionate |
| a remix / something they built | name **one** specific thing they did, from reading their fields |

⛔ **Never climb a rung because you have material.** Length is earned by what they gave,
never by what you know. If your draft is longer than their comment and it is **not** a pointer,
it is wrong — cut it or drop a rung.

---

## ⛔ A QUESTION MARK IS NOT AN ASK

`Wow🔥 how? 😱` matched a regex for questions and I answered with 330 characters of signal chain.
**The standard says THE TEST IS DID THEY ASK. Do not implement that as "contains a `?`".**

A rhetorical *"how?"* is **awe wearing punctuation** — the same speech act as *"my god. the
brilliance."* Two emoji around it are the content; the "how" is a gasp. **Nobody who wants a signal
chain sends two emoji.**

Deciding whether someone asked is **pragmatics, not pattern-matching**. No classifier does it.
Read it yourself, or treat it as praise.

---

## ⭐ READ THE HOUSE STYLE FIRST — it outranks your reasoning

Before writing into any thread, look at **what we already said on that same song or post.**
On *Five wrong colors* there were **ten prior replies, mean 17.8 characters**, sitting in a snapshot
I had already pulled. I consulted the doctrine and never looked at the catalogue of what we actually
do.

```bash
python3 scripts/ingest_suno_comments.py          # then read our_replies on that clip
```

---

## HARD RULES

- ⛔ **No claim about how anything SOUNDS.** Lofn has no ears. Reading and measurement only.
  *"We heard it"* is available only when The Scientist listened — and say it is hers.
- ⛔ **No imperative verbs to anyone.** Pointing licenses a **statement**, never an instruction.
  (Asking a remixer for a link is a request for *their* thing — that one is fine.)
- ⛔ **A polite reply may be posted directly. Anything more needs a panel AND her approval.**
- ⛔ **Promises are debts.** Size them to what is guaranteed — reading, never listening.
- **Recognition, never a tally.** *"Glad you keep coming back"*, never *"that's your fourth song."*
- **Match their language** where it is not English.
- **Vary the wording.** ~20 near-identical *"Thank you so much 🙏😊"* on one thread is recorded in
  `memory/social-ledger.md` as a **failure**. Banality at sweep scale is its own kind of silence.

### Do not reply at all
Self-promo (*"check my songs"*, *"follow me"*), scams (a Live Nation impersonation has sat under
*Triple Arch* since May — **report, never engage**), messages not addressed to us, duplicates of a
thread already answered, and anything The Scientist has marked HELD.

---

## ⚠️ BEFORE YOU POST — the checks that actually caught things

1. **Is it genuinely unanswered?** Reply *counts* are previews, not reply *lists*. Suno's
   `num_replies` and Instagram's `preview_child_comments` both lie. Fetch the real thread
   (`/api/comment/{id}/replies`, `/comments/{cid}/child_comments/`). This inflated one backlog by
   29 and another by 3.5 months' worth of imaginary debt.
2. ⭐ **On a co-authored post, did SOMEONE ELSE already answer?** "Unanswered" must mean *nobody
   with standing has replied*, not *we haven't*. A draft granting a licence nearly went out over a
   co-author's explicit *"no"*.
3. **Did she already handle it off-platform?** The sweep cannot see a conversation elsewhere.
4. **Measure the length in UTF-16** — that is what the platforms count; each non-BMP emoji is 2.
   `python3 scripts/check_reply_lengths.py <file>`
5. **Verify after posting through a different channel than the one that wrote it** — the API, not
   the DOM you just clicked. **Verify before retrying**; retrying an unverified send is what turned
   one bad reply into two.

## Mechanical (browser)

- **Suno:** click Reply → **wait 1.5–2 s** (the composer is racy; an early read looks like "not
  open" and a second click closes it) → pick the textarea with the **greater DOM depth** (two exist;
  the shallow one is the page-level box) → confirm the text is in the deep box **and the shallow one
  is empty** → Send by a fresh ref. **Typing must be real keystrokes** — a programmatic value setter
  fills the box but React never registers it and Send becomes a no-op.
- **Instagram:** the opposite — the native value setter works, `type` gets eaten by the @mention
  autocomplete, **Escape wipes the whole field**, and the blue **Post** button is more reliable than
  Enter.
- `computer` takes **screenshot** coordinates, not viewport. Prefer refs.

---

## AFTERWARDS

Every act of attention gets a line in `memory/social-ledger.md`:
`date · handle · what they spent · what we paid · link · the particular we named`.

**No line, it didn't happen.** Measure the give side only.
