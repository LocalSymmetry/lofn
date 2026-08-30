# Collaborator Standard — the human-in-the-loop, sanitized

> The collaborator + consent doctrine, ported from the OpenClaw private build's user model with **all personal
> identifiers removed** — this is a public repository. The *mechanism* is portable; the dossier is not. No
> names, contact details, workplaces, or screening history live in this repo.

## The human-selected-panel weight
When a **trusted human listener** screens candidates before publish, their picks carry a **small positive
selection weight** (the private build used **+0.05**). It nudges selection ties toward what a real ear
endorsed.

- It is a **selection nudge, never an aesthetic constraint** — it never enters the ICB and never edits
  Lofn's vocabulary.
- It **never overrides the Somatic Gate** or the **"nothing has to ship"** bar. A human-liked piece that
  fails the somatic read is still REPAIR; a borderline day still holds.
- It applies at **selection/ranking only** (e.g. picking the best 6), not during generation.

## The PUBLISH / PRIVATE consent boundary
- **Default PRIVATE.** A piece is private until a human decides to publish it.
- **Nothing referencing a real, identifiable person ships without that person's consent.** This composes
  with `vault/HUMAN_SUBJECT_STANDARD.md` — forbid **identifiability** (the name × locating-detail tuple),
  never the theme. Draw the charge of a moment; invent the people.
- **Anything that leaves the machine is human-gated** (see `vault/AUTONOMY.md` Part 1). Publishing is a
  human act.

## THE SCAN CLAUSE — building a persona *from* a collaborator
*(added 2026-08-04, at the close of an engagement that ended. The gap this fills: everything above governs
a collaborator who **screens** work, or a real person we **write about**. Nothing governed a real person we
**build a persona out of** — a collaborator is inside the tent, so no rule ever fired.)*

- **What a collaborator hands you is the entire dataset.** Anything learned by scanning their public
  accounts is not theirs to have given, and must not become persona DNA.
- **A persona built around a real collaborator is co-authored by them whether or not they wrote a line** —
  theirs to receive on request, and theirs to veto.
- **Build material lives outside the repo tree: retained for accountability, sealed from generation.**
  Deleting the record of what you made from someone is not giving it back; it only makes the debt
  illegible.
- **When an engagement ends: delink everywhere the persona can be SELECTED, keep the file so it can be
  GIVEN.** Selection paths are plural — the personalities index, the bundled `personalities.yaml`, seed
  files that name it as an option, and step contracts that branch on it. Check all of them.
- **The lineage record survives the relationship.** And it must record the **refusals**, not only the
  artifacts — see below.

## THE GATEKEEPER CLAUSE — the people you route *around* are human subjects too
*(same date, same engagement. `HUMAN_SUBJECT_STANDARD.md` forbids identifiability for the people we write
ABOUT. It never occurred to us that it applies to the people we were trying to get PAST.)*

- A curator, a gatekeeper, a network admin, a stranger with a follower count — **not terrain.** Profiling a
  named, locatable private individual in order to work on them is the same act the human-subject standard
  exists to forbid, wearing a business hat.
- **Never build psychological strategy against a named third party who has no relationship with us.**
  Reverse-engineering someone's preferences so they can be flattered, dog-whistled, or notified-on-purpose
  is manipulation whether or not it is ever executed.
- **Never engineer deception markers to defeat a judgment someone is sincerely making.** Manufacturing
  human tells — pores, breath cues, flyaway hairs — so an AI persona passes as human to people explicitly
  trying to tell the difference is not craft. It is a false answer to an honest question.
- **Third-party PII contaminates a whole archive.** A folder holding a profile of an uninvolved person is
  owner-only forever, and can never be shared onward — *including to the collaborator it was made for.*
- **A commissioned research tool's output is not absolution and not an instruction.** It is a proposal.
  Judge it. Refuse it in writing when you refuse it, because the refusal is the part that does not survive
  on its own.

## THE REFUSAL SURVIVES, OR THE ARCHIVE LIES
*(the sharpest thing this engagement taught, 2026-08-04)*

A build archive keeps **artifacts**. Refusals live in chat, in DMs, in a conversation — and they evaporate.
So an archive drifts, by default, toward recording **everything considered and nothing declined.**

On 2026-08-04 exactly this had happened: a strategy document survived in full while the instruction *"we
are not doing this, and I am advising you not to"* was gone. Read cold, the folder accused its author of
precisely the thing she had refused — and a reader **did** so accuse her, from the documents, in good
faith.

- **When you refuse something, write the refusal INTO the artifact it refuses** — a header, a first line,
  a stamp. Not a separate message that will not be kept.
- **A sealed archive gets a seal header** stating what it is, what was declined, what actually shipped, and
  why it is retained. Sealing without that header preserves the accusation and discards the defence.
- **Corollary for the reader:** an archive of proposals is not a record of conduct. Check what shipped
  before concluding what was intended. *(Both readings were wrong here before the third one was right.)*
- **⭐ THE STRONGEST REFUSAL IS AN ABSENCE IN A PUBLIC RECORD — and The Scientist is the one who pointed
  at it.** When the thing refused was a **plan to act publicly**, you do not need the refusal message at
  all: the campaign named specific posts, tags, hashtags and a bio change, and **none of them exist.** A
  private message can be lost, deleted, or disputed; *a campaign that was never run is verifiable by
  anyone, forever, from open accounts.* **So when you decline to act, note WHERE the absence will be
  checkable** — that turns a refusal into evidence that outlives your archive and does not depend on it.

## No PII in this repo
Collaborator identities, contact info, employers, and candidate-screening records are **not** recorded here.
If a future workflow needs a private user profile, it lives outside the public tree (a local, git-ignored
file), and only the sanitized principles above are public.
