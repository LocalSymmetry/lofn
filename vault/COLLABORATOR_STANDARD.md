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

## No PII in this repo
Collaborator identities, contact info, employers, and candidate-screening records are **not** recorded here.
If a future workflow needs a private user profile, it lives outside the public tree (a local, git-ignored
file), and only the sanitized principles above are public.
