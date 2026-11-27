# Changelog

## November 2026

- **Music:** Suno v6 vocal packages target 370–430 sung words. The lyrics-box
  count includes cues and headers; the detailed channel strip goes in a
  separate production sidecar. Run-local legacy line gates remain supported.
  The daily and Step 11 routes use the same contract. See
  [Suno v6 packaging](docs/SUNO_V6_PACKAGING.md).
- **Validation:** Added regression coverage for word counts, variant counts,
  shared context blocks, and output packaging. Instrumental packages do not
  inherit vocal-lyric checks.
- **Image and video:** Check contest eligibility before rendering images; use
  composition and lighting rather than an extra style to strengthen a thumbnail.
  Video plans specify the renderer early and name what each reference controls.
- **Orchestration:** Panel artifacts preserve proposals, meaningful dissent,
  and decisions in concise notes across the creative lanes.
- **Setup:** Active workflow paths resolve from the checkout rather than the
  original OpenClaw workspace. See [the workflow guide](WORKFLOW.md).

- **Optional tools:** Updated the Explorer web dependencies and PDF Augmenter's Pillow requirement. CI verifies the Explorer production build and PDF tests.

Private runs, unpublished media, deployment settings, and account details are
outside this update.
