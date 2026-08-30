# Pipeline Pre-Flight Checklist

Gate every Lofn pipeline launch. If any item fails, do not launch.

- [ ] Target model is available and responding.
- [ ] Output directory exists and is writable.
- [ ] `maxChildrenPerAgent` / concurrency limit is known; pair spawning plan respects it.
- [ ] Canonical daily research exists and is substantive: `output/daily/YYYY-MM-DD/research/DAILY_RESEARCH_BRIEF.md`.
- [ ] `output/daily/YYYY-MM-DD/research/RESEARCH_STATUS.json` exists with `status: complete`.
- [ ] If canonical research is missing/stale/stubbed, creative creation is paused and the daily research job is launched first.
- [ ] Targeted creation search addendum exists: `00_targeted_search_addendum.md`.
- [ ] Targeted search trigger is identified: competition/venue, user request, or Lofn preference.
- [ ] Targeted addendum states how the Golden Seed should change and what move is banned.
- [ ] Golden Seed or seed packet exists and is not checklist sludge.
- [ ] Venue / competition rules are saved and accessible when relevant.
- [ ] Modality is confirmed: music, image, video, story, or mixed.
- [ ] Target output count is confirmed (default: 24 outputs / 6 pairs × 4 variants).
- [ ] Barbell route is set: ACCESSIBLE or AMBITIOUS.
- [ ] If ACCESSIBLE, eligibility target is ≥5/7 properties.

Controller should record pre-flight status in the run directory before spawning agents.
