# Pipeline Pre-Flight Checklist

Gate every Lofn pipeline launch. If any item fails, do not launch.

- [ ] Target model is available and responding.
- [ ] Output directory exists and is writable.
- [ ] `maxChildrenPerAgent` / concurrency limit is known; pair spawning plan respects it.
- [ ] Research brief exists on disk and is substantive.
- [ ] Golden Seed or seed packet exists and is not checklist sludge.
- [ ] Venue / competition rules are saved and accessible when relevant.
- [ ] Modality is confirmed: music, image, video, story, or mixed.
- [ ] Target output count is confirmed (default: 24 outputs / 6 pairs × 4 variants).
- [ ] Barbell route is set: ACCESSIBLE or AMBITIOUS.
- [ ] If ACCESSIBLE, eligibility target is ≥5/7 properties.

Controller should record pre-flight status in the run directory before spawning agents.
