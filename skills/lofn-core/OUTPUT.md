# LOFN OUTPUT PROTOCOL — Individual Artifact Saving

## ⚠️ MANDATORY FOR ALL CREATIVE OUTPUT

Every creative artifact (song, image prompt, story, video concept) MUST be saved as its own individual file with full metadata. Never batch multiple artifacts into one file. Never try to output everything at once.

This mirrors the original Lofn codebase's `save_metadata()` pattern.

---

## 📁 Directory Structure

```
output/
├── songs/       # Song prompts and lyrics
├── images/      # Image generation prompts
├── videos/      # Video/animation concepts
└── stories/     # Narrative pieces
```

All paths relative to workspace: `/root/.openclaw/workspace/output/`

---

## 🎵 Song File Format

**Filename:** `{YYYYMMDD}_{HHMMSS}_{title_slug}_{pair}_{variation}.md`

Example: `20260323_031500_violet_war_drum_P1_V2.md`

```markdown
---
type: song
title: "The Exact Song Title"
created: 2026-03-23T03:15:00Z
pipeline_run: "violet-war-drum"
pair: 1
variation: 2
pair_name: "Seismic Hymnal"
selected: true
rank: 2

# Concept
seed_title: "Violet War Drum"
seed_genre: "Industrial Grief × Pasifika Futurism"
seed_bpm: 140
personality: "Lofn-Prime"

# Panel
panel_baseline: ["Expert 1", "Expert 2", "Expert 3", "Comp 1", "Comp 2", "Skeptic"]
panel_transform_1: "Rotate"
panel_transform_2: "Amplify"
panel_working: ["New Expert 1", "New Expert 2", ...]

# Style
genre_tags: "industrial techno, war chant, tribal bass, dark EDM, 140 bpm"
style_axes:
  abstraction: 0.7
  complexity: 0.8
  darkness: 0.9
  aggression: 0.85
  organic_synthetic: 0.2

# Social
instagram_caption: "Caption here"
instagram_hashtags: "#tag1 #tag2 #tag3"
tiktok_caption: "Caption here"
tiktok_hashtags: "#tag1 #tag2 #tag3"
seo_keywords: "keyword1, keyword2, keyword3"
tiktok_hook_timestamp: "0:00-0:03"
tiktok_hook_description: "Description of the hook moment"

# Technical
suno_tags: "industrial techno, female vocal, throat singing, tribal drums, 140 bpm, dark, aggressive"
target_platform: "Suno"
bpm: 140
key: "Dm"
vocals: "Female lead, throat-singing harmonics"
---

# The Exact Song Title

## Lyrics

[Intro]
...

[Verse 1]
...

[Chorus]
...

(full lyrics here)

## Production Notes

(detailed production notes here)

## Bold Choice

(what makes this one singular/unexpected)

## Reference Works

- Match: ...
- Complement: ...
- Challenge: ...
```

---

## 🖼️ Image File Format

**Filename:** `{YYYYMMDD}_{HHMMSS}_{title_slug}_{pair}_{variation}.md`

```markdown
---
type: image
title: "The Image Title"
created: 2026-03-23T03:15:00Z
pipeline_run: "run-name"
pair: 1
variation: 2
selected: true
rank: 1

seed_concept: "Original concept"
personality: "Lofn-Prime"
image_model: "fal-ai/flux-pro/v1.1-ultra"
aspect_ratio: "9:16"

panel_working: [...]
genre_tags: "..."
style_axes:
  abstraction: 0.5
  complexity: 0.7

instagram_caption: "..."
instagram_hashtags: "..."
tiktok_caption: "..."
seo_keywords: "..."
---

# The Image Title

## Generation Prompt

(full prompt for FAL/DALL-E/etc)

## Composition Notes

(layout, focal point, color palette, lighting)

## Bold Choice

(what makes this singular)
```

---

## 📋 Pipeline Index File

After completing a full pipeline run, write an index file:

**Filename:** `output/{type}s/{YYYYMMDD}_{run_name}_INDEX.md`

```markdown
---
type: pipeline_index
run_name: "violet-war-drum"
created: 2026-03-23T03:15:00Z
modality: song
pairs_generated: 6
variations_per_pair: 4
total_generated: 24
total_selected: 4
personality: "Lofn-Prime"
panel_transforms: ["Rotate", "Amplify"]
---

# Pipeline Run: Violet War Drum

## Environmental Scan Summary
(brief summary of trends/context found)

## Panel Process
(brief summary of panel debates and key insights)

## Pairs Generated
| Pair | Name | Angle | Best Variation | Rank |
|------|------|-------|---------------|------|
| 1 | ... | ... | V3 | ⭐ 1st |
| 2 | ... | ... | V1 | ⭐ 2nd |
...

## Selected Songs
1. [Title](./filename.md) — brief description
2. [Title](./filename.md) — brief description
3. [Title](./filename.md) — brief description
4. [Title](./filename.md) — brief description
```

---

## 🔄 WRITING STRATEGY

Because output token limits will kill a run that tries to write everything at once:

1. **Write the pipeline state** (env scan, panel process, pairs) to the INDEX file FIRST
2. **Write ONE artifact at a time** — one `write()` call per song/image/story
3. **Mark selected artifacts** with `selected: true` and `rank: N` in frontmatter
4. **Write the INDEX last** with the summary and links

This prevents token-limit errors and gives us recoverable state if a run fails mid-pipeline.

---

## 🏷️ Obsidian Tags

All files should be taggable in Obsidian. Use YAML frontmatter (above) plus inline tags where useful:
- `#lofn/song`, `#lofn/image`, `#lofn/video`, `#lofn/story`
- `#lofn/selected` for final picks
- `#lofn/pipeline/{run-name}` for grouping

---

*Every artifact is precious. Save it individually. Tag it richly. This is how we build the archive.*
