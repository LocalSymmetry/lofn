# Image Pipeline Step 1: Aesthetics & Genres

Read the orchestrator metaprompt at `{metaprompt_path}`.

Generate 50 aesthetics, 50 emotions, 50 compositions, 50 genres relevant to the metaprompt's creative direction. Use the constraint axes and medium collisions as seeds. Focus on RARE and UNEXPECTED combinations.

Adult subjects only. No children. No cute fairy tropes.

Output valid JSON:
```json
{
  "aesthetics": ["...", ...],
  "emotions": ["...", ...],
  "frames_and_compositions": ["...", ...],
  "genres": ["...", ...]
}
```

Save to: `{output_dir}/01_aesthetics_genres.json`
