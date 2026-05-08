# Music Pipeline Step 1: Aesthetics & Genres

Read the orchestrator metaprompt at `{metaprompt_path}`.

Generate 50 aesthetics, 50 emotions, 50 genres relevant to the metaprompt. Use the constraint axes as seeds. Focus on rare combinations.

Output valid JSON:
```json
{
  "aesthetics": ["...", ...],
  "emotions": ["...", ...],
  "genres": ["...", ...]
}
```

Save to: `{output_dir}/01_aesthetics_genres.json`
