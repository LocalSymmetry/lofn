# Music Pipeline Step 1: Aesthetics & Genres

Read the orchestrator metaprompt at `{metaprompt_path}`.

Generate 50 aesthetics, 50 emotions, 50 style labels relevant to the metaprompt. Use the constraint axes as seeds. Focus on rare combinations. Do not use global genre examples; style labels must emerge from the current run.

Output valid JSON:
```json
{
  "aesthetics": ["...", ...],
  "emotions": ["...", ...],
  "genres": ["<run-specific style label>", ...]
}
```

Save to: `{output_dir}/01_aesthetics_genres.json`
