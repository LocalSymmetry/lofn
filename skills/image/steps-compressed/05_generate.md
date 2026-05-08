# Image Pipeline Step 5: Generate Images

Read:
1. `{output_dir}/04_final_prompts.md`
2. `skills/image-gen/SKILL.md` (for FAL generation instructions)

For each of the 6 prompts, generate with FAL Flux Pro 1.1 Ultra:

```bash
node skills/image-gen/scripts/fal-generate.cjs \
  --prompt "<prompt text>" \
  --aspect "3:4" \
  --output {output_dir}/0N_<slug>.png \
  --safety 3
```

Run all 6 generations. Save each PNG + JSON metadata.
