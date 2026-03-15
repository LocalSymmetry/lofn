import glob

# The video block starts at `### EMOTION LIST` and ends at `\n**Disinterest**: [` and then `]\n]`
for filepath in glob.glob('lofn/openclaw_skills/video/*.md'):
    if "00_" in filepath: continue
    with open(filepath, 'r') as f:
        content = f.read()

    start_idx = content.find("### EMOTION LIST")
    if start_idx != -1:
        # Actually, let's find `Art serves as a profound conduit` before it
        real_start = content.rfind("Art serves as a profound conduit", 0, start_idx)
        if real_start != -1:
            end_search = content.find("**Disinterest**", start_idx)
            if end_search != -1:
                end_idx = content.find("]\n]", end_search) + 3
                block = content[real_start:end_idx]
                rep = "Art serves as a profound conduit to human emotions. Use the provided list of `[emotions]` in the `USER INPUT` section to select the exact emotional nuance that aligns with your creation.\n"
                content = content.replace(block, rep)
                print(f"Replaced video emotions in {filepath}")

        with open(filepath, 'w') as f:
            f.write(content)
