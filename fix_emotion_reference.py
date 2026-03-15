import glob

# For all files that contain EMOTION REFERENCE LIST
for filepath in glob.glob('lofn/openclaw_skills/**/*.md', recursive=True):
    if "00_" in filepath: continue

    with open(filepath, 'r') as f:
        content = f.read()

    start_str = "────────────────────────────────────────────────────────\nEMOTION REFERENCE LIST\n────────────────────────────────────────────────────────"

    if start_str in content:
        start_idx = content.find(start_str)
        # End is usually around "• DISINTEREST"
        dis_idx = content.find("• DISINTEREST", start_idx)
        if dis_idx != -1:
            end_idx = content.find("\n", dis_idx) + 1
            block = content[start_idx:end_idx]
            rep = "────────────────────────────────────────────────────────\nEMOTION REFERENCE LIST\n────────────────────────────────────────────────────────\nUse the provided list of `[emotions]` in the `USER INPUT` section to select the exact emotional nuance that aligns with your creation.\n"
            content = content.replace(block, rep)
            print(f"Replaced story emotions list in {filepath}")

            with open(filepath, 'w') as f:
                f.write(content)
