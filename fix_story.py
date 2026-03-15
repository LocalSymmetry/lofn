import glob

# The story block:
block = """• HAPPINESS        → Joy, Serenity, Hope, Zeal, Triumph, Wonder, Fulfillment
• SADNESS          → Melancholy, Nostalgia, Regret, Torment, Isolation
• ANGER            → Rage, Frustration, Betrayal, Defiance, Aggression
• FEAR            → Anxiety, Dread, Panic, Insecurity, Existential Angst
• LOVE            → Affection, Longing, Trust, Passion
• SURPRISE         → Amazement, Curiosity, Revelation, Awe, Surrealism
• TRUST           → Assurance, Solidarity, Forgiveness, Empowerment
• ANTICIPATION    → Eagerness, Suspense, Yearning
• DETERMINATION  → Resilience, Ambition, Persistence, Resolve
• INTROSPECTION   → Reflection, Solitude, Identity
• DISCONNECTION   → Alienation, Numbness, Apathy
• DISINTEREST      → Ennui, Boredom, Indifference"""

rep = """Use the provided list of `[emotions]` in the `USER INPUT` section to select the exact emotional nuance that aligns with your creation."""

for filepath in glob.glob('lofn/openclaw_skills/story/*.md'):
    if "00_" in filepath: continue
    with open(filepath, 'r') as f:
        content = f.read()

    if block in content:
        content = content.replace(block, rep)
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Fixed {filepath}")
    else:
        print(f"NOT FOUND in {filepath}")
