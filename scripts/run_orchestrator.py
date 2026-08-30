import json

panel = {
  "concept": ["Aesthetic Philosopher", "Cultural Anthropologist"],
  "medium": ["Audio Engineer", "Suno Prompt Specialist"],
  "context": ["Gen-Z Music Critic", "Psychoacoustician"],
  "skeptic": ["Cynical Industry Veteran"]
}

metaprompt = """
Generate a Solarpunk Healer (AWE mode) music piece. It must be emotionally resonant, healing, using 432Hz tuning, green synths, and complex polyrhythms. Focus on a bio-adaptive soundscape that drives the listener to a state of clarity. Ensure female vocals and a minimum of 50 lines of lyrics.
"""

axes = ["Organic vs Synthetic", "Intimate vs Massive", "Tribal Rhythm vs Ambient Wash", "Ethereal vs Grounded"]

with open("/data/.openclaw/workspace/output/music/orchestrator_output.json", "w") as f:
    json.dump({
        "metaprompt": metaprompt,
        "panel": panel,
        "personality": "Lofn - Solarpunk Healer",
        "axes": axes
    }, f, indent=2)
