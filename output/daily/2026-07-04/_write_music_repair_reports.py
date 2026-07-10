from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from _validate_music_repair import RUN, validate


SELECTED_SONGS = {
    "P01V2": "the lander learned to bounce",
    "P02V2": "house of measured rest",
    "P03V2": "the air said no",
    "P04V2": "wrong account choir",
    "P05V2": "prefusion lock",
    "P06V2": "sky overclock canticle",
}

SELECTED_IMAGES = {
    "I01V2": "house-cat rover control room",
    "I02V2": "measured bread siren control room",
    "I03V2": "air cancelled the parade control room",
    "I04V2": "cache leak psalm control room",
    "I05V2": "square-tile mercy control room",
    "I06V2": "different fading systems control room",
}

PAIR_AGENTS = {
    "P01": {"nickname": "Godel", "agent_id": "019f2f87-e5bf-7af3-a16e-0c6d0fc887b7"},
    "P02": {"nickname": "Heisenberg", "agent_id": "019f2f88-fd34-7801-86ef-f83bd8e4bd41"},
    "P03": {"nickname": "Harvey", "agent_id": "019f2f89-119d-7921-8c02-38db2f3d43be"},
    "P04": {"nickname": "Boyle", "agent_id": "019f2f89-268a-7d01-ba02-e3395f5da26b"},
    "P05": {"nickname": "Kierkegaard", "agent_id": "019f2f89-3bb7-7ef0-8a79-5a0b77ccfe9f"},
    "P06": {"nickname": "Copernicus", "agent_id": "019f2f89-50a1-7122-a2f5-def39a5318a2"},
}


def main() -> int:
    validation = validate()
    song_rows = []
    for row in validation["song_rows"]:
        song_id = row["song_id"]
        song_rows.append(
            {
                "id": song_id,
                "pair_id": song_id[:3],
                "selected": song_id in SELECTED_SONGS,
                "selected_title": SELECTED_SONGS.get(song_id),
                "music_prompt_chars": row["music_prompt_chars"],
                "exclude_prompt_chars": row["exclude_chars"],
                "lyrics_chars": row["lyrics_chars"],
                "sung_lines": row["sung_lines"],
                "numeric_fact_count": row["numeric_fact_count"],
                "checks": row["checks"],
                "passed": row["passed"],
            }
        )

    image_files = sorted((RUN / "images").glob("2026-07-04_I*.md"))
    report = {
        "run_date": "2026-07-04",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "scope": "music repair after initial inline daily run showed lyric bleed",
        "verdict": "SHIP" if validation["passed"] else "REPAIR",
        "music_execution": {
            "steps_05_11_repair": "pair-isolated",
            "pair_count": 6,
            "songs": 24,
            "selected_finalists": len(SELECTED_SONGS),
            "assignment": "LOFN-Prime",
            "pair_agents": PAIR_AGENTS,
        },
        "music_checks": {
            "all_song_packages_pass": validation["passed"],
            "cross_pair_shared_sung_lines": validation["cross_pair_shared_lines"],
            "lyric_bleed_absent": not validation["cross_pair_shared_lines"],
            "banned_scaffold_absent_all": all(
                row["checks"]["banned_scaffold_absent"] for row in validation["song_rows"]
            ),
            "prompt_gates_pass_all": all(
                row["checks"]["music_prompt_850_1000"]
                and row["checks"]["music_prompt_target_870_960"]
                and row["checks"]["exclude_400_900"]
                for row in validation["song_rows"]
            ),
            "lyric_gates_pass_all": all(
                row["checks"]["lyrics_under_5000"]
                and row["checks"]["lyrics_target_4800"]
                and row["checks"]["sung_lines_70_120"]
                and row["checks"]["full_emo_headers"]
                and row["checks"]["has_sfx"]
                and row["checks"]["numeric_facts_lte_1"]
                for row in validation["song_rows"]
            ),
        },
        "selected_songs": SELECTED_SONGS,
        "song_rows": song_rows,
        "image_summary": {
            "image_prompt_files": len(image_files),
            "selected_finalists": len(SELECTED_IMAGES),
            "selected_images": SELECTED_IMAGES,
            "status": "unchanged from daily run",
        },
        "passed": validation["passed"],
    }

    text = json.dumps(report, indent=2) + "\n"
    (RUN / "MUSIC_REPAIR_GATE_REPORT.json").write_text(text, encoding="utf-8")
    (RUN / "GATE_REPORT.json").write_text(text, encoding="utf-8")
    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
