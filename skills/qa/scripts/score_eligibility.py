#!/usr/bin/env python3
"""Score Lofn 7-property eligibility from a JSON score file.

Input JSON format:
{
  "route": "accessible",
  "scores": {
    "body_in_song": 5,
    "adoptable_hook": 4,
    "vast_emotional_tam": 5,
    "specificity_paradox": 3,
    "cognitive_ease": 4,
    "vocal_co_discovery": 4,
    "sonic_threshold": 5
  }
}
"""
import json, sys
from pathlib import Path

KEYS = [
    "body_in_song", "adoptable_hook", "vast_emotional_tam",
    "specificity_paradox", "cognitive_ease", "vocal_co_discovery",
    "sonic_threshold"
]

def main():
    if len(sys.argv) != 2:
        print(json.dumps({"ok": False, "error_code": "USAGE", "message": "Usage: score_eligibility.py scores.json"}))
        return 2
    path = Path(sys.argv[1])
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        print(json.dumps({"ok": False, "error_code": "BAD_JSON", "message": str(e)}))
        return 1
    scores = data.get("scores", {})
    missing = [k for k in KEYS if k not in scores]
    if missing:
        print(json.dumps({"ok": False, "error_code": "MISSING_SCORES", "missing": missing, "fix_suggestions": ["Add all 7 eligibility scores, each 1-5"]}))
        return 1
    bad = {k: scores[k] for k in KEYS if not isinstance(scores[k], (int, float)) or scores[k] < 1 or scores[k] > 5}
    if bad:
        print(json.dumps({"ok": False, "error_code": "BAD_SCORE_RANGE", "bad_scores": bad, "fix_suggestions": ["Scores must be numeric 1-5"]}))
        return 1
    avg = sum(scores[k] for k in KEYS) / len(KEYS)
    count_ge3 = sum(1 for k in KEYS if scores[k] >= 3)
    classification = "ACCESSIBLE" if avg >= 3.5 and count_ge3 >= 5 else "AMBITIOUS"
    route = str(data.get("route", "")).upper()
    verdict = "PASS"
    if route == "ACCESSIBLE" and classification != "ACCESSIBLE":
        verdict = "FAIL_ACCESSIBLE_THRESHOLD"
    print(json.dumps({
        "ok": True,
        "average": round(avg, 2),
        "properties_ge3": count_ge3,
        "classification": classification,
        "route": route or None,
        "verdict": verdict
    }, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
