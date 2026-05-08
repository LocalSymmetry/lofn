#!/usr/bin/env python3
"""Validate presence of a CREATIVE_CHECKPOINT block in a step artifact."""
import json, re, sys
from pathlib import Path

REQUIRED = ['step_completed','decisions_made','rejected_alternatives','building_toward','seed_fidelity_note']

def main():
    if len(sys.argv) != 2:
        print(json.dumps({'ok': False, 'error_code':'USAGE', 'message':'Usage: validate_checkpoint.py artifact.md'}))
        return 2
    text=Path(sys.argv[1]).read_text(errors='ignore')
    m=re.search(r'---CREATIVE_CHECKPOINT---(.*?)---END_CHECKPOINT---', text, re.S)
    if not m:
        print(json.dumps({'ok': False, 'error_code':'MISSING_CHECKPOINT', 'fix_suggestions':['Add CREATIVE_CHECKPOINT block from warm_handoff_checkpoint.md']}))
        return 1
    block=m.group(1)
    missing=[k for k in REQUIRED if k not in block]
    if missing:
        print(json.dumps({'ok': False, 'error_code':'INCOMPLETE_CHECKPOINT', 'missing':missing, 'fix_suggestions':['Complete all required checkpoint fields']}))
        return 1
    print(json.dumps({'ok': True, 'message':'Creative checkpoint present'}))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
