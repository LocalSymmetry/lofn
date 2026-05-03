#!/usr/bin/env python3
"""Validate that required phase artifacts exist and are non-stub."""
import json, sys
from pathlib import Path


def main():
    if len(sys.argv) not in (2,3):
        print(json.dumps({'ok': False, 'error_code':'USAGE', 'message':'Usage: validate_phase_gate.py phase_gate.json [base_dir]'}))
        return 2
    gate_path=Path(sys.argv[1])
    base=Path(sys.argv[2]) if len(sys.argv)==3 else gate_path.parent
    try:
        gate=json.loads(gate_path.read_text())
    except Exception as e:
        print(json.dumps({'ok': False, 'error_code':'BAD_JSON', 'message':str(e)}))
        return 1
    failures=[]
    for item in gate.get('required_files', []):
        rel=item.get('path')
        if not rel:
            failures.append({'path': None, 'issue':'missing_path_field'})
            continue
        p=base/rel
        if not p.exists():
            failures.append({'path': rel, 'issue':'missing'})
            continue
        text=p.read_text(errors='ignore') if p.is_file() else ''
        size=p.stat().st_size
        min_bytes=item.get('min_bytes',0)
        if size < min_bytes:
            failures.append({'path': rel, 'issue':'under_min_bytes', 'size':size, 'min_bytes':min_bytes})
        for needle in item.get('required_text', []):
            if needle not in text:
                failures.append({'path': rel, 'issue':'missing_required_text', 'required_text':needle})
    if failures:
        print(json.dumps({'ok': False, 'error_code':'PHASE_GATE_FAILED', 'phase':gate.get('phase'), 'failures':failures, 'fix_suggestions':['Do not advance phase', 'Repair or rerun missing/stub artifacts']}, indent=2))
        return 1
    print(json.dumps({'ok': True, 'phase': gate.get('phase'), 'message':'Phase gate passed'}, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
