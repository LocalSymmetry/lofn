#!/usr/bin/env python3
"""Validate a Lofn pipeline preflight JSON artifact."""
import json, sys
from pathlib import Path

REQUIRED = [
  'model_available','output_dir_writable','concurrency_known','research_brief_complete',
  'seed_or_seed_packet_exists','rules_saved_if_relevant','modality_confirmed',
  'target_output_count_confirmed','barbell_route_set'
]

def main():
    if len(sys.argv) != 2:
        print(json.dumps({'ok': False, 'error_code': 'USAGE', 'message': 'Usage: validate_preflight.py preflight.json'}))
        return 2
    try:
        data=json.loads(Path(sys.argv[1]).read_text())
    except Exception as e:
        print(json.dumps({'ok': False, 'error_code':'BAD_JSON', 'message':str(e)}))
        return 1
    missing=[k for k in REQUIRED if k not in data]
    failed=[k for k in REQUIRED if data.get(k) is not True]
    route=data.get('barbell_route') or data.get('route')
    if route not in ('accessible','ambitious'):
        failed.append('barbell_route')
    if missing or failed:
        print(json.dumps({'ok': False, 'error_code':'PREFLIGHT_FAILED', 'missing':missing, 'failed':failed, 'fix_suggestions':['Complete every checklist item before launching agents','Set barbell_route to accessible or ambitious']}, indent=2))
        return 1
    print(json.dumps({'ok': True, 'message':'Preflight complete', 'route': route}, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
