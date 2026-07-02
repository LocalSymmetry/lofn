#!/usr/bin/env python3
"""Validate a Lofn seed_packet.json artifact."""
import json, sys
from pathlib import Path

REQUIRED = [
  'run_id','route','archetype','seed_source','scene','body_entry',
  'vulnerable_object_or_addressee','temperature','textures','specific_fact',
  'emotional_tam','hook_address','hook_posture','arc_turn','bridge_revelation',
  'identity_stance','forbidden_abstractions','eligibility_target'
]
ELIGIBILITY = ['body_in_song','adoptable_hook','vast_emotional_tam','specificity_paradox','cognitive_ease','vocal_co_discovery','sonic_threshold']
ROUTES = {'accessible','ambitious'}

def main():
    if len(sys.argv) != 2:
        print(json.dumps({'ok': False, 'error_code': 'USAGE', 'message': 'Usage: validate_seed_packet.py seed_packet.json'}))
        return 2
    try:
        data=json.loads(Path(sys.argv[1]).read_text())
    except Exception as e:
        print(json.dumps({'ok': False, 'error_code': 'BAD_JSON', 'message': str(e)}))
        return 1
    issues=[]
    for k in REQUIRED:
        if k not in data or data[k] in ('', None, []):
            issues.append({'field': k, 'issue': 'missing_or_empty'})
    if data.get('route') not in ROUTES:
        issues.append({'field':'route','issue':'must_be_accessible_or_ambitious'})
    et=data.get('eligibility_target',{})
    for k in ELIGIBILITY:
        v=et.get(k)
        if not isinstance(v,(int,float)) or v < 1 or v > 5:
            issues.append({'field':f'eligibility_target.{k}','issue':'must_be_numeric_1_to_5'})
    if not isinstance(data.get('textures'), list) or len(data.get('textures',[])) < 3:
        issues.append({'field':'textures','issue':'must_have_at_least_3_textures'})
    if issues:
        print(json.dumps({'ok': False, 'error_code': 'INVALID_SEED_PACKET', 'issues': issues, 'fix_suggestions': ['Complete all required fields', 'Use assets/seed_packet.template.json as mold']}, indent=2))
        return 1
    print(json.dumps({'ok': True, 'route': data['route'], 'archetype': data['archetype'], 'message': 'Seed packet valid'}, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
