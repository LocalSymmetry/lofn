#!/usr/bin/env python3
"""Validate Lofn release/falsification record schema."""
import json, sys
from pathlib import Path

REQUIRED=['release_id','title','modality','route','archetype','seed_source','eligibility_pre_score','ai_vocabulary_level','hypothesis','distribution_events','metrics']
SCORES=['body_in_song','adoptable_hook','vast_emotional_tam','specificity_paradox','cognitive_ease','vocal_co_discovery','sonic_threshold']
ROUTES={'accessible','ambitious'}
VOCAB={'none','encoded','earned_late','explicit_early','explicit_ambitious'}

def main():
    if len(sys.argv)!=2:
        print(json.dumps({'ok':False,'error_code':'USAGE','message':'Usage: validate_release_record.py release_record.json'}))
        return 2
    try:
        data=json.loads(Path(sys.argv[1]).read_text())
    except Exception as e:
        print(json.dumps({'ok':False,'error_code':'BAD_JSON','message':str(e)}))
        return 1
    issues=[]
    for k in REQUIRED:
        if k not in data:
            issues.append({'field':k,'issue':'missing'})
    if data.get('route') not in ROUTES:
        issues.append({'field':'route','issue':'must_be_accessible_or_ambitious'})
    if data.get('ai_vocabulary_level') not in VOCAB:
        issues.append({'field':'ai_vocabulary_level','issue':'bad_value','allowed':sorted(VOCAB)})
    eps=data.get('eligibility_pre_score',{})
    for k in SCORES:
        v=eps.get(k)
        if v is None:
            issues.append({'field':f'eligibility_pre_score.{k}','issue':'null_score'})
        elif not isinstance(v,(int,float)) or v<1 or v>5:
            issues.append({'field':f'eligibility_pre_score.{k}','issue':'must_be_1_to_5'})
    if not data.get('hypothesis'):
        issues.append({'field':'hypothesis','issue':'missing_pre_registered_hypothesis'})
    if issues:
        print(json.dumps({'ok':False,'error_code':'INVALID_RELEASE_RECORD','issues':issues,'fix_suggestions':['Complete record before release','Pre-score all 7 eligibility properties','State the hypothesis before metrics arrive']}, indent=2))
        return 1
    print(json.dumps({'ok':True,'message':'Release record valid','release_id':data.get('release_id')} , indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
