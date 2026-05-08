#!/usr/bin/env python3
"""Validate Lofn spawn manifest concurrency and timeout policy."""
import json, sys
from pathlib import Path

VALID_STATUSES={'pending','spawned','done','failed','pending_concurrency_slot'}

def main():
    if len(sys.argv)!=2:
        print(json.dumps({'ok':False,'error_code':'USAGE','message':'Usage: validate_spawn_manifest.py spawn_manifest.json'}))
        return 2
    try:
        data=json.loads(Path(sys.argv[1]).read_text())
    except Exception as e:
        print(json.dumps({'ok':False,'error_code':'BAD_JSON','message':str(e)}))
        return 1
    issues=[]
    max_children=data.get('max_children',5)
    agents=data.get('agents',[])
    spawned=sum(1 for a in agents if a.get('status')=='spawned')
    if spawned>max_children:
        issues.append({'issue':'too_many_spawned','spawned':spawned,'max_children':max_children})
    if len(agents)>=6 and max_children<=5:
        sixth=agents[5]
        if sixth.get('status')=='spawned' and spawned>=6:
            issues.append({'issue':'sixth_spawned_without_slot','pair':sixth.get('pair')})
    for a in agents:
        if a.get('status') not in VALID_STATUSES:
            issues.append({'issue':'bad_status','agent':a})
    t=data.get('timeouts_minutes',{})
    mins={'orchestrator':15,'coordinator':20,'pair_agent':20,'qa':12,'standard_full_run':90,'competition_full_run':120}
    for k,v in mins.items():
        if t.get(k,0)<v:
            issues.append({'issue':'timeout_below_policy','field':k,'value':t.get(k),'minimum':v})
    if issues:
        print(json.dumps({'ok':False,'error_code':'SPAWN_MANIFEST_INVALID','issues':issues,'fix_suggestions':['Use staggered_5_plus_1 strategy','Do not spawn pair 6 until a slot opens','Use policy minimum timeouts']}, indent=2))
        return 1
    print(json.dumps({'ok':True,'message':'Spawn manifest valid','spawned':spawned,'max_children':max_children}, indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
