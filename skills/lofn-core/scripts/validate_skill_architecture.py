#!/usr/bin/env python3
"""Validate OpenClaw/Lofn skill architecture basics.

Checks:
- SKILL.md exists
- YAML frontmatter exists with name and description
- SKILL.md line count <= 500 warning threshold
- references/scripts/assets dirs are flat if present
- no accidental giant reference dump inside SKILL.md (>500 lines)
"""
import json, sys
from pathlib import Path

ROOTS = [Path('skills')]

def frontmatter(text):
    if not text.startswith('---'):
        return None
    parts = text.split('---', 2)
    if len(parts) < 3:
        return None
    return parts[1]

def check_flat(d):
    if not d.exists():
        return []
    return [str(p) for p in d.rglob('*') if p.is_file() and p.parent != d]

def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
    results=[]
    for skill in sorted((root/'skills').glob('*/')):
        smd=skill/'SKILL.md'
        if not smd.exists():
            continue
        text=smd.read_text(errors='ignore')
        fm=frontmatter(text)
        issues=[]
        if fm is None:
            issues.append('MISSING_FRONTMATTER')
        else:
            if 'name:' not in fm: issues.append('MISSING_NAME')
            if 'description:' not in fm: issues.append('MISSING_DESCRIPTION')
            if 'Do NOT use' not in fm and 'Do not use' not in fm:
                issues.append('NO_NEGATIVE_TRIGGER')
        lines=text.count('\n')+1
        if lines > 500:
            issues.append(f'SKILL_MD_OVER_500_LINES:{lines}')
        for sub in ['references','scripts','assets']:
            nested=check_flat(skill/sub)
            if nested:
                issues.append(f'NESTED_{sub.upper()}:{len(nested)}')
        results.append({'skill': str(skill), 'lines': lines, 'issues': issues, 'ok': not issues})
    ok=all(r['ok'] for r in results)
    print(json.dumps({'ok': ok, 'results': results}, indent=2))
    return 0 if ok else 1

if __name__ == '__main__':
    raise SystemExit(main())
