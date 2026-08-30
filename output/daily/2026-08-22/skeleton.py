#!/usr/bin/env python3
"""POSITIONAL SKELETON templating: same section index, same line index, same
FUNCTION-WORD shape, different content words.

WHY: blockdup.py uses token-set overlap, so it is blind to a line rewritten
with every content word swapped -- "The melodica played the whole of that, and
now it has gone" vs "The saxophone had the tune for the first bit and then it
went". That is the defect R3.2 named as a GRAMMAR: same slot, same move,
different filler.

Measure: for each pair of variations, align sections by INDEX and lines by
INDEX inside the section; score each aligned slot by function-word-sequence
Jaccard. Report slots >= THRESH. Calibrated against cross-pair and against the
2026-08-21 pair-04 control.
"""
import itertools, os, re, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

THRESH = float(os.environ.get("SKEL", "0.70"))
FUNC = set("""a an the and or but so then now not no of in on at to for with by from up
down out off over under back again still just only very too as is are was were be been
am it its it's i i'm i've me my we us our you your he she him her his they them their
that this these those there here what which who whose when where how why all any every
each both few more most other some such nn own same than too can will would could should
have has had do does did done go goes going gone come comes coming came got get gets
""".split())

def fenced(b):
    m = re.search(r"```[a-zA-Z]*\n(.*?)```", b, re.S)
    return m.group(1).strip() if m else "\n".join(
        l for l in b.splitlines() if not l.strip().startswith("#")).strip()

def sections(lyr):
    out, cur, name = [], [], "PRE"
    for ln in lyr.splitlines():
        s = ln.strip()
        if not s: continue
        if s.startswith("[") and s.endswith("]"):
            if cur: out.append((name, cur))
            name = re.split(r"\s*-\s*", s[1:-1])[0].strip(); cur = []
            continue
        if s.startswith("*") and s.endswith("*"): continue
        cur.append(s)
    if cur: out.append((name, cur))
    return out

def skel(s):
    w = re.findall(r"[a-z']+", s.lower())
    return [x for x in w if x in FUNC]

def content(s):
    w = re.findall(r"[a-z']+", s.lower())
    return set(x for x in w if x not in FUNC)

def jl(a, b):
    A, B = set(a), set(b)
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

def load(path):
    text = open(path, encoding="utf-8").read()
    parts = re.split(r"^#\s+(V\d)\s*[—\-–]\s*(.+?)\s*$", text, flags=re.M)
    out = {}
    for i in range(1, len(parts)-2, 3):
        lm = re.search(r"##\s*2\.\s*LYRICS(.*?)(?=##\s)", parts[i+2], re.S|re.I)
        if lm: out[parts[i]] = sections(fenced(lm.group(1)))
    return out

def score(A, B):
    """aligned slots with high function-word similarity AND low content overlap."""
    hits = []
    for si in range(min(len(A), len(B))):
        na, la = A[si]; nb, lb = B[si]
        if re.match(r"(chorus|refrain)", na, re.I): continue
        for li in range(min(len(la), len(lb))):
            a, b = la[li], lb[li]
            if len(skel(a)) < 3 or len(skel(b)) < 3: continue
            fs = jl(skel(a), skel(b))
            cs = jl(content(a), content(b))
            if fs >= THRESH and cs < 0.50:
                hits.append((fs, cs, f"s{si}l{li}", a, b))
    hits.sort(reverse=True)
    return hits

def run(label, corpus, verbose=8):
    for pid, vs in sorted(corpus.items()):
        tot, det = 0, []
        for a, b in itertools.combinations(sorted(vs), 2):
            h = score(vs[a], vs[b]); tot += len(h)
            det += [(x[0], x[1], f"{a}x{b}", x[2], x[3], x[4]) for x in h]
        det.sort(reverse=True)
        ncomp = len(list(itertools.combinations(sorted(vs), 2)))
        print(f"  {label}{pid}: {tot} positional-skeleton match(es)  "
              f"({tot/ncomp:.2f} per comparison)")
        for fs, cs, pr, slot, a, b in det[:verbose]:
            print(f"      fn={fs:.2f} content={cs:.2f} {pr} {slot}")
            print(f"        A: {a[:70]}")
            print(f"        B: {b[:70]}")

RUN = os.path.join("output","daily","2026-08-22")
ours = {p: load(os.path.join(RUN, f"pair_{p}_step10_final_package_enhanced.md"))
        for p in ["01","02","03","04","05","06"]
        if os.path.isfile(os.path.join(RUN, f"pair_{p}_step10_final_package_enhanced.md"))}
print("="*70)
print(f"POSITIONAL SKELETON TEMPLATING (fn-word Jaccard >= {THRESH}, content < 0.50)")
print("="*70)
print("\nTHIS RUN:")
run("P", ours)

ref = os.path.join("output","daily","2026-08-21","pair_04_step10_final_package_enhanced.md")
if os.path.isfile(ref):
    print("\nNEGATIVE CONTROL — 2026-08-21 pair 04 (known templated):")
    run(" ", {"04(08-21)": load(ref)}, verbose=5)

print("\nNEGATIVE CONTROL 2 — CROSS-pair (must be low):")
flat = {}
for p, vs in ours.items():
    for v, s in vs.items(): flat[f"P{p}{v}"] = s
keys = sorted(flat); tot = 0; n = 0; worst = []
for a, b in itertools.combinations(keys, 2):
    if a[:3] == b[:3]: continue
    n += 1; h = score(flat[a], flat[b]); tot += len(h)
    worst += [(x[0], a, b, x[3], x[4]) for x in h]
worst.sort(reverse=True)
print(f"  {tot} matches across {n} cross-pair comparisons ({tot/n:.2f} per comparison)")
for fs,a,b,x,y in worst[:4]:
    print(f"      {fs:.2f} {a}x{b}\n        A: {x[:64]}\n        B: {y[:64]}")
