#!/usr/bin/env python3
"""PARAPHRASE-LEVEL repeated lines within a pair's four variations.

WHY: distinctiveness.py test 7 counts EXACT repeated sung lines. A pair that
writes the same sentence four times with one noun swapped scores 0 there and
0 on tests 1-3 (3-gram Jaccard is far too sparse for a 12-line block inside a
~80-line song). This measures line-level token-set Jaccard instead, which
catches "It is not coming back" / "It does not come back" / "It has gone. It
is not coming back on."

Calibration is mandatory: run against the 2026-08-21 pair-04 negative control
(known templated) AND against CROSS-pair comparisons (known unrelated).
"""
import itertools, os, re, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

THRESH = float(os.environ.get("THRESH", "0.50"))

def fenced(block):
    m = re.search(r"```[a-zA-Z]*\n(.*?)```", block, re.S)
    if m: return m.group(1).strip()
    return "\n".join(l for l in block.splitlines() if not l.strip().startswith("#")).strip()

def sections(lyrics):
    """[(section_name, [lines])] preserving order."""
    out, cur, name = [], [], "PRE"
    for ln in lyrics.splitlines():
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

def toks(s):
    return set(re.findall(r"[a-z']+", s.lower()))

def jac(a, b):
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)

def load(path):
    text = open(path, encoding="utf-8").read()
    parts = re.split(r"^#\s+(V\d)\s*[—\-–]\s*(.+?)\s*$", text, flags=re.M)
    out = {}
    for i in range(1, len(parts) - 2, 3):
        tag, body = parts[i], parts[i + 2]
        lm = re.search(r"##\s*2\.\s*LYRICS(.*?)(?=##\s)", body, re.S | re.I)
        if not lm: continue
        out[tag] = sections(fenced(lm.group(1)))
    return out

def body_lines(secs, drop_refrain=True):
    """Unique lines outside Chorus/Refrain sections."""
    seen, out = set(), []
    for name, lines in secs:
        if drop_refrain and re.match(r"(chorus|refrain)", name, re.I): continue
        for ln in lines:
            k = re.sub(r"[^a-z ]", "", ln.lower()).strip()
            if len(k.split()) < 4: continue     # too short to be a signature
            if k in seen: continue
            seen.add(k); out.append(ln)
    return out

def compare(A, B, thresh=THRESH):
    ta = [(l, toks(l)) for l in A]
    tb = [(l, toks(l)) for l in B]
    hits = []
    for la, sa in ta:
        best, bl = 0.0, None
        for lb, sb in tb:
            s = jac(sa, sb)
            if s > best: best, bl = s, lb
        if best >= thresh: hits.append((best, la, bl))
    hits.sort(reverse=True)
    return hits

def run(label, corpus):
    """corpus: {pair_id: {vtag: sections}}"""
    grand = 0
    for pid, variants in sorted(corpus.items()):
        bodies = {v: body_lines(s) for v, s in variants.items()}
        total = 0
        detail = []
        for a, b in itertools.combinations(sorted(bodies), 2):
            hits = compare(bodies[a], bodies[b])
            total += len(hits)
            for s, la, lb in hits:
                detail.append((s, f"{a}x{b}", la, lb))
        detail.sort(reverse=True)
        grand += total
        print(f"  {label}{pid}: {total} paraphrase-matched line-pair(s) >= {THRESH}"
              f"   ({len(bodies)} variations, {sum(len(x) for x in bodies.values())} body lines)")
        for s, pr, la, lb in detail[:14]:
            print(f"      {s:.2f} {pr}  {la[:64]}")
            print(f"           <-> {lb[:64]}")
    return grand

RUN = os.path.join("output", "daily", "2026-08-22")
ours = {}
for p in ["01","02","03","04","05","06"]:
    fp = os.path.join(RUN, f"pair_{p}_step10_final_package_enhanced.md")
    if os.path.isfile(fp): ours[p] = load(fp)

print("="*70)
print(f"PARAPHRASE-BLOCK DUPLICATION  (line token-set Jaccard >= {THRESH}, refrains excluded)")
print("="*70)
print("\nTHIS RUN (2026-08-22):")
g = run("P", ours)

ref = os.path.join("output","daily","2026-08-21","pair_04_step10_final_package_enhanced.md")
if os.path.isfile(ref):
    print("\nNEGATIVE CONTROL — 2026-08-21 pair 04 (independently measured as templated):")
    run(" ", {"04(08-21)": load(ref)})

# Negative control 2: CROSS-pair. Unrelated songs must score ~0 or the
# instrument is just detecting English.
print("\nNEGATIVE CONTROL 2 — CROSS-pair (unrelated songs; must be ~0):")
flat = {}
for p, vs in ours.items():
    for v, s in vs.items(): flat[f"P{p}{v}"] = body_lines(s)
cross_tot, worst = 0, []
keys = sorted(flat)
for a, b in itertools.combinations(keys, 2):
    if a[:3] == b[:3]: continue
    h = compare(flat[a], flat[b])
    cross_tot += len(h)
    for s, la, lb in h: worst.append((s, a, b, la, lb))
worst.sort(reverse=True)
n_cross_pairs = sum(1 for a,b in itertools.combinations(keys,2) if a[:3]!=b[:3])
print(f"  {cross_tot} matches across {n_cross_pairs} cross-pair comparisons"
      f"  ({cross_tot/n_cross_pairs:.2f} per comparison)")
for s,a,b,la,lb in worst[:6]:
    print(f"      {s:.2f} {a}x{b}  {la[:58]}")
    print(f"           <-> {lb[:58]}")
print(f"\nWITHIN-pair total: {g} across 36 comparisons ({g/36:.2f} per comparison)")
