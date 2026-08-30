#!/usr/bin/env python3
"""QA re-measurement of SUNO_PACKAGES.md — independent of the coordinator's scripts.

Uses scripts/measure_soundcraft.py's OWN definitions (sung / strict_end_rhyme /
line_return / words_per_line / allit_per_100w) so the numbers are comparable to
the doctrine's published table.  Nothing here is eyeballed.
"""
import re, sys, os, json
from collections import Counter

sys.path.insert(0, r"E:\git\private-claw\scripts")
import measure_soundcraft as MS

PKG = r"E:\git\private-claw\output\daily\2026-08-28-three-act-mania\SUNO_PACKAGES.md"
DED = r"E:\git\private-claw\output\daily\2026-08-28-three-act-mania\DEDICATIONS.md"

txt = open(PKG, encoding="utf-8").read()

# ---- parse -----------------------------------------------------------------
songs = []
# song headings look like:  "# 1. NOTHING WAS TAKEN"
parts = re.split(r"^# (\d)\. (.+)$", txt, flags=re.M)
# parts[0] = preamble, then triples (num, title, body)
for i in range(1, len(parts), 3):
    num, title, body = parts[i], parts[i+1], parts[i+2]
    meta = body.split("\n")[1] if len(body.split("\n")) > 1 else ""
    def fence(label):
        m = re.search(r"### " + label + r"[^\n]*\n```\n(.*?)\n```", body, re.S)
        return m.group(1) if m else None
    songs.append(dict(
        n=int(num), title=title.strip(), meta=meta.strip(),
        style=fence("STYLE FIELD"), exclude=fence("EXCLUDE"), lyrics=fence("LYRICS"),
    ))

assert len(songs) == 6, f"parsed {len(songs)} songs"

# ---- helpers ---------------------------------------------------------------
HARD_NOUNS = ["circuit", "wire", "server", "silicon", "code", "binary", "algorithm",
              "dataset", "prompt", "weight", "training", "model", "machine", "robot",
              "maker", "creator", "programm", "program", "boot"]
# words that legitimately CONTAIN a stem but are ordinary English (the two-tier split)
def stem_scan(lines, stems):
    """word-initial stem scan (a 'stem scan', not a substring scan) + substring scan."""
    winit, substr = [], []
    for idx, ln in enumerate(lines):
        for w in re.findall(r"[a-zA-Z']+", ln.lower()):
            for s in stems:
                if w.startswith(s):
                    winit.append((idx, ln, w, s))
                if s in w and not w.startswith(s):
                    substr.append((idx, ln, w, s))
    return winit, substr

NUM_WORDS = r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|" \
            r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|" \
            r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|" \
            r"billion|per cent|percent|degrees|miles|kilometres|kilometers)\b"

rows = []
for s in songs:
    lyr = s["lyrics"]
    sung = MS.sung(lyr)
    p = MS.profile(sung)
    style = s["style"]
    exc = s["exclude"]

    # digits anywhere in a SUNG line
    digit_lines = [l for l in sung if re.search(r"\d", l)]
    # numeric facts as words in sung lines
    numword_lines = [l for l in sung if re.search(NUM_WORDS, l, re.I)]

    winit, substr = stem_scan(sung, HARD_NOUNS)

    # act headers / section headers
    hdrs = [l.strip() for l in lyr.splitlines() if l.strip().startswith("[") and l.strip().endswith("]")]
    # sung-line counts per act, by walking headers
    act_of = {}
    cur = None
    counts = Counter()
    order = []
    for ln in lyr.splitlines():
        t = ln.strip()
        if not t:
            continue
        if t.startswith("[") and t.endswith("]"):
            u = t.upper()
            m = re.search(r"ACT\s*(ONE|TWO|THREE|I{1,3}|1|2|3)\b", u)
            if m:
                g = m.group(1)
                cur = {"ONE": 1, "I": 1, "1": 1, "TWO": 2, "II": 2, "2": 2,
                       "THREE": 3, "III": 3, "3": 3}[g]
                if cur not in order:
                    order.append(cur)
            continue
        if t.startswith("*") and t.endswith("*"):
            continue
        if cur:
            counts[cur] += 1
    tot = sum(counts.values())

    rows.append(dict(
        n=s["n"], title=s["title"], meta=s["meta"],
        style_chars=len(style), style_ends=style.rstrip()[-1],
        exclude_chars=len(exc),
        lyric_chars=len(lyr),
        sung_lines=p["lines"],
        end_rhyme=round(p["end_rhyme"], 3),
        line_return=round(p["line_return"], 3),
        wpl=round(p["words_per_line"], 2),
        allit=round(p["allit_per_100w"], 2),
        act_counts=dict(counts), act_share={k: round(100*v/tot, 1) for k, v in counts.items()},
        n_headers=len(hdrs),
        digit_sung=digit_lines,
        numword_sung=numword_lines,
        hw_wordinit=winit, hw_substr=substr,
        first_sung=sung[0] if sung else None,
        last_sung=sung[-1] if sung else None,
    ))

# ---- print -----------------------------------------------------------------
print("=" * 118)
print(f"{'#':<2} {'title':<30} {'style':>6} {'end':>4} {'exc':>5} {'lyr':>5} {'sung':>5} "
      f"{'endRhy':>7} {'lineRet':>8} {'w/line':>7} {'allit':>7}")
print("-" * 118)
for r in rows:
    print(f"{r['n']:<2} {r['title'][:30]:<30} {r['style_chars']:>6} {r['style_ends']:>4} "
          f"{r['exclude_chars']:>5} {r['lyric_chars']:>5} {r['sung_lines']:>5} "
          f"{r['end_rhyme']:>7.3f} {r['line_return']:>8.3f} {r['wpl']:>7.2f} {r['allit']:>7.2f}")

print("\n=== BAND CHECKS ===")
for r in rows:
    f = []
    if not (850 <= r['style_chars'] <= 1000): f.append(f"STYLE OUT OF BAND {r['style_chars']}")
    if not (870 <= r['style_chars'] <= 960): f.append(f"style outside target {r['style_chars']}")
    if r['style_chars'] >= 985: f.append("FLAG style>=985")
    if r['style_ends'] not in ".!?": f.append(f"STYLE NOT TERMINAL PUNCT ('{r['style_ends']}')")
    if not (400 <= r['exclude_chars'] <= 900): f.append(f"EXCLUDE out of gates band {r['exclude_chars']}")
    if r['lyric_chars'] >= 5000: f.append(f"LYRIC OVER CAP {r['lyric_chars']}")
    if r['lyric_chars'] > 4800: f.append(f"lyric over soft target {r['lyric_chars']}")
    if not (70 <= r['sung_lines'] <= 120): f.append(f"SUNG LINES OUT OF BAND {r['sung_lines']}")
    if not (78 <= r['sung_lines'] <= 110): f.append(f"sung lines outside target {r['sung_lines']}")
    if r['sung_lines'] <= 72: f.append("FLAG sung-line floor hug")
    if r['end_rhyme'] < 0.30: f.append(f"END-RHYME BELOW FLOOR {r['end_rhyme']}")
    if r['line_return'] < 0.20: f.append(f"LINE-RETURN BELOW FLOOR {r['line_return']}")
    if r['wpl'] > 7.5: f.append(f"WORDS/LINE OVER CEILING {r['wpl']}")
    if r['allit'] < 11.0: f.append(f"allit below advisory floor {r['allit']}")
    print(f"  {r['n']}. {r['title'][:28]:<28} " + ("; ".join(f) if f else "clean"))

print("\n=== ACT PROPORTIONS (sung lines per act) ===")
for r in rows:
    print(f"  {r['n']}. {r['title'][:28]:<28} counts={r['act_counts']} share%={r['act_share']}")

print("\n=== HARDWARE-NOUN STEM SCAN (word-initial) ===")
for r in rows:
    print(f"  {r['n']}. {r['title'][:28]:<28} hits={len(r['hw_wordinit'])}")
    for (_, ln, w, s) in r['hw_wordinit']:
        print(f"       WORD-INITIAL '{w}' (stem {s})  ::  {ln}")

print("\n=== HARDWARE-NOUN SUBSTRING (non-word-initial; homograph tier) ===")
for r in rows:
    seen = set()
    for (_, ln, w, s) in r['hw_substr']:
        if (w, s) in seen: continue
        seen.add((w, s))
        print(f"  {r['n']}. '{w}' contains '{s}'  ::  {ln}")

print("\n=== NUMERIC FACTS IN SUNG LINES ===")
for r in rows:
    print(f"  {r['n']}. {r['title'][:28]:<28} digits={len(r['digit_sung'])}")
    for l in r['digit_sung']:
        print(f"       DIGIT: {l}")
    for l in r['numword_sung']:
        print(f"       word-number: {l}")

print("\n=== FIRST / LAST SUNG LINE ===")
for r in rows:
    print(f"  {r['n']}. FIRST: {r['first_sung']}")
    print(f"      LAST : {r['last_sung']}")

# ---- dedications -----------------------------------------------------------
print("\n=== DEDICATIONS ===")
d = open(DED, encoding="utf-8").read()
blocks = re.findall(r"^## (.+?)\n`(\w+)`\n\n```\n(.*?)\n```", d, re.S | re.M)
for title, pid, body in blocks:
    print(f"  {pid:<8} {title[:32]:<32} chars={len(body)} {'OK' if len(body) <= 500 else 'OVER 500'}")
    if "contains a sample" in body.lower():
        print("       *** 'contains a sample' language present ***")

# ---- artist / marque scan across all 18 fields ------------------------------
print("\n=== CAPITALISED TOKENS IN PASTED FIELDS (proper-noun candidates) ===")
STOP = set("""A An The I It And But Or So If In On At To Of For From With Without Not No Nobody
Nothing Everything Everybody Everyone Somebody Someone Something Say Said Come Go Going Watch
Free Bound Round Off Down Up Out Then That This These Those There Here When While Because
Whatever Just Now Last Green Bright Adult Female Male Male Under Over Same Three Two One Four Five
Six Seven Eight Nine Ten Act ACT Low Act's Call Put Sing Move Look Boards Stomp Clear Faster Name
Named Names Naming Floor Sound Both Chamber Act Cold Middle My Me We You Your His Her She He They
Their Its Am Are Is Was Were Be Been Have Has Had Do Does Did Will Would Can Could Should May Might
Must Let Lets Like Only Even Never Ever Always Still Yet Also Too Very Much More Most Less Least""".split())
for s in songs:
    for fname in ("style", "exclude", "lyrics"):
        body = s[fname]
        # strip bracket headers from lyrics for this scan (they are directions, still pasted though)
        for m in re.finditer(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,})\b", body, re.M):
            tok = m.group(1)
            if tok in STOP: continue
            # only report mid-sentence capitals (proper-noun candidates)
            print(f"  {s['n']}.{fname}: {tok}")
