#!/usr/bin/env python3
"""Modality hard-gate check: [Theme:]/[SONG FORM:] openers, EMO on every section,
   >=1 SFX cue, [Intro] with sung words, act-header coverage, pole occurrences."""
import re, sys
sys.path.insert(0, r"E:\git\private-claw\scripts")
import measure_soundcraft as MS

PKG = r"E:\git\private-claw\output\daily\2026-08-28-three-act-mania\SUNO_PACKAGES.md"
txt = open(PKG, encoding="utf-8").read()
parts = re.split(r"^# (\d)\. (.+)$", txt, flags=re.M)
songs = []
for i in range(1, len(parts), 3):
    n, title, body = int(parts[i]), parts[i+1].strip(), parts[i+2]
    def fence(l):
        return re.search(r"### " + l + r"[^\n]*\n```\n(.*?)\n```", body, re.S).group(1)
    songs.append(dict(n=n, title=title, style=fence("STYLE FIELD"),
                      exclude=fence("EXCLUDE"), lyrics=fence("LYRICS")))

POLES = {1: "it works better there", 2: "say how far", 3: "the floor is already going",
         4: "come and run me", 5: "i can see you from here", 6: "the water does not know"}

for s in songs:
    L = s["lyrics"]
    hdrs = [l.strip() for l in L.splitlines() if l.strip().startswith("[") and l.strip().endswith("]")]
    sfx  = [l.strip() for l in L.splitlines() if l.strip().startswith("*") and l.strip().endswith("*")]
    first = L.splitlines()[0].strip()
    theme = bool(re.match(r"\[Theme:", first, re.I))
    songform = any(re.match(r"\[(SONG )?FORM:", h, re.I) for h in hdrs[:4])
    songform_exact = any(re.match(r"\[SONG FORM:", h, re.I) for h in hdrs[:4])
    # EMO coverage: headers that look like a SECTION (not Theme/FORM/Disc_/Pole)
    apparatus = re.compile(r"^\[(Theme|SONG FORM|FORM|Disc_|Pole)", re.I)
    sect = [h for h in hdrs if not apparatus.match(h)]
    no_emo = [h for h in sect if "EMO" not in h.upper()]
    intro = [h for h in hdrs if re.search(r"\[\s*intro", h, re.I)]
    # pole per act
    cur, peracte = None, {1: 0, 2: 0, 3: 0}
    pole = POLES[s["n"]]
    for ln in L.splitlines():
        t = ln.strip()
        if t.startswith("[") and t.endswith("]"):
            m = re.search(r"ACT\s*(ONE|TWO|THREE|I{1,3}|1|2|3)\b", t.upper())
            if m:
                cur = {"ONE":1,"I":1,"1":1,"TWO":2,"II":2,"2":2,"THREE":3,"III":3,"3":3}[m.group(1)]
            continue
        if t.startswith("*") and t.endswith("*"): continue
        if cur and pole in t.lower(): peracte[cur] += 1
    print(f"{s['n']}. {s['title']}")
    print(f"    [Theme:] opener .......... {'YES' if theme else '*** NO ***'}   (first line: {first[:70]})")
    print(f"    [SONG FORM:] present ..... {'YES(exact)' if songform_exact else ('YES(as [FORM:])' if songform else '*** NO ***')}")
    print(f"    section headers .......... {len(sect)}   missing EMO: {len(no_emo)} {no_emo if no_emo else ''}")
    print(f"    SFX cues ................. {len(sfx)} {sfx if sfx else '*** NONE ***'}")
    print(f"    [Intro] header ........... {intro if intro else 'none (correct)'}")
    print(f"    pole '{pole}' per act ..... I={peracte[1]} II={peracte[2]} III={peracte[3]}"
          f"   {'OK' if all(peracte.values()) else '*** MISSING FROM AN ACT ***'}")
    print()
