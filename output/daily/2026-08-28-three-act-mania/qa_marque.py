#!/usr/bin/env python3
"""Marque / artist / company scan across all 18 pasted fields, done honestly:
   - bracket section-headers are STRIPPED from the lyric before the proper-noun pass
     (they are directions, and they are full of EMO taxonomy words in Title Case)
   - EXCLUDE fields are scanned too, but a token found there is reported with the
     BAN flag so the reader can tell an instruction-to-imitate from an instruction-not-to.
   - a positive control proves the hardware-noun scanner actually fires.
"""
import re, sys
sys.path.insert(0, r"E:\git\private-claw\scripts")
import measure_soundcraft as MS

PKG = r"E:\git\private-claw\output\daily\2026-08-28-three-act-mania\SUNO_PACKAGES.md"
txt = open(PKG, encoding="utf-8").read()
parts = re.split(r"^# (\d)\. (.+)$", txt, flags=re.M)
songs = []
for i in range(1, len(parts), 3):
    num, title, body = parts[i], parts[i+1], parts[i+2]
    def fence(label):
        m = re.search(r"### " + label + r"[^\n]*\n```\n(.*?)\n```", body, re.S)
        return m.group(1)
    songs.append(dict(n=int(num), title=title.strip(), style=fence("STYLE FIELD"),
                      exclude=fence("EXCLUDE"), lyrics=fence("LYRICS")))

# --- POSITIVE CONTROL for the hardware-noun scanner --------------------------
HARD = ["circuit","wire","server","silicon","code","binary","algorithm","dataset",
        "prompt","weight","training","model","machine","robot","maker","creator",
        "programm","program","boot"]
def hw_scan(lines):
    out=[]
    for ln in lines:
        for w in re.findall(r"[a-zA-Z']+", ln.lower()):
            for s in HARD:
                if s in w: out.append((w,s,ln))
    return out
ctrl = ["My circuit hummed and the server booted.", "a wire in the machine, my maker's code"]
print("POSITIVE CONTROL (scanner must fire):", len(hw_scan(ctrl)), "hits ->",
      sorted({w for w,_,_ in hw_scan(ctrl)}))
neg = ["There's just a warmer room down the hall"]
print("NEGATIVE CONTROL (scanner must be silent):", len(hw_scan(neg)))
print()

# --- real scan ---------------------------------------------------------------
print("=== HARDWARE NOUNS, SUBSTRING SCAN, ALL SUNG LINES ===")
for s in songs:
    sung = MS.sung(s["lyrics"])
    h = hw_scan(sung)
    print(f"  {s['n']}. {s['title'][:30]:<30} sung={len(sung)}  hardware-noun hits={len(h)}")
    for w,st,ln in h: print(f"        {w} ({st}) :: {ln}")

# also scan the STYLE and EXCLUDE fields (not required by R5, reported for completeness)
print("\n=== HARDWARE NOUNS IN STYLE / EXCLUDE (not an R5 gate; reported) ===")
for s in songs:
    for f in ("style","exclude"):
        h = hw_scan([s[f]])
        for w,st,ln in h:
            print(f"  {s['n']}.{f}: '{w}' contains '{st}'")

# --- proper-noun / marque pass ------------------------------------------------
STOPWORDS = set("""a an the i it and but or so if in on at to of for from with without not no
nobody nothing everything everybody everyone somebody someone something say said come go going
watch free bound round off down up out then that this these those there here when while because
whatever just now last green bright adult female male under over same three two one four five six
seven eight nine ten act low call put sing move look boards stomp clear faster name named names
naming floor sound both chamber cold middle my me we you your his her she he they their its am are
is was were be been have has had do does did will would can could should may might must let lets
like only even never ever always still yet also too very much more most less least how what who
why where which whose whom another each every some any all both few many several other others such
own than as by into onto upon about after before during through under above below between among
""".split())

def propernouns(text, strip_brackets=False):
    if strip_brackets:
        text = re.sub(r"^\s*\[[^\]]*\]\s*$", "", text, flags=re.M)
        text = re.sub(r"^\s*\*[^*]*\*\s*$", "", text, flags=re.M)
    hits = []
    # split into sentences; a capitalised word that is NOT the first token of a
    # sentence/line is a proper-noun candidate
    for line in text.splitlines():
        toks = re.findall(r"[A-Za-z'’]+", line)
        for i, t in enumerate(toks):
            if i == 0: continue
            if not t[0].isupper(): continue
            if t.lower() in STOPWORDS: continue
            # skip words that follow terminal punctuation (sentence starts)
            m = re.search(re.escape(t), line)
            hits.append(t)
    return hits

print("\n=== PROPER-NOUN CANDIDATES (mid-line capitals; headers stripped from lyrics) ===")
for s in songs:
    for f, strip in (("style", False), ("exclude", False), ("lyrics", True)):
        # crude sentence-start filter
        body = s[f]
        if strip:
            body = re.sub(r"^\s*\[[^\]]*\]\s*$", "", body, flags=re.M)
            body = re.sub(r"^\s*\*[^*]*\*\s*$", "", body, flags=re.M)
        found = []
        for line in body.splitlines():
            # mask sentence starts
            masked = re.sub(r"(^|[.!?—-]\s+)([A-Z])", lambda m: m.group(1) + m.group(2).lower(), line)
            for t in re.findall(r"\b([A-Z][a-zA-Z'’]+)\b", masked):
                if t.lower() in STOPWORDS: continue
                found.append(t)
        if found:
            print(f"  {s['n']}.{f}: {sorted(set(found))}")

# --- explicit known-marque list ----------------------------------------------
MARQUES = ["hammond","moog","rhodes","wurlitzer","fender","gibson","marshall","roland","korg",
           "yamaha","steinway","leslie","mellotron","juno","jupiter","prophet","serge","buchla",
           "linn","808","909","akai","mpc","neve","ssl","auto-tune","autotune","melodyne",
           "spotify","suno","apple","google","amazon","tesla","meta","openai","nike","adidas",
           "davachi","winstons","coleman","beatles","beyonce","prince","bowie"]
print("\n=== EXPLICIT MARQUE / COMPANY / ARTIST TOKEN LIST, ALL 18 FIELDS ===")
any_hit = False
for s in songs:
    for f in ("style","exclude","lyrics"):
        low = s[f].lower()
        for m in MARQUES:
            if re.search(r"\b" + re.escape(m) + r"\b", low):
                any_hit = True
                ctx = re.search(r".{0,60}\b" + re.escape(m) + r"\b.{0,60}", low, re.S)
                print(f"  HIT  {s['n']}.{f}: '{m}'  ::  ...{ctx.group(0).strip()}...")
if not any_hit:
    print("  none")

# --- hard-pan / non-musical panning token scan --------------------------------
PAN = ["hard-pan","hard pan","hard-panned","hard panned","ping-pong","ping pong",
       "bouncing left-right","panned wide","stereo widening","auto-pan"]
print("\n=== PAN-TOKEN SCAN (field + whether used as a BAN) ===")
for s in songs:
    for f in ("style","exclude","lyrics"):
        low = s[f].lower()
        for m in PAN:
            if m in low:
                ban = (f == "exclude") or bool(re.search(r"\bno " + re.escape(m), low))
                print(f"  {s['n']}.{f}: '{m}'  ->  {'BAN (correct, not a violation)' if ban else '*** PRESCRIBED ***'}")
