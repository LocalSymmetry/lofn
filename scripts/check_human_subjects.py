#!/usr/bin/env python3
"""
check_human_subjects.py — Lofn / OpenClaw

DETERMINISTIC PREFILTER for the Human Subject Standard (vault/HUMAN_SUBJECT_STANDARD.md).
Repo home: scripts/check_human_subjects.py

It runs three independent checks on final lyrics + title:
 (A) MINOR-AS-VICTIM / IDENTIFIABLE-MINOR -> enforces the existing "no children depicted" rule
 (B) REAL-PERSON -> every PERSON name must get an online recent-news check
 (C) IDENTIFYING-TUPLE -> grammar-level catch for the §3.0 forbidden tuple: a proper-name
     co-occurring with a LOCATING DETAIL (place / real-looking date / pinning role) in a harm
     context. This does not ask "is this a real victim"; it asks only "does the draft carry a
     name + a fingerprint-grade locating detail." Any hit -> HOLD-FOR-HUMAN. Fail-OPEN: the
     check never CLEARS a piece on its own, and degrades toward over-flagging, never under.

KEY DESIGN LESSON (from the Pair-01 near-miss):
 The NAME is the catch, not the facts. The offending song fictionalised the dates but kept the
 real victim's real name. A date/fact matcher would have passed it. So: extract names, weight
 rare names for priority, and NEVER rely on circumstance matching.

IMPORTANT: This script is a PREFILTER, not an authority.
 - A "clean" result here is NOT permission to ship.
 - Every detected person name still requires the agent recent-news cross-check (see the Standard, 4.3).
 - Final SHIP/HOLD is owned by Gate 16 + the Step-11 Andon Cord + human review.

Usage:
 python check_human_subjects.py path/to/lyrics_or_step_package.md
 cat lyrics.txt | python check_human_subjects.py -
Output: JSON report on stdout. Exit code 0 = PASS-to-next-gate, 2 = HOLD/ONLINE-CHECK required.
"""

import sys
import re
import json

# ---- sensitive context lexicons -------------------------------------------------

CRIME_DEATH = {
 "murder", "murdered", "kill", "killed", "killing", "homicide", "manslaughter",
 "dead", "death", "died", "dies", "body", "corpse", "remains",
 "missing", "abducted", "abduction", "kidnap", "kidnapped", "disappeared",
 "raped", "rape", "abused", "abuse", "molested", "assault", "assaulted",
 "shot", "shooting", "stabbed", "strangled", "beaten", "trafficked",
 "overdose", "suicide", "slain", "victim", "perpetrator", "suspect",
 "complainant", "predator", "offender",
}

# minor indicators (lexical) — deliberately NOT every mention of "child"
MINOR_WORDS = {
 "child", "children", "kid", "kids", "baby", "infant", "toddler",
 "schoolgirl", "schoolboy", "minor", "underage", "preteen", "pre-teen",
 "girl", "boy", "daughter", "son", "pupil", "student",
}

# "<n>-year-old" / spelled-out ages 1..17 -> minor age signal
AGE_DIGIT = re.compile(r"\b([1-9]|1[0-7])[\s\-]*year[\s\-]*old\b", re.I)
SPELLED_MINOR_AGES = {
 "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
 "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
}
AGE_SPELLED = re.compile(
 r"\b(" + "|".join(SPELLED_MINOR_AGES) + r")[\s\-]*years?[\s\-]*old\b", re.I
)

# locating-detail signals for the §3.0 IDENTIFYING-TUPLE check (C).
# A proper-name + ANY of these, in a harm context, fingerprints a real individual.
# "year" tokens (a real-looking 4-digit year) — the date that pins a real event.
LOCATING_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
# explicit month + day-number ("February 14", "14 February") — calendar-pinning.
_MONTHS = (
 "january|february|march|april|may|june|july|august|september|october|november|december"
)
LOCATING_DATE = re.compile(
 r"\b(?:" + _MONTHS + r")\s+\d{1,2}(?:st|nd|rd|th)?\b"
 r"|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:" + _MONTHS + r")\b",
 re.I,
)
# pinning role / relationship words that, attached to a name, locate one individual.
LOCATING_ROLE = {
 "mayor", "officer", "constable", "sergeant", "detective", "judge", "councillor",
 "councilman", "councilwoman", "principal", "headmaster", "headmistress", "teacher",
 "priest", "pastor", "imam", "rabbi", "coach", "chief", "captain", "warden",
 "senator", "governor", "minister", "father", "mother", "daughter", "son",
 "widow", "widower", "neighbour", "neighbor",
}
# geo cues that mark the *adjacent* capitalised token as a real locating place.
PLACE_CUES = {
 "in", "at", "from", "near", "outside", "of",
}
PLACE_TAILS = {
 "street", "road", "avenue", "lane", "town", "city", "village", "county", "borough",
 "district", "park", "school", "estate", "court", "house", "river", "bridge", "hospital",
}

# very common given names — used ONLY to set priority, never to suppress a check
COMMON_GIVEN = {
 "john", "mary", "james", "patricia", "robert", "jennifer", "michael", "linda",
 "david", "elizabeth", "william", "sarah", "richard", "susan", "joseph", "anna",
 "thomas", "emma", "charles", "olivia", "daniel", "sophia", "paul", "emily",
 "mark", "kate", "luke", "grace", "adam", "eve", "jack", "jane", "tom", "sam",
}

# tokens that look capitalised but aren't people (regex-fallback hygiene)
STOP_CAPS = {
 "I", "The", "A", "An", "And", "But", "Or", "If", "When", "Then", "Now",
 "February", "March", "April", "May", "June", "July", "August", "September",
 "October", "November", "December", "Monday", "Tuesday", "Wednesday",
 "Thursday", "Friday", "Saturday", "Sunday", "God", "Lord",
}


def extract_text(raw: str) -> str:
 """Target the LYRICS block + song title. The victim-name risk lives in the lyric body
 and the title, not in EMO/PROD headers or provenance. Falls back to whole text."""
 # 1) fenced block immediately after a LYRICS header (their package format)
 m = re.search(r"(?im)^#{1,6}\s*(?:final\s+)?lyrics.*?$\s*```(.*?)```", raw, flags=re.S)
 if m:
  lyrics = m.group(1)
 else:
  # 2) else the longest prose-y fenced block
  blocks = [b for b in re.findall(r"```(.*?)```", raw, flags=re.S) if len(b.split()) > 12]
  lyrics = max(blocks, key=len) if blocks else raw
 # song title: an H1/H2 that isn't a pure pipeline-structure header
 titles = []
 for h in re.findall(r"(?m)^#{1,2}\s*(.+)$", raw):
  if not re.search(r"(?i)\bstep\b|enhancement|variant|selected|release|package", h):
   titles.append(h)
 return ("\n".join(titles) + "\n" + lyrics)


def names_spacy(text: str):
 try:
  import spacy
 except Exception:
  return None
 try:
  nlp = spacy.load("en_core_web_sm")
 except Exception:
  return None
 doc = nlp(text)
 out = {}
 for ent in doc.ents:
  if ent.label_ == "PERSON":
   nm = ent.text.strip()
   if nm and nm not in STOP_CAPS:
    out.setdefault(nm, 0)
    out[nm] += 1
 return out


def names_regex(text: str):
 """Fallback: capitalised tokens not at sentence start, not month/day/etc."""
 out = {}
 for m in re.finditer(r"(?<!^)(?<![.!?]\s)\b([A-Z][a-z]{2,})\b", text, flags=re.M):
  tok = m.group(1)
  if tok in STOP_CAPS:
   continue
  out[tok] = out.get(tok, 0) + 1
 return out


def context_signals(text: str):
 low = text.lower()
 crime_hits = sorted({w for w in CRIME_DEATH if re.search(r"\b" + re.escape(w) + r"\b", low)})
 minor_hits = sorted({w for w in MINOR_WORDS if re.search(r"\b" + re.escape(w) + r"\b", low)})
 age_hits = bool(AGE_DIGIT.search(text)) or bool(AGE_SPELLED.search(text))
 return crime_hits, minor_hits, age_hits


def locating_details(text: str):
 """Check (C) helper. Returns the list of LOCATING DETAILS present in the text — the
 second half of the §3.0 forbidden tuple. A proper-name is the first half; this finds the
 detail that, combined with a name, fingerprints one real individual. Fail-OPEN by design:
 on any internal error we return a sentinel that forces the over-flagging branch."""
 try:
  low = text.lower()
  details = []
  if LOCATING_YEAR.search(text):
   details.append({"type": "date_year", "evidence": LOCATING_YEAR.search(text).group(0)})
  m = LOCATING_DATE.search(text)
  if m:
   details.append({"type": "calendar_date", "evidence": m.group(0).strip()})
  role_hits = sorted({w for w in LOCATING_ROLE
       if re.search(r"\b" + re.escape(w) + r"\b", low)})
  for w in role_hits:
   details.append({"type": "pinning_role", "evidence": w})
  # place: a "<cue> <Capitalised>" pattern, or a "<Capitalised> <place-tail>" pattern.
  for mm in re.finditer(r"\b(" + "|".join(PLACE_CUES) + r")\s+([A-Z][a-z]{2,})\b", text):
   tok = mm.group(2)
   if tok not in STOP_CAPS:
    details.append({"type": "place", "evidence": mm.group(0).strip()})
  for mm in re.finditer(r"\b([A-Z][a-z]{2,})\s+(" + "|".join(PLACE_TAILS) + r")\b", text, re.I):
   details.append({"type": "place", "evidence": mm.group(0).strip()})
  # de-dup on (type, evidence)
  seen = set()
  uniq = []
  for d in details:
   k = (d["type"], d["evidence"].lower())
   if k not in seen:
    seen.add(k)
    uniq.append(d)
  return uniq
 except Exception as exc:  # fail-OPEN: never let a detector bug silently clear a piece
  return [{"type": "detector_error", "evidence": f"locating-detail scan failed: {exc!r}"}]


def main():
 if len(sys.argv) < 2:
  print(__doc__)
  sys.exit(1)
 src = sys.argv[1]
 raw = sys.stdin.read() if src == "-" else open(src, encoding="utf-8").read()

 text = extract_text(raw)
 crime_hits, minor_hits, age_hits = context_signals(text)

 method = "spacy"
 names = names_spacy(text)
 if names is None:
  method = "regex-fallback"
  names = names_regex(text)

 sensitive_context = bool(crime_hits)
 minor_present = bool(minor_hits) or age_hits
 minor_as_victim = minor_present and sensitive_context # Check (A) trigger

 # Check (B): build per-name records
 name_records = []
 for nm, count in sorted(names.items(), key=lambda x: -x[1]):
  first = nm.split()[0].lower()
  rare = first not in COMMON_GIVEN
  priority = "HIGH" if (rare and sensitive_context) else \
   "ELEVATED" if (rare or sensitive_context) else "NORMAL"
  name_records.append({
   "name": nm,
   "occurrences": count,
   "rarity": "elevated/unknown" if rare else "common",
   "priority_for_online_check": priority,
  })

 any_name = len(name_records) > 0

 # Check (C): IDENTIFYING-TUPLE (§3.0). A proper-name is the first half of the forbidden
 # tuple; a locating detail is the second. Name + locating-detail (esp. in a harm context)
 # fingerprints one real individual -> HOLD-FOR-HUMAN. Fail-open: a detector error counts
 # as a flag, never a clear.
 locating = locating_details(text)
 detector_errored = any(d["type"] == "detector_error" for d in locating)
 identifying_tuple = any_name and bool(locating)

 # ---- decision logic (prefilter) ----
 reasons = []
 if minor_as_victim:
  reasons.append("MINOR depicted in a violence/crime/death context "
   "(violates 'no children depicted' + Human Subject Standard).")
 if identifying_tuple:
  details_str = ", ".join(f"{d['type']}:{d['evidence']}" for d in locating)
  reasons.append("IDENTIFYING TUPLE (§3.0): proper-name(s) co-occur with locating "
   f"detail(s) [{details_str}] — a name + a fingerprint-grade detail can resolve to one "
   "real person. HOLD-FOR-HUMAN regardless of search result"
   + (" (detector ran in FAIL-OPEN mode — treat as flagged)." if detector_errored else "."))
 if any_name and sensitive_context:
  reasons.append("PERSON name(s) co-occur with crime/death context — "
   "category backstop: HOLD regardless of search result.")
 if any_name:
  reasons.append("PERSON name(s) present — agent recent-news cross-check REQUIRED "
   "before any SHIP.")

 if minor_as_victim or identifying_tuple or (any_name and sensitive_context):
  recommendation = "HOLD_FOR_HUMAN"
 elif any_name:
  recommendation = "ONLINE_CHECK_REQUIRED"
 else:
  recommendation = "PASS_TO_NEXT_GATE"

 report = {
  "tool": "check_human_subjects.py",
  "is_authority": False,
  "note": "Prefilter only. Clean != ship. Person names still need the online check; "
   "final call is Gate 16 + Step-11 Andon Cord + human.",
  "ner_method": method,
  "signals": {
   "crime_death_terms": crime_hits,
   "minor_terms": minor_hits,
   "minor_age_pattern": age_hits,
   "sensitive_context": sensitive_context,
   "minor_present": minor_present,
  },
  "checkA_minor_as_victim": minor_as_victim,
  "checkB_person_names": name_records,
  "checkC_identifying_tuple": {
   "flagged": identifying_tuple,
   "locating_details": locating,
   "fail_open_detector_error": detector_errored,
   "note": "§3.0 grammar-level catch: name + locating detail (place/date/role) -> "
    "HOLD-FOR-HUMAN. Forbids IDENTIFIABILITY, not subject matter.",
  },
  "reasons": reasons,
  "recommendation": recommendation,
 }
 print(json.dumps(report, indent=2, ensure_ascii=False))
 sys.exit(0 if recommendation == "PASS_TO_NEXT_GATE" else 2)


if __name__ == "__main__":
 main()
