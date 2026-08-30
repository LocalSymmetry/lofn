#!/usr/bin/env python3
"""check_brief_line_budget.py -- does the section architecture we PRESCRIBED admit
any song that clears the sung-line floor?

WHY THIS EXISTS. On 2026-08-26 (`2026-08-26-pantheon-rare-registers`) FIVE OF SIX
isolation briefs handed a pair an exact section skeleton -- named sections with
exact stanza lengths -- and never once evaluated what those numbers multiply out
to. Two pairs discovered before writing a line that the arithmetic does not close:

    P05: Refrain x6 + fields 8/6/4/2. The 20 field lines and the SIX refrains are
         fixed; the refrain LENGTH is not. At 7 lines that is 62 against a floor
         of 70. It had to spend a PERMISSION CLAUSE deviation to ship.
    P06: stanzas 6/5/4/3 = 18 fixed lines + 3 choruses. Clearing 70 needs a
         SEVENTEEN-line chorus. It reinterpreted every stanza count as PAIRS.

The brief states the floor in one section and a skeleton in another, and nothing
multiplies. ⛔ A constraint set that has to be adjudicated is a constraint set that
was not finished -- and this one was not even evaluated.

⭐ SO THE CHECK IS THE DECLARATION, NOT A PARSER. A prose matcher over six freely
worded briefs is exactly the brittle instrument that produced four false positives
on the day this was found. Instead, every isolation brief must carry ONE line:

    LINE BUDGET: 20 + 6*9 = 74

The reader evaluates the left side, checks it equals the stated total, and checks
the total lands inside the sung-line band. **The act of writing the line IS the
check** -- the coordinator cannot state a budget without doing the multiplication.

    python3 scripts/check_brief_line_budget.py <run-dir>

Exit 0 = every brief declares a budget that closes. Exit 3 = a brief is missing the
declaration, or its arithmetic is wrong, or its total cannot reach the floor.
Exit 2 = no briefs found, which is reported rather than passed.

Read-only. Writes nothing, renders nothing, spends nothing.
"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LABEL = "LINE BUDGET:"
ALLOWED = set("0123456789+*() ")


def band(repo_root):
    """The sung-line band, from vault/gates.yaml. Never hard-code a threshold."""
    path = os.path.join(repo_root, "vault", "gates.yaml")
    lo, hi = 70, 120
    try:
        for line in open(path, encoding="utf-8"):
            s = line.strip()
            if s.startswith("sung_lines:"):
                nums = [int(t) for t in s.replace("sung_lines:", "").strip(" []").split(",")]
                if len(nums) == 2:
                    lo, hi = nums
                break
    except OSError:
        pass
    return lo, hi


def evaluate(expr):
    """Evaluate a budget expression. Digits, + * and parentheses only -- nothing
    else is accepted, so there is no code path from a brief into this process."""
    if not expr or not set(expr) <= ALLOWED:
        return None
    try:
        return int(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        return None


def main():
    if len(sys.argv) != 2:
        print("usage: check_brief_line_budget.py <run-dir>")
        return 2
    run = sys.argv[1]
    if not os.path.isdir(run):
        print("FAIL: not a directory: " + run)
        return 2

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lo, hi = band(repo_root)

    briefs = sorted(f for f in os.listdir(run)
                    if f.startswith("pair_") and "step05_isolated_brief" in f)
    if not briefs:
        print("FAIL: no isolation briefs found -- nothing measured.")
        print("      A checker that measures nothing must not return success.")
        return 2

    print("BRIEF LINE BUDGET -- band " + str(lo) + "-" + str(hi) + " sung lines")
    bad = []
    for f in briefs:
        pair = f.split("_")[1]
        stated = None
        for line in open(os.path.join(run, f), encoding="utf-8"):
            if LABEL in line:
                stated = line.split(LABEL, 1)[1].strip().strip("*` ")
                break
        if stated is None:
            print("  P" + pair + ": MISSING -- no '" + LABEL + "' declaration")
            bad.append(pair)
            continue
        if "=" not in stated:
            print("  P" + pair + ": MALFORMED -- '" + stated + "' (want: <expr> = <total>)")
            bad.append(pair)
            continue
        expr, total = stated.rsplit("=", 1)
        got = evaluate(expr.strip())
        try:
            claimed = int(total.strip())
        except ValueError:
            got, claimed = got, None
        if got is None or claimed is None:
            print("  P" + pair + ": UNPARSEABLE -- '" + stated + "'")
            bad.append(pair)
        elif got != claimed:
            print("  P" + pair + ": ARITHMETIC WRONG -- " + expr.strip() + " is "
                  + str(got) + ", declared " + str(claimed))
            bad.append(pair)
        elif not (lo <= claimed <= hi):
            print("  P" + pair + ": OUT OF BAND -- " + str(claimed) + " not in "
                  + str(lo) + "-" + str(hi) + "  (" + stated + ")")
            bad.append(pair)
        else:
            print("  P" + pair + ": OK -- " + stated)

    print()
    if bad:
        print("  RESULT: " + str(len(bad)) + " of " + str(len(briefs))
              + " brief(s) do not close: " + ", ".join("P" + p for p in bad))
        print("  Fix the skeleton BEFORE dispatch. A pair that has to break your")
        print("  arithmetic before writing a line is a pair you sent out unfinished.")
        return 3
    print("  RESULT: all " + str(len(briefs)) + " briefs declare a budget that closes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
