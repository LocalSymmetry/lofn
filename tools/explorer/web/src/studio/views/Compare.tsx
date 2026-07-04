import { useEffect, useState } from "react";
import { studioApi } from "../api";
import type { RunSummary, JudgeResult } from "../types";
import { RunPicker } from "./Compare-RunPicker";
import { DiffCompare } from "./Compare-DiffCompare";
import { BlindJudge } from "./Compare-BlindJudge";
import { Promote } from "./Compare-Promote";

type Mode = "diff" | "blind" | "promote";

/**
 * Compare & Judge (plan §8.4). Pick two finished runs, then:
 *   • diff    — the OPEN comparison: aligned per-artifact diffs, a gate-report
 *               delta table, and cost/token/knob deltas (full provenance).
 *   • blind   — the BLIND judge: payload-only cards (provenance stripped, fail-
 *               closed), randomized sides, pick a winner + note → recorded to
 *               both runs. The OEC is the operator's taste, honest at n=1.
 *   • promote — carry winning values back to canon (preview → apply) + the
 *               structural Promotion Brief.
 *
 * One POST /judge fetches BOTH the compare and the blind set; the mode tabs just
 * switch which view of that payload is shown.
 */
export function Compare() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [a, setA] = useState<string | null>(null);
  const [b, setB] = useState<string | null>(null);
  const [seed, setSeed] = useState(0);
  const [mode, setMode] = useState<Mode>("diff");
  const [result, setResult] = useState<JudgeResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    studioApi.runs()
      .then((rs) => {
        setRuns(rs);
        // Convenience: preselect the two most recent runs (list is newest-first
        // in the store; guard either order).
        if (rs.length >= 2) { setA(rs[1].id); setB(rs[0].id); }
        else if (rs.length === 1) { setA(rs[0].id); }
      })
      .catch((e) => setError(e.message || String(e)));
  }, []);

  // Re-fetch the judge payload whenever A, B, or the blind seed changes.
  useEffect(() => {
    if (!a || !b) { setResult(null); return; }
    let live = true;
    setLoading(true); setError(null);
    studioApi.judge({ run_a: a, run_b: b, seed })
      .then((r) => { if (live) setResult(r); })
      .catch((e) => { if (live) { setError(e.message || String(e)); setResult(null); } })
      .finally(() => { if (live) setLoading(false); });
    return () => { live = false; };
  }, [a, b, seed]);

  const swap = () => { const t = a; setA(b); setB(t); };

  return (
    <div className="pad" style={{ display: "flex", flexDirection: "column", gap: 16, maxWidth: 1100 }}>
      <div className="card">
        <div className="card-bd" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div className="row" style={{ justifyContent: "space-between", flexWrap: "wrap", gap: 8 }}>
            <b>Compare &amp; Judge</b>
            {runs && <span className="faint mono" style={{ fontSize: 11 }}>{runs.length} run(s) available</span>}
          </div>
          {runs && (
            <RunPicker runs={runs} a={a} b={b} onA={setA} onB={setB} onSwap={swap} />
          )}
        </div>
      </div>

      {error && (
        <div className="banner err" style={{ borderRadius: "var(--r-sm)", padding: "8px 12px", fontSize: 12 }}>{error}</div>
      )}

      {(!a || !b) && !error && (
        <div className="empty" style={{ padding: 40 }}>Select two runs (A and B) to compare.</div>
      )}

      {a && b && (
        <>
          <div className="row" style={{ gap: 6 }}>
            <ModeTab id="diff" mode={mode} setMode={setMode} label="⇄ Diff compare" />
            <ModeTab id="blind" mode={mode} setMode={setMode} label="◑ Blind judge" />
            <ModeTab id="promote" mode={mode} setMode={setMode} label="↥ Promote" />
            {loading && <span className="spin" style={{ marginLeft: 8 }} />}
          </div>

          {result && mode === "diff" && <DiffCompare compare={result.compare} />}
          {result && mode === "blind" && (
            <BlindJudge blind={result.blind} runA={a} runB={b} onReseed={setSeed} />
          )}
          {result && mode === "promote" && (
            <Promote compare={result.compare} runA={a} runB={b} />
          )}
          {!result && loading && <div className="empty" style={{ padding: 40 }}><span className="spin" /></div>}
        </>
      )}
    </div>
  );
}

function ModeTab({ id, mode, setMode, label }: { id: Mode; mode: Mode; setMode: (m: Mode) => void; label: string }) {
  return (
    <button
      className={`btn sm ${mode === id ? "primary" : ""}`}
      onClick={() => setMode(id)}
    >
      {label}
    </button>
  );
}
