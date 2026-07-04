import { useMemo, useState } from "react";
import { studioApi } from "../api";
import type { BlindSet, Verdict } from "../types";

type Pick = "A" | "B" | "tie";

interface LocalVerdict {
  pick: Pick;
  note: string;
  recorded?: boolean;      // sent to POST /judge/verdict
  revealed?: boolean;      // de-blinded via the coordinator key
}

/**
 * BLIND JUDGE mode (plan §8.4). Payload-only cards from POST /judge (provenance
 * stripped + fail-closed on the backend), A/B sides randomized by the seed. The
 * operator picks a winner per matchup and (optionally) leaves a note; the pick is
 * de-blinded ONLY on demand via the coordinator key, and recorded to BOTH runs'
 * run.json via POST /judge/verdict.
 *
 * The OEC is the operator's TASTE — we do NOT fabricate a win-rate statistic at
 * n=1. The tally below is an honest count of picks, labelled as such.
 */
export function BlindJudge({
  blind, runA, runB, onReseed, onRecorded,
}: {
  blind: BlindSet;
  runA: string;
  runB: string;
  onReseed: (seed: number) => void;
  onRecorded?: () => void;
}) {
  const [verdicts, setVerdicts] = useState<Record<string, LocalVerdict>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const setPick = (mid: string, pick: Pick) =>
    setVerdicts((v) => ({ ...v, [mid]: { ...(v[mid] || { note: "" }), pick } }));
  const setNote = (mid: string, note: string) =>
    setVerdicts((v) => ({ ...v, [mid]: { ...(v[mid] || { pick: "tie" }), note } }));
  const reveal = (mid: string) =>
    setVerdicts((v) => ({ ...v, [mid]: { ...(v[mid] || { pick: "tie", note: "" }), revealed: true } }));

  async function record(mid: string) {
    const lv = verdicts[mid];
    if (!lv?.pick) return;
    const keyEntry = blind.key[mid];
    // De-blind the pick to a concrete run id (or "tie") for the recorded verdict.
    const winnerRun =
      lv.pick === "tie" ? "tie" : keyEntry ? keyEntry[lv.pick] : lv.pick;
    const verdict: Verdict = {
      matchup_id: mid,
      artifact: keyEntry?.artifact,
      winner: winnerRun,
      blind_side: lv.pick,                 // which visible side won, before de-blind
      note: lv.note?.trim() || undefined,
      seed: blind.seed,
      oec: "operator taste (n=1, not a statistic)",
    };
    setBusy(mid);
    setErr(null);
    try {
      await studioApi.judgeVerdict({ run_a: runA, run_b: runB, verdict });
      setVerdicts((v) => ({ ...v, [mid]: { ...v[mid], recorded: true, revealed: true } }));
      onRecorded?.();
    } catch (e: any) {
      setErr(e?.message || String(e));
    } finally {
      setBusy(null);
    }
  }

  const tally = useMemo(() => {
    // Honest pick count, de-blinded, over the picks made so far. NOT a rate.
    let a = 0, b = 0, tie = 0;
    for (const [mid, lv] of Object.entries(verdicts)) {
      if (!lv.pick) continue;
      if (lv.pick === "tie") { tie++; continue; }
      const k = blind.key[mid];
      const run = k ? k[lv.pick] : lv.pick;
      if (run === runA) a++;
      else if (run === runB) b++;
    }
    return { a, b, tie, made: a + b + tie };
  }, [verdicts, blind.key, runA, runB]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <div className="banner warn" style={{ borderRadius: "var(--r-sm)", padding: "8px 12px", fontSize: 12 }}>
        The OEC is <b>your taste</b>. Sides are randomized; provenance is stripped
        and the set fails closed. Picks are recorded honestly at <b>n=1</b> — this
        is not a win-rate statistic.
      </div>

      <div className="card">
        <div className="card-hd row" style={{ justifyContent: "space-between" }}>
          <span>
            Blind set{" "}
            <span className="faint mono" style={{ fontWeight: 400, fontSize: 11 }}>
              seed {blind.seed} · {blind.n_matchups} matchup(s) · {blind.n_dropped} dropped
            </span>
          </span>
          <span className="row" style={{ gap: 6 }}>
            <button className="btn sm" onClick={() => onReseed(blind.seed + 1)} title="Re-randomize A/B sides">
              ↻ reseed
            </button>
          </span>
        </div>
        <div className="card-bd" style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
          <span className="mono" style={{ fontSize: 12 }}>
            picks made: <b>{tally.made}</b> / {blind.n_matchups}
          </span>
          <span className="mono" style={{ fontSize: 12, color: "var(--accent)" }}>A: {tally.a}</span>
          <span className="mono" style={{ fontSize: 12, color: "var(--accent-2)" }}>B: {tally.b}</span>
          <span className="mono" style={{ fontSize: 12, color: "var(--text-faint)" }}>tie: {tally.tie}</span>
        </div>
      </div>

      {err && <div className="banner err" style={{ borderRadius: "var(--r-sm)", padding: "6px 10px", fontSize: 12 }}>{err}</div>}

      {blind.n_matchups === 0 && (
        <div className="empty" style={{ padding: 24 }}>
          No blind cards — every artifact failed the fail-closed payload extraction
          (too short or a residual provenance leak). Nothing to judge blindly here.
        </div>
      )}

      {blind.matchups.map((m) => {
        const lv = verdicts[m.matchup_id];
        const k = blind.key[m.matchup_id];
        return (
          <div key={m.matchup_id} className="card">
            <div className="card-hd row" style={{ justifyContent: "space-between" }}>
              <span className="mono">{m.matchup_id}</span>
              {lv?.revealed && k ? (
                <span className="row" style={{ gap: 8, fontSize: 11 }}>
                  <span className="faint mono">de-blinded:</span>
                  <span className="mono" style={{ color: "var(--accent)" }}>A = {k.A}</span>
                  <span className="mono" style={{ color: "var(--accent-2)" }}>B = {k.B}</span>
                  <span className="faint mono">· {k.artifact}</span>
                </span>
              ) : (
                <button className="btn sm ghost" onClick={() => reveal(m.matchup_id)} title="Show which run each side came from">
                  reveal
                </button>
              )}
            </div>
            <div className="card-bd">
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <PayloadCard side="A" hue="var(--accent)" text={m.A} picked={lv?.pick === "A"} onPick={() => setPick(m.matchup_id, "A")} />
                <PayloadCard side="B" hue="var(--accent-2)" text={m.B} picked={lv?.pick === "B"} onPick={() => setPick(m.matchup_id, "B")} />
              </div>
              <div className="row" style={{ gap: 8, marginTop: 10, flexWrap: "wrap" }}>
                <button
                  className={`btn sm ${lv?.pick === "tie" ? "primary" : ""}`}
                  onClick={() => setPick(m.matchup_id, "tie")}
                >
                  tie / can't tell
                </button>
                <input
                  className="input"
                  placeholder="note (why — optional)"
                  value={lv?.note ?? ""}
                  onChange={(e) => setNote(m.matchup_id, e.target.value)}
                  style={{ flex: 1, minWidth: 180 }}
                />
                <button
                  className="btn sm primary"
                  disabled={!lv?.pick || busy === m.matchup_id || lv?.recorded}
                  onClick={() => record(m.matchup_id)}
                >
                  {lv?.recorded ? "recorded ✓" : busy === m.matchup_id ? "recording…" : "record verdict"}
                </button>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function PayloadCard({
  side, hue, text, picked, onPick,
}: {
  side: string;
  hue: string;
  text: string;
  picked: boolean;
  onPick: () => void;
}) {
  return (
    <div
      style={{
        border: `1px solid ${picked ? hue : "var(--border)"}`,
        borderRadius: "var(--r-sm)",
        background: picked ? "color-mix(in srgb, " + hue + " 8%, var(--surface))" : "var(--surface)",
        display: "flex", flexDirection: "column",
      }}
    >
      <div className="row" style={{ justifyContent: "space-between", padding: "6px 10px", borderBottom: "1px solid var(--border)" }}>
        <span className="mono" style={{ fontWeight: 700, color: hue }}>{side}</span>
        <button className={`btn sm ${picked ? "primary" : ""}`} onClick={onPick}>
          {picked ? "picked ✓" : "pick"}
        </button>
      </div>
      <pre
        className="mono scroll"
        style={{
          margin: 0, padding: "8px 10px", fontSize: 11.5, lineHeight: 1.5,
          whiteSpace: "pre-wrap", wordBreak: "break-word", maxHeight: 360,
          color: "var(--text-dim)",
        }}
      >
        {text}
      </pre>
    </div>
  );
}
