import { useState } from "react";
import { UnifiedDiff } from "../components";
import { DivergingBar, Dumbbell } from "../../components/charts";
import type {
  CompareResult, CompareArtifact, GateCounts, CompareCostDelta,
} from "../types";

const money = (n: number) =>
  Math.abs(n) >= 1 ? `$${n.toFixed(2)}` : `$${n.toFixed(4)}`;

const signed = (n: number, fmt: (x: number) => string) =>
  (n > 0 ? "+" : n < 0 ? "−" : "±") + fmt(Math.abs(n));

const deltaColor = (n: number) =>
  n > 0 ? "var(--sev-flag)" : n < 0 ? "var(--ok)" : "var(--text-faint)";

/**
 * OPEN comparison surface (plan §8.4). Full provenance — cost/token/knob deltas,
 * a per-artifact gate-report delta table, and aligned unified diffs (reusing the
 * DiffPanel coloring via UnifiedDiff). This is the honest side-by-side; the blind
 * judge is a separate mode.
 */
export function DiffCompare({ compare }: { compare: CompareResult }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <CostDeltaCard cost={compare.cost_delta} />
      <KnobDeltaCard delta={compare.knob_delta} />
      <ArtifactDeltaTable artifacts={compare.artifacts} />
    </div>
  );
}

function CostDeltaCard({ cost }: { cost: CompareCostDelta }) {
  const cells: { label: string; a: string; b: string; delta: string; dc: string }[] = [
    {
      label: "cost (USD)",
      a: money(cost.a_cost_usd), b: money(cost.b_cost_usd),
      delta: signed(cost.delta_cost_usd, money), dc: deltaColor(cost.delta_cost_usd),
    },
    {
      label: "tokens in",
      a: cost.a_tokens_in.toLocaleString(), b: cost.b_tokens_in.toLocaleString(),
      delta: signed(cost.delta_tokens_in, (x) => x.toLocaleString()), dc: deltaColor(cost.delta_tokens_in),
    },
    {
      label: "tokens out",
      a: cost.a_tokens_out.toLocaleString(), b: cost.b_tokens_out.toLocaleString(),
      delta: signed(cost.delta_tokens_out, (x) => x.toLocaleString()), dc: deltaColor(cost.delta_tokens_out),
    },
  ];
  return (
    <div className="card">
      <div className="card-hd">Cost &amp; token delta</div>
      <div className="card-bd" style={{ overflowX: "auto" }}>
        <table className="grid" style={{ minWidth: 420 }}>
          <thead>
            <tr>
              <th></th>
              <th style={{ color: "var(--accent)" }}>A</th>
              <th style={{ color: "var(--accent-2)" }}>B</th>
              <th>B − A</th>
            </tr>
          </thead>
          <tbody>
            {cells.map((c) => (
              <tr key={c.label}>
                <td className="faint">{c.label}</td>
                <td className="mono">{c.a}</td>
                <td className="mono">{c.b}</td>
                <td className="mono" style={{ color: c.dc, fontWeight: 600 }}>{c.delta}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function KnobDeltaCard({ delta }: { delta: CompareResult["knob_delta"] }) {
  return (
    <div className="card">
      <div className="card-hd">Knob delta <span className="faint mono" style={{ fontWeight: 400, fontSize: 11 }}>({delta.length})</span></div>
      <div className="card-bd" style={{ overflowX: "auto" }}>
        {delta.length === 0 ? (
          <p className="faint mono" style={{ fontSize: 12, margin: 0 }}>Both runs used identical knobs.</p>
        ) : (
          <table className="grid" style={{ minWidth: 420 }}>
            <thead>
              <tr>
                <th>knob</th>
                <th style={{ color: "var(--accent)" }}>A</th>
                <th style={{ color: "var(--accent-2)" }}>B</th>
                <th>gap</th>
              </tr>
            </thead>
            <tbody>
              {delta.map((d) => {
                const na = numOf(d.a), nb = numOf(d.b);
                return (
                  <tr key={d.knob}>
                    <td className="mono">{d.knob}</td>
                    <td className="mono">{fmtKnob(d.a)}</td>
                    <td className="mono">{fmtKnob(d.b)}</td>
                    <td>{na != null && nb != null && na !== nb
                      ? <Dumbbell a={na} b={nb} aTitle="run A" bTitle="run B" width={150} />
                      : <span className="faint">—</span>}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

function fmtKnob(v: unknown): string {
  if (v == null) return "—";
  if (Array.isArray(v)) return `[${v.join(", ")}]`;
  return String(v);
}

/** first finite number in a scalar knob value (for the gap dumbbell). */
function numOf(v: unknown): number | null {
  if (typeof v === "number") return Number.isFinite(v) ? v : null;
  if (typeof v === "string") { const m = v.match(/-?\d+(?:\.\d+)?/); return m ? parseFloat(m[0]) : null; }
  return null;
}

function GateChips({ g }: { g: GateCounts }) {
  return (
    <span className="row" style={{ gap: 4 }}>
      <span className="mono" style={{ color: "var(--ok)", fontSize: 11 }} title="pass">{g.pass}✓</span>
      <span className="mono" style={{ color: "var(--sev-flag)", fontSize: 11 }} title="flag">{g.flag}⚑</span>
      <span className="mono" style={{ color: "var(--sev-hard)", fontSize: 11 }} title="fail">{g.fail}✗</span>
    </span>
  );
}

function ArtifactDeltaTable({ artifacts }: { artifacts: CompareArtifact[] }) {
  const [open, setOpen] = useState<string | null>(null);
  // shared churn axis so "grew / shrank / gutted" is comparable across artifacts.
  const maxLines = Math.max(1, ...artifacts.map((a) => Math.max(a.diff.added, a.diff.removed)));
  return (
    <div className="card">
      <div className="card-hd">
        Artifact diffs &amp; gate delta{" "}
        <span className="faint mono" style={{ fontWeight: 400, fontSize: 11 }}>({artifacts.length})</span>
      </div>
      <div className="card-bd" style={{ padding: 0 }}>
        {artifacts.length === 0 ? (
          <p className="empty" style={{ padding: 24 }}>No aligned artifacts across these two runs.</p>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table className="grid" style={{ minWidth: 620 }}>
              <thead>
                <tr>
                  <th>artifact</th>
                  <th>presence</th>
                  <th>lines ±</th>
                  <th style={{ color: "var(--accent)" }}>gate A</th>
                  <th style={{ color: "var(--accent-2)" }}>gate B</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {artifacts.map((art) => {
                  const isOpen = open === art.artifact;
                  const only = !art.in_a ? "B only" : !art.in_b ? "A only" : "both";
                  return (
                    <>
                      <tr key={art.artifact}>
                        <td className="mono" style={{ fontSize: 11.5 }}>{art.artifact}</td>
                        <td>
                          {only === "both" ? (
                            <span className="faint mono" style={{ fontSize: 11 }}>both</span>
                          ) : (
                            <span className="chip flag"><span className="d" />{only}</span>
                          )}
                        </td>
                        <td style={{ fontSize: 11 }}>
                          {art.diff.identical ? (
                            <span className="faint mono">identical</span>
                          ) : (
                            <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                              <DivergingBar neg={art.diff.removed} pos={art.diff.added} max={maxLines} />
                              <span className="mono" style={{ fontSize: 10.5 }}>
                                <span style={{ color: "var(--div-pos)" }}>+{art.diff.added}</span>{" "}
                                <span style={{ color: "var(--div-neg)" }}>−{art.diff.removed}</span>
                              </span>
                            </div>
                          )}
                        </td>
                        <td><GateChips g={art.gate_delta.a} /></td>
                        <td><GateChips g={art.gate_delta.b} /></td>
                        <td>
                          <button
                            className="btn sm ghost"
                            disabled={art.diff.identical}
                            onClick={() => setOpen(isOpen ? null : art.artifact)}
                          >
                            {isOpen ? "hide" : "diff"}
                          </button>
                        </td>
                      </tr>
                      {isOpen && (
                        <tr key={art.artifact + "-diff"}>
                          <td colSpan={6} style={{ padding: "6px 10px 12px" }}>
                            <UnifiedDiff diff={art.diff.unified} />
                          </td>
                        </tr>
                      )}
                    </>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
