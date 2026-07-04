import type { ReactNode } from "react";
import type { RunSummary } from "../types";

const money = (n?: number) =>
  n == null ? "—" : n >= 1 ? `$${n.toFixed(2)}` : `$${n.toFixed(3)}`;

const statusCls = (s: string) =>
  s === "complete" ? "ok" : s === "failed" || s === "cancelled" ? "err" : "warn";

/**
 * A / B run selector. Two labelled `<select>`s over the finished runs, with a
 * swap button. Only runs the compare can read are choosable (any status — the
 * backend just reads run.json + artifacts). The `hue` prop tints the side label
 * so A and B stay legible everywhere downstream.
 */
export function RunPicker({
  runs, a, b, onA, onB, onSwap,
}: {
  runs: RunSummary[];
  a: string | null;
  b: string | null;
  onA: (id: string) => void;
  onB: (id: string) => void;
  onSwap: () => void;
}) {
  const opt = (r: RunSummary) => (
    <option key={r.id} value={r.id}>
      {r.title} · {r.flow?.name}@{r.flow?.version} · {r.status} · {money(r.totals?.cost_usd)}
    </option>
  );

  return (
    <div className="row" style={{ gap: 12, alignItems: "flex-end", flexWrap: "wrap" }}>
      <SidePicker side="A" hue="var(--accent)" runs={runs} value={a} onChange={onA} opt={opt} />
      <button
        className="btn sm"
        title="Swap A and B"
        onClick={onSwap}
        disabled={!a && !b}
        style={{ marginBottom: 2 }}
      >
        ⇄ swap
      </button>
      <SidePicker side="B" hue="var(--accent-2)" runs={runs} value={b} onChange={onB} opt={opt} />
      {a && b && a === b && (
        <span className="chip flag" style={{ marginBottom: 3 }}>
          <span className="d" /> A and B are the same run
        </span>
      )}
    </div>
  );
}

function SidePicker({
  side, hue, runs, value, onChange, opt,
}: {
  side: string;
  hue: string;
  runs: RunSummary[];
  value: string | null;
  onChange: (id: string) => void;
  opt: (r: RunSummary) => ReactNode;
}) {
  const cur = runs.find((r) => r.id === value);
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 4, minWidth: 260 }}>
      <span className="row" style={{ gap: 6 }}>
        <span
          className="mono"
          style={{
            fontSize: 11, fontWeight: 700, color: hue,
            border: `1px solid ${hue}`, borderRadius: 4, padding: "0 6px",
          }}
        >
          {side}
        </span>
        {cur && (
          <span className={`badge ${statusCls(cur.status)}`}>
            <span className="d" />
            <span className="mono" style={{ fontSize: 10.5 }}>{cur.status}</span>
          </span>
        )}
      </span>
      <select
        className="input"
        value={value ?? ""}
        onChange={(e) => onChange(e.target.value)}
        style={{ minWidth: 260 }}
      >
        <option value="" disabled>select run {side}…</option>
        {runs.map(opt)}
      </select>
    </label>
  );
}
