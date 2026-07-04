interface CostBarProps {
  /** Dollars spent (or estimated) so far. */
  spend: number;
  /** Hard budget cap in dollars (no run without one — plan §13.4). */
  cap: number;
  /** Label prefix, e.g. "spend" (default) or "est". */
  label?: string;
  /** When true (unpriced models involved), the bar reads in tokens-only mode and
      shows a loud note that the dollar figure is incomplete. */
  unpriced?: boolean;
}

const money = (n: number) =>
  n >= 100 ? `$${n.toFixed(0)}` : n >= 1 ? `$${n.toFixed(2)}` : `$${n.toFixed(3)}`;

/**
 * Spend-vs-cap bar. Fills violet under budget, amber past ~85%, hard-red at/over
 * the cap (the run aborts on cap server-side; this makes the ceiling legible).
 */
export function CostBar({ spend, cap, label = "spend", unpriced }: CostBarProps) {
  const safeCap = cap > 0 ? cap : 0;
  const pct = safeCap > 0 ? Math.min(100, (spend / safeCap) * 100) : 0;
  const over = safeCap > 0 && spend >= safeCap;
  const near = !over && pct >= 85;

  const fill = over ? "var(--sev-hard)" : near ? "var(--sev-flag)" : "var(--accent)";

  return (
    <div style={{ width: "100%" }}>
      <div className="row" style={{ justifyContent: "space-between", fontSize: 11, marginBottom: 4 }}>
        <span className="faint mono">{label}</span>
        <span className="mono" style={{ color: over ? "var(--sev-hard)" : "var(--text-dim)" }}>
          {money(spend)} <span className="faint">/ {safeCap > 0 ? money(safeCap) : "no cap"}</span>
        </span>
      </div>
      <div style={{
        height: 6, borderRadius: 3, background: "var(--surface-3)",
        overflow: "hidden", border: "1px solid var(--border)",
      }}>
        <div style={{ height: "100%", width: `${pct}%`, background: fill, transition: "width .2s ease" }} />
      </div>
      {unpriced && (
        <div className="mono" style={{ fontSize: 10, marginTop: 3, color: "var(--sev-flag)" }}>
          some models unpriced — dollar figure incomplete (tokens counted loudly)
        </div>
      )}
    </div>
  );
}
