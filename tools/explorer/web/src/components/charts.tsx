// Self-contained chart primitives for the Explorer — no chart library, pure
// SVG/CSS, reading ONLY semantic + scale tokens so they re-ink for free on every
// theme (the Medium panel's "theme-agnostic by contract"). Rule of the house:
// no primitive here without a second consumer; if an encoding has one call site
// and could be a table, it stays a table.
//
// Primitives: SevMark · HypsometricKey · ArcFan · Dumbbell.
import type { CSSProperties } from "react";

export type Sev = "hard" | "flag" | "mixed" | "prose";

/** severity ladder, sky→horizon (pale zenith down to the line-stopping fracture). */
export const SEV_LADDER: { key: Sev; label: string }[] = [
  { key: "prose", label: "prose" },
  { key: "mixed", label: "mixed" },
  { key: "flag", label: "flag" },
  { key: "hard", label: "hard" },
];

export const sevVar = (s: Sev | string) => `var(--sev-${s})`;

/**
 * A severity mark that carries meaning by SHAPE as well as hue (deuteranope-safe):
 * hard = stop-block square · flag = triangle · mixed = diamond · prose = hollow.
 * Reusable anywhere a severity needs a glyph (coverage matrix, keys, inline).
 */
export function SevMark({ sev, size = 8, style }: { sev: Sev | string; size?: number; style?: CSSProperties }) {
  return <span className={`sev-mark ${sev}`} style={{ width: size, height: size, ...style }} aria-hidden />;
}

/**
 * THE HYPSOMETRIC KEY — the signature. A tint ladder that means severity, framed
 * by canon's pale brilliance at the top of the sky. `variant="mark"` is the app's
 * brand glyph (one form, every scale); `variant="legend"` is the Atlas legend.
 */
export function HypsometricKey({ variant = "legend" }: { variant?: "mark" | "legend" }) {
  const bands = SEV_LADDER;
  if (variant === "mark") {
    // tiny vertical strata — canon band (gilt) at the zenith, severities descending.
    return (
      <svg viewBox="0 0 12 14" width="100%" height="100%" role="img" aria-label="Lofn Explorer" style={{ display: "block" }}>
        <rect x="1.5" y="0.5" width="9" height="2.4" rx="0.6" fill="var(--canon-fill)" stroke="var(--canon-edge)" strokeWidth="0.6" />
        {bands.map((b, i) => (
          <rect key={b.key} x="1.5" y={3.4 + i * 2.5} width="9" height="2.1" rx="0.6" fill={sevVar(b.key)} />
        ))}
      </svg>
    );
  }
  // legend — a connected tint strip (canon + 4 severities) with labels + shapes.
  return (
    <div className="hyps-key" role="img" aria-label="severity key: canon, prose, mixed, flag, hard">
      <div className="hyps-strip">
        <span className="hyps-seg canon" style={{ background: "var(--canon-fill)", boxShadow: "inset 0 0 0 1px var(--canon-edge)" }} title="canon · immutable" />
        {bands.map((b) => (
          <span key={b.key} className="hyps-seg" style={{ background: sevVar(b.key) }} title={b.label} />
        ))}
      </div>
      <div className="hyps-labels">
        <span className="hyps-lab"><span className="sev-mark canon" /> canon</span>
        {bands.map((b) => (
          <span key={b.key} className="hyps-lab"><span className={`sev-mark ${b.key}`} /> {b.label}</span>
        ))}
      </div>
    </div>
  );
}

/**
 * ArcFan — quadratic arcs from one source to N targets (a 1→N fan-out). Built for
 * the Atlas dispatch (orchestrator freezes once, fans to 4 rails); each arc inks
 * in its target's rail hue. Its own fixed geometry — no coupling to scrolling.
 */
export function ArcFan({ source, targets, width = 176, rowH = 22 }: {
  source: string;
  targets: { label: string; color: string }[];
  width?: number;
  rowH?: number;
}) {
  const h = Math.max(targets.length * rowH, rowH);
  const x0 = 8, y0 = h / 2, x1 = width - 78, xc = (x0 + x1) / 2;
  return (
    <div className="arc-fan" style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <svg viewBox={`0 0 ${width} ${h}`} width={width} height={h} role="img" aria-label={`${source} dispatches to ${targets.length} rails`}>
        <circle cx={x0} cy={y0} r="3.5" fill="var(--rail-orchestrator)" />
        {targets.map((t, i) => {
          const y1 = rowH / 2 + i * rowH;
          return (
            <g key={t.label}>
              <path d={`M ${x0} ${y0} Q ${xc} ${y0} ${xc} ${(y0 + y1) / 2} Q ${xc} ${y1} ${x1} ${y1}`}
                fill="none" stroke={t.color} strokeWidth="1.4" opacity="0.85" />
              <circle cx={x1} cy={y1} r="2.6" fill={t.color} />
              <text x={x1 + 6} y={y1 + 3.2} fontSize="10.5" fill="var(--text-dim)" style={{ fontFamily: "var(--mono)" }}>{t.label}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/**
 * Dumbbell — one row: a mini-axis from lo→hi with two labeled dots and the gap
 * between them drawn to scale. Shows BOTH ends of a disagreement (drift's canon
 * vs restated; Compare's knob A vs B). Per-row normalized — mixed units never
 * share a scale. The exact numbers stay as text (charts are the overview).
 */
export function Dumbbell({ a, b, aLabel, bLabel, aTitle = "canonical", bTitle = "restated", width = 168 }: {
  a: number; b: number;
  aLabel?: string; bLabel?: string;
  aTitle?: string; bTitle?: string;
  width?: number;
}) {
  const lo = Math.min(a, b), hi = Math.max(a, b);
  const span = hi - lo || Math.max(Math.abs(hi), 1);
  const pad = span * 0.35 || 1;
  const dLo = lo - pad, dHi = hi + pad;
  const x = (v: number) => 6 + ((v - dLo) / (dHi - dLo || 1)) * (width - 12);
  const gap = a !== b;
  return (
    <div className="dumb" style={{ width }}>
      <svg viewBox={`0 0 ${width} 20`} width={width} height={20} role="img"
        aria-label={`${aTitle} ${a}, ${bTitle} ${b}`}>
        <line x1="6" y1="10" x2={width - 6} y2="10" stroke="var(--border)" strokeWidth="1" />
        {gap && <line x1={x(a)} y1="10" x2={x(b)} y2="10" stroke="var(--sev-flag)" strokeWidth="2.5" opacity="0.55" />}
        {/* canonical = the immutable dot (gilt); restated = the drifting dot (amber) */}
        <circle cx={x(a)} cy="10" r="4" fill="var(--canon-fill)" stroke="var(--canon-edge)" strokeWidth="1.5">
          <title>{aTitle}: {a}</title>
        </circle>
        <circle cx={x(b)} cy="10" r="3.5" fill={gap ? "var(--sev-flag)" : "var(--ok)"}>
          <title>{bTitle}: {b}</title>
        </circle>
      </svg>
      <div className="dumb-labels mono">
        <span title={aTitle} style={{ color: "var(--canon-edge)" }}>{aLabel ?? a}</span>
        {gap && <span className="dumb-arrow" style={{ color: "var(--sev-flag)" }}>
          {b > a ? "▲" : "▼"}{Math.abs(b - a)}
        </span>}
        <span title={bTitle} style={{ color: gap ? "var(--sev-flag)" : "var(--text-dim)" }}>{bLabel ?? b}</span>
      </div>
    </div>
  );
}

/** sequential-ramp color for the i-th of n ranked segments (deepest = biggest). */
const seqColor = (i: number) => `var(--seq-${Math.max(1, 5 - i)})`;

/**
 * StackedStrip — a horizontal part-to-whole bar (cost composition; any share).
 * Segments sort biggest-first and ink DOWN the sequential ramp (ordered by
 * lightness, not 11 categorical hues); the top few are direct-labeled — the eye
 * needs "these are most of the bill", not a legend. Explicitly NOT a waterfall:
 * spend is monotonic, so a signed waterfall would be theater.
 */
export function StackedStrip({ segments, max, labelTop = 3, height = 14, fmt = (n: number) => String(n) }: {
  segments: { key: string; value: number }[];
  max?: number;
  labelTop?: number;
  height?: number;
  fmt?: (n: number) => string;
}) {
  const sorted = segments.filter((s) => s.value > 0).sort((a, b) => b.value - a.value);
  const total = sorted.reduce((s, x) => s + x.value, 0);
  const scaleMax = max ?? total;
  if (!sorted.length || scaleMax <= 0) return null;
  return (
    <div className="stack-strip">
      <div className="stack-bar" style={{ height }}>
        {sorted.map((s, i) => (
          <div key={s.key} className="stack-seg" title={`${s.key}: ${fmt(s.value)}`}
            style={{ width: `${(s.value / scaleMax) * 100}%`, background: seqColor(i) }} />
        ))}
      </div>
      <div className="stack-labels">
        {sorted.slice(0, labelTop).map((s, i) => (
          <span key={s.key} className="stack-lab">
            <span className="stack-sw" style={{ background: seqColor(i) }} />
            <span className="mono">{s.key}</span> <span className="faint mono">{fmt(s.value)}</span>
          </span>
        ))}
      </div>
    </div>
  );
}

/**
 * DotStrip — dots on a SHARED axis (latency by step; any distribution). The peak
 * (the stall) is highlighted amber — "which step stalls?" at a glance. A shared
 * `max` across the group keeps lanes comparable. Dots beat bars for outlier reads.
 */
export function DotStrip({ values, max, width = 168, height = 14, fmt = (n: number) => String(n) }: {
  values: { label: string; value: number }[];
  max?: number;
  width?: number;
  height?: number;
  fmt?: (n: number) => string;
}) {
  const peak = values.reduce((m, v) => Math.max(m, v.value), 0);
  const hi = max ?? peak;
  const cy = height / 2;
  const x = (v: number) => 4 + (hi > 0 ? v / hi : 0) * (width - 8);
  return (
    <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} role="img" aria-label="per-step timing">
      <line x1="4" y1={cy} x2={width - 4} y2={cy} stroke="var(--border)" strokeWidth="1" />
      {values.map((v, i) => {
        const isPeak = v.value === peak && peak > 0;
        return (
          <circle key={i} cx={x(v.value)} cy={cy} r={isPeak ? 3.6 : 2.4}
            fill={isPeak ? "var(--sev-flag)" : "var(--accent)"} opacity={v.value > 0 ? 0.9 : 0.3}>
            <title>{v.label}: {fmt(v.value)}</title>
          </circle>
        );
      })}
    </svg>
  );
}

/**
 * DivergingBar — a back-to-back bar around a center baseline (Compare churn:
 * removed ◀ | ▶ added). The app's ONE diverging encoding; balanced weight either
 * side of a neutral so a darker pole doesn't read as "more" regardless of length.
 */
export function DivergingBar({ neg, pos, max, width = 108, height = 12 }: {
  neg: number; pos: number; max?: number; width?: number; height?: number;
}) {
  const hi = max ?? Math.max(neg, pos, 1);
  const half = width / 2;
  const wNeg = hi > 0 ? (neg / hi) * (half - 1) : 0;
  const wPos = hi > 0 ? (pos / hi) * (half - 1) : 0;
  return (
    <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} role="img" aria-label={`removed ${neg}, added ${pos}`}>
      <rect x={half - wNeg} y="2" width={wNeg} height={height - 4} fill="var(--div-neg)" rx="1"><title>removed {neg}</title></rect>
      <rect x={half} y="2" width={wPos} height={height - 4} fill="var(--div-pos)" rx="1"><title>added {pos}</title></rect>
      <line x1={half} y1="0" x2={half} y2={height} stroke="var(--border-strong)" strokeWidth="1" />
    </svg>
  );
}
