import type { KnobDef, KnobValue } from "../types";

interface KnobControlProps {
  name: string;
  def: KnobDef;
  /** The live resolved value (from KnobSummary.resolved) — for derived knobs
      this is the computed number shown read-only with the ƒ badge. */
  resolved?: KnobValue;
  derived?: boolean;
  /** Inline invariant/min_expr violation message; renders in the severity-hard
      register when present. */
  violation?: string | null;
  /** Edit a scalar knob's `value`. Omit to render read-only (derived / preset view). */
  onChange?: (value: number) => void;
  disabled?: boolean;
}

function asScalar(v: KnobValue | undefined): number | null {
  if (typeof v === "number") return v;
  if (Array.isArray(v)) return null;      // bands aren't slider-editable here
  return null;
}

function fmt(v: KnobValue | undefined): string {
  if (v == null) return "—";
  if (Array.isArray(v)) return `[${v[0]}, ${v[1]}]`;
  if (typeof v === "boolean") return v ? "true" : "false";
  return String(v);
}

/**
 * One knob: a numeric input + range slider for a scalar `value`, or a read-only
 * derived display (the computed value shown with a ƒ badge). Bands and booleans
 * render read-only in v1 (the slider path is scalar-only). An invariant or
 * min_expr violation renders inline in the hard-severity register.
 */
export function KnobControl({
  name, def, resolved, derived, violation, onChange, disabled,
}: KnobControlProps) {
  const isDerived = derived ?? Boolean(def.expr);
  const scalar = asScalar(resolved ?? def.value);
  const [min, max] = def.range ?? [0, 100];
  const editable = !isDerived && !disabled && onChange != null && scalar != null;

  return (
    <div className="card-bd" style={{ padding: "8px 12px", borderBottom: "1px solid var(--border)" }}>
      <div className="row" style={{ justifyContent: "space-between", gap: 8, alignItems: "baseline" }}>
        <span className="row" style={{ gap: 6, alignItems: "baseline" }}>
          <span className="mono" style={{ fontSize: 12.5 }}>{name}</span>
          {isDerived && (
            <span className="chip prose" title={def.expr ? `ƒ = ${def.expr}` : "derived"}
                  style={{ padding: "1px 6px", fontSize: 10 }}>ƒ</span>
          )}
          {def.gates_key && (
            <span className="faint mono" style={{ fontSize: 10 }} title="binds a vault/gates.yaml key">
              → {def.gates_key}
            </span>
          )}
        </span>
        {editable ? (
          <input
            className="input mono"
            type="number"
            value={scalar ?? 0}
            min={min}
            max={max}
            step={1}
            onChange={(e) => onChange!(Number(e.target.value))}
            style={{ width: 84, padding: "2px 6px", fontSize: 12, textAlign: "right" }}
          />
        ) : (
          <span className="mono" title={isDerived ? "derived (read-only)" : "read-only"}
                style={{ fontSize: 12.5, color: "var(--text-dim)" }}>
            {fmt(resolved ?? def.value)}
          </span>
        )}
      </div>

      {editable && (
        <input
          type="range"
          value={scalar ?? 0}
          min={min}
          max={max}
          step={1}
          onChange={(e) => onChange!(Number(e.target.value))}
          style={{ width: "100%", marginTop: 6, accentColor: "var(--accent)" }}
        />
      )}

      {def.desc && (
        <div className="faint" style={{ fontSize: 10.5, marginTop: 4 }}>{def.desc}</div>
      )}

      {violation && (
        <div className="mono" style={{ fontSize: 10.5, marginTop: 5, color: "var(--sev-hard)" }}>
          ✕ {violation}
        </div>
      )}
    </div>
  );
}
