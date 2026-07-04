import { useMemo, useState } from "react";
import { KnobControl } from "../components";
import type {
  KnobDef, KnobSetSpec, KnobSummary, KnobValue, InvariantViolation,
} from "../types";

/**
 * Knobs panel (plan §8.2). Grouped numeric inputs + range sliders (KnobControl),
 * derived values live-computed and badged ƒ (from the compile response's
 * KnobSummary), invariant violations inline (block Run, NOT Compile), and
 * preset save / load / duplicate. Every scalar edit bubbles up an inline knob
 * override map so the parent re-calls /compile and everything moves at $0.
 *
 * The `spec` is the loaded preset definitions (KnobSetSpec) — the source of
 * ranges/exprs/descriptions. `summary` is the LIVE resolved view from the last
 * compile (resolved values + which are derived + invariant status). `overrides`
 * is the user's in-flight scalar edits layered over the preset baseline.
 */

// Group knobs into legible sections by name prefix / role (dense but calm).
function groupOf(name: string, def: KnobDef): string {
  if (def.expr) return "derived";
  if (name.startsWith("music_") || name.startsWith("sung_") || name.startsWith("suno_"))
    return "music gates";
  if (name.startsWith("image_")) return "image gates";
  if (name.startsWith("unique_line") || name.startsWith("ngram")) return "collapse (flag)";
  if (
    name === "n_pairs" || name === "n_variations_per_pair" || name === "n_concepts" ||
    name.startsWith("barbell") || name === "total_prompts"
  ) return "portfolio shape";
  return "pipeline";
}

const GROUP_ORDER = [
  "portfolio shape", "derived", "pipeline",
  "music gates", "image gates", "collapse (flag)",
];

interface KnobsPanelProps {
  spec: KnobSetSpec;
  summary: KnobSummary | null;               // live resolved view from /compile
  violations: InvariantViolation[];          // from compile result (block Run)
  overrides: Record<string, number>;         // in-flight scalar edits
  onOverride: (name: string, value: number) => void;
  onResetOverrides: () => void;
  dirtyOverrides: boolean;
  presetName: string;
  presets: string[];
  onLoadPreset: (name: string) => void;
  onSavePreset: (name: string) => void;      // save current (overrides folded in)
  compiling: boolean;
}

export function FlowLabKnobsPanel({
  spec, summary, violations, overrides, onOverride, onResetOverrides,
  dirtyOverrides, presetName, presets, onLoadPreset, onSavePreset, compiling,
}: KnobsPanelProps) {
  const [saveAs, setSaveAs] = useState("");

  // Map violation exprs so a knob referenced by a failing invariant highlights.
  const violByExpr = useMemo(() => {
    const m = new Map<string, string>();
    for (const v of violations) m.set(v.expr, v.detail || "violates invariant");
    return m;
  }, [violations]);

  // Which knobs does a violating expr mention? (cheap substring match on names.)
  const knobViolation = (name: string): string | null => {
    for (const [expr, detail] of violByExpr) {
      if (new RegExp(`\\b${name}\\b`).test(expr)) return detail;
    }
    return null;
  };

  const resolved = summary?.resolved ?? {};
  const derivedSet = new Set(summary?.derived ?? []);

  const groups = useMemo(() => {
    const g: Record<string, string[]> = {};
    for (const [name, def] of Object.entries(spec.knobs)) {
      const key = groupOf(name, def);
      (g[key] ||= []).push(name);
    }
    return g;
  }, [spec]);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      {/* preset controls */}
      <div className="row" style={{ padding: "8px 10px", borderBottom: "1px solid var(--border)", gap: 6, flexWrap: "wrap", background: "var(--surface)" }}>
        <span className="faint mono" style={{ fontSize: 10.5 }}>preset</span>
        <select
          className="input mono"
          value={presets.includes(presetName) ? presetName : ""}
          onChange={(e) => onLoadPreset(e.target.value)}
          style={{ width: "auto", padding: "2px 6px", fontSize: 12 }}
        >
          {!presets.includes(presetName) && <option value="">{presetName} (edited)</option>}
          {presets.map((p) => <option key={p} value={p}>{p}</option>)}
        </select>
        {dirtyOverrides && <span className="badge dirty"><span className="d" />edited</span>}
        <button className="btn sm ghost" disabled={!dirtyOverrides} onClick={onResetOverrides}
                title="discard in-flight knob edits, back to the preset baseline">reset</button>
        <div className="spacer" />
        {compiling && <span className="spin" style={{ width: 12, height: 12 }} title="recompiling" />}
      </div>

      {/* save/duplicate */}
      <div className="row" style={{ padding: "6px 10px", borderBottom: "1px solid var(--border)", gap: 6 }}>
        <input
          className="input mono"
          placeholder="save as… (new preset name)"
          value={saveAs}
          onChange={(e) => setSaveAs(e.target.value)}
          style={{ fontSize: 11.5, padding: "3px 8px" }}
        />
        <button
          className="btn sm"
          disabled={!saveAs.trim()}
          onClick={() => { onSavePreset(saveAs.trim()); setSaveAs(""); }}
          title="write the current knobs (edits folded in) to a new preset"
        >save</button>
        <button
          className="btn sm ghost"
          disabled={!presetName}
          onClick={() => onSavePreset(`${presetName}-copy`)}
          title="duplicate this preset under <name>-copy"
        >dup</button>
      </div>

      {/* invariant banner (blocks Run, never Compile) */}
      {violations.length > 0 && (
        <div className="banner warn" style={{ flexDirection: "column", alignItems: "stretch", gap: 3 }}>
          <b style={{ color: "var(--sev-hard)" }}>✕ {violations.length} invariant violation(s) — Run is blocked (Compile still runs).</b>
          {violations.map((v, i) => (
            <div key={i} className="mono" style={{ fontSize: 10.5, color: "#ffd7d9" }}>{v.expr} — {v.detail}</div>
          ))}
        </div>
      )}

      <div className="scroll">
        {GROUP_ORDER.filter((k) => groups[k]?.length).map((groupKey) => (
          <div key={groupKey}>
            <div className="nav-section" style={{ margin: 0, position: "sticky", top: 0, background: "var(--surface-2)", zIndex: 1 }}>
              {groupKey}
            </div>
            {groups[groupKey].map((name) => {
              const def = spec.knobs[name];
              const isDerived = derivedSet.has(name) || Boolean(def.expr);
              const live = resolved[name] as KnobValue | undefined;
              // scalar-editable knobs carry an override that layers over the preset value.
              const editableScalar =
                !isDerived && typeof (def.value) === "number" && !Array.isArray(def.value);
              const shownValue: KnobValue | undefined =
                editableScalar && name in overrides ? overrides[name] : live;
              return (
                <KnobControl
                  key={name}
                  name={name}
                  def={def}
                  resolved={shownValue}
                  derived={isDerived}
                  violation={knobViolation(name)}
                  onChange={editableScalar ? (v) => onOverride(name, v) : undefined}
                />
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}
