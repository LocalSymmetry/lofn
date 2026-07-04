import type { FlowSpec, FlowStep, StepCompile } from "../types";

/**
 * The pipeline rail rendered FROM the FlowSpec (plan §8.1). One row per step,
 * grouped by stage; a fan-out stage shows a `foreach: pair × n_pairs` badge.
 * In EDIT mode each step gets insert / remove / reorder buttons (buttons, not
 * drag-physics). Selecting a step opens its drawer. Per-step compile badges
 * (resolved tokens + cost, and a diff dot when a baseline diff is non-empty)
 * ride each row so the signature "turn a knob, watch step-05 move" is legible.
 *
 * Gates guarding a step render as ONE severity-colored chip PER gate (its real
 * title + severity from the manifest; full title/note in the tooltip), matching
 * the Atlas — not an opaque count. Heavily-gated steps wrap the chips to a
 * second line under the title.
 */

/** id -> gate metadata, sourced from the v1 manifest (parent passes it in). */
export type GateMeta = Map<string, { title: string; severity: string; note?: string }>;

const money = (n: number | null) =>
  n == null ? "—" : n >= 1 ? `$${n.toFixed(2)}` : `$${n.toFixed(3)}`;

// Trim the modality prefix the Atlas also strips, so "Music prompt band" reads
// as "prompt band" in the tight rail. Full title stays in the tooltip.
const gateLabel = (title: string) => title.replace(/^(Music|Image)\s+/i, "");

function GateChips({ ids, gateMeta }: { ids: string[]; gateMeta: GateMeta }) {
  return (
    <div className="row wrap" style={{ gap: 4, rowGap: 3 }}>
      {ids.map((id) => {
        const g = gateMeta.get(id);
        const sev = g?.severity || "prose";
        const label = g ? gateLabel(g.title) : id;
        const tip = g ? (g.note ? `${g.title} — ${g.note}` : g.title) : id;
        return (
          <span key={id} className={`chip ${sev}`} title={tip}
                style={{ padding: "0 6px", maxWidth: 150 }}>
            <span className="d" />
            <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{label}</span>
          </span>
        );
      })}
    </div>
  );
}

interface RailProps {
  flow: FlowSpec;
  nPairs: number;
  selected: string | null;
  editMode: boolean;
  readOnly: boolean;
  compileByStep: Map<string, StepCompile>;
  gateMeta: GateMeta;
  onSelect: (stepId: string) => void;
  onInsertAfter: (stageId: string, stepId: string) => void;
  onRemove: (stageId: string, stepId: string) => void;
  onMove: (stageId: string, stepId: string, dir: -1 | 1) => void;
}

export function FlowLabPipelineRail({
  flow, nPairs, selected, editMode, readOnly, compileByStep, gateMeta,
  onSelect, onInsertAfter, onRemove, onMove,
}: RailProps) {
  return (
    <div className="scroll" style={{ padding: "6px 0" }}>
      {flow.stages.map((stage) => {
        const isFanout = stage.foreach === "pair";
        const steps = stage.steps || [];
        return (
          <div key={stage.id} style={{ marginBottom: 8 }}>
            <div className="row" style={{ padding: "4px 12px", gap: 8 }}>
              <span className="mono" style={{ fontSize: 11, color: "var(--rail-orchestrator)" }}>▤ {stage.id}</span>
              {isFanout && (
                <span className="chip mixed" title={`this stage runs once per pair (foreach: pair) × ${nPairs} pairs`}>
                  <span className="d" />foreach: pair × {nPairs}
                </span>
              )}
              {stage.concurrency != null && (
                <span className="faint mono" style={{ fontSize: 10 }}>‖ {String(stage.concurrency)}</span>
              )}
            </div>

            {steps.map((step, i) => {
              const sc = compileByStep.get(step.id);
              const active = selected === step.id;
              const hasDiff = Boolean(sc?.diff && sc.diff.trim());
              const gates = step.gates ?? [];
              return (
                <div
                  key={step.id}
                  className="rail-step"
                  onClick={() => onSelect(step.id)}
                  style={{
                    margin: "1px 8px", padding: "6px 10px", cursor: "pointer",
                    borderRadius: "var(--r-sm)",
                    background: active ? "var(--surface-3)" : "transparent",
                    border: active ? "1px solid var(--border-strong)" : "1px solid transparent",
                  }}
                >
                  <div className="row" style={{ gap: 8, alignItems: "center" }}>
                    <span className="mono" style={{ fontSize: 12, minWidth: 34, color: active ? "var(--text)" : "var(--text-dim)" }}>
                      {step.id}
                    </span>
                    <span style={{ flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {step.title || (step.kind === "validator" ? "(validator)" : step.prompt) || step.id}
                    </span>

                    {step.kind === "validator" && <span className="chip prose" style={{ padding: "0 5px" }}>val</span>}
                    {step.overlay && <span className="chip flag" style={{ padding: "0 5px" }} title="has an overlay patch">ov</span>}
                    {hasDiff && <span className="d-dot" title="resolved prompt differs from baseline" />}

                    {sc && (
                      <span className="faint mono" style={{ fontSize: 10, minWidth: 96, textAlign: "right" }}
                            title={sc.cost_note}>
                        {sc.est_tokens_in}t · {money(sc.cost_estimate)}
                      </span>
                    )}

                    {editMode && !readOnly && (
                      <span className="row" style={{ gap: 2 }} onClick={(e) => e.stopPropagation()}>
                        <button className="btn sm ghost" title="move up" disabled={i === 0}
                                onClick={() => onMove(stage.id, step.id, -1)}>↑</button>
                        <button className="btn sm ghost" title="move down" disabled={i === steps.length - 1}
                                onClick={() => onMove(stage.id, step.id, 1)}>↓</button>
                        <button className="btn sm ghost" title="insert a step after this one"
                                onClick={() => onInsertAfter(stage.id, step.id)}>＋</button>
                        <button className="btn sm ghost" title="remove this step"
                                style={{ color: "var(--sev-hard)" }}
                                onClick={() => onRemove(stage.id, step.id)}>✕</button>
                      </span>
                    )}
                  </div>

                  {gates.length > 0 && (
                    <div style={{ paddingLeft: 42, marginTop: 3 }}>
                      <GateChips ids={gates} gateMeta={gateMeta} />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        );
      })}
    </div>
  );
}

export function nextStepId(flow: FlowSpec): string {
  // Find an unused numeric-ish id for a freshly inserted step.
  const used = new Set<string>();
  for (const s of flow.stages) for (const st of s.steps || []) used.add(st.id);
  let n = 1;
  while (used.has(`new-${n}`)) n += 1;
  return `new-${n}`;
}

export function emptyStep(id: string): FlowStep {
  return {
    id, title: "New step", kind: "llm", prompt: null, overlay: null,
    tier: null, gates: [], on_fail: { policy: "retry", max_attempts: 3 },
  };
}
