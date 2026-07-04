import Monaco from "@monaco-editor/react";
import type { FlowStep, FlowSpec, OnFailPolicy } from "../types";

/**
 * Per-step drawer (plan §8.1): prompt source + overlay editor (Monaco), model
 * tier, gates multi-select, on_fail policy. Edits mutate an in-memory FlowStep
 * draft; the parent tracks the dirty draft and only persists on "New version".
 * Read-only when the flow is immutable (has ever run).
 */

const POLICIES: OnFailPolicy[] = ["block", "flag", "retry", "quarantine_pair"];

// A pragmatic gate vocabulary drawn from the canon flows (music/image/etc).
const GATE_VOCAB = [
  "g.step00", "g.tree_branches", "g.pairs_distinct", "g.portfolio",
  "g.music_prompt", "g.music_lyrics", "g.music_house_lexicon", "g.music_one_fact",
  "g.collapse", "g.human_subject", "g.andon",
  "g.image_scene", "g.image_words", "g.story_words",
];

interface StepDrawerProps {
  flow: FlowSpec;
  step: FlowStep;
  stageId: string;
  readOnly: boolean;
  onEdit: (patch: Partial<FlowStep>) => void;
  onClose: () => void;
}

export function FlowLabStepDrawer({ flow, step, stageId, readOnly, onEdit, onClose }: StepDrawerProps) {
  const tiers = Object.keys(flow.model_tiers || {});
  const gatesSet = new Set(step.gates || []);
  const allGates = Array.from(new Set([...GATE_VOCAB, ...(step.gates || [])])).sort();

  const toggleGate = (g: string) => {
    const next = new Set(step.gates || []);
    if (next.has(g)) next.delete(g); else next.add(g);
    onEdit({ gates: Array.from(next) });
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      <div className="row" style={{ padding: "8px 12px", borderBottom: "1px solid var(--border)", background: "var(--surface)", gap: 8 }}>
        <b className="mono">step {step.id}</b>
        {step.kind === "validator" && <span className="chip prose">validator</span>}
        {readOnly && <span className="badge">read-only</span>}
        <div className="spacer" />
        <button className="btn sm ghost" onClick={onClose}>close ✕</button>
      </div>

      <div className="scroll" style={{ padding: 12, display: "flex", flexDirection: "column", gap: 12 }}>
        <label className="col">
          <span className="faint mono lbl">title</span>
          <input className="input" value={step.title ?? ""} disabled={readOnly}
                 onChange={(e) => onEdit({ title: e.target.value })} />
        </label>

        <div className="row" style={{ gap: 12, alignItems: "flex-start", flexWrap: "wrap" }}>
          <label className="col" style={{ minWidth: 160 }}>
            <span className="faint mono lbl">model tier</span>
            <select className="input mono" value={step.tier ?? ""} disabled={readOnly || step.kind === "validator"}
                    onChange={(e) => onEdit({ tier: e.target.value || null })} style={{ padding: "4px 8px" }}>
              <option value="">(none)</option>
              {tiers.map((t) => {
                const mt = flow.model_tiers[t];
                return <option key={t} value={t}>{t} · {mt.model}</option>;
              })}
            </select>
          </label>

          <label className="col" style={{ minWidth: 160 }}>
            <span className="faint mono lbl">on_fail policy</span>
            <select className="input mono" value={step.on_fail?.policy ?? "retry"} disabled={readOnly}
                    onChange={(e) => onEdit({ on_fail: { ...step.on_fail, policy: e.target.value as OnFailPolicy } })}
                    style={{ padding: "4px 8px" }}>
              {POLICIES.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </label>

          {(step.on_fail?.policy === "retry") && (
            <label className="col" style={{ width: 120 }}>
              <span className="faint mono lbl">max_attempts</span>
              <input className="input mono" type="number" min={1} max={5}
                     value={step.on_fail?.max_attempts ?? 3} disabled={readOnly}
                     onChange={(e) => onEdit({ on_fail: { policy: "retry", max_attempts: Number(e.target.value) } })} />
            </label>
          )}
        </div>

        <label className="col">
          <span className="faint mono lbl">prompt source</span>
          <input className="input mono" value={step.prompt ?? ""} disabled={readOnly}
                 placeholder={step.kind === "validator" ? "(validator — no prompt)" : "skills/…/steps/NN_….md"}
                 onChange={(e) => onEdit({ prompt: e.target.value || null })} style={{ fontSize: 11.5 }} />
        </label>

        {step.kind === "validator" && (
          <label className="col">
            <span className="faint mono lbl">validator script</span>
            <input className="input mono" value={step.run ?? ""} disabled={readOnly}
                   placeholder="scripts/….py"
                   onChange={(e) => onEdit({ run: e.target.value || null })} style={{ fontSize: 11.5 }} />
          </label>
        )}

        {/* gates multi-select */}
        <div className="col">
          <span className="faint mono lbl">gates</span>
          <div className="row wrap" style={{ gap: 5, marginTop: 3 }}>
            {allGates.map((g) => (
              <button key={g}
                      className={`chip ${gatesSet.has(g) ? "hard" : "prose"}`}
                      disabled={readOnly}
                      onClick={() => toggleGate(g)}
                      style={{ opacity: readOnly ? 0.6 : 1 }}>
                <span className="d" />{g}
              </button>
            ))}
          </div>
        </div>

        {/* overlay editor (Monaco) — prompt tweak layered over the source file */}
        {step.kind !== "validator" && (
          <div className="col" style={{ minHeight: 200 }}>
            <span className="faint mono lbl">overlay (prompt patch, applied over the source)</span>
            <div style={{ border: "1px solid var(--border)", borderRadius: "var(--r-sm)", overflow: "hidden", height: 200, marginTop: 3 }}>
              <Monaco
                language="markdown"
                value={step.overlay ?? ""}
                theme="lofn-dark"
                onChange={(v) => onEdit({ overlay: v ?? null })}
                beforeMount={(m) => {
                  m.editor.defineTheme("lofn-dark", {
                    base: "vs-dark", inherit: true, rules: [],
                    colors: { "editor.background": "#15171e", "editorGutter.background": "#15171e" },
                  });
                }}
                options={{
                  fontSize: 12, fontFamily: "ui-monospace, Cascadia Code, Consolas, monospace",
                  minimap: { enabled: false }, wordWrap: "on", readOnly,
                  scrollBeyondLastLine: false, tabSize: 2, padding: { top: 6 },
                  lineNumbers: "off", folding: false,
                }}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
