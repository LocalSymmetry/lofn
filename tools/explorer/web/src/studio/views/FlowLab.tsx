import Monaco from "@monaco-editor/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { studioApi } from "../api";
import { useStore } from "../../store";
import { CostBar } from "../components";
import type {
  FlowSummary, FlowSpec, FlowStep, KnobSetSpec, CompileResult, StepCompile,
} from "../types";
import { FlowLabKnobsPanel } from "./FlowLab-KnobsPanel";
import { FlowLabStepDrawer } from "./FlowLab-StepDrawer";
import {
  FlowLabPipelineRail, nextStepId, emptyStep, type GateMeta,
} from "./FlowLab-PipelineRail";

/**
 * Flow Lab (plan §8.1 + §8.2). The flow + knobs authoring surface:
 *   · flow picker + version list (immutable flows are read-only — offer a fork)
 *   · the pipeline rail rendered FROM the FlowSpec, EDIT mode (insert/remove/
 *     reorder + a per-step drawer), a raw-YAML tab (schema-validated on fork)
 *   · the Knobs panel — every knob change re-calls /compile at $0 and moves the
 *     resolved prompt + derived values + gates + cost estimate together.
 *
 * Immutability: a flow that has ever run is read-only; edits fork to @N+1.
 * All structural edits live in an in-memory draft; nothing persists until
 * "New version from this" writes the draft (raw or structured) to @N+1.
 */

type Mode = "view" | "edit" | "raw";

// deep-clone a FlowSpec for the in-memory editable draft.
const cloneFlow = (f: FlowSpec): FlowSpec => JSON.parse(JSON.stringify(f));

export function FlowLab() {
  const manifest = useStore((s) => s.manifest);  // v1 manifest → gate titles/severities
  const [flows, setFlows] = useState<FlowSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [sel, setSel] = useState<{ flow: string; version: number } | null>(null);
  const [draft, setDraft] = useState<FlowSpec | null>(null);   // editable in-memory flow
  const [pristine, setPristine] = useState<FlowSpec | null>(null); // last-loaded (for dirty check)
  const [mode, setMode] = useState<Mode>("view");
  const [selStep, setSelStep] = useState<string | null>(null);

  // knobs
  const [presetSpec, setPresetSpec] = useState<KnobSetSpec | null>(null);
  const [presetName, setPresetName] = useState<string>("");
  const [presets, setPresets] = useState<string[]>([]);
  const [overrides, setOverrides] = useState<Record<string, number>>({});

  // raw-yaml buffer
  const [rawYaml, setRawYaml] = useState<string>("");

  // compile
  const [compile, setCompile] = useState<CompileResult | null>(null);
  const [compiling, setCompiling] = useState(false);
  const [compileErr, setCompileErr] = useState<string | null>(null);

  const [notice, setNotice] = useState<string | null>(null);

  // -- initial flows list -------------------------------------------------
  useEffect(() => {
    studioApi.flows()
      .then((fs) => {
        setFlows(fs);
        if (fs.length && !sel) setSel({ flow: fs[0].flow, version: fs[0].version });
      })
      .catch((e) => setError(e.message || String(e)));
    studioApi.presets()
      .then((ps) => setPresets(ps.map((p) => p.name)))
      .catch(() => { /* presets optional */ });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // -- load selected flow + its default preset ----------------------------
  useEffect(() => {
    if (!sel) return;
    let live = true;
    setSelStep(null); setMode("view"); setOverrides({});
    studioApi.flow(sel.flow, sel.version)
      .then((f) => {
        if (!live) return;
        setDraft(cloneFlow(f));
        setPristine(cloneFlow(f));
        // load the flow's declared default preset
        const ref = f.knobs_defaults || "";
        const name = ref.split("/").pop()?.replace(/\.knobs\.yaml$/, "") || "classic-6x4";
        setPresetName(name);
        studioApi.preset(name)
          .then((p) => live && setPresetSpec(p))
          .catch(() => live && setPresetSpec(null));
      })
      .catch((e) => live && setError(e.message || String(e)));
    studioApi.flowRaw(sel.flow, sel.version)
      .then((r) => live && setRawYaml(r.yaml))
      .catch(() => { /* raw optional */ });
    return () => { live = false; };
  }, [sel]);

  const readOnly = Boolean(draft?.immutable);
  const nPairs = useMemo(() => {
    const v = compile?.knobs.resolved.n_pairs;
    return typeof v === "number" ? v : 6;
  }, [compile]);

  // -- the effective knobs sent to /compile: preset + inline scalar overrides.
  const knobsRequest = useMemo(() => {
    if (!presetSpec) return presetName || null;
    if (!Object.keys(overrides).length) return presetName || presetSpec;
    // inline: start from the preset spec, layer the scalar edits over .value
    const merged: KnobSetSpec = JSON.parse(JSON.stringify(presetSpec));
    for (const [k, v] of Object.entries(overrides)) {
      if (merged.knobs[k]) merged.knobs[k] = { ...merged.knobs[k], value: v };
    }
    return merged;
  }, [presetSpec, presetName, overrides]);

  // -- debounced compile (the $0 signature interaction) -------------------
  const compileTimer = useRef<any>(null);
  const runCompile = useCallback(() => {
    if (!sel) return;
    setCompiling(true); setCompileErr(null);
    studioApi.compile({
      flow: sel.flow, version: sel.version,
      knobs: knobsRequest,
      context: {},
      baseline: presetName || undefined,   // "feel the knob": diff vs the clean preset
    })
      .then((r) => { setCompile(r); setCompiling(false); })
      .catch((e) => { setCompileErr(e.message || String(e)); setCompiling(false); });
  }, [sel, knobsRequest, presetName]);

  useEffect(() => {
    if (!sel || !presetSpec) return;
    clearTimeout(compileTimer.current);
    compileTimer.current = setTimeout(runCompile, 180);
    return () => clearTimeout(compileTimer.current);
  }, [sel, presetSpec, knobsRequest, runCompile]);

  const compileByStep = useMemo(() => {
    const m = new Map<string, StepCompile>();
    for (const s of compile?.steps ?? []) m.set(s.step, s);
    return m;
  }, [compile]);

  // gate id -> {title, severity, note} from the v1 manifest, so the rail can
  // show a real severity-colored chip per gate instead of an opaque count.
  const gateMeta = useMemo<GateMeta>(() => {
    const m: GateMeta = new Map();
    for (const g of manifest?.gates ?? []) m.set(g.id, { title: g.title, severity: g.severity, note: g.note });
    return m;
  }, [manifest]);

  // -- draft edits (structural) -------------------------------------------
  const patchStep = (stageId: string, stepId: string, patch: Partial<FlowStep>) => {
    setDraft((d) => {
      if (!d) return d;
      const nd = cloneFlow(d);
      const st = nd.stages.find((s) => s.id === stageId);
      const step = st?.steps.find((x) => x.id === stepId);
      if (step) Object.assign(step, patch);
      return nd;
    });
  };
  const insertAfter = (stageId: string, stepId: string) => {
    setDraft((d) => {
      if (!d) return d;
      const nd = cloneFlow(d);
      const st = nd.stages.find((s) => s.id === stageId);
      if (!st) return nd;
      const idx = st.steps.findIndex((x) => x.id === stepId);
      const ns = emptyStep(nextStepId(nd));
      st.steps.splice(idx + 1, 0, ns);
      setSelStep(ns.id);
      return nd;
    });
  };
  const removeStep = (stageId: string, stepId: string) => {
    setDraft((d) => {
      if (!d) return d;
      const nd = cloneFlow(d);
      const st = nd.stages.find((s) => s.id === stageId);
      if (st) st.steps = st.steps.filter((x) => x.id !== stepId);
      if (selStep === stepId) setSelStep(null);
      return nd;
    });
  };
  const moveStep = (stageId: string, stepId: string, dir: -1 | 1) => {
    setDraft((d) => {
      if (!d) return d;
      const nd = cloneFlow(d);
      const st = nd.stages.find((s) => s.id === stageId);
      if (!st) return nd;
      const i = st.steps.findIndex((x) => x.id === stepId);
      const j = i + dir;
      if (i < 0 || j < 0 || j >= st.steps.length) return nd;
      [st.steps[i], st.steps[j]] = [st.steps[j], st.steps[i]];
      return nd;
    });
  };

  const structurallyDirty = useMemo(
    () => JSON.stringify(draft) !== JSON.stringify(pristine),
    [draft, pristine],
  );

  // -- fork to @N+1 -------------------------------------------------------
  const [forking, setForking] = useState(false);
  const forkFromRaw = async () => {
    if (!sel) return;
    setForking(true); setNotice(null); setError(null);
    try {
      const r = await studioApi.forkFlow(sel.flow, sel.version, rawYaml);
      await refreshFlows();
      setSel({ flow: r.flow, version: r.version });
      setNotice(`Forked to ${r.flow}@${r.version} (${r.file}).`);
    } catch (e: any) {
      setError(e.body?.detail || e.message || String(e));
    } finally { setForking(false); }
  };
  const forkFromDraft = async () => {
    if (!sel || !draft) return;
    setForking(true); setNotice(null); setError(null);
    try {
      // serialize the structured draft to YAML via the raw path: send the draft
      // as YAML text so the backend schema-validates it.
      const yaml = draftToYaml(draft);
      const r = await studioApi.forkFlow(sel.flow, sel.version, yaml);
      await refreshFlows();
      setSel({ flow: r.flow, version: r.version });
      setNotice(`New version ${r.flow}@${r.version} written (${r.file}).`);
    } catch (e: any) {
      setError(e.body?.detail || e.message || String(e));
    } finally { setForking(false); }
  };

  const refreshFlows = async () => {
    const fs = await studioApi.flows();
    setFlows(fs);
  };

  // -- knobs handlers -----------------------------------------------------
  const onOverride = (name: string, value: number) =>
    setOverrides((o) => ({ ...o, [name]: value }));
  const resetOverrides = () => setOverrides({});
  const loadPreset = (name: string) => {
    setPresetName(name); setOverrides({});
    studioApi.preset(name).then(setPresetSpec).catch(() => setPresetSpec(null));
  };
  const savePreset = async (name: string) => {
    if (!presetSpec) return;
    const merged: KnobSetSpec = JSON.parse(JSON.stringify(presetSpec));
    merged.name = name;
    for (const [k, v] of Object.entries(overrides)) {
      if (merged.knobs[k]) merged.knobs[k] = { ...merged.knobs[k], value: v };
    }
    try {
      await studioApi.savePreset(name, merged);
      const ps = await studioApi.presets();
      setPresets(ps.map((p) => p.name));
      setNotice(`Preset '${name}' saved.`);
      loadPreset(name);
    } catch (e: any) {
      setError(e.body?.detail || e.message || String(e));
    }
  };

  // -- version list for the selected flow family --------------------------
  const versions = useMemo(() => {
    if (!sel) return [];
    const fam = flows?.filter((f) => f.flow === sel.flow) ?? [];
    return fam.map((f) => f.version).sort((a, b) => a - b);
  }, [flows, sel]);

  // diff-vs-version selector
  const [diffVersion, setDiffVersion] = useState<number | null>(null);

  if (error && !flows) {
    return <div className="pad"><div className="card"><div className="card-bd">
      <b style={{ color: "var(--sev-hard)" }}>Flow Lab could not load flows.</b>
      <p className="mono" style={{ fontSize: 11, color: "var(--sev-hard)" }}>{error}</p>
    </div></div></div>;
  }
  if (!flows || !sel || !draft) {
    return <div className="empty row" style={{ justifyContent: "center" }}><span className="spin" /> loading flows…</div>;
  }

  const selectedStepObj: { step: FlowStep; stageId: string } | null = (() => {
    if (!selStep) return null;
    for (const st of draft.stages) {
      const s = st.steps.find((x) => x.id === selStep);
      if (s) return { step: s, stageId: st.id };
    }
    return null;
  })();

  const runBlocked = (compile?.invariant_violations.length ?? 0) > 0;
  const totals = compile?.totals;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      {/* header: flow picker + version list + mode + fork */}
      <div className="hd" style={{ gap: 10, flexWrap: "wrap" }}>
        <h1>⚗ Flow Lab</h1>
        <select className="input mono" value={sel.flow}
                onChange={(e) => {
                  const fam = flows.filter((f) => f.flow === e.target.value);
                  const v = Math.max(...fam.map((f) => f.version));
                  setSel({ flow: e.target.value, version: v });
                }}
                style={{ width: "auto", padding: "3px 8px" }}>
          {[...new Set(flows.map((f) => f.flow))].map((name) => (
            <option key={name} value={name}>{name}</option>
          ))}
        </select>

        <div className="row" style={{ gap: 3 }}>
          {versions.map((v) => (
            <button key={v}
                    className={`btn sm ${v === sel.version ? "primary" : ""}`}
                    onClick={() => setSel({ flow: sel.flow, version: v })}
                    title={`version ${v}`}>@{v}</button>
          ))}
        </div>

        {readOnly
          ? <span className="badge canon" title="this flow has produced a run — it is immutable; fork to edit"><span className="d" />immutable</span>
          : <span className="badge ok"><span className="d" />editable</span>}

        <div className="spacer" />

        {/* mode toggle */}
        <div className="row" style={{ gap: 3 }}>
          {(["view", "edit", "raw"] as Mode[]).map((m) => (
            <button key={m} className={`btn sm ${mode === m ? "primary" : ""}`}
                    onClick={() => { setMode(m); if (m !== "edit") setSelStep(null); }}>{m}</button>
          ))}
        </div>

        <button className="btn sm" disabled={forking}
                onClick={mode === "raw" ? forkFromRaw : forkFromDraft}
                title="copy-on-write this flow to @N+1 (structured draft or raw YAML)">
          {forking ? "forking…" : "＋ New version"}
        </button>
      </div>

      {notice && <div className="banner" style={{ background: "color-mix(in srgb, var(--ok) 10%, var(--bg))", color: "#c8f5df" }}>
        {notice} <button className="btn sm ghost" onClick={() => setNotice(null)}>dismiss</button>
      </div>}
      {error && <div className="banner err">{error} <button className="btn sm ghost" onClick={() => setError(null)}>dismiss</button></div>}
      {compileErr && <div className="banner warn">compile: {compileErr}</div>}
      {structurallyDirty && (
        <div className="banner warn">
          {readOnly
            ? "This flow is immutable (it has produced a run) — "
            : "Edits are copy-on-write — "}
          your structural changes persist only via <b>＋ New version</b> (@{sel.version + 1}), never overwriting @{sel.version}.
        </div>
      )}

      {/* main: [ center rail/raw | drawer/diff ] + [ right knobs ] */}
      <div style={{ flex: 1, minHeight: 0, display: "grid", gridTemplateColumns: "1.15fr 1fr 340px" }}>
        {/* CENTER: pipeline rail OR raw-yaml */}
        <div style={{ borderRight: "1px solid var(--border)", minHeight: 0, display: "flex", flexDirection: "column" }}>
          {mode === "raw" ? (
            <RawYamlTab yaml={rawYaml} readOnly={false} onChange={setRawYaml} />
          ) : (
            <FlowLabPipelineRail
              flow={draft}
              nPairs={nPairs}
              selected={selStep}
              editMode={mode === "edit"}
              // edit affordances are enabled in edit mode even for an immutable
              // flow — edits persist only via a fork (New version), never in place.
              readOnly={false}
              compileByStep={compileByStep}
              gateMeta={gateMeta}
              onSelect={(id) => setSelStep(id === selStep ? null : id)}
              onInsertAfter={insertAfter}
              onRemove={removeStep}
              onMove={moveStep}
            />
          )}
        </div>

        {/* MIDDLE: step drawer (when a step is selected) or the resolved-prompt diff */}
        <div style={{ borderRight: "1px solid var(--border)", minHeight: 0, display: "flex", flexDirection: "column" }}>
          {selectedStepObj ? (
            <FlowLabStepDrawer
              flow={draft}
              step={selectedStepObj.step}
              stageId={selectedStepObj.stageId}
              // In edit mode the drawer is always editable — edits live in the
              // draft and only ever persist via a fork, so mutating an immutable
              // flow's draft is safe (it writes @N+1, never overwrites).
              readOnly={mode !== "edit"}
              onEdit={(patch) => patchStep(selectedStepObj.stageId, selectedStepObj.step.id, patch)}
              onClose={() => setSelStep(null)}
            />
          ) : (
            <ResolvedDiffPane
              compile={compile}
              step={selStep}
              versions={versions.filter((v) => v !== sel.version)}
              diffVersion={diffVersion}
              onDiffVersion={setDiffVersion}
              flow={sel.flow}
            />
          )}
        </div>

        {/* RIGHT: knobs */}
        <div style={{ minHeight: 0, display: "flex", flexDirection: "column" }}>
          {presetSpec ? (
            <FlowLabKnobsPanel
              spec={presetSpec}
              summary={compile?.knobs ?? null}
              violations={compile?.invariant_violations ?? []}
              overrides={overrides}
              onOverride={onOverride}
              onResetOverrides={resetOverrides}
              dirtyOverrides={Object.keys(overrides).length > 0}
              presetName={presetName}
              presets={presets}
              onLoadPreset={loadPreset}
              onSavePreset={savePreset}
              compiling={compiling}
            />
          ) : (
            <div className="empty">no preset loaded</div>
          )}
        </div>
      </div>

      {/* bottom: cost estimate + run affordance */}
      <div className="row" style={{ padding: "8px 14px", borderTop: "1px solid var(--border)", background: "var(--surface)", gap: 16 }}>
        <div style={{ width: 260 }}>
          <CostBar
            label="est cost (worst-case)"
            spend={totals?.cost_estimate ?? 0}
            cap={0}
            unpriced={Boolean(totals && totals.cost_estimate == null)}
          />
        </div>
        <span className="faint mono" style={{ fontSize: 10.5 }} title={totals?.cost_estimate_note}>
          {totals ? `${totals.est_tokens_in}t in · ${totals.est_tokens_out_ceiling}t out (ceiling)` : "compiling…"}
        </span>
        <div className="spacer" />
        {runBlocked
          ? <span className="badge err" title="invariant violations block Run — fix the knobs"><span className="d" />Run blocked</span>
          : <span className="badge ok"><span className="d" />Run allowed</span>}
        <button className="btn primary sm" disabled={runBlocked}
                title={runBlocked ? "resolve invariant violations first" : "hand off to Run Bench with these knobs"}
                onClick={() => {
                  const params = new URLSearchParams({
                    flow: sel.flow, version: String(sel.version),
                    preset: presetName,
                  });
                  window.location.hash = `#/studio/run?${params.toString()}`;
                }}>
          ▷ Run…
        </button>
      </div>
    </div>
  );
}

// -- raw-yaml Monaco tab ----------------------------------------------------
function RawYamlTab({ yaml, readOnly, onChange }: { yaml: string; readOnly: boolean; onChange: (v: string) => void }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      <div className="row" style={{ padding: "6px 10px", borderBottom: "1px solid var(--border)", background: "var(--surface)" }}>
        <span className="faint mono" style={{ fontSize: 10.5 }}>raw FlowSpec YAML — schema-validated on "New version"</span>
      </div>
      <div style={{ flex: 1, minHeight: 0 }}>
        <Monaco
          language="yaml"
          value={yaml}
          theme="lofn-dark"
          onChange={(v) => onChange(v ?? "")}
          beforeMount={(m) => {
            m.editor.defineTheme("lofn-dark", {
              base: "vs-dark", inherit: true, rules: [],
              colors: { "editor.background": "#15171e", "editorGutter.background": "#15171e" },
            });
          }}
          options={{
            fontSize: 12.5, fontFamily: "ui-monospace, Cascadia Code, Consolas, monospace",
            minimap: { enabled: false }, wordWrap: "off", readOnly,
            scrollBeyondLastLine: false, tabSize: 2, padding: { top: 8 },
          }}
        />
      </div>
    </div>
  );
}

// -- resolved-prompt diff pane (the "feel the knob" view) -------------------
function ResolvedDiffPane({
  compile, step, versions, diffVersion, onDiffVersion, flow,
}: {
  compile: CompileResult | null;
  step: string | null;
  versions: number[];
  diffVersion: number | null;
  onDiffVersion: (v: number | null) => void;
  flow: string;
}) {
  const sc = useMemo(() => {
    if (!compile) return null;
    if (step) return compile.steps.find((s) => s.step === step) ?? null;
    // default to the coordinator step that shows knob movement best (step 05)
    return compile.steps.find((s) => s.step === "05") ?? compile.steps[0] ?? null;
  }, [compile, step]);

  if (!compile) {
    return <div className="empty row" style={{ justifyContent: "center" }}><span className="spin" /></div>;
  }
  if (!sc) return <div className="empty">no compiled step to show.</div>;

  const diff = sc.diff;
  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      <div className="row" style={{ padding: "6px 10px", borderBottom: "1px solid var(--border)", background: "var(--surface)", gap: 8, flexWrap: "wrap" }}>
        <b className="mono">resolved · step {sc.step}{sc.pair != null ? ` (pair ${sc.pair})` : ""}</b>
        <span className="faint mono" style={{ fontSize: 10 }}>{sc.est_tokens_in}t · {sc.resolved.text_sha.slice(0, 8)}</span>
        <div className="spacer" />
        {versions.length > 0 && (
          <>
            <span className="faint mono" style={{ fontSize: 10 }}>diff vs</span>
            <select className="input mono" value={diffVersion ?? ""} onChange={(e) => onDiffVersion(e.target.value ? Number(e.target.value) : null)}
                    style={{ width: "auto", padding: "1px 6px", fontSize: 11 }}>
              <option value="">baseline preset</option>
              {versions.map((v) => <option key={v} value={v}>@{v}</option>)}
            </select>
          </>
        )}
      </div>

      {diffVersion != null ? (
        <CrossVersionDiff flow={flow} step={sc.step} versionB={diffVersion} compile={compile} />
      ) : diff && diff.trim() ? (
        <DiffText diff={diff} />
      ) : (
        <div className="scroll mono" style={{ fontSize: 11.5, padding: "8px 12px", whiteSpace: "pre-wrap", color: "var(--text-dim)" }}>
          <div className="faint" style={{ marginBottom: 6 }}>no diff vs baseline preset — resolved prompt shown below:</div>
          {sc.resolved.user}
        </div>
      )}
    </div>
  );
}

// unified-diff renderer (mirrors the v1 DiffPanel colors, on inline text).
function DiffText({ diff }: { diff: string }) {
  return (
    <div className="scroll mono" style={{ fontSize: 11.5, padding: "8px 0", background: "var(--surface)" }}>
      {diff.split("\n").map((line, i) => {
        let color = "var(--text-dim)"; let bg = "transparent";
        if (line.startsWith("+") && !line.startsWith("+++")) { color = "var(--ok)"; bg = "color-mix(in srgb, var(--ok) 10%, transparent)"; }
        else if (line.startsWith("-") && !line.startsWith("---")) { color = "var(--sev-hard)"; bg = "color-mix(in srgb, var(--sev-hard) 10%, transparent)"; }
        else if (line.startsWith("@@")) color = "var(--accent)";
        else if (line.startsWith("+++") || line.startsWith("---")) color = "var(--text-faint)";
        return <div key={i} style={{ color, background: bg, padding: "0 10px", whiteSpace: "pre" }}>{line || " "}</div>;
      })}
    </div>
  );
}

// diff the current resolved step against the SAME step compiled from another
// flow version (the "diff vs another version" affordance).
function CrossVersionDiff({ flow, step, versionB, compile }: {
  flow: string; step: string; versionB: number; compile: CompileResult;
}) {
  const [other, setOther] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    studioApi.compile({ flow, version: versionB, step, context: {} })
      .then((r) => { if (live) setOther(r.steps.find((s) => s.step === step)?.resolved.user ?? ""); })
      .catch((e) => live && setErr(e.message || String(e)));
    return () => { live = false; };
  }, [flow, step, versionB]);

  if (err) return <div className="empty" style={{ color: "var(--sev-hard)" }}>{err}</div>;
  if (other == null) return <div className="empty row" style={{ justifyContent: "center" }}><span className="spin" /></div>;

  const current = compile.steps.find((s) => s.step === step)?.resolved.user ?? "";
  const diff = simpleLineDiff(other, current, `@${versionB}`, "current");
  return diff.trim()
    ? <DiffText diff={diff} />
    : <div className="empty">identical to @{versionB} for step {step}.</div>;
}

// tiny LCS-free line diff (added/removed markers) — enough to see version drift.
function simpleLineDiff(a: string, b: string, la: string, lb: string): string {
  const al = a.split("\n"); const bl = b.split("\n");
  const bset = new Set(bl); const aset = new Set(al);
  const out: string[] = [`--- ${la}`, `+++ ${lb}`];
  for (const line of al) if (!bset.has(line)) out.push(`-${line}`);
  for (const line of bl) if (!aset.has(line)) out.push(`+${line}`);
  return out.length > 2 ? out.join("\n") : "";
}

// Serialize a structured FlowSpec draft to YAML text for the fork endpoint.
// Minimal, deterministic emitter — the backend re-parses + schema-validates it,
// so we only need valid YAML, not canonical formatting.
function draftToYaml(f: FlowSpec): string {
  // Quote a string scalar whenever leaving it bare would change its YAML type
  // (numeric ids like "00"/"06", booleans, null-ish, or special chars).
  const esc = (s: unknown): string => {
    if (s == null) return "";
    const str = String(s);
    const looksNonString =
      /^[+-]?(\d+\.?\d*|\.\d+)$/.test(str) ||                  // number
      /^(true|false|null|yes|no|on|off|~)$/i.test(str) ||      // bool/null keywords
      str === "";
    if (looksNonString || /[:#{}\[\],&*?|<>=!%@`"']/.test(str) || str.trim() !== str)
      return JSON.stringify(str);
    return str;
  };
  const L: string[] = [];
  L.push(`flow: ${esc(f.flow)}`);
  L.push(`version: ${f.version}`);
  L.push(`modality: ${esc(f.modality)}`);
  L.push(`base: ${esc(f.base)}`);
  if (f.knobs_defaults) L.push(`knobs_defaults: ${esc(f.knobs_defaults)}`);
  L.push(`context:`);
  L.push(`  template: ${esc(f.context.template)}`);
  L.push(`  slots:`);
  for (const s of f.context.slots) L.push(`    - ${esc(s)}`);
  L.push(`model_tiers:`);
  for (const [t, mt] of Object.entries(f.model_tiers)) {
    L.push(`  ${t}:`);
    L.push(`    provider: ${esc(mt.provider)}`);
    L.push(`    model: ${esc(mt.model)}`);
    if (mt.max_tokens != null) L.push(`    max_tokens: ${mt.max_tokens}`);
  }
  L.push(`stages:`);
  for (const st of f.stages) {
    L.push(`  - id: ${esc(st.id)}`);
    if (st.foreach) L.push(`    foreach: ${esc(st.foreach)}`);
    if (st.concurrency != null) L.push(`    concurrency: ${esc(st.concurrency)}`);
    L.push(`    steps:`);
    for (const step of st.steps) {
      L.push(`      - id: ${esc(step.id)}`);
      if (step.title) L.push(`        title: ${esc(step.title)}`);
      L.push(`        kind: ${esc(step.kind || "llm")}`);
      L.push(`        prompt: ${step.prompt ? esc(step.prompt) : "null"}`);
      L.push(`        overlay: ${step.overlay ? esc(step.overlay) : "null"}`);
      L.push(`        tier: ${step.tier ? esc(step.tier) : "null"}`);
      if (step.emits) L.push(`        emits: ${esc(step.emits)}`);
      const consumes = step.consumes || [];
      if (consumes.length === 0) L.push(`        consumes: []`);
      else { L.push(`        consumes:`); for (const c of consumes) L.push(`          - ${esc(c)}`); }
      if (step.run) L.push(`        run: ${esc(step.run)}`);
      const gates = step.gates || [];
      if (gates.length === 0) L.push(`        gates: []`);
      else { L.push(`        gates:`); for (const g of gates) L.push(`          - ${esc(g)}`); }
      if (step.on_fail) {
        L.push(`        on_fail:`);
        L.push(`          policy: ${esc(step.on_fail.policy)}`);
        if (step.on_fail.max_attempts != null) L.push(`          max_attempts: ${step.on_fail.max_attempts}`);
      }
    }
  }
  return L.join("\n") + "\n";
}
