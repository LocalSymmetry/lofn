import { useState } from "react";
import { studioApi } from "../api";
import { UnifiedDiff } from "../components";
import type {
  CompareResult, PromotePreview, PromoteApplyResult,
} from "../types";

type PatchRow = { file: string; old: string; new: string };

/**
 * PROMOTION surface (plan §8.4 / §8.5). Carry a winning run's knob/threshold
 * VALUES back toward canon:
 *   1. gate_values — vault/gates.yaml scalar edits (existing keys only), the
 *      byte-safe ruamel round-trip.
 *   2. patches — exact, unique prose number/word swaps in allowlisted files.
 *   3. structural — NOT auto-applied; a Promotion Brief is generated instead.
 *
 * Flow: PREVIEW (POST /promote/preview) shows every diff first; APPLY (POST
 * /promote/apply) auto-writes value + simple-patch changes through the byte-safe
 * path, runs --meta-check, and reports git status. Structural changes go to a
 * brief the maintainer applies by hand.
 */
export function Promote({
  compare, runA, runB,
}: {
  compare: CompareResult;
  runA: string;
  runB: string;
}) {
  // Seed gate-value edits from the knob delta whose knob name looks like a gate
  // key; the operator confirms/edits and includes only what they want to promote.
  const [gateValues, setGateValues] = useState<Record<string, string>>({});
  const [patches, setPatches] = useState<PatchRow[]>([]);
  const [preview, setPreview] = useState<PromotePreview | null>(null);
  const [applied, setApplied] = useState<PromoteApplyResult | null>(null);
  const [brief, setBrief] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const setGate = (k: string, v: string) =>
    setGateValues((g) => ({ ...g, [k]: v }));
  const removeGate = (k: string) =>
    setGateValues((g) => { const n = { ...g }; delete n[k]; return n; });

  const seedFromKnob = (knob: string, val: unknown) =>
    setGate(knob, val == null ? "" : String(val));

  function buildBody() {
    const gate_values: Record<string, number | boolean> = {};
    for (const [k, raw] of Object.entries(gateValues)) {
      if (raw.trim() === "") continue;
      if (raw === "true" || raw === "false") gate_values[k] = raw === "true";
      else if (!Number.isNaN(Number(raw))) gate_values[k] = Number(raw);
      // non-numeric, non-bool values are left out (gates are numeric/bool thresholds)
    }
    const cleanPatches = patches.filter((p) => p.file.trim() && p.old.trim());
    return {
      gate_values: Object.keys(gate_values).length ? gate_values : undefined,
      patches: cleanPatches.length ? cleanPatches : undefined,
    };
  }

  async function doPreview() {
    setBusy("preview"); setErr(null); setApplied(null);
    try {
      setPreview(await studioApi.promotePreview(buildBody()));
    } catch (e: any) {
      setErr(e?.message || String(e));
    } finally { setBusy(null); }
  }

  async function doApply() {
    setBusy("apply"); setErr(null);
    try {
      setApplied(await studioApi.promoteApply(buildBody()));
    } catch (e: any) {
      setErr(e?.message || String(e));
    } finally { setBusy(null); }
  }

  async function doBrief() {
    setBusy("brief"); setErr(null);
    try {
      const r = await studioApi.promoteBrief({
        run_a: runA, run_b: runB,
        title: `Promotion ${runB} over ${runA}`,
        summary: "Structural changes require a deliberate canon edit; values/patches auto-apply.",
        knob_delta: compare.knob_delta,
      });
      setBrief(r.brief);
    } catch (e: any) {
      setErr(e?.message || String(e));
    } finally { setBusy(null); }
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <div className="banner warn" style={{ borderRadius: "var(--r-sm)", padding: "8px 12px", fontSize: 12 }}>
        Promotion writes winning <b>values</b> back to canon through the byte-safe
        path. Preview shows every diff first. Structural flow changes are never
        auto-applied — they get a <b>Promotion Brief</b> to apply by hand.
      </div>

      {/* knob-delta shortcut: seed a gate-value edit from any differing knob */}
      {compare.knob_delta.length > 0 && (
        <div className="card">
          <div className="card-hd">Knob delta → seed a gate value</div>
          <div className="card-bd" style={{ overflowX: "auto" }}>
            <table className="grid" style={{ minWidth: 420 }}>
              <thead>
                <tr>
                  <th>knob</th>
                  <th style={{ color: "var(--accent)" }}>A</th>
                  <th style={{ color: "var(--accent-2)" }}>B (promote)</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {compare.knob_delta.map((d) => (
                  <tr key={d.knob}>
                    <td className="mono">{d.knob}</td>
                    <td className="mono faint">{fmt(d.a)}</td>
                    <td className="mono">{fmt(d.b)}</td>
                    <td>
                      <button
                        className="btn sm"
                        disabled={d.knob in gateValues}
                        onClick={() => seedFromKnob(d.knob, d.b)}
                      >
                        {d.knob in gateValues ? "queued" : "queue B →"}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* gate value edits */}
      <div className="card">
        <div className="card-hd">gates.yaml value edits</div>
        <div className="card-bd" style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {Object.keys(gateValues).length === 0 ? (
            <p className="faint mono" style={{ fontSize: 12, margin: 0 }}>
              No queued gate values. Queue one from the knob delta above, or add a key below.
            </p>
          ) : (
            Object.entries(gateValues).map(([k, v]) => (
              <div key={k} className="row" style={{ gap: 8 }}>
                <input className="input mono" value={k} readOnly style={{ flex: 1, opacity: 0.85 }} />
                <input
                  className="input mono"
                  value={v}
                  onChange={(e) => setGate(k, e.target.value)}
                  placeholder="value (number / true / false)"
                  style={{ width: 200 }}
                />
                <button className="btn sm ghost" onClick={() => removeGate(k)}>remove</button>
              </div>
            ))
          )}
          <AddGateKey onAdd={(k) => setGate(k, "")} existing={gateValues} />
        </div>
      </div>

      {/* prose patches */}
      <div className="card">
        <div className="card-hd">
          prose patches{" "}
          <span className="faint mono" style={{ fontWeight: 400, fontSize: 11 }}>(exact, unique swap)</span>
        </div>
        <div className="card-bd" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {patches.map((p, i) => (
            <div key={i} className="row" style={{ gap: 8, alignItems: "flex-start", flexWrap: "wrap" }}>
              <input className="input mono" placeholder="path (allowlisted)" value={p.file}
                onChange={(e) => setPatches((ps) => ps.map((x, j) => j === i ? { ...x, file: e.target.value } : x))}
                style={{ flex: "1 1 220px" }} />
              <input className="input mono" placeholder="old (unique)" value={p.old}
                onChange={(e) => setPatches((ps) => ps.map((x, j) => j === i ? { ...x, old: e.target.value } : x))}
                style={{ flex: "1 1 140px" }} />
              <input className="input mono" placeholder="new" value={p.new}
                onChange={(e) => setPatches((ps) => ps.map((x, j) => j === i ? { ...x, new: e.target.value } : x))}
                style={{ flex: "1 1 140px" }} />
              <button className="btn sm ghost" onClick={() => setPatches((ps) => ps.filter((_, j) => j !== i))}>remove</button>
            </div>
          ))}
          <button className="btn sm" onClick={() => setPatches((ps) => [...ps, { file: "", old: "", new: "" }])}>
            + add patch
          </button>
        </div>
      </div>

      {/* actions */}
      <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
        <button className="btn" onClick={doPreview} disabled={busy === "preview"}>
          {busy === "preview" ? "previewing…" : "preview diffs"}
        </button>
        <button className="btn primary" onClick={doApply} disabled={busy === "apply" || !preview}
          title={!preview ? "preview first" : "auto-apply value + simple-patch changes"}>
          {busy === "apply" ? "applying…" : "promote (apply)"}
        </button>
        <button className="btn ghost" onClick={doBrief} disabled={busy === "brief"}>
          {busy === "brief" ? "…" : "generate structural brief"}
        </button>
      </div>

      {err && <div className="banner err" style={{ borderRadius: "var(--r-sm)", padding: "6px 10px", fontSize: 12 }}>{err}</div>}

      {preview && !applied && <PreviewPanel preview={preview} />}
      {applied && <ApplyPanel applied={applied} />}
      {brief && (
        <div className="card">
          <div className="card-hd">PROMOTION BRIEF <span className="chip prose" style={{ marginLeft: 6 }}>hand-apply</span></div>
          <div className="card-bd">
            <pre className="mono scroll" style={{ margin: 0, fontSize: 11.5, whiteSpace: "pre-wrap", maxHeight: 360, color: "var(--text-dim)" }}>
              {brief}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}

function PreviewPanel({ preview }: { preview: PromotePreview }) {
  return (
    <div className="card">
      <div className="card-hd">Preview — diffs a promotion WOULD produce (no write yet)</div>
      <div className="card-bd" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {preview.skipped_keys.length > 0 && (
          <div className="chip flag"><span className="d" />skipped (not in canon gates): {preview.skipped_keys.join(", ")}</div>
        )}
        {preview.gates && (
          <div>
            <div className="row" style={{ justifyContent: "space-between", marginBottom: 4 }}>
              <span className="mono faint" style={{ fontSize: 11 }}>{preview.gates.file}</span>
              <span className="mono faint" style={{ fontSize: 11 }}>
                {preview.gates.changed ? `applying ${Object.keys(preview.gates.applied).length} value(s)` : "no change"}
              </span>
            </div>
            <UnifiedDiff diff={preview.gates.diff} empty="No gates.yaml change." />
          </div>
        )}
        {preview.patches.map((p, i) => (
          <div key={i}>
            <div className="row" style={{ justifyContent: "space-between", marginBottom: 4 }}>
              <span className="mono faint" style={{ fontSize: 11 }}>{p.file}</span>
              <span className={`chip ${p.ok ? "" : "hard"}`}><span className="d" />{p.ok ? "ready" : p.detail}</span>
            </div>
            {p.ok && <UnifiedDiff diff={p.diff} empty="No change." />}
          </div>
        ))}
        {!preview.gates && preview.patches.length === 0 && (
          <p className="faint mono" style={{ fontSize: 12, margin: 0 }}>Nothing queued to promote.</p>
        )}
      </div>
    </div>
  );
}

function ApplyPanel({ applied }: { applied: PromoteApplyResult }) {
  const meta = applied.meta_check;
  return (
    <div className="card">
      <div className="card-hd row" style={{ justifyContent: "space-between" }}>
        <span>Applied</span>
        <span className={`badge ${meta.ok ? "ok" : "err"}`}>
          <span className="d" />
          <span className="mono" style={{ fontSize: 10.5 }}>meta-check {meta.ok ? "clean" : `exit ${meta.exit_code}`}</span>
        </span>
      </div>
      <div className="card-bd" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {applied.errors.length > 0 && (
          <div className="banner err" style={{ borderRadius: "var(--r-sm)", padding: "6px 10px", fontSize: 11.5 }}>
            {applied.errors.map((e, i) => <div key={i} className="mono">{e}</div>)}
          </div>
        )}
        {Object.keys(applied.applied_gates).length > 0 && (
          <div className="mono" style={{ fontSize: 12 }}>
            <span className="faint">gates written: </span>
            {Object.entries(applied.applied_gates).map(([k, v]) => `${k}=${fmt(v)}`).join(", ")}
          </div>
        )}
        {applied.applied_patches.length > 0 && (
          <div style={{ fontSize: 12 }}>
            {applied.applied_patches.map((p, i) => (
              <div key={i} className="mono">
                <span style={{ color: p.applied ? "var(--ok)" : "var(--sev-hard)" }}>{p.applied ? "✓" : "✗"}</span>{" "}
                {p.file} <span className="faint">— {p.detail}</span>
              </div>
            ))}
          </div>
        )}
        <div>
          <div className="faint mono" style={{ fontSize: 11, marginBottom: 4 }}>git status --porcelain</div>
          <pre className="mono scroll" style={{ margin: 0, fontSize: 11, background: "var(--surface)", border: "1px solid var(--border)", borderRadius: "var(--r-sm)", padding: 8, maxHeight: 160, color: "var(--text-dim)" }}>
            {applied.git_status || "(clean working tree)"}
          </pre>
        </div>
        {meta.stdout && (
          <details>
            <summary className="faint mono" style={{ fontSize: 11, cursor: "pointer" }}>meta-check output</summary>
            <pre className="mono scroll" style={{ margin: "6px 0 0", fontSize: 10.5, maxHeight: 160, color: "var(--text-faint)" }}>{meta.stdout}</pre>
          </details>
        )}
      </div>
    </div>
  );
}

function AddGateKey({ onAdd, existing }: { onAdd: (k: string) => void; existing: Record<string, string> }) {
  const [k, setK] = useState("");
  return (
    <div className="row" style={{ gap: 8 }}>
      <input className="input mono" placeholder="add gates.yaml key…" value={k}
        onChange={(e) => setK(e.target.value)} style={{ flex: 1 }} />
      <button className="btn sm" disabled={!k.trim() || k in existing}
        onClick={() => { onAdd(k.trim()); setK(""); }}>+ add key</button>
    </div>
  );
}

function fmt(v: unknown): string {
  if (v == null) return "—";
  if (Array.isArray(v)) return `[${v.join(", ")}]`;
  return String(v);
}
