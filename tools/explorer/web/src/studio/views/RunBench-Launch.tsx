import { useEffect, useMemo, useState } from "react";
import { studioApi } from "../api";
import { api as v1api } from "../../api";
import { CostBar, ProviderLight } from "../components";
import type {
  FlowSummary, PresetSummary, ProviderStatus, ModelTier,
  LaunchRequest, CompileResult, ProviderTestResult,
} from "../types";
import type { LibraryItemRef } from "../../types";

/** The four flows the studio ships. Used as a fallback picker when the
    /flows list route isn't wired yet (it currently 404s — see api.ts note),
    so the launch form is fully usable today and auto-upgrades when it lands. */
const KNOWN_FLOWS: FlowSummary[] = [
  { flow: "lofn-music", version: 1, modality: "music", title: "Lofn Music" },
  { flow: "lofn-image", version: 1, modality: "image", title: "Lofn Image" },
  { flow: "lofn-video", version: 1, modality: "video", title: "Lofn Video" },
  { flow: "lofn-story", version: 1, modality: "story", title: "Lofn Story" },
];

const BUDGET_SMOKE = 2;
const BUDGET_STANDARD = 25;

export interface LaunchParams extends LaunchRequest {}

interface Props {
  /** Called with the freshly-created run.json id once a launch succeeds. */
  onLaunched: (runId: string) => void;
}

/**
 * Run Bench — launch form (plan §8.3). flow@version · preset · brief/seed ·
 * personality & panel pickers (v1 Library endpoints) · model-tier map with
 * provider status lights · REQUIRED budget cap · dry-run toggle (compile-all,
 * no spend). Nothing launches without a non-negative cap.
 */
export function LaunchForm({ onLaunched }: Props) {
  const [flows, setFlows] = useState<FlowSummary[]>(KNOWN_FLOWS);
  const [flowName, setFlowName] = useState("lofn-music");
  const [flowVersion, setFlowVersion] = useState(1);

  const [presets, setPresets] = useState<PresetSummary[]>([]);
  const [preset, setPreset] = useState<string>("");   // "" = flow default

  const [providers, setProviders] = useState<ProviderStatus[]>([]);
  const [tierMap, setTierMap] = useState<Record<string, ModelTier>>({});
  const [pings, setPings] = useState<Record<string, ProviderTestResult>>({});
  const [pinging, setPinging] = useState<string | null>(null);

  const [personalities, setPersonalities] = useState<LibraryItemRef[]>([]);
  const [panels, setPanels] = useState<LibraryItemRef[]>([]);
  const [personality, setPersonality] = useState("");
  const [panel, setPanel] = useState("");

  const [brief, setBrief] = useState("");
  const [seed, setSeed] = useState("");
  const [title, setTitle] = useState("");

  const [budgetCap, setBudgetCap] = useState<string>(String(BUDGET_SMOKE));
  const [dryRun, setDryRun] = useState(true);
  const [experimentalQa, setExperimentalQa] = useState(false);

  const [estimate, setEstimate] = useState<CompileResult | null>(null);
  const [estimating, setEstimating] = useState(false);
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // -- load flows / providers / libraries once -----------------------------
  useEffect(() => {
    studioApi.flows()
      .then((f) => { if (f && f.length) setFlows(f); })
      .catch(() => { /* route not wired yet — keep KNOWN_FLOWS */ });
    studioApi.providers().then(setProviders).catch(() => setProviders([]));
    v1api.library("personalities")
      .then((l) => setPersonalities(asRefs(l.items)))
      .catch(() => setPersonalities([]));
    v1api.library("panels")
      .then((l) => setPanels(asRefs(l.items)))
      .catch(() => setPanels([]));
  }, []);

  // -- load presets + model tiers when the flow changes --------------------
  useEffect(() => {
    setPreset("");
    studioApi.presets(flowName)
      .then((p) => setPresets(p))
      .catch(() => setPresets([]));
    studioApi.flow(flowName, flowVersion)
      .then((spec) => setTierMap(spec.model_tiers || {}))
      .catch(() => setTierMap({}));   // /flows/{name} not wired -> tier map hidden
  }, [flowName, flowVersion]);

  const capNum = parseFloat(budgetCap);
  const capValid = Number.isFinite(capNum) && capNum >= 0;

  const context = useMemo(() => {
    const ctx: Record<string, string> = {};
    if (brief.trim()) ctx.input = brief.trim();
    if (seed.trim()) ctx.seed = seed.trim();
    if (personality) ctx.personality = personality;
    if (panel) ctx.panel = panel;
    return ctx;
  }, [brief, seed, personality, panel]);

  // -- dry-cost estimate (compile all, priced) -----------------------------
  const runEstimate = () => {
    setEstimating(true);
    setError(null);
    studioApi.compile({
      flow: flowName,
      version: flowVersion,
      knobs: preset || undefined,
      context,
    })
      .then(setEstimate)
      .catch((e) => setError(estMsg(e)))
      .finally(() => setEstimating(false));
  };

  const pingProvider = (p: string) => {
    setPinging(p);
    studioApi.testProvider(p)
      .then((r) => setPings((prev) => ({ ...prev, [p]: r })))
      .catch((e) => setPings((prev) => ({ ...prev, [p]: { provider: p, ok: false, note: String(e?.message || e) } })))
      .finally(() => setPinging(null));
  };

  const launch = () => {
    if (!capValid || launching) return;
    setLaunching(true);
    setError(null);
    const body: LaunchRequest = {
      flow: flowName,
      version: flowVersion,
      knobs: preset || undefined,
      context,
      title: title.trim() || undefined,
      budget_cap_usd: capNum,
      dry_run: dryRun,
      experimental_qa: experimentalQa,
    };
    studioApi.launchRun(body)
      .then((run) => onLaunched(run.id))
      .catch((e) => setError(e?.message || String(e)))
      .finally(() => setLaunching(false));
  };

  // providers referenced by this flow's tiers (light them up loud)
  const flowProviders = useMemo(() => {
    const set = new Set<string>();
    for (const t of Object.values(tierMap)) if (t?.provider) set.add(String(t.provider));
    return set;
  }, [tierMap]);

  const est = estimate?.totals;
  const estUnpriced = est?.cost_estimate == null;

  return (
    <div className="card"><div className="card-bd" style={{ display: "grid", gap: 14 }}>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <b>Launch a run</b>
        <span className="faint mono" style={{ fontSize: 11 }}>plan §8.3</span>
      </div>

      {/* flow @ version · preset */}
      <div className="row" style={{ gap: 12, flexWrap: "wrap" }}>
        <label className="field">
          <span className="field-lbl">flow</span>
          <select className="input" value={flowName} onChange={(e) => setFlowName(e.target.value)}>
            {flows.map((f) => (
              <option key={`${f.flow}@${f.version}`} value={f.flow}>
                {f.title || f.flow}{f.modality ? ` · ${f.modality}` : ""}
              </option>
            ))}
          </select>
        </label>
        <label className="field" style={{ maxWidth: 90 }}>
          <span className="field-lbl">version</span>
          <input className="input mono" type="number" min={1} value={flowVersion}
                 onChange={(e) => setFlowVersion(Math.max(1, parseInt(e.target.value || "1", 10)))} />
        </label>
        <label className="field">
          <span className="field-lbl">preset (knobs)</span>
          <select className="input" value={preset} onChange={(e) => setPreset(e.target.value)}>
            <option value="">— flow default —</option>
            {presets.map((p) => (
              <option key={p.name} value={p.name}>{p.name}{p.desc ? ` — ${p.desc}` : ""}</option>
            ))}
          </select>
        </label>
      </div>

      {/* seed / brief */}
      <label className="field">
        <span className="field-lbl">brief / input <span className="faint">(ICB `input` slot)</span></span>
        <textarea className="input mono" rows={3} value={brief} placeholder="One-line intent, prompt, or the day's facts…"
                  onChange={(e) => setBrief(e.target.value)} />
      </label>
      <label className="field">
        <span className="field-lbl">seed <span className="faint">(optional — ICB `seed` slot)</span></span>
        <textarea className="input mono" rows={2} value={seed} placeholder="Golden Seed text (leave blank to let Phase 0 generate)…"
                  onChange={(e) => setSeed(e.target.value)} />
      </label>

      {/* personality + panel pickers (v1 Library endpoints) */}
      <div className="row" style={{ gap: 12, flexWrap: "wrap" }}>
        <label className="field">
          <span className="field-lbl">personality <span className="faint">({personalities.length})</span></span>
          <select className="input" value={personality} onChange={(e) => setPersonality(e.target.value)}>
            <option value="">— auto-select (Phase 1) —</option>
            {personalities.map((p) => <option key={p.slug} value={p.name}>{p.name}</option>)}
          </select>
        </label>
        <label className="field">
          <span className="field-lbl">panel <span className="faint">({panels.length})</span></span>
          <select className="input" value={panel} onChange={(e) => setPanel(e.target.value)}>
            <option value="">— auto-select (Phase 1) —</option>
            {panels.map((p) => <option key={p.slug} value={p.name}>{p.name}</option>)}
          </select>
        </label>
        <label className="field">
          <span className="field-lbl">title <span className="faint">(optional)</span></span>
          <input className="input" value={title} placeholder={flowName}
                 onChange={(e) => setTitle(e.target.value)} />
        </label>
      </div>

      {/* model-tier map + provider status lights */}
      <div className="field">
        <span className="field-lbl">model tiers · provider status</span>
        <div className="row" style={{ gap: 8, flexWrap: "wrap", marginTop: 2 }}>
          {Object.keys(tierMap).length === 0 && (
            <span className="faint mono" style={{ fontSize: 11 }}>
              tier map unavailable (flow detail route not live) — provider lights below
            </span>
          )}
          {Object.entries(tierMap).map(([tier, t]) => {
            const prov = providers.find((p) => String(p.provider) === String(t.provider));
            const ping = pings[String(t.provider)];
            return (
              <span key={tier} className="chip" title={`${tier} → ${t.provider}/${t.model}`}>
                <span className="mono" style={{ fontSize: 10.5 }}>{tier}</span>
                <span className="faint mono" style={{ fontSize: 10 }}>{t.provider}/{t.model}</span>
                {prov && !prov.configured && <span className="badge warn" style={{ marginLeft: 4 }}><span className="d" />no key</span>}
                {ping && <span className={`badge ${ping.ok ? "ok" : "err"}`} style={{ marginLeft: 4 }}><span className="d" />{ping.ok ? "ok" : "fail"}</span>}
              </span>
            );
          })}
        </div>
        <div className="row" style={{ gap: 8, flexWrap: "wrap", marginTop: 8 }}>
          {providers.map((p) => {
            const inFlow = flowProviders.has(String(p.provider));
            const ping = pings[String(p.provider)];
            return (
              <span key={String(p.provider)} style={{ display: "inline-flex", alignItems: "center", gap: 4, opacity: inFlow || flowProviders.size === 0 ? 1 : 0.5 }}>
                <ProviderLight p={p} ok={ping?.ok} note={ping?.note} />
                <button className="btn sm ghost" disabled={pinging === String(p.provider)}
                        onClick={() => pingProvider(String(p.provider))}>
                  {pinging === String(p.provider) ? "…" : "ping"}
                </button>
              </span>
            );
          })}
        </div>
      </div>

      {/* budget cap (REQUIRED) + dry-run */}
      <div className="row" style={{ gap: 14, flexWrap: "wrap", alignItems: "flex-end" }}>
        <label className="field" style={{ maxWidth: 200 }}>
          <span className="field-lbl">budget cap (USD) <span style={{ color: "var(--sev-hard)" }}>*required</span></span>
          <div className="row" style={{ gap: 6 }}>
            <input className="input mono" type="number" min={0} step="0.5" value={budgetCap}
                   onChange={(e) => setBudgetCap(e.target.value)}
                   style={!capValid ? { borderColor: "var(--sev-hard)" } : undefined} />
          </div>
          <div className="row" style={{ gap: 6, marginTop: 4 }}>
            <button className="btn sm" onClick={() => setBudgetCap(String(BUDGET_SMOKE))}>smoke ${BUDGET_SMOKE}</button>
            <button className="btn sm" onClick={() => setBudgetCap(String(BUDGET_STANDARD))}>standard ${BUDGET_STANDARD}</button>
          </div>
        </label>
        <label className="row" style={{ gap: 6 }}>
          <input type="checkbox" checked={dryRun} onChange={(e) => setDryRun(e.target.checked)} />
          <span>dry-run <span className="faint">(compile-all, no API spend)</span></span>
        </label>
        <label className="row" style={{ gap: 6 }}>
          <input type="checkbox" checked={experimentalQa} onChange={(e) => setExperimentalQa(e.target.checked)} />
          <span className="faint">experimental QA judge</span>
        </label>
      </div>

      {/* estimate + launch */}
      <div className="row" style={{ gap: 10, flexWrap: "wrap", justifyContent: "space-between" }}>
        <button className="btn sm" onClick={runEstimate} disabled={estimating}>
          {estimating ? "estimating…" : "estimate cost (compile)"}
        </button>
        {est && (
          <div style={{ minWidth: 220, flex: 1, maxWidth: 340 }}>
            <CostBar
              label="est"
              spend={est.cost_estimate ?? 0}
              cap={capValid ? capNum : 0}
              unpriced={estUnpriced}
            />
            <div className="faint mono" style={{ fontSize: 10.5, marginTop: 2 }}>
              {est.est_tokens_in.toLocaleString()} in · {est.est_tokens_out_ceiling.toLocaleString()} out ceiling
              {estimate && !estimate.run_allowed && <span style={{ color: "var(--sev-hard)" }}> · invariant violation</span>}
            </div>
          </div>
        )}
        <button className="btn primary" onClick={launch}
                disabled={!capValid || launching}
                title={!capValid ? "a non-negative budget cap is required" : undefined}>
          {launching ? "launching…" : dryRun ? "▷ dry-run" : "▷ launch"}
        </button>
      </div>

      {estimate && estimate.invariant_violations.length > 0 && (
        <div className="banner warn" style={{ borderRadius: 6 }}>
          <b>invariants</b>
          <span className="mono" style={{ fontSize: 11 }}>
            {estimate.invariant_violations.map((v) => v.expr).join(" · ")}
          </span>
        </div>
      )}
      {error && <div className="banner err" style={{ borderRadius: 6 }}><span className="mono" style={{ fontSize: 11 }}>{error}</span></div>}
    </div></div>
  );
}

function asRefs(items: LibraryItemRef[] | string[]): LibraryItemRef[] {
  if (!items || items.length === 0) return [];
  if (typeof items[0] === "string") {
    return (items as string[]).map((s) => ({ name: s, file: s, slug: s, is_variant: false }));
  }
  return (items as LibraryItemRef[]).filter((it) => !it.is_variant);
}

function estMsg(e: any): string {
  if (e?.status === 404) return "compile/flow not found — is this flow@version present in tools/explorer/flows/?";
  return e?.message || String(e);
}
