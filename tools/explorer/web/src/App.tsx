import { useEffect, useMemo, useState } from "react";
import { NavLink, Route, Routes, useNavigate } from "react-router-dom";
import { useStore } from "./store";
import { useHealth } from "./health";
import { Bridge } from "./views/Bridge";
import { WorkingSet } from "./views/WorkingSet";
import { Atlas } from "./views/Atlas";
import { EditorView } from "./views/EditorView";
import { GateCenter } from "./views/GateCenter";
import { Library } from "./views/Library";
import { CommandPalette } from "./components/CommandPalette";
import { HypsometricKey } from "./components/charts";
import { FlowLab } from "./studio/views/FlowLab";
import { RunBench } from "./studio/views/RunBench";
import { Compare } from "./studio/views/Compare";
import {
  applyExplorerTheme,
  EXPLORER_THEMES,
  EXPLORER_THEME_STORAGE_KEY,
  getExplorerTheme,
  type ExplorerTheme,
} from "./themes";

export function App() {
  const { loading, error, verify, load, repoRoot } = useStore();
  const [themeId, setThemeId] = useState(() => {
    try { return localStorage.getItem(EXPLORER_THEME_STORAGE_KEY) || undefined; }
    catch { return undefined; }
  });
  const theme = useMemo(() => getExplorerTheme(themeId), [themeId]);

  useEffect(() => { load(); }, [load]);
  useEffect(() => {
    applyExplorerTheme(theme);
    try { localStorage.setItem(EXPLORER_THEME_STORAGE_KEY, theme.id); }
    catch { /* localStorage can be unavailable in hardened shells */ }
  }, [theme]);

  if (loading) {
    return <div className="app" data-theme={theme.id}><div /><div className="main"><div className="empty row" style={{ justifyContent: "center", gap: 10 }}><span className="spin" /> loading pipeline…</div></div></div>;
  }
  if (error) {
    return <div className="pad"><div className="card"><div className="card-bd">
      <b style={{ color: "var(--sev-hard)" }}>Backend unreachable.</b>
      <p className="muted">{error}</p>
      <p className="faint">Start it: <code className="mono">python -m tools.explorer.server</code> (or run.ps1), then reload.</p>
    </div></div></div>;
  }

  return (
    <div className="app" data-theme={theme.id}>
      <nav className="nav">
        <div className="nav-brand"><span className="dot" aria-hidden><HypsometricKey variant="mark" /></span> Lofn Explorer</div>
        <ThemeNavLink to="/" end glyph="⌂" label="Bridge" shortcut="g h" />
        <div className="nav-section">Pipeline</div>
        <ThemeNavLink to="/atlas" glyph={theme.navGlyphs.atlas} label="Atlas" shortcut="g a" />
        <ThemeNavLink to="/gates" glyph={theme.navGlyphs.gates} label="Gate Center" shortcut="g g" />
        <ThemeNavLink to="/library" glyph={theme.navGlyphs.library} label="Library" shortcut="g l" />
        <ThemeNavLink to="/changes" glyph="±" label="Working set" shortcut="g w" />
        <div className="nav-section">Studio</div>
        <ThemeNavLink to="/studio/flow" glyph={theme.navGlyphs.flow} label="Flow Lab" shortcut="g s f" />
        <ThemeNavLink to="/studio/run" glyph={theme.navGlyphs.run} label="Run Bench" shortcut="g s r" />
        <ThemeNavLink to="/studio/compare" glyph={theme.navGlyphs.compare} label="Compare" shortcut="g s c" />
        <div className="nav-spacer" />
        <ThemeSelector theme={theme} value={theme.id} onChange={setThemeId} />
        <button className="nav-link nav-button" onClick={() => window.dispatchEvent(new CustomEvent("open-palette"))}>
          <ThemeIcon glyph={theme.navGlyphs.search} />
          <span className="nav-label">Search</span>
          <span className="k">⌘K</span>
        </button>
        <StateSpine repoRoot={repoRoot} />
      </nav>
      <div className="main">
        <Routes>
          <Route path="/" element={<Bridge />} />
          <Route path="/atlas" element={<Atlas />} />
          <Route path="/changes" element={<WorkingSet />} />
          <Route path="/edit" element={<EditorView />} />
          <Route path="/gates" element={<GateCenter />} />
          <Route path="/library" element={<Library />} />
          <Route path="/studio/flow" element={<FlowLab />} />
          <Route path="/studio/run" element={<RunBench />} />
          <Route path="/studio/compare" element={<Compare />} />
        </Routes>
      </div>
      <CommandPalette />
      <NavShortcuts />
    </div>
  );
}

function ThemeNavLink({ to, glyph, label, shortcut, end }: {
  to: string; glyph: string; label: string; shortcut: string; end?: boolean;
}) {
  return (
    <NavLink to={to} end={end} className="nav-link">
      <ThemeIcon glyph={glyph} />
      <span className="nav-label">{label}</span>
      <span className="k">{shortcut}</span>
    </NavLink>
  );
}

function ThemeIcon({ glyph }: { glyph: string }) {
  return <span className="theme-icon" aria-hidden="true">{glyph}</span>;
}

function ThemeSelector({ theme, value, onChange }: {
  theme: ExplorerTheme; value: string; onChange: (id: string) => void;
}) {
  return (
    <div className="theme-switcher">
      <div className="theme-preview" style={theme.art ? { backgroundImage: `url("${theme.art}")` } : undefined}>
        <span className="theme-preview-mark">{theme.brandGlyph}</span>
      </div>
      <div className="theme-copy">
        <label className="field-lbl" htmlFor="theme-select">Theme</label>
        <select id="theme-select" className="input theme-select" value={value} onChange={(e) => onChange(e.target.value)}>
          {EXPLORER_THEMES.map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}
        </select>
        <div className="theme-swatches" aria-hidden="true">
          {theme.swatches.slice(0, 5).map((color) => <span key={color} className="theme-swatch" style={{ background: color }} />)}
        </div>
      </div>
    </div>
  );
}

/** The state spine — the instrument's five vitals, encoded BY EXCEPTION: quiet
   and near-invisible when green, recruiting color + a count only on deviation (a
   smoke detector, not a gauge cluster). Always present; clicking opens the Bridge.
   Drift/sync read "unknown" (idle) until checked — a stale green is never shown. */
type SpineTone = "ok" | "warn" | "err" | "info" | "idle";

function StateSpine({ repoRoot }: { repoRoot: string }) {
  const { verify, git } = useStore();
  const drift = useHealth((s) => s.drift);
  const sync = useHealth((s) => s.sync);
  const lastRun = useHealth((s) => s.lastRun);
  const nav = useNavigate();
  const short = repoRoot.replace(/\\/g, "/").split("/").slice(-2).join("/");
  const missing = (verify?.missing?.length || 0) + (verify?.orphan_step_files?.length || 0);
  const dirty = Object.keys(git).length;

  const signals: { key: string; label: string; tone: SpineTone; n: number }[] = [
    { key: "manifest", label: "manifest", tone: verify && (!verify.ok || missing > 0) ? "err" : "ok", n: missing },
    { key: "drift", label: "prose↔yaml drift", tone: drift.state !== "done" ? "idle" : drift.count ? "warn" : "ok", n: drift.count },
    { key: "sync", label: "library sync", tone: sync.state !== "done" ? "idle" : sync.drifted.length ? "warn" : "ok", n: sync.drifted.length },
    { key: "tree", label: "working tree", tone: dirty ? "info" : "ok", n: dirty },
    { key: "run", label: "last run", tone: runSpineTone(lastRun), n: 0 },
  ];
  const anyDegraded = signals.some((s) => s.tone === "err" || s.tone === "warn");

  return (
    <div className="nav-foot">
      <button className="spine" onClick={() => nav("/")} title="Bridge — the instrument's vitals (g h)">
        {signals.map((s) => <SpineLed key={s.key} tone={s.tone} n={s.n} label={s.label} />)}
        <span className="spine-cap">{anyDegraded ? "needs attention" : "vitals"}</span>
      </button>
      <div className="mono faint" style={{ marginTop: 6, fontSize: 10.5 }} title={repoRoot}>…/{short}</div>
    </div>
  );
}

function SpineLed({ tone, n, label }: { tone: SpineTone; n: number; label: string }) {
  const color =
    tone === "err" ? "var(--sev-hard)" :
    tone === "warn" ? "var(--sev-flag)" :
    tone === "info" ? "var(--dirty)" :
    tone === "idle" ? "var(--text-faint)" : "var(--border-strong)";
  const loud = tone === "err" || tone === "warn" || (tone === "info" && n > 0);
  return (
    <span className="spine-led" title={`${label}${n ? `: ${n}` : ""}`}>
      <span className="spine-dot" style={{ background: color, boxShadow: loud ? `0 0 6px ${color}` : "none", opacity: tone === "ok" ? 0.5 : 1 }} />
      {loud && n > 0 && <span className="spine-n" style={{ color }}>{n}</span>}
    </span>
  );
}

function runSpineTone(lr: { state: string; summary: any }): SpineTone {
  if (lr.state !== "done" || !lr.summary) return "idle";
  if (lr.summary.budget?.aborted_on_cap) return "warn";
  if (lr.summary.status === "failed" || lr.summary.status === "cancelled") return "err";
  return "ok";
}

/** `g a/g/l` chords + `⌘K` palette (keyboard-first, EXPERT_OPERATOR). */
function NavShortcuts() {
  const nav = useNavigate();
  useEffect(() => {
    // mode: "" idle, "g" after g, "gs" after g→s (studio sub-chord)
    let mode: "" | "g" | "gs" = "";
    let t: any;
    const arm = (m: "g" | "gs") => { mode = m; clearTimeout(t); t = setTimeout(() => (mode = ""), 800); };
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      const inField = tag === "INPUT" || tag === "TEXTAREA" || (e.target as HTMLElement)?.isContentEditable;
      if ((e.key === "k" || e.key === "K") && (e.metaKey || e.ctrlKey)) {
        e.preventDefault(); window.dispatchEvent(new CustomEvent("open-palette")); return;
      }
      if (inField) return;
      if (mode === "gs") {
        if (e.key === "f") nav("/studio/flow");
        else if (e.key === "r") nav("/studio/run");
        else if (e.key === "c") nav("/studio/compare");
        mode = "";
      } else if (mode === "g") {
        if (e.key === "s") { arm("gs"); return; }
        if (e.key === "h") nav("/");
        if (e.key === "a") nav("/atlas");
        if (e.key === "g") nav("/gates");
        if (e.key === "l") nav("/library");
        if (e.key === "w") nav("/changes");
        mode = "";
      } else if (e.key === "g") {
        arm("g");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [nav]);
  return null;
}
