import { useEffect } from "react";
import { NavLink, Route, Routes, useNavigate } from "react-router-dom";
import { useStore } from "./store";
import { Atlas } from "./views/Atlas";
import { EditorView } from "./views/EditorView";
import { GateCenter } from "./views/GateCenter";
import { Library } from "./views/Library";
import { CommandPalette } from "./components/CommandPalette";
import { FlowLab } from "./studio/views/FlowLab";
import { RunBench } from "./studio/views/RunBench";
import { Compare } from "./studio/views/Compare";

export function App() {
  const { loading, error, verify, load, repoRoot } = useStore();
  useEffect(() => { load(); }, [load]);

  if (loading) {
    return <div className="app"><div /><div className="main"><div className="empty row" style={{ justifyContent: "center", gap: 10 }}><span className="spin" /> loading pipeline…</div></div></div>;
  }
  if (error) {
    return <div className="pad"><div className="card"><div className="card-bd">
      <b style={{ color: "var(--sev-hard)" }}>Backend unreachable.</b>
      <p className="muted">{error}</p>
      <p className="faint">Start it: <code className="mono">python -m tools.explorer.server</code> (or run.ps1), then reload.</p>
    </div></div></div>;
  }

  return (
    <div className="app">
      <nav className="nav">
        <div className="nav-brand"><span className="dot" /> Lofn Explorer</div>
        <div className="nav-section">Pipeline</div>
        <NavLink to="/" end className="nav-link">◆ Atlas <span className="k">g a</span></NavLink>
        <NavLink to="/gates" className="nav-link">▤ Gate Center <span className="k">g g</span></NavLink>
        <NavLink to="/library" className="nav-link">▦ Library <span className="k">g l</span></NavLink>
        <div className="nav-section">Studio</div>
        <NavLink to="/studio/flow" className="nav-link">⚗ Flow Lab <span className="k">g s f</span></NavLink>
        <NavLink to="/studio/run" className="nav-link">▷ Run Bench <span className="k">g s r</span></NavLink>
        <NavLink to="/studio/compare" className="nav-link">⇄ Compare <span className="k">g s c</span></NavLink>
        <div className="nav-spacer" />
        <button className="nav-link" style={{ width: "auto" }} onClick={() => window.dispatchEvent(new CustomEvent("open-palette"))}>
          ⌕ Search <span className="k">⌘K</span>
        </button>
        <VerifyFoot ok={verify?.ok ?? true} repoRoot={repoRoot} />
      </nav>
      <div className="main">
        <Routes>
          <Route path="/" element={<Atlas />} />
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

function VerifyFoot({ ok, repoRoot }: { ok: boolean; repoRoot: string }) {
  const short = repoRoot.replace(/\\/g, "/").split("/").slice(-2).join("/");
  return (
    <div className="nav-foot">
      <div className="row" style={{ gap: 6 }}>
        <span className={`badge ${ok ? "ok" : "err"}`}><span className="d" />{ok ? "manifest ok" : "manifest drift"}</span>
      </div>
      <div className="mono faint" style={{ marginTop: 6, fontSize: 10.5 }} title={repoRoot}>…/{short}</div>
    </div>
  );
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
        if (e.key === "a") nav("/");
        if (e.key === "g") nav("/gates");
        if (e.key === "l") nav("/library");
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
