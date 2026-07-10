import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useStore } from "../store";
import { DiffPanel } from "../components/DiffPanel";

/**
 * Working set (`g w`) — the maintainer's most common mid-session question, answered
 * diff-first: "what have I actually touched, and will any of it break a gate?"
 * Repo-wide changed files grouped by district → the existing DiffPanel, entered
 * diff-first (not file-first), with a change→consequence line coupling an edit to
 * the surface it puts at risk (edit gates.yaml → drift may now disagree).
 */
export function WorkingSet() {
  const git = useStore((s) => s.git);
  const refreshGit = useStore((s) => s.refreshGit);
  const paths = Object.keys(git);
  const [sel, setSel] = useState<string | null>(paths[0] ?? null);

  const groups = useMemo(() => {
    const by: Record<string, { path: string; code: string }[]> = {};
    for (const p of paths) {
      const top = p.split("/")[0] || "·";
      (by[top] ||= []).push({ path: p, code: git[p] });
    }
    return Object.entries(by)
      .map(([name, files]) => ({ name, files: files.sort((a, b) => a.path.localeCompare(b.path)) }))
      .sort((a, b) => b.files.length - a.files.length);
  }, [git, paths]);

  return (
    <>
      <div className="hd">
        <h1>Working set</h1>
        <span className="faint">{paths.length} uncommitted file(s) · diff-first</span>
        <div className="spacer" />
        <button className="btn sm" onClick={refreshGit}>↻ refresh</button>
      </div>
      <div style={{ flex: 1, minHeight: 0, display: "grid", gridTemplateColumns: "324px 1fr" }}>
        <div className="scroll" style={{ borderRight: "1px solid var(--border)" }}>
          {groups.map((g) => (
            <div key={g.name}>
              <div className="nav-section" style={{ display: "flex", justifyContent: "space-between", paddingRight: 12 }}>
                <span>{g.name}</span><span className="mono">{g.files.length}</span>
              </div>
              {g.files.map((f) => (
                <button key={f.path} onClick={() => setSel(f.path)}
                  className="row" style={{
                    width: "100%", textAlign: "left", border: 0, gap: 8, padding: "6px 12px", cursor: "pointer",
                    background: sel === f.path ? "var(--surface-3)" : "transparent",
                    borderBottom: "1px solid var(--surface-2)",
                  }}>
                  <span className="mono" style={{ fontSize: 10, color: codeColor(f.code), width: 18, flex: "none" }}>{f.code.trim() || "??"}</span>
                  <span className="mono" style={{ fontSize: 11, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", direction: "rtl" }}>{f.path}</span>
                </button>
              ))}
            </div>
          ))}
          {paths.length === 0 && <div className="empty">HEAD clean — nothing changed.</div>}
        </div>
        <div style={{ display: "grid", gridTemplateRows: "auto 1fr", minHeight: 0 }}>
          {sel ? (
            <>
              <ConsequenceBar path={sel} />
              <DiffPanel key={sel} path={sel} />
            </>
          ) : <div className="empty">select a changed file to see its diff vs HEAD.</div>}
        </div>
      </div>
    </>
  );
}

/** change → consequence: couple an edit to the surface it puts at risk. */
function ConsequenceBar({ path }: { path: string }) {
  const nav = useNavigate();
  const isGates = path.endsWith("gates.yaml");
  const isStep = /skills\/(music|image|video|story|orchestration)\/steps\//.test(path) || /steps\/\d\d_/.test(path);
  const isLib = /orchestration\/(personalities|panels)\//.test(path);
  return (
    <div className="row" style={{ gap: 10, padding: "8px 12px", borderBottom: "1px solid var(--border)", background: "var(--surface)", flexWrap: "wrap" }}>
      <span className="mono" style={{ fontSize: 11 }}>{path}</span>
      <div className="spacer" />
      {isGates && <button className="btn sm" title="threshold edits can make restated prose disagree" onClick={() => nav("/gates?tab=drift")}>consequence: re-check Drift →</button>}
      {isStep && <button className="btn sm" title="see the gates guarding this step" onClick={() => nav("/atlas")}>consequence: gates on the Atlas →</button>}
      {isLib && <button className="btn sm" title="canonical edit → derived index/monolith go stale" onClick={() => nav("/library")}>consequence: regenerate derived →</button>}
      <button className="btn sm primary" onClick={() => nav(`/edit?path=${encodeURIComponent(path)}`)}>open in editor →</button>
    </div>
  );
}

function codeColor(code: string): string {
  const c = code.trim();
  if (c === "??") return "var(--ok)";              // new/untracked
  if (c.includes("D")) return "var(--sev-hard)";   // deleted
  if (c.includes("A")) return "var(--ok)";         // added
  return "var(--dirty)";                            // modified
}
