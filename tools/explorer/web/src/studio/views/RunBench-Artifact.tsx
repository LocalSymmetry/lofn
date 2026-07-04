import Monaco from "@monaco-editor/react";
import { useEffect, useRef, useState } from "react";
import { api as v1api } from "../../api";
import type { FileView } from "../../types";

const RUNS_ROOT = "output/studio";

function langOf(path: string): string {
  const ext = path.split(".").pop()?.toLowerCase();
  return ({ md: "markdown", markdown: "markdown", yaml: "yaml", yml: "yaml",
    json: "json", py: "python", csv: "plaintext", txt: "plaintext" } as Record<string, string>)[ext || ""] || "markdown";
}

/**
 * Read-only Monaco viewer for a run artifact (plan §8.3: click an artifact →
 * Monaco viewer). Artifacts live under output/studio/<run-id>/; the v1 byte-safe
 * /api/file read path is jailed to the repo but not to the write-allowlist, so
 * these load straight through. Reuses the v1 offline Monaco (monaco.ts) — no CDN.
 */
export function ArtifactViewer({ runId, artifact, onClose }: {
  runId: string;
  artifact: string;
  onClose: () => void;
}) {
  const [view, setView] = useState<FileView | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const themedRef = useRef(false);
  const path = `${RUNS_ROOT}/${runId}/${artifact}`;

  useEffect(() => {
    let live = true;
    setView(null); setErr(null);
    v1api.readFile(path)
      .then((v) => { if (live) setView(v); })
      .catch((e) => { if (live) setErr(e?.message || String(e)); });
    return () => { live = false; };
  }, [path]);

  // Esc closes.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div style={{
      position: "fixed", inset: 0, background: "rgba(0,0,0,0.6)", zIndex: 60,
      display: "flex", alignItems: "center", justifyContent: "center", padding: 24,
    }} onClick={onClose}>
      <div className="card" style={{ width: "min(920px, 96vw)", height: "min(80vh, 720px)", display: "flex", flexDirection: "column" }}
           onClick={(e) => e.stopPropagation()}>
        <div className="card-hd row" style={{ justifyContent: "space-between" }}>
          <span className="row" style={{ gap: 8 }}>
            <span className="mono" style={{ fontSize: 12 }}>{artifact}</span>
            {view && <span className="faint mono" style={{ fontSize: 10.5 }}>{view.eol} · {view.size}b · {view.sha.slice(0, 8)}</span>}
            <span className="badge">read-only</span>
          </span>
          <button className="btn sm" onClick={onClose}>close <span className="kbd">Esc</span></button>
        </div>
        <div style={{ flex: 1, minHeight: 0 }}>
          {err && <div className="empty" style={{ color: "var(--sev-hard)" }}>{err}</div>}
          {!err && !view && <div className="empty row" style={{ justifyContent: "center" }}><span className="spin" /></div>}
          {view && (
            <Monaco
              language={langOf(path)}
              value={view.text}
              theme="lofn-dark"
              beforeMount={(monaco) => {
                if (themedRef.current) return;
                themedRef.current = true;
                monaco.editor.defineTheme("lofn-dark", {
                  base: "vs-dark", inherit: true, rules: [],
                  colors: { "editor.background": "#15171e", "editorGutter.background": "#15171e" },
                });
              }}
              options={{
                readOnly: true, fontSize: 12.5,
                fontFamily: "ui-monospace, Cascadia Code, Consolas, monospace",
                minimap: { enabled: false }, wordWrap: "on",
                scrollBeyondLastLine: false, tabSize: 2, smoothScrolling: true,
                padding: { top: 8 },
              }}
            />
          )}
        </div>
      </div>
    </div>
  );
}
