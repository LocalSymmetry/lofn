import { useEffect, useState } from "react";
import { api } from "../api";

/** Unified diff vs HEAD, lightly colored. Simpler than a dual-model editor and
    enough to see what changed before saving/committing. */
export function DiffPanel({ path }: { path: string }) {
  const [diff, setDiff] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    api.diff(path).then((d) => live && setDiff(d.diff)).catch(() => live && setDiff(""));
    return () => { live = false; };
  }, [path]);

  if (diff == null) return <div className="empty row" style={{ justifyContent: "center", borderLeft: "1px solid var(--border)" }}><span className="spin" /></div>;
  if (!diff.trim()) return <div className="empty" style={{ borderLeft: "1px solid var(--border)" }}>No diff vs HEAD.</div>;

  return (
    <div className="scroll mono" style={{ borderLeft: "1px solid var(--border)", background: "var(--surface)", fontSize: 11.5, padding: "8px 0" }}>
      {diff.split("\n").map((line, i) => {
        let color = "var(--text-dim)";
        let bg = "transparent";
        if (line.startsWith("+") && !line.startsWith("+++")) { color = "var(--ok)"; bg = "color-mix(in srgb, var(--ok) 10%, transparent)"; }
        else if (line.startsWith("-") && !line.startsWith("---")) { color = "var(--sev-hard)"; bg = "color-mix(in srgb, var(--sev-hard) 10%, transparent)"; }
        else if (line.startsWith("@@")) color = "var(--accent)";
        else if (line.startsWith("diff") || line.startsWith("index") || line.startsWith("+++") || line.startsWith("---")) color = "var(--text-faint)";
        return <div key={i} style={{ color, background: bg, padding: "0 10px", whiteSpace: "pre" }}>{line || " "}</div>;
      })}
    </div>
  );
}
