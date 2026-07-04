/**
 * UnifiedDiff — render a raw unified-diff STRING with the v1 DiffPanel coloring.
 *
 * The v1 `components/DiffPanel` fetches a diff by path from the file API; the
 * studio's compare/promote endpoints already hand back the unified block, so this
 * sibling renders a supplied string (no fetch) while reusing the exact same
 * severity register — added green, removed red, hunk header violet, file headers
 * faint. Kept dependency-light and calm to match the cockpit.
 */
export function UnifiedDiff({
  diff,
  empty = "Identical — no differences.",
  maxHeight,
}: {
  diff: string;
  empty?: string;
  maxHeight?: number | string;
}) {
  if (!diff || !diff.trim()) {
    return <div className="empty" style={{ padding: 24, fontSize: 12 }}>{empty}</div>;
  }
  return (
    <div
      className="scroll mono"
      style={{
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderRadius: "var(--r-sm)",
        fontSize: 11.5,
        padding: "8px 0",
        maxHeight: maxHeight ?? 420,
      }}
    >
      {diff.split("\n").map((line, i) => {
        let color = "var(--text-dim)";
        let bg = "transparent";
        if (line.startsWith("+") && !line.startsWith("+++")) {
          color = "var(--ok)";
          bg = "color-mix(in srgb, var(--ok) 10%, transparent)";
        } else if (line.startsWith("-") && !line.startsWith("---")) {
          color = "var(--sev-hard)";
          bg = "color-mix(in srgb, var(--sev-hard) 10%, transparent)";
        } else if (line.startsWith("@@")) {
          color = "var(--accent)";
        } else if (
          line.startsWith("diff") || line.startsWith("index") ||
          line.startsWith("+++") || line.startsWith("---")
        ) {
          color = "var(--text-faint)";
        }
        return (
          <div key={i} style={{ color, background: bg, padding: "0 10px", whiteSpace: "pre" }}>
            {line || " "}
          </div>
        );
      })}
    </div>
  );
}
