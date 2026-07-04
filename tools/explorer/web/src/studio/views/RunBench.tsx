import { useCallback } from "react";
import { useSearchParams } from "react-router-dom";
import { LaunchForm } from "./RunBench-Launch";
import { LiveMonitor } from "./RunBench-Live";
import { RunHistory } from "./RunBench-History";

/**
 * Run Bench (plan §8.3) — launch + live-monitor + replay a studio run.
 *
 * Two surfaces, selected by the `?run=<id>` query param:
 *   - no run selected: the launch form + the run-history table.
 *   - a run selected: the live monitor (which replays identically from the
 *     runlog for a finished run — run.json + the SSE replay reconstruct the
 *     same view with no live-only state).
 *
 * The selection lives in the URL so a run view is shareable / bookmarkable and
 * survives a reload (deep-link replay).
 */
export function RunBench() {
  const [params, setParams] = useSearchParams();
  const runId = params.get("run");

  const openRun = useCallback((id: string) => {
    setParams((prev) => {
      const p = new URLSearchParams(prev);
      p.set("run", id);
      return p;
    });
  }, [setParams]);

  const clearRun = useCallback(() => {
    setParams((prev) => {
      const p = new URLSearchParams(prev);
      p.delete("run");
      return p;
    });
  }, [setParams]);

  if (runId) {
    return (
      <div className="pad">
        <LiveMonitor runId={runId} onExit={clearRun} />
      </div>
    );
  }

  return (
    <div className="pad" style={{ display: "grid", gap: 14 }}>
      <LaunchForm onLaunched={openRun} />
      {/* refreshKey unused-but-stable: the history reloads on mount; a launch
          navigates into the run view, and returning remounts this branch. */}
      <RunHistory onOpen={openRun} refreshKey={0} />
    </div>
  );
}
