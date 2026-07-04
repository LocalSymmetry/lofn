import { useEffect, useRef, useState } from "react";
import { studioApi, type RunlogEvent } from "../api";

export interface RunStreamState {
  events: RunlogEvent[];
  last: RunlogEvent | null;
  connected: boolean;
  ended: boolean;                 // a `run_end` frame arrived (or a finished run drained)
  error: string | null;
}

/**
 * Tail GET /runs/{id}/events (SSE) and accumulate runlog frames (plan §5: one
 * event stream, two consumers). Replays the existing log then streams live;
 * the stream self-terminates after `run_end` for a finished run. Every frame is
 * already key-safe (redaction happens at write time, S2).
 *
 * Pass `id = null` to hold the hook inactive (no connection). Uses the native
 * EventSource — same-origin, no auth header needed; the vite proxy forwards /api.
 */
export function useRunStream(id: string | null): RunStreamState {
  const [events, setEvents] = useState<RunlogEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [ended, setEnded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    // Reset per run id.
    setEvents([]);
    setConnected(false);
    setEnded(false);
    setError(null);
    if (!id) return;

    const es = new EventSource(studioApi.runEventsUrl(id));
    esRef.current = es;

    es.onopen = () => setConnected(true);

    es.onmessage = (e: MessageEvent) => {
      if (!e.data) return;              // keep-alive comment frames don't fire onmessage
      let ev: RunlogEvent;
      try {
        ev = JSON.parse(e.data) as RunlogEvent;
      } catch {
        return;
      }
      setEvents((prev) => [...prev, ev]);
      if (ev.event === "run_end") {
        setEnded(true);
        es.close();
        setConnected(false);
      }
    };

    es.onerror = () => {
      // EventSource auto-reconnects; only surface an error if we never ended.
      // A finished-run stream closes cleanly server-side and lands here with
      // readyState CLOSED — treat that as ended, not an error.
      if (es.readyState === EventSource.CLOSED) {
        setConnected(false);
        setEnded((wasEnded) => {
          if (!wasEnded) setError("stream closed");
          return wasEnded;
        });
      }
    };

    return () => {
      es.close();
      esRef.current = null;
    };
  }, [id]);

  return {
    events,
    last: events.length ? events[events.length - 1] : null,
    connected,
    ended,
    error,
  };
}
