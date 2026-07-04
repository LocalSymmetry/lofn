import type { ProviderStatus } from "../types";

interface StatusLightProps {
  /** Provider name. */
  provider: string;
  /** Configured (a key resolved in env/.env.studio). Keyless providers (mock)
      report configured:true from the backend. */
  configured: boolean;
  /** Optional live-test result: true=ping ok, false=ping failed, undefined=untested. */
  ok?: boolean;
  note?: string;
}

/**
 * A provider status dot + label. Green = configured (or test ok), grey =
 * unconfigured/untested, red = a live test failed. Reuses the v1 `.badge`
 * register so it sits naturally in the cockpit.
 */
export function StatusLight({ provider, configured, ok, note }: StatusLightProps) {
  const failed = ok === false;
  const good = ok === true || (ok === undefined && configured);
  const cls = failed ? "err" : good ? "ok" : "";
  const text = failed
    ? "unreachable"
    : ok === true
      ? "ok"
      : configured
        ? "configured"
        : "not configured";

  return (
    <span className={`badge ${cls}`} title={note || `${provider}: ${text}`}>
      <span className="d" />
      <span className="mono" style={{ fontSize: 11 }}>{provider}</span>
      <span className="faint" style={{ fontSize: 10.5 }}>{text}</span>
    </span>
  );
}

/** Convenience: render a StatusLight straight from a /providers entry. */
export function ProviderLight({ p, ok, note }: { p: ProviderStatus; ok?: boolean; note?: string }) {
  return <StatusLight provider={String(p.provider)} configured={p.configured} ok={ok} note={note} />;
}
