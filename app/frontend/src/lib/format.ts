export function isNil(v: unknown): v is null | undefined {
  return v === null || v === undefined || (typeof v === "number" && Number.isNaN(v));
}

export function num(v: unknown, digits = 2): string {
  if (isNil(v)) return "—";
  const n = Number(v);
  if (Number.isNaN(n)) return "—";
  return n.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function pct(v: unknown, digits = 2): string {
  if (isNil(v)) return "—";
  const n = Number(v);
  if (Number.isNaN(n)) return "—";
  return `${(n * 100).toFixed(digits)}%`;
}

export function signedPct(v: unknown, digits = 2): string {
  if (isNil(v)) return "—";
  const n = Number(v);
  if (Number.isNaN(n)) return "—";
  const s = (n * 100).toFixed(digits);
  return `${n > 0 ? "+" : ""}${s}%`;
}

export function compact(v: unknown): string {
  if (isNil(v)) return "—";
  const n = Number(v);
  if (Number.isNaN(n)) return "—";
  return Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(n);
}

export function fmtDate(iso?: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function fmtDuration(sec?: number | null): string {
  if (isNil(sec)) return "—";
  const s = Number(sec);
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  const r = Math.round(s % 60);
  return `${m}m ${r}s`;
}

export function toneFor(v: unknown, invert = false): "pos" | "neg" | "flat" {
  if (isNil(v)) return "flat";
  const n = Number(v);
  if (Number.isNaN(n) || n === 0) return "flat";
  const positive = invert ? n < 0 : n > 0;
  return positive ? "pos" : "neg";
}
