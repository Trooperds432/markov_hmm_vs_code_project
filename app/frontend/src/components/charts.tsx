import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { SeriesData } from "../lib/types";

const REGIME_COLORS: Record<string, string> = {
  Bull: "#13c98a",
  Bear: "#f0445a",
  Sideways: "#e0a83b",
};

export interface ChartRow {
  date: string;
  price?: number | null;
  strat?: number | null;
  bh?: number | null;
  dd?: number | null;
  bull?: number | null;
  bear?: number | null;
  side?: number | null;
  conf?: number | null;
  regime?: string | null;
}

export function buildRows(s?: SeriesData): ChartRow[] {
  if (!s || !s.dates) return [];
  return s.dates.map((date, i) => ({
    date,
    price: s.price?.[i] ?? null,
    strat: s.strategy_equity?.[i] ?? null,
    bh: s.buy_hold_equity?.[i] ?? null,
    dd: s.drawdown?.[i] ?? null,
    bull: s.prob_bull?.[i] ?? null,
    bear: s.prob_bear?.[i] ?? null,
    side: s.prob_sideways?.[i] ?? null,
    conf: s.confidence?.[i] ?? null,
    regime: s.regime?.[i] ?? null,
  }));
}

function regimeRuns(rows: ChartRow[]) {
  const runs: { start: string; end: string; label: string }[] = [];
  let cur: { start: string; end: string; label: string } | null = null;
  for (const r of rows) {
    if (!r.regime) {
      if (cur) {
        runs.push(cur);
        cur = null;
      }
      continue;
    }
    if (cur && cur.label === r.regime) {
      cur.end = r.date;
    } else {
      if (cur) runs.push(cur);
      cur = { start: r.date, end: r.date, label: r.regime };
    }
  }
  if (cur) runs.push(cur);
  return runs;
}

const axisProps = {
  stroke: "#3a465a",
  tick: { fill: "#7c8aa0", fontSize: 10 },
  tickLine: false,
};

const dateTick = (v: string) => (v ? v.slice(2, 7) : "");

function TooltipBox({ active, payload, label, fmt }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded border border-terminal-borderlit bg-terminal-panel2 px-2.5 py-2 text-[11px] shadow-lg">
      <div className="mb-1 font-mono text-terminal-muted">{label}</div>
      {payload.map((p: any) => (
        <div key={p.dataKey} className="flex items-center justify-between gap-3">
          <span className="flex items-center gap-1.5" style={{ color: p.color }}>
            <span className="h-2 w-2 rounded-sm" style={{ background: p.color }} />
            {p.name}
          </span>
          <span className="font-mono tnum text-terminal-text">
            {fmt ? fmt(p.value) : p.value?.toFixed?.(3) ?? p.value}
          </span>
        </div>
      ))}
    </div>
  );
}

export function EquityChart({ rows, height = 260 }: { rows: ChartRow[]; height?: number }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={rows} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="#161e2b" vertical={false} />
        <XAxis dataKey="date" {...axisProps} tickFormatter={dateTick} minTickGap={48} />
        <YAxis {...axisProps} width={48} domain={["auto", "auto"]} tickFormatter={(v) => v.toFixed(2)} />
        <Tooltip content={<TooltipBox fmt={(v: number) => v?.toFixed?.(3)} />} />
        <Line type="monotone" dataKey="strat" name="Strategy" stroke="#3fa7ff" dot={false} strokeWidth={1.6} isAnimationActive={false} connectNulls />
        <Line type="monotone" dataKey="bh" name="Buy & Hold" stroke="#7c8aa0" dot={false} strokeWidth={1.2} strokeDasharray="4 3" isAnimationActive={false} connectNulls />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function DrawdownChart({ rows, height = 160 }: { rows: ChartRow[]; height?: number }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={rows} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="ddFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#f0445a" stopOpacity={0.05} />
            <stop offset="100%" stopColor="#f0445a" stopOpacity={0.4} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke="#161e2b" vertical={false} />
        <XAxis dataKey="date" {...axisProps} tickFormatter={dateTick} minTickGap={48} />
        <YAxis {...axisProps} width={48} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
        <Tooltip content={<TooltipBox fmt={(v: number) => `${(v * 100).toFixed(2)}%`} />} />
        <Area type="monotone" dataKey="dd" name="Drawdown" stroke="#f0445a" fill="url(#ddFill)" strokeWidth={1.2} isAnimationActive={false} connectNulls />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function ProbabilityChart({ rows, height = 180 }: { rows: ChartRow[]; height?: number }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={rows} margin={{ top: 8, right: 12, left: 0, bottom: 0 }} stackOffset="expand">
        <CartesianGrid stroke="#161e2b" vertical={false} />
        <XAxis dataKey="date" {...axisProps} tickFormatter={dateTick} minTickGap={48} />
        <YAxis {...axisProps} width={48} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} domain={[0, 1]} />
        <Tooltip content={<TooltipBox fmt={(v: number) => (v != null ? v.toFixed(2) : "—")} />} />
        <Area type="monotone" dataKey="bull" name="Bull" stackId="1" stroke="#13c98a" fill="#13c98a" fillOpacity={0.55} isAnimationActive={false} connectNulls />
        <Area type="monotone" dataKey="side" name="Sideways" stackId="1" stroke="#e0a83b" fill="#e0a83b" fillOpacity={0.45} isAnimationActive={false} connectNulls />
        <Area type="monotone" dataKey="bear" name="Bear" stackId="1" stroke="#f0445a" fill="#f0445a" fillOpacity={0.5} isAnimationActive={false} connectNulls />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function RegimePriceChart({ rows, height = 240 }: { rows: ChartRow[]; height?: number }) {
  const runs = regimeRuns(rows);
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={rows} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="#161e2b" vertical={false} />
        {runs.map((run, i) => (
          <ReferenceArea
            key={i}
            x1={run.start}
            x2={run.end}
            fill={REGIME_COLORS[run.label] || "#56627a"}
            fillOpacity={0.1}
            stroke="none"
          />
        ))}
        <XAxis dataKey="date" {...axisProps} tickFormatter={dateTick} minTickGap={48} />
        <YAxis {...axisProps} width={48} domain={["auto", "auto"]} tickFormatter={(v) => v.toFixed(0)} />
        <Tooltip content={<TooltipBox fmt={(v: number) => v?.toFixed?.(2)} />} />
        <Line type="monotone" dataKey="price" name="Price" stroke="#d6dde8" dot={false} strokeWidth={1.5} isAnimationActive={false} connectNulls />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function RegimeLegend() {
  return (
    <div className="flex items-center gap-3">
      {Object.entries(REGIME_COLORS).map(([k, c]) => (
        <span key={k} className="flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-terminal-muted">
          <span className="h-2 w-2 rounded-sm" style={{ background: c }} />
          {k}
        </span>
      ))}
    </div>
  );
}

export function MiniBars({ data, color = "#3fa7ff", height = 40 }: { data: number[]; color?: string; height?: number }) {
  const rows = data.map((v, i) => ({ i, v }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={rows} margin={{ top: 2, right: 0, left: 0, bottom: 0 }}>
        <Area type="monotone" dataKey="v" stroke={color} fill={color} fillOpacity={0.15} strokeWidth={1.2} isAnimationActive={false} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export { REGIME_COLORS };
