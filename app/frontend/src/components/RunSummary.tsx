import { AlertTriangle } from "lucide-react";
import type { Metrics, RunMeta } from "../lib/types";
import { KpiCard, Tag } from "./ui";
import { fmtDuration, num, pct, signedPct, toneFor } from "../lib/format";

export function MetricsGrid({ metrics }: { metrics?: Metrics | null }) {
  const m = metrics || {};
  const cards: { label: string; value: string; tone: "pos" | "neg" | "flat"; sub?: string }[] = [
    { label: "Sharpe", value: num(m.strategy_sharpe), tone: toneFor(m.strategy_sharpe) },
    { label: "Total Return", value: signedPct(m.strategy_total_return), tone: toneFor(m.strategy_total_return), sub: `B&H ${signedPct(m.buy_hold_total_return)}` },
    { label: "CAGR", value: signedPct(m.strategy_cagr), tone: toneFor(m.strategy_cagr) },
    { label: "Max Drawdown", value: pct(m.strategy_max_drawdown), tone: toneFor(m.strategy_max_drawdown) },
    { label: "Sortino", value: num(m.strategy_sortino), tone: toneFor(m.strategy_sortino) },
    { label: "Calmar", value: num(m.strategy_calmar), tone: toneFor(m.strategy_calmar) },
    { label: "Win Rate", value: pct(m.win_rate), tone: "flat" },
    { label: "Turnover", value: num(m.turnover, 3), tone: "flat", sub: `${num(m.trades, 0)} trades` },
  ];
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
      {cards.map((c) => (
        <KpiCard key={c.label} label={c.label} value={c.value} tone={c.tone} sub={c.sub} />
      ))}
    </div>
  );
}

export function RunMetaBar({ run }: { run: RunMeta }) {
  const req = run.request;
  const facts: [string, string][] = [
    ["Mode", run.mode],
    ["Config", req.config],
    ["Ticker", run.ticker || req.ticker || "—"],
    ["Benchmark", run.benchmark || req.benchmark || "—"],
    ["Quick", req.quick ? "yes" : "no"],
    ["Profile", req.profile || "—"],
    ["Duration", fmtDuration(run.duration_sec)],
    ["Return code", run.return_code != null ? String(run.return_code) : "—"],
  ];
  return (
    <div className="rounded-lg border border-terminal-border bg-terminal-panel p-3">
      <div className="flex flex-wrap gap-x-6 gap-y-2">
        {facts.map(([k, v]) => (
          <div key={k}>
            <div className="micro">{k}</div>
            <div className="font-mono text-xs text-terminal-text">{v}</div>
          </div>
        ))}
      </div>
      {run.error && (
        <div className="mt-3 flex items-start gap-2 rounded border border-bear/30 bg-bear/5 px-3 py-2 text-xs text-bear">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span className="font-mono">{run.error}</span>
        </div>
      )}
      {run.config_warnings && run.config_warnings.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-2">
          {run.config_warnings.map((w, i) => (
            <Tag key={i} tone="side">
              {w.field}: {w.message}
            </Tag>
          ))}
        </div>
      )}
    </div>
  );
}
