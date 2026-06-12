import { Download, RotateCw, Trash2, XCircle, ChevronRight } from "lucide-react";
import type { RunMeta } from "../lib/types";
import { useRunActions } from "../lib/hooks";
import { useRunContext } from "../lib/runContext";
import { downloadFile } from "../lib/api";
import { fmtDate, fmtDuration, num, signedPct, toneFor } from "../lib/format";
import { Button, EmptyState, StatusBadge, cn } from "./ui";

function MetricCell({ value, kind }: { value?: number | null; kind: "pct" | "num" }) {
  const tone = toneFor(value);
  const text = kind === "pct" ? signedPct(value) : num(value);
  return (
    <span className={cn("font-mono tnum", tone === "pos" ? "text-bull" : tone === "neg" ? "text-bear" : "text-terminal-muted")}>
      {text}
    </span>
  );
}

export function RunsTable({ runs }: { runs: RunMeta[] }) {
  const { rerun, cancel, remove } = useRunActions();
  const { selectedId, setSelectedId } = useRunContext();

  if (runs.length === 0) {
    return <EmptyState title="No runs yet" hint="Launch a backtest to populate the registry." />;
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-terminal-border text-left text-terminal-dim">
            <th className="px-3 py-2 font-medium">Run</th>
            <th className="px-3 py-2 font-medium">Mode</th>
            <th className="px-3 py-2 font-medium">Status</th>
            <th className="px-3 py-2 text-right font-medium">Sharpe</th>
            <th className="px-3 py-2 text-right font-medium">Return</th>
            <th className="px-3 py-2 text-right font-medium">Max DD</th>
            <th className="px-3 py-2 text-right font-medium">Created</th>
            <th className="px-3 py-2 text-right font-medium">Dur</th>
            <th className="px-3 py-2 text-right font-medium">Actions</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => {
            const active = r.run_id === selectedId;
            const isActive = r.status === "running" || r.status === "queued";
            return (
              <tr
                key={r.run_id}
                onClick={() => setSelectedId(r.run_id)}
                className={cn(
                  "cursor-pointer border-b border-terminal-border/60 transition-colors hover:bg-terminal-panel2/60",
                  active && "bg-accent/5"
                )}
              >
                <td className="px-3 py-2">
                  <div className="flex items-center gap-1.5">
                    {active ? <ChevronRight className="h-3 w-3 text-accent" /> : <span className="w-3" />}
                    <div>
                      <div className="font-mono text-terminal-text">{r.run_id}</div>
                      <div className="text-[10px] text-terminal-dim">{r.label || "—"} · {r.ticker || "?"}</div>
                    </div>
                  </div>
                </td>
                <td className="px-3 py-2 text-terminal-muted">{r.mode}</td>
                <td className="px-3 py-2"><StatusBadge status={r.status} /></td>
                <td className="px-3 py-2 text-right"><MetricCell value={r.metrics?.strategy_sharpe} kind="num" /></td>
                <td className="px-3 py-2 text-right"><MetricCell value={r.metrics?.strategy_total_return} kind="pct" /></td>
                <td className="px-3 py-2 text-right"><MetricCell value={r.metrics?.strategy_max_drawdown} kind="pct" /></td>
                <td className="px-3 py-2 text-right text-terminal-muted">{fmtDate(r.created_at)}</td>
                <td className="px-3 py-2 text-right font-mono text-terminal-muted">{fmtDuration(r.duration_sec)}</td>
                <td className="px-3 py-2">
                  <div className="flex items-center justify-end gap-1" onClick={(e) => e.stopPropagation()}>
                    {isActive ? (
                      <Button variant="danger" className="px-1.5 py-1" onClick={() => cancel.mutate(r.run_id)} title="Cancel">
                        <XCircle className="h-3.5 w-3.5" />
                      </Button>
                    ) : (
                      <Button variant="ghost" className="px-1.5 py-1" onClick={() => rerun.mutate(r.run_id)} title="Re-run">
                        <RotateCw className="h-3.5 w-3.5" />
                      </Button>
                    )}
                    {r.status === "completed" && (
                      <Button
                        variant="ghost"
                        className="px-1.5 py-1"
                        onClick={() => downloadFile(r.run_id, "stats.csv")}
                        title="Download stats.csv"
                      >
                        <Download className="h-3.5 w-3.5" />
                      </Button>
                    )}
                    {!isActive && (
                      <Button
                        variant="danger"
                        className="px-1.5 py-1"
                        onClick={() => {
                          if (confirm(`Delete run ${r.run_id}? This removes its output files.`)) remove.mutate(r.run_id);
                        }}
                        title="Delete"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
                    )}
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
