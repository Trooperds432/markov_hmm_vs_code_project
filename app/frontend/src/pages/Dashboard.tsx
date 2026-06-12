import { useMemo } from "react";
import { useRunContext } from "../lib/runContext";
import { useSeries } from "../lib/hooks";
import { PageHeader, NoRun, RunningNotice } from "../components/PageHeader";
import { MetricsGrid } from "../components/RunSummary";
import { Panel, StatusBadge } from "../components/ui";
import { DataState } from "../components/DataState";
import {
  buildRows,
  DrawdownChart,
  EquityChart,
  ProbabilityChart,
  RegimeLegend,
  RegimePriceChart,
} from "../components/charts";
import { RunsTable } from "../components/RunsTable";

export function Dashboard() {
  const { selected, runs } = useRunContext();
  const { data: series, isLoading, error } = useSeries(selected?.run_id, selected?.status);
  const rows = useMemo(() => buildRows(series), [series]);

  const stats = useMemo(() => {
    const completed = runs.filter((r) => r.status === "completed");
    const active = runs.filter((r) => r.status === "running" || r.status === "queued");
    return { total: runs.length, completed: completed.length, active: active.length };
  }, [runs]);

  return (
    <div className="space-y-5">
      <PageHeader
        title="Dashboard"
        desc="HMM-driven regime detection and strategy performance for the active research run."
        right={
          selected && (
            <div className="flex items-center gap-2 text-xs text-terminal-muted">
              <span className="font-mono">{selected.run_id}</span>
              <StatusBadge status={selected.status} />
            </div>
          )
        }
      />

      <div className="grid grid-cols-3 gap-3">
        <Panel className="px-4 py-3">
          <div className="micro">Total Runs</div>
          <div className="mt-1 font-mono text-2xl font-semibold tnum">{stats.total}</div>
        </Panel>
        <Panel className="px-4 py-3">
          <div className="micro">Completed</div>
          <div className="mt-1 font-mono text-2xl font-semibold tnum text-bull">{stats.completed}</div>
        </Panel>
        <Panel className="px-4 py-3">
          <div className="micro">Active</div>
          <div className="mt-1 font-mono text-2xl font-semibold tnum text-accent">{stats.active}</div>
        </Panel>
      </div>

      {!selected ? (
        <NoRun />
      ) : selected.status !== "completed" ? (
        <Panel title="Active Run">
          <RunningNotice />
        </Panel>
      ) : (
        <>
          <MetricsGrid metrics={selected.metrics} />

          <Panel
            title="Regime Detector"
            subtitle="Price with HMM regime shading"
            right={<RegimeLegend />}
          >
            <DataState loading={isLoading} error={error} available={series?.available} reason={series?.reason}>
              <div className="px-2 pb-3 pt-1">
                <RegimePriceChart rows={rows} />
              </div>
            </DataState>
          </Panel>

          <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
            <Panel title="Equity Curve" subtitle="Strategy vs Buy & Hold">
              <DataState loading={isLoading} available={series?.available} reason={series?.reason}>
                <div className="px-2 pb-3 pt-1">
                  <EquityChart rows={rows} />
                </div>
              </DataState>
            </Panel>
            <Panel title="State Probabilities" subtitle="HMM posterior distribution">
              <DataState loading={isLoading} available={series?.available} reason={series?.reason}>
                <div className="px-2 pb-3 pt-1">
                  <ProbabilityChart rows={rows} />
                </div>
              </DataState>
            </Panel>
          </div>

          <Panel title="Drawdown" subtitle="Rolling strategy drawdown">
            <DataState loading={isLoading} available={series?.available} reason={series?.reason}>
              <div className="px-2 pb-3 pt-1">
                <DrawdownChart rows={rows} />
              </div>
            </DataState>
          </Panel>
        </>
      )}

      <Panel title="Recent Runs" subtitle="Click a row to make it active">
        <RunsTable runs={runs.slice(0, 8)} />
      </Panel>
    </div>
  );
}
