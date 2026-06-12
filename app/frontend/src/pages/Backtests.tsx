import { useMemo } from "react";
import { useRunContext } from "../lib/runContext";
import { useSeries } from "../lib/hooks";
import { PageHeader } from "../components/PageHeader";
import { Panel, StatusBadge } from "../components/ui";
import { RunForm } from "../components/RunForm";
import { RunsTable } from "../components/RunsTable";
import { MetricsGrid, RunMetaBar } from "../components/RunSummary";
import { DataState } from "../components/DataState";
import { buildRows, DrawdownChart, EquityChart } from "../components/charts";
import { LogViewer, OutputsList } from "../components/RunArtifacts";

export function Backtests() {
  const { selected, runs } = useRunContext();
  const { data: series, isLoading } = useSeries(selected?.run_id, selected?.status);
  const rows = useMemo(() => buildRows(series), [series]);

  return (
    <div className="space-y-5">
      <PageHeader
        title="Backtests"
        desc="Configure and launch engine runs as background jobs. Results stream live and are stored per-run."
      />

      <div className="grid grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)]">
        <Panel title="Run Configuration" subtitle="Engine launches via approved CLI flags">
          <RunForm />
        </Panel>

        <div className="space-y-3">
          {selected ? (
            <>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-sm text-terminal-text">{selected.run_id}</span>
                  <StatusBadge status={selected.status} />
                </div>
              </div>
              <RunMetaBar run={selected} />
              {selected.status === "completed" && <MetricsGrid metrics={selected.metrics} />}
              <Panel title="Equity Curve">
                <DataState loading={isLoading} available={series?.available} reason={series?.reason}>
                  <div className="px-2 pb-3 pt-1">
                    <EquityChart rows={rows} height={220} />
                  </div>
                </DataState>
              </Panel>
              <Panel title="Drawdown">
                <DataState loading={isLoading} available={series?.available} reason={series?.reason}>
                  <div className="px-2 pb-3 pt-1">
                    <DrawdownChart rows={rows} />
                  </div>
                </DataState>
              </Panel>
            </>
          ) : (
            <Panel title="Selected Run">
              <div className="p-8 text-center text-sm text-terminal-muted">
                Launch a run to see live results here.
              </div>
            </Panel>
          )}
        </div>
      </div>

      {selected && (
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
          <LogViewer run={selected} />
          <OutputsList run={selected} />
        </div>
      )}

      <Panel title="Run Registry" subtitle={`${runs.length} runs`}>
        <RunsTable runs={runs} />
      </Panel>
    </div>
  );
}
