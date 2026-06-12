import { useMemo } from "react";
import { useRunContext } from "../lib/runContext";
import { useSeries } from "../lib/hooks";
import { PageHeader } from "../components/PageHeader";
import { Panel } from "../components/ui";
import { RunForm } from "../components/RunForm";
import { MetricsGrid } from "../components/RunSummary";
import { DataState } from "../components/DataState";
import { TableCard } from "../components/DataTable";
import { buildRows, EquityChart } from "../components/charts";

export function Portfolio() {
  const { selected } = useRunContext();
  const { data: series, isLoading } = useSeries(selected?.run_id, selected?.status);
  const rows = useMemo(() => buildRows(series), [series]);

  return (
    <div className="space-y-5">
      <PageHeader
        title="Portfolio"
        desc="Multi-asset and universe runs. Launch portfolio mode with a portfolio config to allocate across tickers."
      />

      <div className="grid grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)]">
        <Panel title="Portfolio Run" subtitle="Use mode = portfolio / universe">
          <RunForm defaultMode="portfolio" />
        </Panel>
        <div className="space-y-3">
          {selected?.status === "completed" ? (
            <>
              <MetricsGrid metrics={selected.metrics} />
              <Panel title="Equity Curve">
                <DataState loading={isLoading} available={series?.available} reason={series?.reason}>
                  <div className="px-2 pb-3 pt-1">
                    <EquityChart rows={rows} height={240} />
                  </div>
                </DataState>
              </Panel>
            </>
          ) : (
            <Panel title="Results">
              <div className="p-8 text-center text-sm text-terminal-muted">
                Launch a portfolio run to view allocation results.
              </div>
            </Panel>
          )}
        </div>
      </div>

      {selected && (
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
          <TableCard run={selected} file="stats" title="Statistics" subtitle="stats.csv" />
          <TableCard run={selected} file="state_summary" title="State Summary" subtitle="state_summary.csv" />
        </div>
      )}
    </div>
  );
}
