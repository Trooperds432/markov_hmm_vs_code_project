import { useMemo } from "react";
import { ShieldAlert } from "lucide-react";
import { useRunContext } from "../lib/runContext";
import { useSeries } from "../lib/hooks";
import { PageHeader } from "../components/PageHeader";
import { Panel } from "../components/ui";
import { RunForm } from "../components/RunForm";
import { MetricsGrid } from "../components/RunSummary";
import { DataState } from "../components/DataState";
import { buildRows, EquityChart } from "../components/charts";
import { LogViewer } from "../components/RunArtifacts";

export function PaperTrading() {
  const { selected } = useRunContext();
  const { data: series, isLoading } = useSeries(selected?.run_id, selected?.status);
  const rows = useMemo(() => buildRows(series), [series]);

  return (
    <div className="space-y-5">
      <PageHeader title="Paper Trading" desc="Simulated forward execution using the same engine. No broker, no real orders." />

      <div className="flex items-start gap-3 rounded-lg border border-side/30 bg-side/5 px-4 py-3">
        <ShieldAlert className="mt-0.5 h-5 w-5 shrink-0 text-side" />
        <div className="text-xs text-terminal-text">
          <span className="font-semibold text-side">Simulation only.</span> Paper trading runs the
          research engine over historical/simulated data. It does not connect to any broker, does
          not place real or paper orders with an exchange, and is not investment advice.
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)]">
        <Panel title="Paper Run" subtitle="mode = paper">
          <RunForm defaultMode="paper" />
        </Panel>
        <div className="space-y-3">
          {selected?.status === "completed" ? (
            <>
              <MetricsGrid metrics={selected.metrics} />
              <Panel title="Simulated Equity">
                <DataState loading={isLoading} available={series?.available} reason={series?.reason}>
                  <div className="px-2 pb-3 pt-1">
                    <EquityChart rows={rows} height={240} />
                  </div>
                </DataState>
              </Panel>
            </>
          ) : selected ? (
            <LogViewer run={selected} />
          ) : (
            <Panel title="Results">
              <div className="p-8 text-center text-sm text-terminal-muted">
                Launch a paper run to view the simulated session.
              </div>
            </Panel>
          )}
        </div>
      </div>
    </div>
  );
}
