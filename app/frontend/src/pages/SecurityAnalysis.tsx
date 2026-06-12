import { useMemo } from "react";
import { useRunContext } from "../lib/runContext";
import { useSeries } from "../lib/hooks";
import { PageHeader, NoRun } from "../components/PageHeader";
import { Panel, StatusBadge } from "../components/ui";
import { DataState } from "../components/DataState";
import { TableCard } from "../components/DataTable";
import { MetricsGrid } from "../components/RunSummary";
import { buildRows, EquityChart, RegimeLegend, RegimePriceChart } from "../components/charts";

export function SecurityAnalysis() {
  const { selected } = useRunContext();
  const { data: series, isLoading } = useSeries(selected?.run_id, selected?.status);
  const rows = useMemo(() => buildRows(series), [series]);

  if (!selected) {
    return (
      <div>
        <PageHeader title="Security Analysis" desc="Per-security performance, feature coverage, and data provenance." />
        <NoRun />
      </div>
    );
  }

  return (
    <div className="space-y-5">
      <PageHeader
        title="Security Analysis"
        desc="Per-security performance, feature coverage, and data provenance for the active run."
        right={
          <div className="flex items-center gap-2 text-xs text-terminal-muted">
            <span className="font-mono">{selected.ticker || "—"}</span>
            <StatusBadge status={selected.status} />
          </div>
        }
      />

      {selected.status === "completed" && <MetricsGrid metrics={selected.metrics} />}

      <Panel title="Price & Regime" right={<RegimeLegend />}>
        <DataState loading={isLoading} available={series?.available} reason={series?.reason}>
          <div className="px-2 pb-3 pt-1">
            <RegimePriceChart rows={rows} height={260} />
          </div>
        </DataState>
      </Panel>

      <Panel title="Equity Curve" subtitle="Strategy vs Buy & Hold">
        <DataState loading={isLoading} available={series?.available} reason={series?.reason}>
          <div className="px-2 pb-3 pt-1">
            <EquityChart rows={rows} height={220} />
          </div>
        </DataState>
      </Panel>

      <TableCard run={selected} file="stats" title="Full Statistics" subtitle="stats.csv" maxHeight={460} />

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <TableCard run={selected} file="feature_summary" title="Feature Summary" subtitle="feature_summary.csv" />
        <TableCard run={selected} file="factor_data_coverage" title="Factor Coverage" subtitle="factor_data_coverage.csv" />
        <TableCard run={selected} file="feature_group_metadata" title="Feature Groups" subtitle="feature_group_metadata.csv" />
        <TableCard run={selected} file="data_provider_metadata" title="Data Provenance" subtitle="data_provider_metadata.csv" />
      </div>
    </div>
  );
}
