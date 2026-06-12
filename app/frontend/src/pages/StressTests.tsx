import { useRunContext } from "../lib/runContext";
import { PageHeader, NoRun } from "../components/PageHeader";
import { StatusBadge, Tag } from "../components/ui";
import { TableCard } from "../components/DataTable";

export function StressTests() {
  const { selected } = useRunContext();

  if (!selected) {
    return (
      <div>
        <PageHeader title="Stress Tests" desc="Robustness checks: synthetic-sample stress, seed/window stability, and volatility forecasts." />
        <NoRun />
      </div>
    );
  }

  return (
    <div className="space-y-5">
      <PageHeader
        title="Stress Tests"
        desc="Robustness diagnostics — synthetic-sample stress, seed/window stability, overfit signals, and volatility forecasting."
        right={
          <div className="flex items-center gap-2">
            <Tag tone={selected.request.enable_stress_tests ? "accent" : "neutral"}>
              Stress {selected.request.enable_stress_tests ? "ON" : "default"}
            </Tag>
            <StatusBadge status={selected.status} />
          </div>
        }
      />

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <TableCard run={selected} file="hmm_synthetic_sample_stress_summary" title="Synthetic Stress Summary" subtitle="hmm_synthetic_sample_stress_summary.csv" />
        <TableCard run={selected} file="model_overfit_report" title="Overfit Report" subtitle="model_overfit_report.csv" />
        <TableCard run={selected} file="seed_diagnostics" title="Seed Diagnostics" subtitle="seed_diagnostics.csv" />
        <TableCard run={selected} file="window_diagnostics" title="Window Diagnostics" subtitle="window_diagnostics.csv" />
      </div>

      <TableCard run={selected} file="hmm_synthetic_sample_stress" title="Synthetic Sample Stress" subtitle="hmm_synthetic_sample_stress.csv" limit={300} maxHeight={420} />
      <TableCard run={selected} file="volatility_forecasts" title="Volatility Forecasts" subtitle="volatility_forecasts.csv" limit={300} maxHeight={420} />
    </div>
  );
}
