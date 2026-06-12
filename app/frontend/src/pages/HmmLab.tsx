import { useMemo } from "react";
import { useRunContext } from "../lib/runContext";
import { useSeries } from "../lib/hooks";
import { PageHeader, NoRun } from "../components/PageHeader";
import { Panel, StatusBadge } from "../components/ui";
import { DataState } from "../components/DataState";
import { TableCard } from "../components/DataTable";
import { buildRows, ProbabilityChart, RegimeLegend, RegimePriceChart } from "../components/charts";

export function HmmLab() {
  const { selected } = useRunContext();
  const { data: series, isLoading } = useSeries(selected?.run_id, selected?.status);
  const rows = useMemo(() => buildRows(series), [series]);

  if (!selected) {
    return (
      <div>
        <PageHeader title="HMM Regime Lab" desc="Hidden Markov model internals: transitions, emissions, and decoded states." />
        <NoRun />
      </div>
    );
  }

  return (
    <div className="space-y-5">
      <PageHeader
        title="HMM Regime Lab"
        desc="Hidden Markov model internals — transition dynamics, emissions, stationary distribution, and decoded regimes."
        right={<StatusBadge status={selected.status} />}
      />

      <Panel title="Decoded Regimes" subtitle="Viterbi path over price" right={<RegimeLegend />}>
        <DataState loading={isLoading} available={series?.available} reason={series?.reason}>
          <div className="px-2 pb-3 pt-1">
            <RegimePriceChart rows={rows} height={240} />
          </div>
        </DataState>
      </Panel>

      <Panel title="State Probabilities" subtitle="Posterior distribution over time">
        <DataState loading={isLoading} available={series?.available} reason={series?.reason}>
          <div className="px-2 pb-3 pt-1">
            <ProbabilityChart rows={rows} height={200} />
          </div>
        </DataState>
      </Panel>

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <TableCard run={selected} file="hmm_transition_matrices" title="Transition Matrix" subtitle="hmm_transition_matrices.csv" />
        <TableCard run={selected} file="hmm_stationary_distribution" title="Stationary Distribution" subtitle="hmm_stationary_distribution.csv" />
        <TableCard run={selected} file="state_summary" title="State Summary" subtitle="state_summary.csv" />
        <TableCard run={selected} file="hmm_state_stability" title="State Stability" subtitle="hmm_state_stability.csv" />
        <TableCard run={selected} file="hmm_emissions" title="Emissions" subtitle="hmm_emissions.csv" />
        <TableCard run={selected} file="hmm_decode_comparison" title="Decode Comparison" subtitle="hmm_decode_comparison.csv" />
        <TableCard run={selected} file="hmm_component_diagnostics" title="Component Diagnostics" subtitle="hmm_component_diagnostics.csv" />
        <TableCard run={selected} file="hmm_model_selection" title="Model Selection" subtitle="hmm_model_selection.csv" />
      </div>
    </div>
  );
}
