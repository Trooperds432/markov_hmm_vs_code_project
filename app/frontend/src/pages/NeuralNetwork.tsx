import { useRunContext } from "../lib/runContext";
import { PageHeader, NoRun } from "../components/PageHeader";
import { StatusBadge, Tag } from "../components/ui";
import { TableCard } from "../components/DataTable";

export function NeuralNetwork() {
  const { selected } = useRunContext();

  if (!selected) {
    return (
      <div>
        <PageHeader title="Neural Network" desc="Neural confirmation model forecasts, training metrics, and ensemble diagnostics." />
        <NoRun />
      </div>
    );
  }

  return (
    <div className="space-y-5">
      <PageHeader
        title="Neural Network"
        desc="Neural confirmation forecasts and ensemble diagnostics. Requires the optional torch backend; tables degrade gracefully when not produced."
        right={
          <div className="flex items-center gap-2">
            <Tag tone={selected.request.enable_nn_confirmation ? "accent" : "neutral"}>
              NN {selected.request.enable_nn_confirmation ? "ON" : "default"}
            </Tag>
            <StatusBadge status={selected.status} />
          </div>
        }
      />

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <TableCard run={selected} file="nn_training_metrics" title="Training Metrics" subtitle="nn_training_metrics.csv" />
        <TableCard run={selected} file="nn_oos_split_info" title="OOS Split Info" subtitle="nn_oos_split_info.csv" />
      </div>

      <TableCard run={selected} file="nn_predictions" title="NN Predictions" subtitle="nn_predictions.csv" limit={300} maxHeight={420} />
      <TableCard run={selected} file="nn_forecasts" title="NN Forecasts" subtitle="nn_forecasts.csv" limit={300} maxHeight={420} />

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <TableCard run={selected} file="ensemble_predictions" title="Ensemble Predictions" subtitle="ensemble_predictions.csv" limit={300} />
        <TableCard run={selected} file="ensemble_diagnostics" title="Ensemble Diagnostics" subtitle="ensemble_diagnostics.csv" limit={300} />
      </div>

      <TableCard run={selected} file="model_overfit_report" title="Overfit Report" subtitle="model_overfit_report.csv" />
    </div>
  );
}
