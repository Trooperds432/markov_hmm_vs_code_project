# Experiment Tracking

Experiment tracking is local-first. Runs save config snapshots, environment metadata,
selected model metadata, file inventory, and summary stats.

Outputs include:

- `experiment_manifest.json`
- `metrics.json`
- `artifacts_index.json`
- `dependency_report.json`
- `warnings.json`
- `feature_manifest.csv`
- `model_selection_candidates.csv`
- `environment.json`
- copied run CSVs under the configured experiment root
- model registry rows at `model_registry_path`

Main output directories also receive `experiment_metadata.json`, `artifacts_index.json`,
`warnings.csv`, and `feature_manifest.csv` when the relevant pipeline stage writes them.

Models are not approved for paper trading automatically. Approval must be an explicit
manual registry decision.
