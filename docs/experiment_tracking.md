# Experiment Tracking

Experiment tracking is local-first. Runs save config snapshots, environment metadata,
selected model metadata, file inventory, and summary stats.

Outputs include:

- `experiment_manifest.json`
- copied run CSVs under the configured experiment root
- model registry rows at `model_registry_path`

Models are not approved for paper trading automatically. Approval must be an explicit
manual registry decision.
