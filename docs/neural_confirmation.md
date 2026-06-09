# Neural Confirmation

The neural layer is optional and does not affect the base install. If `torch` is not
available, NN mode writes clear skip metadata and the default HMM path continues.

The NN output contract includes predictive mean probability, predictive standard
deviation, and `adjusted_prob = pred_prob - penalty * pred_prob_std`. The adjusted
probability is used for conservative signal gating.

Planned backends share one output contract:

- `mlp`
- `conv1d_lstm`

Outputs include `nn_predictions.csv`, `nn_training_metrics.csv`, and ensemble files.
