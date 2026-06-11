# Neural Confirmation

The neural layer is optional and does not affect the base install. If `torch` is not
available, NN mode writes clear skip metadata and the default HMM path continues.

The NN output contract includes predictive mean probability, predictive standard
deviation, and `adjusted_prob = pred_prob - penalty * pred_prob_std`. The adjusted
probability is used for conservative signal gating.

Backends share one output contract:

- `mlp`
- `conv_lstm`

The Torch path translates the local LSTM stock-predictor repo's practical research
patterns: sequence windows, strict train/validation/OOS metadata, MC-dropout inference,
`adjusted_prob = pred_prob - penalty * pred_prob_std`, and optional exponential
recency weights for training samples.

Outputs include `nn_predictions.csv`, `nn_forecasts.csv`, `nn_oos_split_info.csv`,
`nn_training_metrics.csv`, and ensemble files.

Ensemble modes remain backward compatible:

- `hmm_only`
- `nn_only`
- `hmm_and_nn_agree`
- `weighted_blend`
- `multi_signal`

`multi_signal` blends HMM, NN, observable Markov, forecast, and volatility risk-gate
columns when present. The HMM-only default is unchanged.
