# Observable Markov

The observable Markov baseline labels each row as Bull, Bear, or Sideways from rolling
asset returns. It builds a transparent 3x3 transition matrix, computes a stationary
distribution, and produces n-step state forecasts by matrix powers.

Outputs:

- `observable_markov_baseline.csv`
- `observable_markov_transition_matrix.csv`
- `observable_markov_stationary_distribution.csv`
- `observable_markov_forecasts.csv`
- `observable_markov_equity_curve.csv`

The Stage 2A config aliases `observable_markov_window`,
`observable_markov_threshold`, and `observable_markov_forecast_horizons` are accepted
alongside the original field names. The baseline file stores the per-date observable
state, one-hot semantic labels, signal, and independent position.

This baseline is intended as a sanity check and fallback comparator, not a replacement
for the latent HMM engine.
