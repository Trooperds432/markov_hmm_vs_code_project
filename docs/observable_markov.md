# Observable Markov

The observable Markov baseline labels each row as Bull, Bear, or Sideways from rolling
asset returns. It builds a transparent 3x3 transition matrix, computes a stationary
distribution, and produces n-step state forecasts by matrix powers.

Outputs:

- `observable_markov_transition_matrix.csv`
- `observable_markov_forecasts.csv`
- `observable_markov_equity_curve.csv`

This baseline is intended as a sanity check and fallback comparator, not a replacement
for the latent HMM engine.
