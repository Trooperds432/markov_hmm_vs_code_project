# HMM Engine

The HMM engine supports `GaussianHMM` and optional `GMMHMM` through a local factory
wrapper. This keeps direct `hmmlearn` API calls in one place.

Supported covariance types are `spherical`, `diag`, `full`, and `tied`. `min_covar`
regularizes covariance estimates. Start and transition priors help discourage unstable
transition matrices. `params` and `init_params` control which parameters are trained
and initialized by `hmmlearn`.

The diagnostics path uses `score_samples()` to capture log likelihood and posterior
state probabilities. Viterbi and MAP decoded paths are compared in
`hmm_decode_comparison.csv`.

Model selection writes candidate likelihoods, AIC/BIC, selected architecture, and
errors to `hmm_model_selection.csv`. Stationary probabilities, dwell times, semantic
state maps, component diagnostics, and sample-path stress outputs are written as
separate CSV files.
