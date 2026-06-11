# HMM Engine

The HMM engine supports `GaussianHMM` and optional `GMMHMM` through a local factory
wrapper. This keeps direct `hmmlearn` API calls in one place.

Supported covariance types are `spherical`, `diag`, `full`, and `tied`. `min_covar`
regularizes covariance estimates. Start and transition priors help discourage unstable
transition matrices. `params` and `init_params` control which parameters are trained
and initialized by `hmmlearn`.

The diagnostics path uses `score_samples()` to capture log likelihood and posterior
state probabilities. Viterbi and MAP decoded paths are compared in
`hmm_decode_comparison.csv`; the main `equity_curve.csv` also carries
`decoded_state_viterbi`, `decoded_state_map`, and `state_decode_disagreement`.
The legacy-compatible `hidden_state` column remains the Viterbi semantic state.

Model selection writes candidate likelihoods, AIC/BIC, selected architecture, and
errors to `hmm_model_selection.csv`. Stationary probabilities, dwell times, semantic
state maps, component diagnostics, and sample-path stress outputs are written as
separate CSV files. `hmm_stationary_distribution.csv` includes both component-style
columns and compatibility aliases: `state`, `probability`, `model_type`, `n_states`,
`covariance_type`, `seed`, and `n_mix`.

Configs may use either `hmm_model_types` or the alias
`hmm_model_type_candidates`; both feed the same model-selection grid.
