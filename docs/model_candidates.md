# Model Candidates

| Candidate | Role | Default status |
|---|---|---|
| Observable Markov | Transparent baseline and fallback | Optional |
| GaussianHMM | Main latent-regime engine | Default |
| GMMHMM | Richer mixture-emission regime engine | Optional |
| AutoARIMA | Classical return forecast benchmark | Optional |
| AutoETS | Classical return forecast benchmark | Optional |
| AutoTheta | Classical return forecast benchmark | Optional |
| Realised volatility | Simple volatility baseline | Default diagnostic |
| EWMA volatility | Stable no-lookahead volatility baseline | Default diagnostic |
| GARCH / ARCH | Optional econometric volatility benchmark | Optional |
| MLP | Optional neural confirmer | Optional |
| Conv1D+LSTM | Optional advanced neural confirmer | Optional |
| MarkovRegression | Econometric regime comparator | Optional |

Recommended quick grid:

| Parameter | Values |
|---|---|
| `hmm_model_types` | `gaussian`, `gmm` |
| `hmm_state_candidates` | `3`, `4`, `5` |
| `hmm_covariance_candidates` | `diag`, `full`, `tied` |
| `hmm_n_mix_candidates` | `2`, `3` |
| `hmm_min_covar_candidates` | `0.001`, `0.01` |
| `hmm_transmat_prior_candidates` | `1.0`, `5.0`, `10.0` |
| `vol_model_candidates` | `realised_vol`, `ewma_vol`, `garch`, `arch` |
