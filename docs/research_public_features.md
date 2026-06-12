# Research Public Features

The optional `research_public` feature group adds compact statistical diagnostics inspired by
Quantopian-style lecture material from `Trooperds432/research_public`. The project reimplements
small formulas locally and does not vendor notebook code.

Enable it with `enable_research_public_features=true` or by adding `research_public` to
`feature_groups_enabled`. The default HMM observation set is unchanged.

Feature families include stationarity, autocorrelation, volatility clustering, market residuals,
correlation regimes, PCA market mode, cointegration spread stress, factor ranks, regression
diagnostics, and parameter instability. All observation features use trailing windows only.

Safe HMM inputs are selected with `use_research_public_hmm_features=true` and the explicit
`research_public_hmm_features` allowlist. Validation-only fields such as forward returns,
long-short spread returns, factor hit rates, labels, targets, and hypothesis outputs are blocked.
