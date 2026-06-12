# Statistical Confidence

`use_statistical_confidence_scaling=true` applies an additional exposure multiplier after the
existing HMM, volatility, and drawdown controls. It is disabled by default.

The multiplier starts at `1.0` and is reduced when stationarity risk, parameter instability,
market-model R², PCA market-mode dominance, systemic correlation, or residual tail risk is high.
It is clipped between `min_statistical_confidence_mult` and `max_statistical_confidence_mult`.

This does not claim a signal is profitable. It only reduces exposure when diagnostics suggest the
regime estimate may be less reliable or less stock-specific.
