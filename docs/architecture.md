# Architecture

The pipeline is:

`data -> features -> HMM -> optional NN -> ensemble -> position sizing -> backtest -> diagnostics -> stress -> reporting`

`data.py` loads asset OHLCV, benchmark close, and optional factor closes. `features.py`
creates base technical/regime features plus optional cross-asset factors. `hmm_model.py`
and `model_selection.py` fit Gaussian HMM regimes and map raw components into Bull,
Bear, and Sideways semantic probabilities.

`backtest.py` turns regime probabilities into positions with volatility targeting,
drawdown throttles, costs, and optional ensemble confirmation. `runner.py` writes the
standard reports and delegates to nested optimisation, portfolio, model comparison,
experiment tracking, and paper simulation modes.

```mermaid
flowchart LR
    A[Data Loader] --> B[Feature Engineering]
    B --> C1[Observable Markov]
    B --> C2[GaussianHMM / GMMHMM]
    B --> C3[AutoARIMA / ETS / Theta]
    B --> C4[Realised / EWMA / GARCH / ARCH]
    B --> C5[MLP / Conv1D+LSTM]
    C1 --> D[Ensemble + Risk Gates]
    C2 --> D
    C3 --> D
    C4 --> D
    C5 --> D
    D --> E[Walk-Forward Backtest]
    D --> F[Portfolio Layer]
    E --> G[Diagnostics + CSV Outputs]
    F --> G
    G --> H[Experiment Tracking / Registry]
    H --> I[Paper Trading Scaffold]
```

```mermaid
flowchart LR
    R[Research Run] --> S[Model Selection]
    S --> M[Metrics + Overfit Report]
    M --> X[Experiment Manifest]
    X --> G[Model Registry]
    G --> P[Paper Trading Scaffold]
    P --> O[CSV Orders / Fills / Positions]
```
