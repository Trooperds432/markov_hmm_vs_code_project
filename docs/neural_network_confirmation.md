# Neural Network Confirmation

Neural-network confirmation is optional and disabled by default. The base install works
without `torch`; when NN mode is enabled without the optional dependency, the framework
prints a clear skip message.

Labels are forward returns shifted from each decision date. Feature windows use only
data available at or before the decision date. The Torch implementation follows the
useful patterns from the local LSTM stock-predictor project: sequence windows, strict
OOS split metadata, MC-dropout uncertainty, adjusted probabilities, and optional
exponential recency weighting for training samples.

The backtest writes `nn_probability_positive`, `nn_probability_std`,
`nn_adjusted_probability`, `nn_expected_return_proxy`, `nn_signal`, and `nn_confidence`
columns, then `ensemble.py` can combine HMM and NN signals through `hmm_only`,
`nn_only`, `hmm_and_nn_agree`, or `weighted_blend`.

Diagnostics include:

- `nn_predictions.csv`: canonical per-row probability, adjusted probability, label, OOS, and error fields
- `nn_forecasts.csv`: LSTM-style `Pred_Prob_*`, `Pred_Prob_Std_*`, and `Adj_Prob_*` columns
- `nn_oos_split_info.csv`: train, validation, and OOS date boundaries for every NN fold
- `nn_training_metrics.csv`: epoch losses, best epoch, early stopping, learning rate, and sample-weight metadata

Install optional dependencies with:

```bash
pip install -r requirements-ml.txt
```
