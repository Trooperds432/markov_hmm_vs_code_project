# Neural Network Confirmation

Neural-network confirmation is optional and disabled by default. The base install works
without `torch`; when NN mode is enabled without the optional dependency, the framework
prints a clear skip message.

Labels are forward returns shifted from each decision date. Feature windows use only
data available at or before the decision date. The initial implementation writes
`nn_probability_positive`, `nn_expected_return_proxy`, `nn_signal`, and `nn_confidence`
columns, then `ensemble.py` can combine HMM and NN signals through `hmm_only`,
`nn_only`, `hmm_and_nn_agree`, or `weighted_blend`.

Install optional dependencies with:

```bash
pip install -r requirements-ml.txt
```
