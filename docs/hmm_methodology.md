# HMM Methodology

A Markov chain models transitions between states where the next state depends on the
current state. A Hidden Markov Model assumes those states are not directly observed;
instead, observed feature vectors are emitted from state-specific probability
distributions.

This project uses `hmmlearn` GaussianHMM. The Forward algorithm scores state
probabilities, Baum-Welch fits parameters by expectation maximisation, and Viterbi
decodes the most likely state path. Transition matrices describe persistence, and
expected dwell time is estimated as `1 / (1 - self_transition_probability)`.

Raw HMM components are mapped into semantic Bull, Bear, and Sideways regimes using
forward return, volatility, drawdown behaviour, occupancy, trend score, and skew. When
the selected HMM has more than three components, several raw components may aggregate
into the same semantic state. State stability reports compare semantic maps,
transition matrices, and emission means across walk-forward windows.
