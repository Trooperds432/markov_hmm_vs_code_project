# Markov / HMM Algorithmic Trading Pipeline

This is a VS Code-ready Python project for researching a Markov Chain + Hidden Markov Model regime trading strategy.

It includes:

- Yahoo Finance OHLCV downloader
- Feature engineering
- Gaussian Hidden Markov Model using `hmmlearn`
- Multi-seed model fitting
- Walk-forward retraining
- Viterbi-style decoded regimes
- Posterior hidden-state probabilities
- Adaptive position sizing
- Volatility targeting
- Drawdown throttle
- Transaction-cost and slippage model
- Buy-and-hold comparison
- Sharpe, drawdown, win rate, profit factor, CAGR
- Equity / regime / position plots
- Optional universe test

## 1. Open in VS Code

Open the folder:

```bash
code markov_hmm_vs_code_project
```

Or open VS Code, then choose **File > Open Folder** and select this folder.

## 2. Create a virtual environment

In the VS Code terminal:

```bash
python -m venv .venv
```

Activate it.

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bash
.venv\Scripts\activate.bat
```

macOS / Linux:

```bash
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

If `hmmlearn` fails to install, update pip first:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## 4. Run the strategy

```bash
python main.py
```

This will run the default config in `configs/default.json`.

Optional run flags:

```bash
python main.py --config configs/default.json --output-dir outputs --no-plots
python main.py --no-diagnostics --no-stress
```

## 5. Change ticker/settings

Edit:

```text
configs/default.json
```

Example:

```json
"ticker": "NVDA",
"benchmark": "QQQ",
"long_only": false
```

## 6. Run a universe test

In `main.py`, set:

```python
RUN_UNIVERSE = True
```

Then run:

```bash
python main.py
```

## 7. Outputs

Results are saved into:

```text
outputs/
```

You should see:

- `equity_curve.csv`
- `stats.csv`
- `equity_plot.png`
- `regime_probabilities.png`
- `position_exposure.png`

Robust baseline diagnostics also save:

- `window_diagnostics.csv`
- `seed_diagnostics.csv`
- `state_summary.csv`
- `stress_tests.csv`

Set `"enable_benchmark": true` in the config to also write `markov_regression_benchmark.csv`.

## Important warning

This is a research pipeline, not a live execution bot. Do not use real money until you have done paper trading, out-of-sample validation, slippage analysis, and risk testing.
