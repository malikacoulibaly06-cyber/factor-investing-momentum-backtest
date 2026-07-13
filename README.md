# Factor Investing & Momentum Backtest

Research/experimentation project testing the CAPM on beta-sorted
portfolios, estimating Fama-French 3-factor exposures for individual stocks
(including a 60-year rolling regression on Berkshire Hathaway), and
backtesting a simple moving-average crossover strategy against Fama-French
benchmark portfolios. Originated from a group project in FINA 4335 (Asset
Pricing, Prof. Marius Popescu).

## What this demonstrates

- Pulling Ken French data library portfolios via `pandas_datareader`
- Testing whether **CAPM** holds across beta-sorted quintile/decile
  portfolios
- Estimating the **Fama-French 3-factor model** (`Mkt-RF`, `SMB`, `HML`)
  with `statsmodels`, including annualizing daily alpha
- **Rolling-window regression** (`RollingOLS`) to see how factor loadings
  drift over multi-decade horizons
- Building and backtesting a **moving-average crossover** trading signal,
  correctly lagged to avoid look-ahead bias
- Refactoring the notebook's regression and signal logic into a tested
  module (`src/factor_models/core.py`), with unit tests that recover known
  simulated betas and verify the backtest signal is properly lagged

## Layout

```
src/factor_models/core.py   CAPM, FF3, rolling regression, SMA signal & backtest
tests/test_core.py          pytest suite (simulated factor data with known coefficients)
project2_analysis.ipynb     Final submitted notebook
```

The git history of `project2_analysis.ipynb` preserves the iteration from
the working draft through to the final submitted version.

## Running the tests

```bash
pip install -r requirements.txt
pip install -e .
pytest -v
```

CI runs the suite on every push (`.github/workflows/tests.yml`).
