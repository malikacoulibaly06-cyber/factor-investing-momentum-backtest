import numpy as np
import pandas as pd
import pytest

from factor_models.core import (
    capm_alpha_beta,
    fama_french_regression,
    rolling_regression,
    beta_quintile_average_returns,
    sma_crossover_signal,
    backtest_signal_returns,
)


def test_capm_alpha_beta_recovers_known_beta():
    rng = np.random.default_rng(7)
    n = 400
    idx = pd.date_range("2000-01-01", periods=n, freq="D")
    mkt = pd.Series(rng.normal(0, 0.01, n), index=idx)
    asset = 0.9 * mkt + rng.normal(0, 0.001, n)
    result = capm_alpha_beta(asset, mkt)
    assert result["beta"] == pytest.approx(0.9, abs=0.05)


def test_fama_french_regression_annualized_alpha():
    rng = np.random.default_rng(3)
    n = 600
    idx = pd.date_range("2000-01-01", periods=n, freq="D")
    factors = pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0, 0.01, n),
            "SMB": rng.normal(0, 0.005, n),
            "HML": rng.normal(0, 0.005, n),
        },
        index=idx,
    )
    daily_alpha = 0.0002
    asset = daily_alpha + 1.0 * factors["Mkt-RF"] + rng.normal(0, 0.0005, n)
    result = fama_french_regression(asset, factors)
    assert result["annualized_alpha"] == pytest.approx(result["alpha"] * 252)
    assert result["mktrf_beta"] == pytest.approx(1.0, abs=0.1)


def test_rolling_regression_no_lookahead():
    rng = np.random.default_rng(5)
    n = 200
    idx = pd.date_range("2000-01-01", periods=n, freq="D")
    factors = pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0, 0.01, n),
            "SMB": rng.normal(0, 0.005, n),
            "HML": rng.normal(0, 0.005, n),
        },
        index=idx,
    )
    asset = 1.1 * factors["Mkt-RF"] + rng.normal(0, 0.001, n)
    window = 60
    rolling = rolling_regression(asset, factors, window)
    assert rolling["Mkt-RF"].iloc[: window - 1].isna().all()


def test_beta_quintile_average_returns():
    df = pd.DataFrame({"Lo20": [0.01, 0.02], "Hi20": [0.03, 0.05]})
    avg = beta_quintile_average_returns(df)
    assert avg["Hi20"] > avg["Lo20"]


def test_sma_crossover_signal_is_binary():
    prices = pd.Series([10, 11, 12, 11, 9, 8, 10, 13, 14, 15])
    signal = sma_crossover_signal(prices, window=3)
    assert set(signal.dropna().unique()).issubset({0, 1})


def test_backtest_signal_returns_shifts_signal():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    returns = pd.Series([0.1, 0.2, -0.1, 0.05], index=idx)
    signal = pd.Series([1, 1, 0, 1], index=idx)
    result = backtest_signal_returns(returns, signal)
    assert np.isnan(result.iloc[0])  # no prior signal to apply
    assert result.iloc[1] == pytest.approx(0.2 * 1)
    assert result.iloc[2] == pytest.approx(-0.1 * 1)
    assert result.iloc[3] == pytest.approx(0.05 * 0)
