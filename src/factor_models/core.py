"""Factor-model estimation and simple technical-signal backtesting.

Generalizes the FINA 4335 Project 2 analysis: testing CAPM on
beta-sorted portfolios, estimating Fama-French factor exposures (including
rolling regressions), and evaluating a moving-average crossover trading
signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS


def capm_alpha_beta(excess_asset_ret: pd.Series, excess_mkt_ret: pd.Series) -> dict:
    """Single-factor CAPM regression: returns alpha, beta and their p-values."""
    df = pd.concat([excess_asset_ret.rename("y"), excess_mkt_ret.rename("x")], axis=1).dropna()
    X = sm.add_constant(df["x"])
    model = sm.OLS(df["y"], X).fit()
    return {
        "alpha": model.params["const"],
        "beta": model.params["x"],
        "alpha_pvalue": model.pvalues["const"],
        "beta_pvalue": model.pvalues["x"],
    }


def fama_french_regression(excess_asset_ret: pd.Series, factors: pd.DataFrame) -> dict:
    """Fama-French 3-factor regression; ``factors`` must have Mkt-RF, SMB, HML."""
    df = pd.concat([excess_asset_ret.rename("y"), factors[["Mkt-RF", "SMB", "HML"]]], axis=1).dropna()
    X = sm.add_constant(df[["Mkt-RF", "SMB", "HML"]])
    model = sm.OLS(df["y"], X).fit()
    return {
        "alpha": model.params["const"],
        "annualized_alpha": model.params["const"] * 252,
        "mktrf_beta": model.params["Mkt-RF"],
        "smb_beta": model.params["SMB"],
        "hml_beta": model.params["HML"],
        "rsquared_adj": model.rsquared_adj,
    }


def rolling_regression(excess_asset_ret: pd.Series, factors: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling Fama-French 3-factor regression, returning a DataFrame of rolling coefficients."""
    df = pd.concat([excess_asset_ret.rename("y"), factors[["Mkt-RF", "SMB", "HML"]]], axis=1).dropna()
    X = sm.add_constant(df[["Mkt-RF", "SMB", "HML"]])
    rolling = RollingOLS(df["y"], X, window=window).fit()
    return rolling.params.rename(columns={"const": "alpha"})


def beta_quintile_average_returns(quintile_portfolios: pd.DataFrame) -> pd.Series:
    """Average return of each beta-sorted quintile portfolio (columns = quintiles)."""
    return quintile_portfolios.mean()


def sma_crossover_signal(prices: pd.Series, window: int = 12) -> pd.Series:
    """Long/flat position signal: long (1) when price is above its trailing SMA, else flat (0).

    The signal at time t only uses information available up to t, and should
    be lagged by the caller before applying it to next-period returns to
    avoid look-ahead bias.
    """
    sma = prices.rolling(window=window).mean()
    return (prices > sma).astype(int)


def backtest_signal_returns(returns: pd.Series, signal: pd.Series) -> pd.Series:
    """Apply a (already-lagged) position signal to a return series."""
    return returns.mul(signal.shift(1))
