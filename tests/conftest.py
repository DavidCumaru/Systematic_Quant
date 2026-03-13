"""
tests/conftest.py
=================
Shared pytest fixtures for the systematic_alpha test suite.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_engineering import build_features
from labeling import apply_triple_barrier


# ---------------------------------------------------------------------------
# Synthetic OHLCV generators
# ---------------------------------------------------------------------------

def make_ohlcv(n: int = 600, seed: int = 42, freq: str = "1D") -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with tz-aware DatetimeIndex."""
    rng = np.random.default_rng(seed)

    if freq == "1D":
        idx = pd.bdate_range(start="2020-01-02", periods=n, tz="America/New_York")
    else:
        idx = pd.date_range(
            start="2020-01-02 09:30",
            periods=n,
            freq=freq,
            tz="America/New_York",
        )

    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    close = np.maximum(close, 1.0)
    high   = close + rng.uniform(0.1, 1.5, n)
    low    = close - rng.uniform(0.1, 1.5, n)
    low    = np.maximum(low, 0.5)
    open_  = close + rng.normal(0, 0.3, n)
    volume = rng.integers(500_000, 5_000_000, n).astype(float)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Session-scoped fixtures (expensive to build, shared across the module)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def raw_df():
    return make_ohlcv(600)


@pytest.fixture(scope="session")
def raw_df_intraday():
    return make_ohlcv(600, freq="1h")


@pytest.fixture(scope="session")
def spy_df():
    return make_ohlcv(600, seed=99)


@pytest.fixture(scope="session")
def featured_df(raw_df):
    return build_features(raw_df)


@pytest.fixture(scope="session")
def labeled_df(featured_df):
    return apply_triple_barrier(featured_df)


@pytest.fixture(scope="session")
def returns_df(raw_df, spy_df):
    """Multi-ticker returns DataFrame for portfolio tests."""
    tickers = ["SPY", "QQQ", "TLT", "GLD"]
    rng = np.random.default_rng(7)
    n = 300
    idx = pd.bdate_range(start="2022-01-03", periods=n, tz="America/New_York")
    data = {t: rng.normal(0.0005, 0.01, n) for t in tickers}
    return pd.DataFrame(data, index=idx)
