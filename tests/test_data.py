"""
Offline tests for the data layer — yfinance is fully mocked.
No network access required.
"""

from __future__ import annotations

import warnings
from collections import namedtuple
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from volforecast.data.loader import (
    MarketDataSource,
    YFinanceSource,
    _normalize_chain,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]
_CHAIN_COLS = ["strike", "bid", "ask", "lastPrice", "impliedVolatility", "volume", "openInterest"]


def _make_ohlcv(n: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": [400.0 + i for i in range(n)],
            "High": [405.0 + i for i in range(n)],
            "Low": [395.0 + i for i in range(n)],
            "Close": [402.0 + i for i in range(n)],
            "Volume": [1_000_000 + i * 1000 for i in range(n)],
        },
        index=idx,
    )


def _make_chain_df(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strike": [400.0 + i * 5 for i in range(n)],
            "bid": [2.0 + i for i in range(n)],
            "ask": [2.1 + i for i in range(n)],
            "lastPrice": [2.05 + i for i in range(n)],
            "impliedVolatility": [0.20 + i * 0.01 for i in range(n)],
            "volume": [100 + i * 10 for i in range(n)],
            "openInterest": [500 + i * 50 for i in range(n)],
        }
    )


_ChainTuple = namedtuple("OptionChain", ["calls", "puts"])


# ---------------------------------------------------------------------------
# ABC / interface
# ---------------------------------------------------------------------------


def test_is_abstract_base():
    assert issubclass(YFinanceSource, MarketDataSource)


# ---------------------------------------------------------------------------
# _normalize_chain
# ---------------------------------------------------------------------------


def test_normalize_chain_full_columns():
    df = _make_chain_df(3)
    out = _normalize_chain(df)
    assert list(out.columns) == _CHAIN_COLS
    assert len(out) == 3


def test_normalize_chain_missing_columns():
    """Columns absent in source must be NaN-filled, not raise."""
    df = _make_chain_df(3)[["strike", "bid"]]  # only 2 of 7 cols
    out = _normalize_chain(df)
    assert list(out.columns) == _CHAIN_COLS
    assert out["ask"].isna().all()
    assert out["impliedVolatility"].isna().all()


def test_normalize_chain_drops_null_strike():
    df = _make_chain_df(3)
    df.loc[1, "strike"] = float("nan")
    out = _normalize_chain(df)
    assert len(out) == 2
    assert out["strike"].notna().all()


def test_normalize_chain_resets_index():
    df = _make_chain_df(3)
    df.loc[1, "strike"] = float("nan")
    out = _normalize_chain(df)
    assert list(out.index) == list(range(len(out)))


# ---------------------------------------------------------------------------
# OHLCV — cache hit
# ---------------------------------------------------------------------------


def test_ohlcv_cache_hit(tmp_path):
    """Pre-write a parquet; verify loader returns it without touching yfinance."""
    src = YFinanceSource(cache_dir=tmp_path)
    cache_path = tmp_path / "SPY_ohlcv_2024-01-01_2024-02-01_1d.parquet"
    expected = _make_ohlcv(5)
    expected.to_parquet(cache_path)

    with patch("yfinance.Ticker") as mock_ticker:
        result = src.get_ohlcv("SPY", "2024-01-01", "2024-02-01")

    mock_ticker.assert_not_called()
    # Parquet doesn't preserve index freq; compare values and column names only.
    pd.testing.assert_frame_equal(result, expected, check_freq=False)


# ---------------------------------------------------------------------------
# OHLCV — fetch and cache
# ---------------------------------------------------------------------------


def test_ohlcv_fetches_and_caches(tmp_path):
    src = YFinanceSource(cache_dir=tmp_path)
    fake_df = _make_ohlcv(10)

    mock_tkr = MagicMock()
    mock_tkr.history.return_value = fake_df

    with patch("yfinance.Ticker", return_value=mock_tkr):
        result = src.get_ohlcv("SPY", "2024-01-01", "2024-04-01")

    assert list(result.columns) == _OHLCV_COLS
    assert len(result) == len(fake_df)
    # Cache file must exist now
    cache_path = tmp_path / "SPY_ohlcv_2024-01-01_2024-04-01_1d.parquet"
    assert cache_path.exists()


def test_ohlcv_index_is_tz_naive(tmp_path):
    src = YFinanceSource(cache_dir=tmp_path)
    fake_df = _make_ohlcv(5)
    # Simulate yfinance returning a tz-aware index
    fake_df.index = pd.to_datetime(fake_df.index).tz_localize("America/New_York")

    mock_tkr = MagicMock()
    mock_tkr.history.return_value = fake_df

    with patch("yfinance.Ticker", return_value=mock_tkr):
        result = src.get_ohlcv("SPY", "2024-01-01", "2024-02-01")

    assert result.index.tz is None


# ---------------------------------------------------------------------------
# OHLCV — empty response
# ---------------------------------------------------------------------------


def test_ohlcv_empty_response_warns(tmp_path):
    src = YFinanceSource(cache_dir=tmp_path)
    mock_tkr = MagicMock()
    mock_tkr.history.return_value = pd.DataFrame()

    with patch("yfinance.Ticker", return_value=mock_tkr):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = src.get_ohlcv("FAKE", "2024-01-01", "2024-02-01")

    assert len(w) == 1
    assert "no ohlcv data" in str(w[0].message).lower()
    assert result.empty
    assert list(result.columns) == _OHLCV_COLS


# ---------------------------------------------------------------------------
# Option chain — cache hit
# ---------------------------------------------------------------------------


def test_option_chain_cache_hit(tmp_path):
    src = YFinanceSource(cache_dir=tmp_path)
    expiry = "2024-03-15"
    calls = _make_chain_df(3)
    puts = _make_chain_df(3)
    calls.to_parquet(tmp_path / f"SPY_{expiry}_calls.parquet")
    puts.to_parquet(tmp_path / f"SPY_{expiry}_puts.parquet")

    with patch("yfinance.Ticker") as mock_ticker:
        result = src.get_option_chain("SPY", expiry=expiry)

    mock_ticker.assert_not_called()
    assert result["expiry"] == expiry
    assert list(result["calls"].columns) == _CHAIN_COLS


# ---------------------------------------------------------------------------
# Option chain — fetch and cache
# ---------------------------------------------------------------------------


def test_option_chain_fetches_and_caches(tmp_path):
    src = YFinanceSource(cache_dir=tmp_path)
    expiry = "2024-03-15"
    fake_calls = _make_chain_df(4)
    fake_puts = _make_chain_df(4)

    mock_tkr = MagicMock()
    mock_tkr.options = [expiry, "2024-04-19"]
    mock_tkr.option_chain.return_value = _ChainTuple(calls=fake_calls, puts=fake_puts)

    with patch("yfinance.Ticker", return_value=mock_tkr):
        result = src.get_option_chain("SPY", expiry=expiry)

    assert result["expiry"] == expiry
    assert list(result["calls"].columns) == _CHAIN_COLS
    assert len(result["calls"]) == 4
    assert (tmp_path / f"SPY_{expiry}_calls.parquet").exists()
    assert (tmp_path / f"SPY_{expiry}_puts.parquet").exists()


def test_option_chain_default_uses_nearest(tmp_path):
    """When expiry=None, the first (nearest) available expiry is chosen."""
    src = YFinanceSource(cache_dir=tmp_path)
    nearest = "2024-03-15"
    fake_calls = _make_chain_df(2)
    fake_puts = _make_chain_df(2)

    mock_tkr = MagicMock()
    mock_tkr.options = [nearest, "2024-04-19"]
    mock_tkr.option_chain.return_value = _ChainTuple(calls=fake_calls, puts=fake_puts)

    with patch("yfinance.Ticker", return_value=mock_tkr):
        result = src.get_option_chain("SPY")

    assert result["expiry"] == nearest


# ---------------------------------------------------------------------------
# Option chain — missing / bad expiry
# ---------------------------------------------------------------------------


def test_option_chain_empty_options_warns(tmp_path):
    src = YFinanceSource(cache_dir=tmp_path)
    mock_tkr = MagicMock()
    mock_tkr.options = []

    with patch("yfinance.Ticker", return_value=mock_tkr):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = src.get_option_chain("FAKE")

    assert any("no options" in str(x.message).lower() for x in w)
    assert result["calls"].empty
    assert result["puts"].empty
    assert result["expiry"] is None


def test_option_chain_bad_expiry_warns(tmp_path):
    src = YFinanceSource(cache_dir=tmp_path)
    mock_tkr = MagicMock()
    mock_tkr.options = ["2024-03-15", "2024-04-19"]

    with patch("yfinance.Ticker", return_value=mock_tkr):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = src.get_option_chain("SPY", expiry="1999-01-01")

    assert len(w) == 1
    assert result["calls"].empty
    assert result["expiry"] is None


# ---------------------------------------------------------------------------
# get_available_expiries
# ---------------------------------------------------------------------------


def test_get_available_expiries(tmp_path):
    src = YFinanceSource(cache_dir=tmp_path)
    expected = ["2024-03-15", "2024-04-19", "2024-06-21"]
    mock_tkr = MagicMock()
    mock_tkr.options = expected

    with patch("yfinance.Ticker", return_value=mock_tkr):
        result = src.get_available_expiries("SPY")

    assert result == expected


def test_get_available_expiries_empty_warns(tmp_path):
    src = YFinanceSource(cache_dir=tmp_path)
    mock_tkr = MagicMock()
    mock_tkr.options = []

    with patch("yfinance.Ticker", return_value=mock_tkr):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = src.get_available_expiries("FAKE")

    assert result == []
    assert len(w) == 1
