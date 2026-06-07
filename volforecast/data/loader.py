"""
Market data interface and yfinance implementation with disk caching.

Swap in a different source by subclassing MarketDataSource and implementing the
three abstract methods; no other code needs to change.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

# Columns we expose from an option chain — superset of what downstream code uses.
_CHAIN_COLS = [
    "strike",
    "bid",
    "ask",
    "lastPrice",
    "impliedVolatility",
    "volume",
    "openInterest",
]

_OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class MarketDataSource(ABC):
    @abstractmethod
    def get_ohlcv(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Return OHLCV with tz-naive DatetimeIndex and columns
        [Open, High, Low, Close, Volume].  Empty DataFrame (correct columns)
        if data is unavailable.
        """

    @abstractmethod
    def get_option_chain(
        self,
        ticker: str,
        expiry: str | None = None,
    ) -> dict:
        """
        Return {"calls": DataFrame, "puts": DataFrame, "expiry": str | None}.
        Both DataFrames share the standardised columns in _CHAIN_COLS.
        If expiry is None, use the nearest available expiry.
        Returns empty DataFrames (with correct columns) on failure, never raises.
        """

    @abstractmethod
    def get_available_expiries(self, ticker: str) -> list[str]:
        """Return sorted list of available option expiry strings ('YYYY-MM-DD')."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_chain(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise an option chain DataFrame to _CHAIN_COLS, NaN-filling gaps."""
    out = pd.DataFrame()
    for col in _CHAIN_COLS:
        out[col] = df[col] if col in df.columns else float("nan")
    # Strike must be present to be useful
    out = out.dropna(subset=["strike"]).reset_index(drop=True)
    return out


def _empty_chain() -> dict:
    calls = pd.DataFrame(columns=_CHAIN_COLS)
    puts = pd.DataFrame(columns=_CHAIN_COLS)
    return {"calls": calls, "puts": puts, "expiry": None}


# ---------------------------------------------------------------------------
# yfinance implementation
# ---------------------------------------------------------------------------


class YFinanceSource(MarketDataSource):
    """
    Fetches OHLCV and option chains from Yahoo Finance via yfinance.
    Responses are cached as parquet files to avoid redundant network calls.
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        if cache_dir is None:
            cache_dir = Path(__file__).parent / "cache"
        self._cache = Path(cache_dir)
        self._cache.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # OHLCV
    # ------------------------------------------------------------------

    def get_ohlcv(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        ticker = ticker.upper()
        safe_start = start.replace("/", "-")
        safe_end = end.replace("/", "-")
        cache_path = self._cache / f"{ticker}_ohlcv_{safe_start}_{safe_end}_{interval}.parquet"

        if cache_path.exists():
            return pd.read_parquet(cache_path)

        import yfinance as yf  # lazy import — keeps module importable without network

        df = yf.Ticker(ticker).history(start=start, end=end, interval=interval)

        if df.empty:
            warnings.warn(
                f"yfinance returned no OHLCV data for {ticker} [{start} → {end}]",
                stacklevel=2,
            )
            return pd.DataFrame(columns=_OHLCV_COLS)

        df = df[_OHLCV_COLS].copy()
        # Normalise index: drop timezone so downstream code doesn't need to care
        df.index = pd.to_datetime(df.index).tz_localize(None)

        df.to_parquet(cache_path)
        return df

    # ------------------------------------------------------------------
    # Option chains
    # ------------------------------------------------------------------

    def get_available_expiries(self, ticker: str) -> list[str]:
        import yfinance as yf

        ticker = ticker.upper()
        available = list(yf.Ticker(ticker).options)
        if not available:
            warnings.warn(f"No option expiries available for {ticker}", stacklevel=2)
        return available

    def get_option_chain(
        self,
        ticker: str,
        expiry: str | None = None,
    ) -> dict:
        ticker = ticker.upper()

        # Fast path: if caller specified an expiry and both cache files exist,
        # skip the network call entirely.
        if expiry is not None:
            calls_path = self._cache / f"{ticker}_{expiry}_calls.parquet"
            puts_path = self._cache / f"{ticker}_{expiry}_puts.parquet"
            if calls_path.exists() and puts_path.exists():
                return {
                    "calls": pd.read_parquet(calls_path),
                    "puts": pd.read_parquet(puts_path),
                    "expiry": expiry,
                }

        import yfinance as yf

        tkr = yf.Ticker(ticker)
        available = list(tkr.options)
        if not available:
            warnings.warn(f"No options available for {ticker}", stacklevel=2)
            return _empty_chain()

        if expiry is None:
            expiry = available[0]
        elif expiry not in available:
            warnings.warn(
                f"Expiry {expiry!r} not available for {ticker}. "
                f"Nearest available: {available[:3]}",
                stacklevel=2,
            )
            return _empty_chain()

        calls_path = self._cache / f"{ticker}_{expiry}_calls.parquet"
        puts_path = self._cache / f"{ticker}_{expiry}_puts.parquet"

        if calls_path.exists() and puts_path.exists():
            return {
                "calls": pd.read_parquet(calls_path),
                "puts": pd.read_parquet(puts_path),
                "expiry": expiry,
            }

        chain = tkr.option_chain(expiry)
        calls = _normalize_chain(chain.calls)
        puts = _normalize_chain(chain.puts)

        calls.to_parquet(calls_path)
        puts.to_parquet(puts_path)

        return {"calls": calls, "puts": puts, "expiry": expiry}
