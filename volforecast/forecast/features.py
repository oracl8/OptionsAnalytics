from __future__ import annotations

import numpy as np
import pandas as pd

from volforecast.vol.realized_vol import realized_vol


def _rolling_parkinson(
    high: pd.Series,
    low: pd.Series,
    window: int,
    trading_days: int = 252,
) -> pd.Series:
    """
    Annualized rolling Parkinson vol.  σ² = mean[(log H/L)²] / (4·ln 2)
    """
    log_hl_sq = np.log(high / low) ** 2
    daily_var = log_hl_sq.rolling(window).mean() / (4.0 * np.log(2.0))
    return np.sqrt(daily_var * trading_days)


def _rolling_garman_klass(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
    trading_days: int = 252,
) -> pd.Series:
    """
    Annualized rolling Garman-Klass vol.
    σ² = mean[0.5·(log H/L)² − k·(log C/O)²],  k = 2·ln2 − 1 ≈ 0.386
    Clamped to ≥ 0 before sqrt to handle large overnight-gap samples.
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    _k = 2.0 * np.log(2.0) - 1.0
    daily_term = 0.5 * log_hl ** 2 - _k * log_co ** 2
    daily_var = daily_term.rolling(window).mean().clip(lower=0.0)
    return np.sqrt(daily_var * trading_days)


def build_features(
    ohlcv: pd.DataFrame,
    horizon: int,
    iv_series: pd.Series | None = None,
    trading_days: int = 252,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build a lookahead-free feature matrix and the realized-vol target.

    Invariant
    ---------
    feature_matrix.iloc[i]  depends only on  ohlcv.iloc[:i+1]  (data ≤ day i)
    target.iloc[i]          depends only on  ohlcv.iloc[i+1:i+horizon+1]  (data > day i)

    This means (features, target) pairs can be fed into any walk-forward or
    purged CV without additional lookahead guards on the feature side.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Daily OHLCV with columns [Open, High, Low, Close, Volume].
    horizon : int
        Forecast horizon in trading days (e.g. 5 for one week, 21 for one month).
    iv_series : pd.Series, optional
        ATM implied-vol series aligned to ohlcv.index.  Included as a feature
        lagged by 1 day so that IV at day i uses only data available at i-1.
    trading_days : int
        Annualisation factor (252 for equities).

    Returns
    -------
    features : pd.DataFrame
        Feature matrix indexed by ohlcv.index.  NaN in the first ~63 rows
        (rolling windows not yet filled) and optionally in later rows.
    target : pd.Series
        Annualized forward realized vol; NaN in the last `horizon` rows.
    """
    close = ohlcv["Close"]
    high = ohlcv["High"]
    low = ohlcv["Low"]
    open_ = ohlcv["Open"]
    volume = ohlcv["Volume"]

    # log_ret[i] = log(close[i] / close[i-1]) — available at end of day i
    log_ret = np.log(close).diff()

    ann = np.sqrt(trading_days)
    feats: dict[str, pd.Series] = {}

    for w in [5, 10, 21, 63]:
        feats[f"rv_{w}"] = log_ret.rolling(w).std() * ann

    feats["ret_1"] = log_ret
    for w in [5, 21]:
        feats[f"ret_{w}_cum"] = log_ret.rolling(w).sum()

    for w in [5, 21]:
        feats[f"park_{w}"] = _rolling_parkinson(high, low, w, trading_days)
        feats[f"gk_{w}"] = _rolling_garman_klass(open_, high, low, close, w, trading_days)

    for w in [5, 21]:
        feats[f"vol_ratio_{w}"] = volume / volume.rolling(w).mean()

    if iv_series is not None:
        # shift(1): IV at day i uses IV from day i-1 (available at open on day i)
        feats["iv_lag1"] = iv_series.shift(1)

    feature_df = pd.DataFrame(feats, index=ohlcv.index)
    target = realized_vol(close, horizon, trading_days)

    return feature_df, target
