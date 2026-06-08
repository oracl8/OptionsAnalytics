from __future__ import annotations

import numpy as np


def parkinson_vol(
    high: np.ndarray,
    low: np.ndarray,
    trading_days: int = 252,
) -> float:
    """
    Parkinson (1980) range-based volatility estimator.

    Uses the daily high-low range, which is more efficient than close-to-close
    for assets with significant intraday price movement.

    σ² = (1 / (4n · ln2)) · Σ [ln(H_i / L_i)]²

    Annualized: σ_annual = sqrt(σ² · trading_days)

    Parameters
    ----------
    high, low : array-like
        Daily high and low prices.  Must satisfy high >= low > 0.
    trading_days : int
        Annualisation factor.

    Returns
    -------
    float
        Annualized Parkinson volatility estimate.
    """
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    log_hl = np.log(h / l)
    daily_var = np.mean(log_hl**2) / (4.0 * np.log(2.0))
    return float(np.sqrt(daily_var * trading_days))


def garman_klass_vol(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    trading_days: int = 252,
) -> float:
    """
    Garman-Klass (1980) range-based volatility estimator.

    Extends Parkinson by incorporating the open-to-close drift, improving
    efficiency when overnight gaps and drift are non-trivial.

    σ² = (1/n) · Σ [ 0.5·(ln(H/L))² − (2·ln2 − 1)·(ln(C/O))² ]

    The (2ln2 − 1) ≈ 0.386 coefficient corrects for the drift component.
    Note: the per-day term can be negative when the close-to-open drift is
    large, so we clamp the average to ≥ 0 before taking the square root.

    Annualized: σ_annual = sqrt(σ² · trading_days)

    Parameters
    ----------
    open_, high, low, close : array-like
        Daily OHLC prices.  All must be > 0.
    trading_days : int
        Annualisation factor.

    Returns
    -------
    float
        Annualized Garman-Klass volatility estimate.
    """
    o = np.asarray(open_, dtype=float)
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)

    log_hl = np.log(h / l)
    log_co = np.log(c / o)

    _k = 2.0 * np.log(2.0) - 1.0  # ≈ 0.3863
    daily_var = np.mean(0.5 * log_hl**2 - _k * log_co**2)

    # Clamp: negative daily_var is possible for large overnight gaps in short samples
    daily_var = max(daily_var, 0.0)
    return float(np.sqrt(daily_var * trading_days))
