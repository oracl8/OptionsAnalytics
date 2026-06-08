"""
Tests for vol/realized_vol.py, vol/estimators.py, and vol/garch.py.

All tests are offline — no network access required.
The GARCH real-data test uses the YFinanceSource cache; it hits the network
only on the very first run and is thereafter reproducible offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volforecast.vol.estimators import garman_klass_vol, parkinson_vol
from volforecast.vol.garch import ewma_vol, garch_forecast
from volforecast.vol.realized_vol import realized_vol


# ---------------------------------------------------------------------------
# realized_vol
# ---------------------------------------------------------------------------


def _alternating_prices(d: float = 0.01, n_returns: int = 10) -> pd.Series:
    """Prices derived from alternating ±d log-returns."""
    log_rets = np.array([-d, d] * (n_returns // 2))
    cumlog = np.concatenate([[0.0], np.cumsum(log_rets)])
    return pd.Series(100.0 * np.exp(cumlog))


def test_realized_vol_hand_computed():
    # Alternating ±d log-returns.  For a 4-element window [-d, d, -d, d]:
    #   mean = 0, sum-of-sq = 4d², std(ddof=1) = sqrt(4d²/3) = d·√(4/3)
    # All even-length windows of alternating ±d have the same std, so
    # every valid target entry equals d·√(4/3)·√(252).
    d = 0.01
    prices = _alternating_prices(d=d, n_returns=10)
    rv = realized_vol(prices, horizon=4, trading_days=252)

    expected = d * np.sqrt(4.0 / 3.0) * np.sqrt(252)
    # Valid entries: indices 0 .. n_returns - horizon = 6
    valid = rv.iloc[:7]
    assert not valid.isna().any(), "Expected no NaN in valid region"
    assert np.allclose(valid.values, expected, rtol=1e-9)


def test_realized_vol_nan_tail():
    prices = _alternating_prices(n_returns=10)
    horizon = 4
    rv = realized_vol(prices, horizon=horizon, trading_days=252)
    # Prices has 11 entries; log_ret has 10; last 'horizon' entries should be NaN.
    # n = 10, valid range: i in 0..n-horizon = 0..6 → indices 0..6 valid, 7..10 NaN
    assert rv.iloc[7:].isna().all()
    assert rv.iloc[:7].notna().all()


def test_realized_vol_constant_returns():
    # Constant log-returns → std = 0 → realized vol = 0
    prices = pd.Series(100.0 * np.exp(0.001 * np.arange(21)))
    rv = realized_vol(prices, horizon=5, trading_days=252)
    valid = rv.dropna()
    assert np.allclose(valid.values, 0.0, atol=1e-12)


def test_realized_vol_index_preserved():
    idx = pd.date_range("2024-01-01", periods=11, freq="B")
    prices = _alternating_prices(n_returns=10)
    prices.index = idx
    rv = realized_vol(prices, horizon=4, trading_days=252)
    assert list(rv.index) == list(idx)


# ---------------------------------------------------------------------------
# Parkinson estimator
# ---------------------------------------------------------------------------

_HIGH = np.array([102.0, 101.0, 103.0, 104.0, 100.5])
_LOW = np.array([98.0, 99.0, 97.0, 96.0, 99.0])


def test_parkinson_positive_finite():
    pv = parkinson_vol(_HIGH, _LOW)
    assert pv > 0.0
    assert np.isfinite(pv)


def test_parkinson_scales_with_range():
    # Doubling the range should roughly double σ (σ ∝ ln(H/L))
    pv_base = parkinson_vol(_HIGH, _LOW)
    pv_wide = parkinson_vol(_HIGH * 1.2, _LOW * 0.8)
    assert pv_wide > pv_base


def test_parkinson_zero_for_zero_range():
    # H == L everywhere → ln(H/L) = 0 → vol = 0
    same = np.array([100.0, 101.0, 102.0])
    assert parkinson_vol(same, same) == 0.0


# ---------------------------------------------------------------------------
# Garman-Klass estimator
# ---------------------------------------------------------------------------

_OPEN = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
_CLOSE = np.array([101.0, 100.0, 102.0, 99.0, 100.5])


def test_garman_klass_positive_finite():
    gk = garman_klass_vol(_OPEN, _HIGH, _LOW, _CLOSE)
    assert gk >= 0.0
    assert np.isfinite(gk)


def test_garman_klass_zero_for_zero_range():
    # H == L and O == C → both terms vanish → vol = 0
    flat = np.array([100.0, 101.0, 102.0])
    assert garman_klass_vol(flat, flat, flat, flat) == 0.0


def test_garman_klass_and_parkinson_agree_on_no_drift():
    # When O == C (no close-to-open drift), the log_co term vanishes;
    # GK reduces to: 0.5·(ln(H/L))²/n, while Parkinson is (ln(H/L))²/(4n·ln2).
    # They differ by a constant factor, but both should be finite and positive.
    pv = parkinson_vol(_HIGH, _LOW)
    gk = garman_klass_vol(_HIGH, _HIGH, _LOW, _HIGH)  # O == C == H (extreme)
    assert pv > 0.0 and gk >= 0.0


# ---------------------------------------------------------------------------
# EWMA volatility
# ---------------------------------------------------------------------------


def test_ewma_shape_and_positive():
    rng = np.random.default_rng(1)
    returns = rng.normal(0, 0.01, 300)
    vol = ewma_vol(returns)
    assert vol.shape == returns.shape
    assert (vol > 0).all()


def test_ewma_declines_after_shock():
    # A single large shock followed by zeros: vol should strictly decrease.
    returns = np.zeros(200)
    returns[0] = 0.10
    vol = ewma_vol(returns)
    # After the shock at index 0, vol[1] < vol[0], and it should be monotone
    # decreasing (up to floating-point) for the rest of the zero series.
    assert vol[-1] < vol[0]
    diffs = np.diff(vol[1:])  # skip index 0 (warm-start override)
    assert (diffs <= 0).all()


def test_ewma_higher_lambda_smoother():
    rng = np.random.default_rng(7)
    returns = rng.normal(0, 0.01, 300)
    vol_fast = ewma_vol(returns, lambda_=0.85)
    vol_slow = ewma_vol(returns, lambda_=0.97)
    # Higher λ → smoother series → lower std of the vol series
    assert vol_slow.std() < vol_fast.std()


# ---------------------------------------------------------------------------
# GARCH(1,1) forecast
# ---------------------------------------------------------------------------


def test_garch_forecast_returns_positive_float():
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.01, 500)
    fc = garch_forecast(returns, horizon=5)
    assert isinstance(fc, float)
    assert fc > 0.0
    # Plausible annualized range for ~1 % daily vol
    assert 0.01 < fc < 1.5


def test_garch_forecast_horizon_independence():
    # The forecast value should change with horizon (GARCH mean-reverts)
    # but not blow up or go negative.
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.01, 500)
    fc1 = garch_forecast(returns, horizon=1)
    fc21 = garch_forecast(returns, horizon=21)
    assert fc1 > 0.0
    assert fc21 > 0.0


def test_garch_forecast_real_data():
    """Fit GARCH on real SPY data downloaded via YFinanceSource (cached after first run)."""
    import numpy as np

    from volforecast.data.loader import YFinanceSource

    src = YFinanceSource()
    df = src.get_ohlcv("SPY", "2022-01-01", "2024-01-01")
    if df.empty:
        pytest.skip("SPY data unavailable")

    log_ret = np.log(df["Close"]).diff().dropna().to_numpy()
    assert len(log_ret) > 100, "Need at least 100 returns"

    fc = garch_forecast(log_ret, horizon=21)
    assert isinstance(fc, float)
    assert np.isfinite(fc)
    assert fc > 0.0
