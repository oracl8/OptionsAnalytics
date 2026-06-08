"""Tests for volforecast.forecast.implied_vs_forecast (no network required)."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from volforecast.forecast.implied_vs_forecast import compare_chain_to_forecast, _OUTPUT_COLS
from volforecast.pricing.black_scholes import bsm_price


def _make_chain(
    spot: float,
    strikes: list[float],
    sigma: float,
    T: float,
    r: float = 0.045,
    option_type: str = "call",
) -> pd.DataFrame:
    """Synthetic chain: mid-prices from BSM, bid/ask bracketing the mid by 1%."""
    mids = [float(bsm_price(spot, k, T, r, sigma, option_type=option_type)) for k in strikes]
    records = []
    for k, mid in zip(strikes, mids):
        records.append({"strike": k, "bid": mid * 0.99, "ask": mid * 1.01, "lastPrice": mid})
    return pd.DataFrame(records)


def _future_expiry(days: int = 30) -> str:
    return (date.today() + timedelta(days=days)).isoformat()


def _past_expiry() -> str:
    return (date.today() - timedelta(days=1)).isoformat()


# ---------------------------------------------------------------------------


def test_richness_sign_rich():
    """When market IV (sigma=0.25) > forecast (0.15), richness should be positive."""
    spot = 100.0
    sigma = 0.25
    forecast_rv = 0.15
    T_days = 45
    strikes = [95.0, 100.0, 105.0]

    calls = _make_chain(spot, strikes, sigma, T_days / 365)
    puts = pd.DataFrame(columns=["strike", "bid", "ask"])

    result = compare_chain_to_forecast(
        calls=calls,
        puts=puts,
        spot=spot,
        expiry=_future_expiry(T_days),
        forecast_rv=forecast_rv,
    )

    assert not result.empty
    assert (result["richness"] > 0).all(), "All options should look rich when IV > forecast"
    # richness ≈ sigma - forecast_rv within a small tolerance (solver accuracy)
    np.testing.assert_allclose(result["richness"], sigma - forecast_rv, atol=5e-3)


def test_richness_sign_cheap():
    """When market IV (sigma=0.10) < forecast (0.20), richness should be negative."""
    spot = 100.0
    sigma = 0.10
    forecast_rv = 0.20
    strikes = [95.0, 100.0, 105.0]

    calls = _make_chain(spot, strikes, sigma, 45 / 365)
    puts = pd.DataFrame(columns=["strike", "bid", "ask"])

    result = compare_chain_to_forecast(
        calls=calls,
        puts=puts,
        spot=spot,
        expiry=_future_expiry(45),
        forecast_rv=forecast_rv,
    )

    assert not result.empty
    assert (result["richness"] < 0).all(), "All options should look cheap when IV < forecast"


def test_moneyness_filter():
    """Strikes outside moneyness_range should be excluded."""
    spot = 100.0
    strikes = [70.0, 85.0, 100.0, 115.0, 135.0]  # 70 and 135 are outside (0.8, 1.2)
    sigma = 0.20

    calls = _make_chain(spot, strikes, sigma, 45 / 365)
    puts = pd.DataFrame(columns=["strike", "bid", "ask"])

    result = compare_chain_to_forecast(
        calls=calls,
        puts=puts,
        spot=spot,
        expiry=_future_expiry(45),
        forecast_rv=0.20,
        moneyness_range=(0.8, 1.2),
    )

    assert not result.empty
    moneyness = result["moneyness"].values
    assert moneyness.min() >= 0.8
    assert moneyness.max() <= 1.2
    # Strike=70 (moneyness=0.70) and 135 (moneyness=1.35) must be excluded
    assert 70.0 not in result["strike"].values
    assert 135.0 not in result["strike"].values


def test_expired_expiry_returns_empty():
    """An already-expired option chain should return an empty DataFrame."""
    spot = 100.0
    calls = _make_chain(spot, [100.0], 0.20, 30 / 365)
    puts = pd.DataFrame(columns=["strike", "bid", "ask"])

    result = compare_chain_to_forecast(
        calls=calls,
        puts=puts,
        spot=spot,
        expiry=_past_expiry(),
        forecast_rv=0.20,
    )

    assert result.empty
    assert list(result.columns) == _OUTPUT_COLS


def test_output_columns():
    """Result must have exactly the documented columns."""
    spot = 100.0
    calls = _make_chain(spot, [100.0], 0.20, 30 / 365)
    puts = _make_chain(spot, [100.0], 0.20, 30 / 365, option_type="put")

    result = compare_chain_to_forecast(
        calls=calls,
        puts=puts,
        spot=spot,
        expiry=_future_expiry(30),
        forecast_rv=0.20,
    )

    assert list(result.columns) == _OUTPUT_COLS


def test_both_legs_included():
    """Both calls and puts should appear when both are provided."""
    spot = 100.0
    strikes = [98.0, 100.0, 102.0]
    sigma = 0.20

    calls = _make_chain(spot, strikes, sigma, 30 / 365, option_type="call")
    puts = _make_chain(spot, strikes, sigma, 30 / 365, option_type="put")

    result = compare_chain_to_forecast(
        calls=calls,
        puts=puts,
        spot=spot,
        expiry=_future_expiry(30),
        forecast_rv=0.20,
    )

    assert "call" in result["option_type"].values
    assert "put" in result["option_type"].values


def test_invalid_quotes_filtered():
    """Options with zero bid or crossed quotes (bid > ask) must be dropped."""
    spot = 100.0
    calls = pd.DataFrame([
        {"strike": 100.0, "bid": 0.0, "ask": 5.0},   # zero bid → invalid
        {"strike": 101.0, "bid": 6.0, "ask": 4.0},   # crossed → invalid
        {"strike": 102.0, "bid": 3.0, "ask": 5.0},   # valid
    ])
    puts = pd.DataFrame(columns=["strike", "bid", "ask"])

    result = compare_chain_to_forecast(
        calls=calls,
        puts=puts,
        spot=spot,
        expiry=_future_expiry(30),
        forecast_rv=0.20,
    )

    # Only strike 102 should survive
    assert len(result) <= 1
    if not result.empty:
        assert result["strike"].iloc[0] == pytest.approx(102.0)
