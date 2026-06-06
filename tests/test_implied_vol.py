import numpy as np
import pytest

from volforecast.pricing.black_scholes import bsm_price
from volforecast.pricing.implied_vol import implied_vol

S0, K0, T0, r0, sig0, q0 = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0


# ---------------------------------------------------------------------------
# Round-trip: price → implied_vol → price must recover input sigma
# ---------------------------------------------------------------------------


def test_round_trip_atm_call():
    price = float(bsm_price(S0, K0, T0, r0, sig0, q0, "call"))
    iv = float(implied_vol(price, S0, K0, T0, r0, q0, "call"))
    assert abs(iv - sig0) < 1e-6


def test_round_trip_atm_put():
    price = float(bsm_price(S0, K0, T0, r0, sig0, q0, "put"))
    iv = float(implied_vol(price, S0, K0, T0, r0, q0, "put"))
    assert abs(iv - sig0) < 1e-6


def test_round_trip_grid():
    # Grid covering a variety of moneyness, maturities, and vols
    S   = np.array([80.0, 100.0, 120.0, 100.0, 100.0, 100.0])
    K   = np.array([100.0, 100.0, 100.0,  80.0, 120.0, 100.0])
    T   = np.array([0.25,  0.5,   1.0,   1.0,   1.0,   2.0])
    r   = np.array([0.02,  0.05,  0.05,  0.05,  0.05,  0.03])
    q   = np.array([0.0,   0.0,   0.0,   0.0,   0.0,   0.01])
    sig = np.array([0.15,  0.20,  0.25,  0.30,  0.10,  0.35])

    for ot in ("call", "put"):
        prices = np.asarray(bsm_price(S, K, T, r, sig, q, ot), dtype=float)
        ivs = implied_vol(prices, S, K, T, r, q, ot)
        np.testing.assert_allclose(ivs, sig, atol=1e-6, err_msg=f"round-trip failed for {ot}")


# ---------------------------------------------------------------------------
# Bisection fallback path: deep OTM where vega is essentially zero
# ---------------------------------------------------------------------------


def test_bisection_path_deep_otm_call():
    # d1 at sigma=0.2 initial guess ≈ −10.8 → vega ≈ 0 → NR routes to bisection.
    # Deep OTM has tiny vega at the solution too, so sigma accuracy is limited (~1e-3);
    # we verify in price space instead, which is what the solver guarantees.
    S, K, T, r, q, sig = 100.0, 200.0, 0.1, 0.05, 0.0, 0.40
    market_price = float(bsm_price(S, K, T, r, sig, q, "call"))
    iv = float(implied_vol(market_price, S, K, T, r, q, "call"))
    assert not np.isnan(iv), "bisection should find a solution for a valid deep-OTM call"
    recovered = float(bsm_price(S, K, T, r, iv, q, "call"))
    assert abs(recovered - market_price) < 1e-7


def test_bisection_path_deep_otm_put():
    # Deeply OTM put: S >> K
    S, K, T, r, q, sig = 200.0, 100.0, 0.1, 0.05, 0.0, 0.40
    market_price = float(bsm_price(S, K, T, r, sig, q, "put"))
    iv = float(implied_vol(market_price, S, K, T, r, q, "put"))
    assert not np.isnan(iv), "bisection should find a solution for a valid deep-OTM put"
    recovered = float(bsm_price(S, K, T, r, iv, q, "put"))
    assert abs(recovered - market_price) < 1e-7


def test_bisection_path_deep_itm_call():
    # Deep ITM call: S >> K → d1 at initial guess very large → vega ≈ 0 → bisection fallback
    S, K, T, r, q, sig = 200.0, 100.0, 0.1, 0.05, 0.0, 0.40
    market_price = float(bsm_price(S, K, T, r, sig, q, "call"))
    iv = float(implied_vol(market_price, S, K, T, r, q, "call"))
    assert not np.isnan(iv), "bisection should find a solution for a valid deep-ITM call"
    recovered = float(bsm_price(S, K, T, r, iv, q, "call"))
    assert abs(recovered - market_price) < 1e-7


# ---------------------------------------------------------------------------
# Degenerate inputs must return NaN without raising
# ---------------------------------------------------------------------------


def test_degenerate_price_below_intrinsic_call():
    # Call intrinsic ≈ max(S − K·e^(−rT), 0); pass a price well below it
    intrinsic = max(S0 - K0 * np.exp(-r0 * T0), 0.0)
    bad_price = intrinsic - 1.0
    iv = float(implied_vol(bad_price, S0, K0, T0, r0, q0, "call"))
    assert np.isnan(iv)


def test_degenerate_price_below_intrinsic_put():
    intrinsic = max(K0 * np.exp(-r0 * T0) - S0, 0.0)
    bad_price = intrinsic - 1.0
    iv = float(implied_vol(bad_price, S0, K0, T0, r0, q0, "put"))
    assert np.isnan(iv)


def test_degenerate_zero_T():
    price = float(bsm_price(S0, K0, T0, r0, sig0, q0, "call"))
    iv = float(implied_vol(price, S0, K0, 0.0, r0, q0, "call"))
    assert np.isnan(iv)


def test_degenerate_price_above_upper_bound_call():
    # Call price cannot exceed S·e^(−qT) = S (for q=0)
    bad_price = S0 + 10.0
    iv = float(implied_vol(bad_price, S0, K0, T0, r0, q0, "call"))
    assert np.isnan(iv)


def test_degenerate_negative_price():
    iv = float(implied_vol(-1.0, S0, K0, T0, r0, q0, "call"))
    assert np.isnan(iv)


# ---------------------------------------------------------------------------
# Vectorization: array input preserves shape, all elements correct
# ---------------------------------------------------------------------------


def test_vectorized_shape_and_values():
    K = np.linspace(80, 120, 20)
    sig_in = np.full(20, 0.25)
    prices = np.asarray(bsm_price(S0, K, T0, r0, sig_in, q0, "call"), dtype=float)
    ivs = implied_vol(prices, S0, K, T0, r0, q0, "call")
    assert ivs.shape == (20,)
    np.testing.assert_allclose(ivs, sig_in, atol=1e-6)


def test_scalar_input_returns_0d_array():
    price = float(bsm_price(S0, K0, T0, r0, sig0, q0, "call"))
    iv = implied_vol(price, S0, K0, T0, r0, q0, "call")
    assert isinstance(iv, np.ndarray)
    assert iv.ndim == 0


# ---------------------------------------------------------------------------
# Mixed valid / invalid in same array → NaN only for bad ones
# ---------------------------------------------------------------------------


def test_mixed_valid_invalid():
    sig_in = np.array([0.20, 0.30])
    K = np.array([100.0, 100.0])
    prices_good = np.asarray(bsm_price(S0, K, T0, r0, sig_in, q0, "call"), dtype=float)
    # Replace second price with something below intrinsic
    prices_mixed = prices_good.copy()
    prices_mixed[1] = -5.0
    ivs = implied_vol(prices_mixed, S0, K, T0, r0, q0, "call")
    assert abs(ivs[0] - sig_in[0]) < 1e-6
    assert np.isnan(ivs[1])
