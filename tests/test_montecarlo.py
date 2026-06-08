import numpy as np
import pytest

from volforecast.montecarlo.exotic import mc_asian_price
from volforecast.montecarlo.gbm import mc_price
from volforecast.montecarlo.variance_reduction import (
    mc_price_antithetic,
    mc_price_control_variate,
)
from volforecast.pricing.black_scholes import bsm_price

# Standard ATM reference parameters
S0, K0, T0, r0, sig0, q0 = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0


# ---------------------------------------------------------------------------
# Convergence to Black-Scholes (the primary correctness check)
# ---------------------------------------------------------------------------
# If the simulator has wrong drift, missing discount, or sign error, the
# MC price will be biased by far more than 4 * stderr.


def test_mc_call_converges_to_bs():
    bs = float(bsm_price(S0, K0, T0, r0, sig0, q0, "call"))
    price, stderr = mc_price(S0, K0, T0, r0, sig0, q0, "call", n_paths=200_000, seed=42)
    assert abs(price - bs) < 4 * stderr


def test_mc_put_converges_to_bs():
    bs = float(bsm_price(S0, K0, T0, r0, sig0, q0, "put"))
    price, stderr = mc_price(S0, K0, T0, r0, sig0, q0, "put", n_paths=200_000, seed=42)
    assert abs(price - bs) < 4 * stderr


# ---------------------------------------------------------------------------
# Standard-error scaling: 4× paths should halve stderr
# ---------------------------------------------------------------------------


def test_stderr_scales_sqrt_n():
    # The 1/√N scaling is a fundamental property; we allow 15 % tolerance for
    # the finite-sample estimate with a fixed seed.
    _, se1 = mc_price(S0, K0, T0, r0, sig0, q0, n_paths=10_000, seed=0)
    _, se4 = mc_price(S0, K0, T0, r0, sig0, q0, n_paths=40_000, seed=0)
    assert abs(se4 / se1 - 0.5) < 0.15


# ---------------------------------------------------------------------------
# Antithetic variates
# ---------------------------------------------------------------------------


def test_antithetic_reduces_variance():
    result = mc_price_antithetic(S0, K0, T0, r0, sig0, q0, n_paths=100_000, seed=42)
    assert result["var_antithetic"] < result["var_naive"]
    # Require a meaningful reduction, not just numerical noise
    assert result["reduction_pct"] > 10.0


def test_antithetic_price_close_to_bs():
    bs = float(bsm_price(S0, K0, T0, r0, sig0, q0, "call"))
    result = mc_price_antithetic(S0, K0, T0, r0, sig0, q0, n_paths=100_000, seed=42)
    assert abs(result["price"] - bs) < 4 * result["stderr"]


def test_antithetic_put_reduces_variance():
    result = mc_price_antithetic(S0, K0, T0, r0, sig0, q0, option_type="put", n_paths=100_000, seed=42)
    assert result["var_antithetic"] < result["var_naive"]


# ---------------------------------------------------------------------------
# Control variate
# ---------------------------------------------------------------------------


def test_cv_reduces_variance():
    # CV exploits high correlation between call payoff and S_T;
    # expect > 50 % reduction for an ATM call.
    result = mc_price_control_variate(S0, K0, T0, r0, sig0, q0, n_paths=100_000, seed=42)
    assert result["var_cv"] < result["var_naive"]
    assert result["reduction_pct"] > 50.0


def test_cv_price_close_to_bs():
    bs = float(bsm_price(S0, K0, T0, r0, sig0, q0, "call"))
    result = mc_price_control_variate(S0, K0, T0, r0, sig0, q0, n_paths=100_000, seed=42)
    assert abs(result["price"] - bs) < 4 * result["stderr"]


def test_cv_beats_antithetic_variance():
    # Control variate should out-reduce antithetic for ATM calls under BSM.
    anti = mc_price_antithetic(S0, K0, T0, r0, sig0, q0, n_paths=100_000, seed=42)
    cv = mc_price_control_variate(S0, K0, T0, r0, sig0, q0, n_paths=100_000, seed=42)
    assert cv["reduction_pct"] > anti["reduction_pct"]


# ---------------------------------------------------------------------------
# Asian option
# ---------------------------------------------------------------------------


def test_asian_call_positive():
    result = mc_asian_price(S0, K0, T0, r0, sig0, q0, "call", n_paths=50_000, seed=0)
    assert result["price"] > 0.0


def test_asian_call_cheaper_than_european():
    # The arithmetic average of a GBM path has lower variance than the terminal
    # value, so the Asian call is always cheaper than the European call (for the
    # same K and T).
    euro_price, _ = mc_price(S0, K0, T0, r0, sig0, q0, "call", n_paths=100_000, seed=42)
    asian = mc_asian_price(S0, K0, T0, r0, sig0, q0, "call", n_paths=100_000, n_steps=50, seed=42)
    assert asian["price"] < euro_price


def test_asian_put_positive():
    result = mc_asian_price(S0, K0, T0, r0, sig0, q0, "put", n_paths=50_000, seed=0)
    assert result["price"] > 0.0


def test_asian_stderr_decreases_with_paths():
    r1 = mc_asian_price(S0, K0, T0, r0, sig0, q0, n_paths=10_000, n_steps=50, seed=0)
    r2 = mc_asian_price(S0, K0, T0, r0, sig0, q0, n_paths=40_000, n_steps=50, seed=0)
    assert r2["stderr"] < r1["stderr"]


def test_asian_bad_option_type():
    with pytest.raises(ValueError):
        mc_asian_price(S0, K0, T0, r0, sig0, q0, option_type="straddle")
