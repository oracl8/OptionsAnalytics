import numpy as np
import pytest

from volforecast.pricing.black_scholes import bsm_price
from volforecast.pricing.greeks import delta, gamma, rho, theta, vega

# Standard ATM reference case: S=K=100, T=1y, r=5%, q=0, σ=20%
# d1=0.35, d2=0.15  →  call≈10.4506, put≈5.5735
S0, K0, T0, r0, sig0, q0 = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0


# ---------------------------------------------------------------------------
# Known reference values
# ---------------------------------------------------------------------------


def test_known_call_price():
    assert abs(float(bsm_price(S0, K0, T0, r0, sig0, q0, "call")) - 10.4506) < 5e-4


def test_known_put_price():
    assert abs(float(bsm_price(S0, K0, T0, r0, sig0, q0, "put")) - 5.5735) < 5e-4


def test_known_delta():
    # N(d1) = N(0.35) ≈ 0.6368
    assert abs(float(delta(S0, K0, T0, r0, sig0, q0, "call")) - 0.6368) < 5e-4


def test_known_gamma():
    # φ(0.35) / (100 * 0.20) ≈ 0.0188
    assert abs(float(gamma(S0, K0, T0, r0, sig0, q0)) - 0.0188) < 5e-4


def test_known_vega():
    # 100 * φ(0.35) * 1 ≈ 37.524
    assert abs(float(vega(S0, K0, T0, r0, sig0, q0)) - 37.524) < 5e-3


# ---------------------------------------------------------------------------
# Put-call parity over a grid
# ---------------------------------------------------------------------------


def test_put_call_parity():
    S = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    K = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
    T = np.array([0.25, 0.5, 1.0, 1.5, 2.0])
    r = np.array([0.03, 0.04, 0.05, 0.05, 0.06])
    q = np.array([0.0, 0.01, 0.0, 0.02, 0.0])
    sig = np.array([0.15, 0.20, 0.25, 0.20, 0.30])

    call = bsm_price(S, K, T, r, sig, q, "call")
    put = bsm_price(S, K, T, r, sig, q, "put")
    # C − P = S·e^(−qT) − K·e^(−rT)
    np.testing.assert_allclose(call - put, S * np.exp(-q * T) - K * np.exp(-r * T), atol=1e-10)


# ---------------------------------------------------------------------------
# Finite-difference checks — closed-form Greeks vs. central differences
# ---------------------------------------------------------------------------


def test_delta_fd():
    h = 0.01
    fd = (bsm_price(S0 + h, K0, T0, r0, sig0, q0) - bsm_price(S0 - h, K0, T0, r0, sig0, q0)) / (2 * h)
    assert abs(float(delta(S0, K0, T0, r0, sig0, q0)) - float(fd)) < 1e-5


def test_gamma_fd():
    h = 0.01
    # Second-order central difference for second derivative
    c_up = bsm_price(S0 + h, K0, T0, r0, sig0, q0)
    c_0 = bsm_price(S0, K0, T0, r0, sig0, q0)
    c_dn = bsm_price(S0 - h, K0, T0, r0, sig0, q0)
    fd = (c_up - 2 * c_0 + c_dn) / h**2
    assert abs(float(gamma(S0, K0, T0, r0, sig0, q0)) - float(fd)) < 1e-4


def test_vega_fd():
    h = 1e-4
    fd = (bsm_price(S0, K0, T0, r0, sig0 + h, q0) - bsm_price(S0, K0, T0, r0, sig0 - h, q0)) / (2 * h)
    assert abs(float(vega(S0, K0, T0, r0, sig0, q0)) - float(fd)) < 1e-5


def test_theta_fd():
    h = 1.0 / 252
    # theta = ∂C/∂t = −∂C/∂T; FD bumps T, so negate to get conventional theta
    fd = -(bsm_price(S0, K0, T0 + h, r0, sig0, q0) - bsm_price(S0, K0, T0 - h, r0, sig0, q0)) / (2 * h)
    assert abs(float(theta(S0, K0, T0, r0, sig0, q0)) - float(fd)) < 1e-4


def test_rho_fd():
    h = 1e-4
    fd = (bsm_price(S0, K0, T0, r0 + h, sig0, q0) - bsm_price(S0, K0, T0, r0 - h, sig0, q0)) / (2 * h)
    assert abs(float(rho(S0, K0, T0, r0, sig0, q0)) - float(fd)) < 1e-5


def test_put_delta_fd():
    h = 0.01
    fd = (bsm_price(S0 + h, K0, T0, r0, sig0, q0, "put") - bsm_price(S0 - h, K0, T0, r0, sig0, q0, "put")) / (2 * h)
    assert abs(float(delta(S0, K0, T0, r0, sig0, q0, "put")) - float(fd)) < 1e-5


# ---------------------------------------------------------------------------
# Vectorised call — shape and monotonicity
# ---------------------------------------------------------------------------


def test_vectorized_strikes():
    K = np.linspace(80, 120, 50)
    prices = bsm_price(S0, K, T0, r0, sig0, q0, "call")
    assert prices.shape == (50,)
    # Higher strike → lower call price (all else equal)
    assert np.all(np.diff(prices) < 0)
