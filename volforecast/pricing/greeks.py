import numpy as np
from scipy.stats import norm

from .black_scholes import _d1_d2


def delta(S, K, T, r, sigma, q=0.0, option_type="call"):
    """
    BSM delta: ∂price/∂S.
    call: e^(−qT)·N(d1)   put: e^(−qT)·(N(d1) − 1)
    """
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    S, K, T, r, q = (np.asarray(x, dtype=float) for x in (S, K, T, r, q))
    disc = np.exp(-q * T)
    nd1 = norm.cdf(d1)
    ot = option_type.lower()
    if ot == "call":
        return disc * nd1
    if ot == "put":
        return disc * (nd1 - 1.0)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def gamma(S, K, T, r, sigma, q=0.0):
    """
    BSM gamma: ∂²price/∂S² — identical for calls and puts.
    γ = e^(−qT)·φ(d1) / (S·σ·√T)
    """
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    S, sigma, T, q = (np.asarray(x, dtype=float) for x in (S, sigma, T, q))
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma, q=0.0):
    """
    BSM vega: ∂price/∂σ — identical for calls and puts.
    ν = S·e^(−qT)·φ(d1)·√T

    Returned as ∂price/∂σ (not per vol point). Divide by 100 for "per 1% move in vol".
    """
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    S, T, q = (np.asarray(x, dtype=float) for x in (S, T, q))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def theta(S, K, T, r, sigma, q=0.0, option_type="call"):
    """
    BSM theta: ∂price/∂t (rate of change as calendar time advances, T shrinks).
    Returned annualised; divide by 252 for daily decay.

    call: −S·e^(−qT)·φ(d1)·σ/(2√T) − r·K·e^(−rT)·N(d2) + q·S·e^(−qT)·N(d1)
    put:  −S·e^(−qT)·φ(d1)·σ/(2√T) + r·K·e^(−rT)·N(−d2) − q·S·e^(−qT)·N(−d1)
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    S, K, T, r, sigma, q = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q))
    sqrt_T = np.sqrt(T)
    # common decay term (negative); same sign for calls and puts
    decay = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2.0 * sqrt_T)
    ot = option_type.lower()
    if ot == "call":
        return decay - r * K * np.exp(-r * T) * norm.cdf(d2) + q * S * np.exp(-q * T) * norm.cdf(d1)
    if ot == "put":
        return decay + r * K * np.exp(-r * T) * norm.cdf(-d2) - q * S * np.exp(-q * T) * norm.cdf(-d1)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def rho(S, K, T, r, sigma, q=0.0, option_type="call"):
    """
    BSM rho: ∂price/∂r.
    call:  K·T·e^(−rT)·N(d2)
    put:  −K·T·e^(−rT)·N(−d2)
    """
    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    K, T, r = (np.asarray(x, dtype=float) for x in (K, T, r))
    base = K * T * np.exp(-r * T)
    ot = option_type.lower()
    if ot == "call":
        return base * norm.cdf(d2)
    if ot == "put":
        return -base * norm.cdf(-d2)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
