import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma, q):
    """Return (d1, d2) from the BSM formula. All inputs converted to float arrays."""
    S, K, T, r, sigma, q = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q))
    sqrt_T = np.sqrt(T)
    # d1 = [ln(S/K) + (r - q + σ²/2)·T] / (σ·√T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bsm_price(S, K, T, r, sigma, q=0.0, option_type="call"):
    """
    Black-Scholes-Merton price for a European call or put.

    Parameters
    ----------
    S, K     : spot and strike (scalar or ndarray)
    T        : time to expiry in years
    r        : continuously compounded risk-free rate
    sigma    : annualised volatility
    q        : continuous dividend yield (default 0)
    option_type : "call" or "put"

    Returns
    -------
    ndarray  (scalar input → 0-d array)
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    S, K, T, r, q = (np.asarray(x, dtype=float) for x in (S, K, T, r, q))

    ot = option_type.lower()
    if ot == "call":
        # C = S·e^(−qT)·N(d1) − K·e^(−rT)·N(d2)
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    if ot == "put":
        # P = K·e^(−rT)·N(−d2) − S·e^(−qT)·N(−d1)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
