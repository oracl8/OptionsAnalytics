import numpy as np


def _payoff(S_T: np.ndarray, K: float, option_type: str) -> np.ndarray:
    if option_type == "call":
        return np.maximum(S_T - K, 0.0)
    return np.maximum(K - S_T, 0.0)


def mc_price_antithetic(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    n_paths: int = 100_000,
    seed: int = 42,
) -> dict:
    """
    Price a European option with antithetic variates.

    For each draw Z, also evaluate the payoff at −Z.  Cov(f(Z), f(−Z)) < 0
    for monotone payoffs (calls/puts), so averaging the paired payoffs reduces
    variance relative to using Z alone.

    Fair comparison: var_naive uses the same n_paths draws from Z (same RNG
    budget); var_antithetic uses those same draws paired with their negations.

    Returns
    -------
    dict with keys: price, stderr, var_naive, var_antithetic, reduction_pct
    """
    ot = option_type.lower()
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)

    drift = (r - q - 0.5 * sigma**2) * T
    diff = sigma * np.sqrt(T)
    disc = np.exp(-r * T)

    S_T_pos = S * np.exp(drift + diff * Z)
    S_T_neg = S * np.exp(drift - diff * Z)

    pay_pos = _payoff(S_T_pos, K, ot)
    pay_neg = _payoff(S_T_neg, K, ot)
    paired = (pay_pos + pay_neg) * 0.5

    var_naive = float(np.var(disc * pay_pos, ddof=1))
    var_anti = float(np.var(disc * paired, ddof=1))

    price = disc * paired.mean()
    stderr = disc * paired.std(ddof=1) / np.sqrt(n_paths)

    return {
        "price": float(price),
        "stderr": float(stderr),
        "var_naive": var_naive,
        "var_antithetic": var_anti,
        "reduction_pct": float((1.0 - var_anti / var_naive) * 100.0),
    }


def mc_price_control_variate(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    n_paths: int = 100_000,
    seed: int = 42,
) -> dict:
    """
    Price a European option using the discounted stock price as a control variate.

    Control:   X  = e^(−rT) · S_T
    Known E[X] = S · e^(−qT)  (the prepaid forward / risk-neutral expectation)

    Adjusted estimator:
        Y_cv = Y − c* · (X − E[X]),   c* = Cov(Y, X) / Var(X)

    Variance reduction ≈ 1 − ρ²(Y, X).  For an ATM call the correlation is
    high (≈ 0.9+), giving >50 % variance reduction in practice.

    c* is estimated in-sample; the resulting bias is O(1/N) and negligible
    for N ≥ 50 000.

    Returns
    -------
    dict with keys: price, stderr, var_naive, var_cv, reduction_pct
    """
    ot = option_type.lower()
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)

    drift = (r - q - 0.5 * sigma**2) * T
    S_T = S * np.exp(drift + sigma * np.sqrt(T) * Z)
    disc = np.exp(-r * T)

    payoffs = _payoff(S_T, K, ot)
    Y = disc * payoffs               # discounted payoffs
    X = disc * S_T                   # control: discounted stock
    E_X = S * np.exp(-q * T)        # known expectation of X

    # Optimal coefficient (in-sample OLS estimate)
    cov_mat = np.cov(Y, X, ddof=1)
    c_star = cov_mat[0, 1] / cov_mat[1, 1]

    Y_cv = Y - c_star * (X - E_X)

    var_naive = float(Y.var(ddof=1))
    var_cv = float(Y_cv.var(ddof=1))

    price = Y_cv.mean()
    stderr = Y_cv.std(ddof=1) / np.sqrt(n_paths)

    return {
        "price": float(price),
        "stderr": float(stderr),
        "var_naive": var_naive,
        "var_cv": var_cv,
        "reduction_pct": float((1.0 - var_cv / var_naive) * 100.0),
    }
