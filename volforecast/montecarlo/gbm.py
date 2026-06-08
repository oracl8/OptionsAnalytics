import numpy as np


def mc_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    n_paths: int = 100_000,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Price a European option by Monte Carlo simulation of GBM terminal prices.

    Risk-neutral dynamics:
        S_T = S · exp((r − q − σ²/2)·T + σ·√T·Z),   Z ~ N(0,1)

    Returns
    -------
    (price, stderr)
        price  : discounted mean payoff
        stderr : standard error of the mean (one standard deviation of the
                 MC estimator), useful for convergence checks
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)

    drift = (r - q - 0.5 * sigma**2) * T
    S_T = S * np.exp(drift + sigma * np.sqrt(T) * Z)

    ot = option_type.lower()
    if ot == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    elif ot == "put":
        payoffs = np.maximum(K - S_T, 0.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    disc = np.exp(-r * T)
    price = disc * payoffs.mean()
    stderr = disc * payoffs.std(ddof=1) / np.sqrt(n_paths)
    return float(price), float(stderr)
