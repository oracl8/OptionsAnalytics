import numpy as np


def mc_asian_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    n_paths: int = 50_000,
    n_steps: int = 252,
    seed: int = 42,
    antithetic: bool = True,
) -> dict:
    """
    Price a fixed-strike arithmetic-average Asian option by Monte Carlo.

    No closed form exists for arithmetic Asians under GBM (unlike geometric
    Asians), so MC is the natural pricer.

    Payoff:
        call: max(A − K, 0)   where A = (1/n_steps) · Σ S_{t_i}
        put:  max(K − A, 0)

    The average is over n_steps equally-spaced dates in (0, T] — S₀ is not
    included in the average.

    Antithetic variates:
        When antithetic=True, generate Z of shape (n_paths//2, n_steps) then
        stack with -Z.  Negating every increment of a path reflects it, which
        is negatively correlated with the original.  Variance reduction is
        moderate (less than for European options because the averaging smooths
        out the correlation).

    Memory: n_paths × n_steps × 8 bytes  (~100 MB at 50k × 252).

    Returns
    -------
    dict with keys: price, stderr
    """
    ot = option_type.lower()
    if ot not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    diff = sigma * np.sqrt(dt)

    if antithetic:
        half = n_paths // 2
        Z_half = rng.standard_normal((half, n_steps))
        # shape: (n_paths, n_steps) — first half positive, second half negative
        Z = np.vstack([Z_half, -Z_half])
        n_paths = Z.shape[0]   # in case n_paths was odd
    else:
        Z = rng.standard_normal((n_paths, n_steps))

    # Log-returns for each step: shape (n_paths, n_steps)
    log_increments = drift + diff * Z
    # Cumulative log-price relative to S₀: shape (n_paths, n_steps)
    log_paths = np.cumsum(log_increments, axis=1)
    S_paths = S * np.exp(log_paths)   # shape (n_paths, n_steps)

    # Arithmetic average over monitoring dates (excluding S₀)
    A = S_paths.mean(axis=1)  # shape (n_paths,)

    if ot == "call":
        payoffs = np.maximum(A - K, 0.0)
    else:
        payoffs = np.maximum(K - A, 0.0)

    disc = np.exp(-r * T)
    price = disc * payoffs.mean()
    stderr = disc * payoffs.std(ddof=1) / np.sqrt(n_paths)

    return {"price": float(price), "stderr": float(stderr)}
