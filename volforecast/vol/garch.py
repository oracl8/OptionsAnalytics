from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_vol(
    returns: np.ndarray,
    lambda_: float = 0.94,
    trading_days: int = 252,
) -> np.ndarray:
    """
    RiskMetrics EWMA volatility (annualized).

    Recursion: σ²_t = λ·σ²_{t-1} + (1−λ)·r²_t

    The series is warm-started with the sample mean of r² to avoid cold-start
    bias from an arbitrary σ²_0 = 0.

    Parameters
    ----------
    returns : array-like
        Daily log-returns (as fractions, not percent).
    lambda_ : float
        Decay factor; 0.94 is the RiskMetrics daily default.
    trading_days : int
        Annualisation factor.

    Returns
    -------
    np.ndarray
        Annualized EWMA volatility, same length as `returns`.
    """
    r = np.asarray(returns, dtype=float)
    r2 = r**2

    # pandas ewm with adjust=False implements the RiskMetrics recursion exactly,
    # seeding with r2[0] (which equals the sample mean for a length-1 window).
    # We override the first element with the full-sample mean to warm-start.
    r2_series = pd.Series(r2)
    r2_series.iloc[0] = r2.mean()

    var = r2_series.ewm(alpha=1.0 - lambda_, adjust=False).mean().to_numpy()
    return np.sqrt(var * trading_days)


def garch_forecast(
    returns: np.ndarray,
    horizon: int,
    trading_days: int = 252,
) -> float:
    """
    Fit GARCH(1,1) and return the h-step-ahead annualized volatility forecast.

    The model is fit on the full `returns` series and the forecast is extracted
    for step `horizon`.  In the ML pipeline this will be refitted inside each
    walk-forward fold; here it is exposed as a standalone baseline.

    Parameters
    ----------
    returns : array-like
        Daily log-returns (fractions).  Minimum ~100 observations recommended
        for stable GARCH parameter estimation.
    horizon : int
        Number of trading days ahead to forecast.
    trading_days : int
        Annualisation factor.

    Returns
    -------
    float
        Annualized GARCH(1,1) h-step-ahead conditional volatility forecast.
    """
    from arch import arch_model

    r = np.asarray(returns, dtype=float)
    # Scale to percentage returns; arch has better numerical behaviour near 1.0
    scaled = r * 100.0
    model = arch_model(scaled, vol="Garch", p=1, q=1, rescale=False)
    res = model.fit(disp="off")
    fc = res.forecast(horizon=horizon, reindex=False)

    # fc.variance has shape (1, horizon); [-1, -1] is the h-step-ahead variance
    # in (pct)².  Convert: var_frac = var_pct / 100², then annualize.
    var_pct = float(fc.variance.values[-1, -1])
    return float(np.sqrt(var_pct / 10_000.0 * trading_days))


if __name__ == "__main__":
    # Quick sanity check: fit on synthetic data and print forecast
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, 500)
    vol = garch_forecast(r, horizon=21)
    ewma = ewma_vol(r)
    print(f"GARCH(1,1) 21-day forecast: {vol:.4f}  ({vol*100:.2f}%)")
    print(f"EWMA vol (last):            {ewma[-1]:.4f}  ({ewma[-1]*100:.2f}%)")
