from __future__ import annotations

import numpy as np
import pandas as pd


def realized_vol(
    prices: pd.Series,
    horizon: int,
    trading_days: int = 252,
) -> pd.Series:
    """
    Forward-looking realized-volatility target (annualized).

    At each date t_i, returns the annualized std of log-returns over the NEXT
    `horizon` days — i.e., std(log_ret[t_{i+1} ... t_{i+horizon}]) * sqrt(trading_days).

    This is the lookahead-free forecasting target: the value at t_i uses only
    data that lies strictly in the future relative to t_i.  The last `horizon`
    entries of the output are NaN because they have insufficient future data.

    Parameters
    ----------
    prices : pd.Series
        Daily closing prices with a DatetimeIndex (or any monotonic index).
        Length N+1 produces N log-returns.
    horizon : int
        Forecast horizon in trading days (e.g. 5 for one week, 21 for one month).
    trading_days : int
        Annualisation factor (252 for equities).

    Returns
    -------
    pd.Series
        Same index as `prices`.  Value at index i = annualized realized vol
        over the horizon starting the day after i.  NaN for i > N - horizon.
    """
    log_ret = np.log(prices).diff().dropna()  # N values
    n = len(log_ret)

    # Output is aligned to prices.index.  Entry i corresponds to the prediction
    # made after observing prices[i], targeting future returns log_ret[i..i+h-1].
    target = pd.Series(np.nan, index=prices.index, dtype=float)

    ann = np.sqrt(trading_days)
    for i in range(n - horizon + 1):
        target.iloc[i] = log_ret.iloc[i : i + horizon].std(ddof=1) * ann

    return target
