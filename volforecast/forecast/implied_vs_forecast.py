"""
Compare model-forecast realized volatility to market implied volatility.

Research output: identifies which options look rich (IV > forecast) or cheap
(IV < forecast) relative to the model's view of future realized vol.
This is a research signal, not investment advice.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from volforecast.data.loader import MarketDataSource, YFinanceSource
from volforecast.pricing.implied_vol import implied_vol as _solve_iv

_OUTPUT_COLS = ["strike", "option_type", "moneyness", "implied_vol", "forecast_rv", "richness"]


def compare_chain_to_forecast(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spot: float,
    expiry: str,
    forecast_rv: float,
    r: float = 0.045,
    q: float = 0.0,
    moneyness_range: tuple[float, float] = (0.8, 1.2),
) -> pd.DataFrame:
    """
    Compare one expiry's option chain to a realized-vol forecast.

    Parameters
    ----------
    calls, puts     : normalized chain DataFrames (columns: strike, bid, ask)
    spot            : current underlying price
    expiry          : option expiry date string "YYYY-MM-DD"
    forecast_rv     : annualized realized-vol forecast (e.g. 0.18 = 18%)
    r               : risk-free rate (continuously compounded, annualized)
    q               : continuous dividend yield
    moneyness_range : (lo, hi) filter on K/S; default keeps strikes within ±20% of spot

    Returns
    -------
    DataFrame with columns:
        strike       : option strike price
        option_type  : "call" or "put"
        moneyness    : K / S
        implied_vol  : market-implied annualized vol from mid-price
        forecast_rv  : the forecast (same scalar broadcast across all rows)
        richness     : implied_vol − forecast_rv
                       positive → option IV exceeds forecast (looks rich)
                       negative → option IV below forecast (looks cheap)
    """
    exp_date = date.fromisoformat(expiry)
    T = (exp_date - date.today()).days / 365.0
    if T <= 0:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    lo, hi = moneyness_range
    pieces: list[pd.DataFrame] = []

    for leg_df, ot in [(calls, "call"), (puts, "put")]:
        if leg_df.empty:
            continue

        df = leg_df.copy()
        df["mid"] = (df["bid"] + df["ask"]) / 2.0

        # Valid, non-crossed quotes only
        valid = (df["bid"] > 0) & (df["ask"] > 0) & (df["bid"] <= df["ask"])
        df = df[valid].reset_index(drop=True)
        if df.empty:
            continue

        strikes = df["strike"].values.astype(float)
        mids = df["mid"].values.astype(float)

        ivs = _solve_iv(
            market_price=mids,
            S=spot,
            K=strikes,
            T=T,
            r=r,
            q=q,
            option_type=ot,
        )
        ivs = np.asarray(ivs, dtype=float)

        moneyness = strikes / spot
        keep = ~np.isnan(ivs) & (moneyness >= lo) & (moneyness <= hi)
        if not keep.any():
            continue

        piece = pd.DataFrame(
            {
                "strike": strikes[keep],
                "option_type": ot,
                "moneyness": moneyness[keep],
                "implied_vol": ivs[keep],
                "forecast_rv": forecast_rv,
                "richness": ivs[keep] - forecast_rv,
            }
        )
        pieces.append(piece)

    if not pieces:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    return pd.concat(pieces, ignore_index=True).sort_values("strike").reset_index(drop=True)


def build_richness_table(
    ticker: str,
    forecast_rv: float,
    source: MarketDataSource | None = None,
    r: float = 0.045,
    q: float = 0.0,
    expiry: str | None = None,
    moneyness_range: tuple[float, float] = (0.8, 1.2),
) -> pd.DataFrame:
    """
    Fetch a live option chain and return the implied-vs-forecast richness table.

    Parameters
    ----------
    ticker       : equity ticker (e.g. "SPY")
    forecast_rv  : annualized realized-vol forecast
    source       : data source; defaults to YFinanceSource()
    r, q         : risk-free rate and dividend yield
    expiry       : "YYYY-MM-DD"; None → nearest available expiry
    moneyness_range : (lo, hi) filter on K/S

    Returns
    -------
    DataFrame as described in compare_chain_to_forecast.
    Empty DataFrame if chain or spot data is unavailable.
    """
    if source is None:
        source = YFinanceSource()

    today = date.today()
    start = (today - timedelta(days=7)).isoformat()
    ohlcv = source.get_ohlcv(ticker, start=start, end=today.isoformat())
    if ohlcv.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)
    spot = float(ohlcv["Close"].iloc[-1])

    chain = source.get_option_chain(ticker, expiry=expiry)
    if chain["expiry"] is None:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    return compare_chain_to_forecast(
        calls=chain["calls"],
        puts=chain["puts"],
        spot=spot,
        expiry=chain["expiry"],
        forecast_rv=forecast_rv,
        r=r,
        q=q,
        moneyness_range=moneyness_range,
    )
