from __future__ import annotations

import warnings
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from ..data.loader import MarketDataSource, YFinanceSource
from ..pricing.greeks import vega as bsm_vega
from ..pricing.implied_vol import implied_vol

_SURFACE_COLS = [
    "expiry", "T", "strike", "moneyness", "log_moneyness",
    "iv", "bid", "ask", "mid", "option_type",
]

# Vega threshold below which IV estimates are considered unreliable.
# Deep-OTM/ITM options with near-zero vega produce noisy IV even when
# the solver converges, because a small price error maps to a huge sigma error.
_VEGA_MIN = 1e-6


def build_surface(
    ticker: str,
    source: MarketDataSource | None = None,
    r: float = 0.045,
    q: float = 0.0,
    option_type: str = "call",
    max_expiries: int | None = 12,
    max_iv: float = 2.0,
) -> pd.DataFrame:
    """
    Compute the implied-volatility surface for *ticker* across available expiries.

    Parameters
    ----------
    ticker       : equity ticker symbol (e.g. "SPY")
    source       : data source; defaults to YFinanceSource()
    r            : risk-free rate (continuously compounded annualised)
    q            : continuous dividend yield
    option_type  : "call", "put", or "both"
    max_expiries : cap the number of expiries fetched (None = all)
    max_iv       : drop options whose solved IV exceeds this (default 2.0 = 200%).
                   Deep-OTM short-dated options with non-zero bid can produce
                   arbitrarily large IV values that are not meaningful for surface
                   fitting.

    Returns
    -------
    DataFrame with columns: expiry, T, strike, moneyness, log_moneyness,
    iv, bid, ask, mid, option_type
    Bad quotes and solver failures are silently dropped.
    """
    if source is None:
        source = YFinanceSource()

    # Spot price: last available close
    today = date.today()
    start = (today - timedelta(days=7)).isoformat()
    ohlcv = source.get_ohlcv(ticker, start=start, end=today.isoformat())
    if ohlcv.empty:
        raise ValueError(f"No OHLCV data returned for {ticker}; cannot determine spot price.")
    S = float(ohlcv["Close"].iloc[-1])

    expiries = source.get_available_expiries(ticker)
    if not expiries:
        warnings.warn(f"No option expiries available for {ticker}", stacklevel=2)
        return pd.DataFrame(columns=_SURFACE_COLS)

    if max_expiries is not None:
        expiries = expiries[:max_expiries]

    types_to_process = ["call", "put"] if option_type == "both" else [option_type]
    dfs: list[pd.DataFrame] = []

    for expiry in expiries:
        exp_dt = datetime.strptime(expiry, "%Y-%m-%d").date()
        T_days = (exp_dt - today).days
        if T_days <= 0:
            continue
        T = T_days / 365.0

        chain = source.get_option_chain(ticker, expiry=expiry)

        for ot in types_to_process:
            df = chain[ot + "s"].copy()
            if df.empty:
                continue

            # Mid-price
            df["mid"] = (df["bid"] + df["ask"]) / 2.0

            # Filter crossed / no-market quotes
            good = (df["bid"] > 0) & (df["ask"] > 0) & (df["bid"] <= df["ask"])
            df = df[good].reset_index(drop=True)
            if df.empty:
                continue

            strikes = df["strike"].values.astype(float)
            mids = df["mid"].values.astype(float)
            bids = df["bid"].values.astype(float)
            asks = df["ask"].values.astype(float)

            ivs = implied_vol(
                market_price=mids,
                S=S,
                K=strikes,
                T=T,
                r=r,
                q=q,
                option_type=ot,
            )

            # Near-zero vega filter: IV is numerically unreliable where vega ≈ 0
            vega_vals = np.asarray(bsm_vega(S, strikes, T, r, ivs, q), dtype=float).ravel()

            keep = ~np.isnan(ivs) & (vega_vals >= _VEGA_MIN) & (ivs <= max_iv)
            if not keep.any():
                continue

            strikes = strikes[keep]
            ivs = ivs[keep]
            bids = bids[keep]
            asks = asks[keep]
            mids = mids[keep]

            expiry_df = pd.DataFrame(
                {
                    "expiry": expiry,
                    "T": T,
                    "strike": strikes,
                    "moneyness": strikes / S,
                    "log_moneyness": np.log(strikes / S),
                    "iv": ivs,
                    "bid": bids,
                    "ask": asks,
                    "mid": mids,
                    "option_type": ot,
                }
            )
            dfs.append(expiry_df)

    if not dfs:
        return pd.DataFrame(columns=_SURFACE_COLS)

    surface = pd.concat(dfs, ignore_index=True)
    surface = surface.sort_values(["T", "strike"]).reset_index(drop=True)
    return surface
