"""
Verify realized-vol target and baseline models on real SPY data.

Usage:
    python scripts/vol_baselines.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volforecast.data.loader import YFinanceSource
from volforecast.vol.estimators import garman_klass_vol, parkinson_vol
from volforecast.vol.garch import ewma_vol, garch_forecast
from volforecast.vol.realized_vol import realized_vol

TICKER = "SPY"
START = "2022-01-01"
END = "2024-01-01"
HORIZON = 21  # ~1 trading month


def main() -> None:
    src = YFinanceSource()
    df = src.get_ohlcv(TICKER, START, END)
    if df.empty:
        print(f"No data for {TICKER}")
        return

    print(f"\n{TICKER}  |  {START} to {END}  |  {len(df)} trading days\n")

    # ── Realized vol target ──────────────────────────────────────────────
    rv = realized_vol(df["Close"], horizon=HORIZON)
    print(f"Forward realized vol ({HORIZON}-day), last 5 valid rows:")
    print(rv.dropna().tail(5).apply(lambda x: f"{x:.4f}").to_frame("rv").to_string())

    # ── Range-based estimators (scalar over full sample) ─────────────────
    pv = parkinson_vol(df["High"].values, df["Low"].values)
    gk = garman_klass_vol(
        df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    )
    print(f"\nRange-based estimators (full sample):")
    print(f"  Parkinson    : {pv:.4f}  ({pv*100:.2f}%)")
    print(f"  Garman-Klass : {gk:.4f}  ({gk*100:.2f}%)")

    # ── EWMA baseline ────────────────────────────────────────────────────
    log_ret = np.log(df["Close"]).diff().dropna().to_numpy()
    ewma = ewma_vol(log_ret)
    ewma_series = pd.Series(ewma, index=df.index[1:])
    print(f"\nEWMA vol (lambda=0.94), last 5 rows:")
    print(ewma_series.tail(5).apply(lambda x: f"{x:.4f}").to_frame("ewma_vol").to_string())

    # ── GARCH(1,1) baseline ──────────────────────────────────────────────
    print(f"\nFitting GARCH(1,1) on {len(log_ret)} daily returns...")
    fc = garch_forecast(log_ret, horizon=HORIZON)
    print(f"  GARCH(1,1) {HORIZON}-day-ahead forecast: {fc:.4f}  ({fc*100:.2f}%)")

    # ── Quick comparison: EWMA vs GARCH vs recent realized ───────────────
    recent_rv = rv.dropna().iloc[-1]
    recent_ewma = ewma_series.iloc[-1]
    print(f"\nMost-recent comparison:")
    print(f"  Realized vol (last valid {HORIZON}-day window): {recent_rv:.4f}  ({recent_rv*100:.2f}%)")
    print(f"  EWMA vol (most recent):                        {recent_ewma:.4f}  ({recent_ewma*100:.2f}%)")
    print(f"  GARCH forecast ({HORIZON}-day ahead):           {fc:.4f}  ({fc*100:.2f}%)")


if __name__ == "__main__":
    main()
