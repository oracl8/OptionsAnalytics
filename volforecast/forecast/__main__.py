from __future__ import annotations

import argparse

from volforecast.forecast.metrics import run_and_print


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward realized-vol forecast: ML vs EWMA vs GARCH baselines."
    )
    parser.add_argument("ticker", nargs="?", default="SPY", help="Ticker symbol (default: SPY)")
    parser.add_argument("--horizon", type=int, default=5, help="Forecast horizon in trading days")
    parser.add_argument("--start", default="2019-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2024-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--min-train", type=int, default=252, help="Minimum training set size")
    parser.add_argument("--step", type=int, default=21, help="Fold step size in trading days")
    args = parser.parse_args()

    run_and_print(
        ticker=args.ticker,
        horizon=args.horizon,
        start=args.start,
        end=args.end,
        min_train=args.min_train,
        step=args.step,
    )


if __name__ == "__main__":
    main()
