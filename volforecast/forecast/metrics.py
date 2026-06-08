from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    QLIKE loss (Patton 2011, Eq. 2): mean(true_var / pred_var + log(pred_var))
    where *_var = vol².

    Derived from the Gaussian quasi-log-likelihood; smaller is better.
    Predictions are clamped to ≥ 1e-16 to avoid division by zero and log(0).
    """
    tv = np.asarray(y_true, dtype=float) ** 2
    pv = np.clip(np.asarray(y_pred, dtype=float) ** 2, 1e-16, None)
    return float(np.mean(tv / pv + np.log(pv)))


def corr_vol(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def make_results_table(cv_results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate all fold predictions and compute metrics for each model.

    Parameters
    ----------
    cv_results : dict
        {model_name: DataFrame with columns [date, y_true, y_pred]}

    Returns
    -------
    pd.DataFrame
        Indexed by model name, columns [RMSE, MAE, QLIKE, Corr, N].
        Sorted ascending by RMSE so the best model appears first.
    """
    rows = []
    for model, df in cv_results.items():
        yt = df["y_true"].to_numpy(dtype=float)
        yp = df["y_pred"].to_numpy(dtype=float)
        mask = np.isfinite(yt) & np.isfinite(yp)
        yt, yp = yt[mask], yp[mask]
        rows.append({
            "Model": model,
            "RMSE": rmse(yt, yp),
            "MAE": mae(yt, yp),
            "QLIKE": qlike(yt, yp),
            "Corr": corr_vol(yt, yp),
            "N": int(mask.sum()),
        })
    return pd.DataFrame(rows).set_index("Model").sort_values("RMSE")


def run_and_print(
    ticker: str = "SPY",
    horizon: int = 5,
    start: str = "2019-01-01",
    end: str = "2024-12-31",
    min_train: int = 252,
    step: int = 21,
) -> pd.DataFrame:
    """
    Load OHLCV, run walk-forward CV for ML/EWMA/GARCH, print and return results.

    Uses YFinanceSource with on-disk caching; no network required after the
    first run.
    """
    from volforecast.data.loader import YFinanceSource
    from volforecast.forecast.validation import run_cv

    src = YFinanceSource()
    ohlcv = src.get_ohlcv(ticker, start, end)
    if ohlcv.empty:
        raise ValueError(f"No data returned for {ticker} [{start}, {end}]")

    n_obs = len(ohlcv)
    print(
        f"\nWalk-forward CV | {ticker}  horizon={horizon}d  "
        f"obs={n_obs}  train>={min_train}d  step={step}d\n"
    )

    cv_results = run_cv(ohlcv, horizon=horizon, min_train=min_train, step=step)
    table = make_results_table(cv_results)

    print(table.to_string(float_format="{:.6f}".format))
    print()
    return table
