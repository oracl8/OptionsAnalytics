from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from volforecast.vol.garch import ewma_vol, garch_forecast
from volforecast.forecast.features import build_features
from volforecast.forecast.models import VolForecaster


def walk_forward_splits(
    n: int,
    min_train: int = 252,
    step: int = 21,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window walk-forward CV splits.

    Yields (train_idx, test_idx) as integer arrays.  The training set always
    starts at 0 and grows with each fold; the test window advances by `step`.

    Parameters
    ----------
    n : int
        Total number of observations.
    min_train : int
        Minimum training set size before the first test fold begins.
    step : int
        Number of test observations per fold.
    """
    fold_start = min_train
    while fold_start < n:
        fold_end = min(fold_start + step, n)
        yield np.arange(fold_start), np.arange(fold_start, fold_end)
        fold_start = fold_end


def purged_embargo_splits(
    n: int,
    min_train: int = 252,
    step: int = 21,
    horizon: int = 5,
    embargo: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window splits with purging of target-overlapping train samples.

    For a horizon-h target, the sample at t-k (k < h) has a forward window
    that overlaps with the test set starting at t.  These samples are removed
    from training to prevent label leakage through correlated targets.

    Purge range: indices [fold_start - (horizon-1),  fold_start - 1] inclusive.

    An optional `embargo` of additional rows after the test end can be applied,
    though in an expanding-window scheme this has no practical effect because the
    embargoed indices never fall inside a previous fold's test set.

    Parameters
    ----------
    n, min_train, step : same as walk_forward_splits
    horizon : int
        Forecast horizon; determines how many end-of-train rows are purged.
    embargo : int
        Extra rows to drop after test end (default 0).
    """
    fold_start = min_train
    while fold_start < n:
        fold_end = min(fold_start + step, n)

        all_train = np.arange(fold_start)

        # Purge the (horizon-1) samples whose targets overlap with [fold_start, fold_end)
        purge_lo = max(0, fold_start - (horizon - 1))
        purge_hi = fold_start  # exclusive upper bound
        keep = (all_train < purge_lo) | (all_train >= purge_hi)
        train_idx = all_train[keep]

        # Embargo: drop rows immediately after test end (optional, affects next fold)
        if embargo > 0:
            emb_end = min(fold_end + embargo, n)
            train_idx = train_idx[(train_idx < fold_end) | (train_idx >= emb_end)]

        yield train_idx, np.arange(fold_start, fold_end)
        fold_start = fold_end


def run_cv(
    ohlcv: pd.DataFrame,
    horizon: int,
    min_train: int = 252,
    step: int = 21,
) -> dict[str, pd.DataFrame]:
    """
    Walk-forward CV producing out-of-sample predictions for ML, EWMA, and GARCH
    on identical purged folds.

    All fitting (ML model, GARCH parameters) is done strictly on train data
    inside each fold.  EWMA is a causal infinite-memory filter that requires no
    fitting.

    Returns
    -------
    dict mapping model name → DataFrame with columns [date, y_true, y_pred].
    Rows correspond to individual test-set observations across all folds.
    """
    features, target = build_features(ohlcv, horizon)
    close = ohlcv["Close"]

    # returns aligned to close.index[1:] (first diff is NaN-dropped)
    log_ret_full = np.log(close).diff().dropna()

    # EWMA: causal filter — value at date d uses returns up to and including d
    ewma_arr = ewma_vol(log_ret_full.values)
    ewma_series = pd.Series(ewma_arr, index=log_ret_full.index)

    n = len(features)
    splits = list(purged_embargo_splits(n, min_train=min_train, step=step, horizon=horizon))

    records: dict[str, list[dict]] = {"ML": [], "EWMA": [], "GARCH": []}

    for train_idx, test_idx in splits:
        # --- filter NaN rows from training ---
        feat_train = features.iloc[train_idx]
        tgt_train = target.iloc[train_idx]
        train_valid = ~feat_train.isna().any(axis=1) & ~tgt_train.isna()
        train_valid_idx = train_idx[train_valid.to_numpy()]

        # --- filter NaN target rows from test ---
        tgt_test = target.iloc[test_idx]
        test_valid = ~tgt_test.isna()
        test_valid_idx = test_idx[test_valid.to_numpy()]

        if len(train_valid_idx) < 50 or len(test_valid_idx) == 0:
            continue

        X_train = features.iloc[train_valid_idx]
        y_train = target.iloc[train_valid_idx]
        X_test = features.iloc[test_valid_idx]
        y_test = target.iloc[test_valid_idx]
        test_dates = ohlcv.index[test_valid_idx]

        # --- ML ---
        forecaster = VolForecaster()
        forecaster.fit(X_train, y_train)
        ml_pred = forecaster.predict(X_test)
        for date, yt, yp in zip(test_dates, y_test.to_numpy(), ml_pred):
            records["ML"].append({"date": date, "y_true": float(yt), "y_pred": float(yp)})

        # --- EWMA: look up causal value at each test date ---
        ewma_pred = ewma_series.reindex(test_dates)
        for date, yt, ep in zip(test_dates, y_test.to_numpy(), ewma_pred.to_numpy()):
            records["EWMA"].append({"date": date, "y_true": float(yt), "y_pred": float(ep)})

        # --- GARCH: refit on train returns, single h-step forecast for the fold ---
        # Using returns up to the last training date keeps the fit strictly in-sample.
        train_end_date = ohlcv.index[train_valid_idx[-1]]
        train_returns = log_ret_full.loc[:train_end_date].to_numpy()
        try:
            g_pred = float(garch_forecast(train_returns, horizon))
        except Exception:
            g_pred = float("nan")
        for date, yt in zip(test_dates, y_test.to_numpy()):
            records["GARCH"].append({"date": date, "y_true": float(yt), "y_pred": g_pred})

    return {
        model: pd.DataFrame(recs)
        for model, recs in records.items()
        if recs
    }
