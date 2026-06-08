from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volforecast.forecast.features import build_features
from volforecast.forecast.metrics import corr_vol, mae, make_results_table, qlike, rmse
from volforecast.forecast.validation import purged_embargo_splits, run_cv, walk_forward_splits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV with a DatetimeIndex; no network required."""
    rng = np.random.default_rng(seed)
    close_vals = np.cumprod(1.0 + rng.normal(0.0, 0.01, n)) * 100.0
    noise = rng.uniform(0.001, 0.015, n)
    close = pd.Series(close_vals, index=pd.date_range("2018-01-01", periods=n, freq="B"))
    high = close * (1.0 + noise)
    low = close * (1.0 - noise)
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(
        rng.integers(1_000_000, 10_000_000, n).astype(float), index=close.index
    )
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}
    )


# ---------------------------------------------------------------------------
# 1. Zero-lookahead: features at t must not use prices after t
# ---------------------------------------------------------------------------

def test_features_no_lookahead() -> None:
    """
    Corrupt all OHLCV data from day 100 onwards and verify that features at
    days 0–99 are bit-identical to the originals.  Any lookahead in a rolling
    computation would cause features to change here.
    """
    ohlcv = _make_ohlcv(300)
    features_orig, _ = build_features(ohlcv, horizon=5)

    ohlcv_bad = ohlcv.copy()
    # Multiply ALL price/volume columns from index 100 onwards by 1000.
    # A 1000x shock propagates into any window that spans across day 100.
    ohlcv_bad.iloc[100:] = ohlcv_bad.iloc[100:] * 1000.0

    features_bad, _ = build_features(ohlcv_bad, horizon=5)

    pd.testing.assert_frame_equal(
        features_orig.iloc[:100],
        features_bad.iloc[:100],
        check_exact=True,
    )


# ---------------------------------------------------------------------------
# 2. Target is genuinely forward-looking
# ---------------------------------------------------------------------------

def test_target_uses_only_future() -> None:
    """
    Verify that target[t] changes when prices strictly after t are altered.
    horizon=5: target at index 94 uses prices[95..99]; corrupting those changes it.
    target at index 0 uses prices[1..5]; corrupting prices[100+] leaves it intact.
    """
    ohlcv = _make_ohlcv(300)
    _, target_orig = build_features(ohlcv, horizon=5)

    ohlcv_alt = ohlcv.copy()
    idx_from_95 = ohlcv_alt.index[95:]
    ohlcv_alt.loc[idx_from_95, "Close"] = ohlcv_alt.loc[idx_from_95, "Close"] * 5.0

    _, target_alt = build_features(ohlcv_alt, horizon=5)

    # target[94] = std of returns [95..99] — must change
    assert target_orig.iloc[94] != target_alt.iloc[94], (
        "target[94] should change when prices[95:] are corrupted"
    )

    # target[0] uses prices[1..5] — unaffected by corruption at [95:]
    assert target_orig.iloc[0] == pytest.approx(target_alt.iloc[0]), (
        "target[0] should be unchanged when only prices[95:] are corrupted"
    )


# ---------------------------------------------------------------------------
# 3. Walk-forward splits: disjoint and ordered
# ---------------------------------------------------------------------------

def test_walk_forward_disjoint_and_ordered() -> None:
    splits = list(walk_forward_splits(400, min_train=100, step=50))

    assert len(splits) > 0, "Expected at least one fold"

    test_sets: list[set] = []
    for train_idx, test_idx in splits:
        train_set = set(train_idx.tolist())
        test_set = set(test_idx.tolist())

        # train and test must not overlap
        assert train_set.isdisjoint(test_set), "Train and test overlap within a fold"

        # every train index is strictly before every test index
        assert max(train_idx) < min(test_idx), (
            f"Train bleeds into test: max train={max(train_idx)}, min test={min(test_idx)}"
        )

        # test folds must not overlap each other
        for prev in test_sets:
            assert test_set.isdisjoint(prev), "Test sets overlap across folds"

        test_sets.append(test_set)


# ---------------------------------------------------------------------------
# 4. Purged splits: purge window respected
# ---------------------------------------------------------------------------

def test_purged_splits_respect_horizon() -> None:
    """
    For every fold, the (horizon-1) rows immediately before fold_start must
    not appear in train_idx.
    """
    horizon = 5
    splits = list(purged_embargo_splits(400, min_train=100, step=50, horizon=horizon))

    assert len(splits) > 0

    # Reconstruct fold_start values
    fold_start = 100
    for i, (train_idx, test_idx) in enumerate(splits):
        expected_fold_start = test_idx[0]
        purge_lo = max(0, expected_fold_start - (horizon - 1))
        purge_hi = expected_fold_start  # exclusive

        train_set = set(train_idx.tolist())
        forbidden = set(range(purge_lo, purge_hi))

        assert train_set.isdisjoint(forbidden), (
            f"Fold {i}: purged indices {forbidden} found in train_idx"
        )

        # basic ordering still holds
        if len(train_idx) > 0:
            assert max(train_idx) < min(test_idx)


# ---------------------------------------------------------------------------
# 5. QLIKE formula correctness
# ---------------------------------------------------------------------------

def test_qlike_formula() -> None:
    """
    QLIKE at perfect prediction: mean(tv/pv + log(pv)) = mean(1 + log(v²))
    Also verify wrong predictions give strictly higher loss.
    """
    vols = np.array([0.10, 0.20, 0.15, 0.25])
    expected_perfect = float(np.mean(1.0 + np.log(vols ** 2)))
    assert qlike(vols, vols) == pytest.approx(expected_perfect, rel=1e-9)

    # Doubling predictions gives higher QLIKE (worse calibration)
    assert qlike(vols, vols * 2.0) > qlike(vols, vols)
    assert qlike(vols, vols * 0.5) > qlike(vols, vols)


# ---------------------------------------------------------------------------
# 6. Scalar metrics basic properties
# ---------------------------------------------------------------------------

def test_scalar_metrics_basic() -> None:
    y = np.array([0.15, 0.20, 0.18, 0.12])
    assert rmse(y, y) == pytest.approx(0.0)
    assert mae(y, y) == pytest.approx(0.0)
    assert corr_vol(y, y) == pytest.approx(1.0)

    y_pred = y + 0.02
    assert rmse(y, y_pred) == pytest.approx(0.02)
    assert mae(y, y_pred) == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# 7. Integration: run_cv produces a valid results table
# ---------------------------------------------------------------------------

def test_run_cv_produces_table() -> None:
    """
    End-to-end CV on synthetic data: table must have 3 rows (EWMA/GARCH/ML),
    all metric columns, and all finite values.
    Uses a small dataset to keep GARCH fitting fast (~4 folds).
    """
    ohlcv = _make_ohlcv(n=500, seed=7)
    cv_results = run_cv(ohlcv, horizon=5, min_train=200, step=50)

    table = make_results_table(cv_results)

    assert set(table.index) == {"ML", "EWMA", "GARCH"}, (
        f"Expected models ML/EWMA/GARCH, got {list(table.index)}"
    )
    for col in ("RMSE", "MAE", "QLIKE", "Corr"):
        assert col in table.columns
        assert table[col].notna().all(), f"NaN in column {col}"
        assert np.isfinite(table[col].to_numpy()).all(), f"Non-finite value in column {col}"

    # Predictions must be positive (vol is non-negative)
    for model, df in cv_results.items():
        assert (df["y_pred"] > 0).all() or df["y_pred"].isna().any(), (
            f"{model} produced non-positive predictions"
        )
