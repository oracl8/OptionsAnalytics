from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


class VolForecaster:
    """
    Gradient-boosting realized-vol forecaster.

    Wraps HistGradientBoostingRegressor: tree-based (scale-invariant) and
    handles NaN features natively, so no StandardScaler or imputer is required.

    The instance must be fit inside each walk-forward fold — never on the full
    dataset.  A fresh VolForecaster is created per fold in run_cv.
    """

    def __init__(self, **hgbr_kwargs: object) -> None:
        defaults: dict[str, object] = {"random_state": 42, "max_iter": 200}
        defaults.update(hgbr_kwargs)
        self._kwargs = defaults
        self.model_: HistGradientBoostingRegressor | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "VolForecaster":
        self.model_ = HistGradientBoostingRegressor(**self._kwargs)  # type: ignore[arg-type]
        self.model_.fit(X.to_numpy(), y.to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict().")
        return self.model_.predict(X.to_numpy())
