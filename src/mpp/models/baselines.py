"""Baseline predictors a learned lap-time model must beat (architecture
report SS10): a weak pair that ignores every feature, plus the strong
baseline the report singles out as the actual bar to clear.

| Baseline | What it predicts | Role |
|---|---|---|
| mean_baseline | training set's global mean lap time, for every row | weak |
| median_baseline | training set's global median lap time, for every row | weak |
| PerGroupMedianRegressor(circuit_id) | training set's median lap time *for that row's circuit* | strong |

Lap time is dominated by which circuit a lap is on (a street circuit and a
high-speed circuit differ by tens of seconds), so a model that only beats the
global mean/median is not a reportable result - it must also beat the
per-circuit median, which already captures that single largest source of
variance with zero learned parameters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor


def mean_baseline() -> DummyRegressor:
    return DummyRegressor(strategy="mean")


def median_baseline() -> DummyRegressor:
    return DummyRegressor(strategy="median")


class PerGroupMedianRegressor(BaseEstimator, RegressorMixin):
    """Predicts the training-fold median of y within X[group_column].

    Falls back to the training fold's global median for a group unseen at
    fit time - required because the grouped/circuit-holdout split
    (docs/05_split_strategy.md) can legitimately put a circuit in test that
    never appeared in train.
    """

    def __init__(self, group_column: str = "circuit_id"):
        self.group_column = group_column

    def fit(self, X: pd.DataFrame, y) -> "PerGroupMedianRegressor":
        X = pd.DataFrame(X).reset_index(drop=True)
        y = np.asarray(y, dtype=float)
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        if len(y) == 0:
            raise ValueError("Cannot fit PerGroupMedianRegressor on zero rows.")
        frame = pd.DataFrame({self.group_column: X[self.group_column].to_numpy(), "_y": y})
        self.group_medians_ = frame.groupby(self.group_column)["_y"].median().to_dict()
        self.global_median_ = float(np.median(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X)
        groups = X[self.group_column].to_numpy()
        return np.array(
            [self.group_medians_.get(group, self.global_median_) for group in groups], dtype=float
        )
