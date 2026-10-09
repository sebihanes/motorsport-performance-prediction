"""Baseline predictors a learned lap-time model must beat (architecture
report SS10): a weak pair that ignores every feature, plus the strong
baseline the report singles out as the actual bar to clear.

| Baseline | What it predicts | Role |
|---|---|---|
| mean_baseline | training set's global mean lap time, for every row | weak |
| median_baseline | training set's global median lap time, for every row | weak |
| PerGroupMedianRegressor(circuit_id) | training set's median lap time *for that row's circuit* | strong |
| StintPacePersistenceRegressor | the stint's own mean lap time so far (``stint_prior_mean_lap_time_s``) | strong |
| PerGroupLinearTrendRegressor(compound, tyre_age_delta) | per-compound straight-line fit of the target vs. tyre age (tyre degradation's architecture SS10 strong baseline) | strong |

Lap time is dominated by which circuit a lap is on (a street circuit and a
high-speed circuit differ by tens of seconds), so a model that only beats the
global mean/median is not a reportable result - it must also beat the
per-circuit median, which already captures that single largest source of
variance with zero learned parameters.

Pace persistence was added in the final audit (docs/FINAL_AUDIT.md): it is
the first thing a race engineer would predict next lap with ("same pace as
this stint so far"), it uses only information the learned model is also
given, and in the current scope - where no circuit repeats, so the
per-circuit median degenerates to the global median - it is by far the
strongest baseline. Architecture SS10 requires a learned model to beat the
*strongest* reasonable baseline, so it is part of the bar.
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


class StintPacePersistenceRegressor(BaseEstimator, RegressorMixin):
    """Predicts ``X[pace_column]`` - the running mean of the current stint's
    earlier clean laps - unchanged: "the next lap will be as fast as this
    stint has been so far". Where it is missing (the first clean lap of a
    stint) it falls back to the training fold's global median. No parameter
    is learned from the target except that fallback.
    """

    def __init__(self, pace_column: str = "stint_prior_mean_lap_time_s"):
        self.pace_column = pace_column

    def fit(self, X: pd.DataFrame, y) -> "StintPacePersistenceRegressor":
        y = np.asarray(y, dtype=float)
        if len(y) == 0:
            raise ValueError("Cannot fit StintPacePersistenceRegressor on zero rows.")
        self.global_median_ = float(np.median(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pace = pd.to_numeric(pd.DataFrame(X)[self.pace_column], errors="coerce").astype(float)
        return pace.fillna(self.global_median_).to_numpy(dtype=float)


class PerGroupLinearTrendRegressor(BaseEstimator, RegressorMixin):
    """Per-group ordinary least-squares line of y on one numeric column
    (architecture SS10's tyre-degradation strong baseline: "per-compound
    linear fit of delta vs. tyre age"). A group with fewer than
    ``min_rows`` training rows, a group unseen at fit time, or a row whose
    x is missing falls back to the training fold's global median.
    """

    def __init__(self, group_column: str = "compound", x_column: str = "tyre_age_delta", min_rows: int = 5):
        self.group_column = group_column
        self.x_column = x_column
        self.min_rows = min_rows

    def fit(self, X: pd.DataFrame, y) -> "PerGroupLinearTrendRegressor":
        X = pd.DataFrame(X).reset_index(drop=True)
        y = np.asarray(y, dtype=float)
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        if len(y) == 0:
            raise ValueError("Cannot fit PerGroupLinearTrendRegressor on zero rows.")
        frame = pd.DataFrame(
            {
                "g": X[self.group_column].astype(str).to_numpy(),
                "x": pd.to_numeric(X[self.x_column], errors="coerce").astype(float).to_numpy(),
                "y": y,
            }
        ).dropna(subset=["x"])
        self.global_median_ = float(np.median(y))
        self.lines_ = {}
        for group, rows in frame.groupby("g"):
            if len(rows) < self.min_rows or rows["x"].nunique() < 2:
                continue
            slope, intercept = np.polyfit(rows["x"].to_numpy(), rows["y"].to_numpy(), deg=1)
            self.lines_[group] = (float(slope), float(intercept))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X)
        groups = X[self.group_column].astype(str).to_numpy()
        xs = pd.to_numeric(X[self.x_column], errors="coerce").astype(float).to_numpy()
        out = np.full(len(X), self.global_median_, dtype=float)
        for i, (group, x) in enumerate(zip(groups, xs)):
            line = self.lines_.get(group)
            if line is not None and not np.isnan(x):
                out[i] = line[0] * x + line[1]
        return out
