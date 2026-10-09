"""Model interpretability utilities (Phase 6): permutation importance,
model-native feature importance, and partial dependence.

Everything here measures **predictive reliance**, not causation: "how much
worse does this model's held-out error get when this feature's information
is destroyed", or "how does this model's prediction move when this input
is varied". A feature can score high because it is a *proxy* for something
the model cannot see (e.g. weather standing in for circuit identity when
every race is at a different circuit), and correlated features share or
split credit unpredictably. See docs/PHASE_6_REPORT.md for how each result
is interpreted in this project.

Design choices, each for a stated reason:

- **Permutation importance is computed on out-of-fold predictions** - each
  row is predicted by the leave-one-race-out model that never saw its race
  (``FoldRoutedPredictor``). Importance on training rows would measure what
  the model memorised, not what it relies on to generalize.
- **Feature groups are permuted jointly** - ``tyre_age``, ``tyre_age_sq``
  and the ``tyre_age_x_<compound>`` interactions are deterministic
  functions of each other. Permuting one while its siblings stay intact
  leaves the information recoverable and under-states the group's
  importance; permuting them together with the same row index does not.
- **Optional stratified permutation** (``strata=``) - shuffling only
  *within* each race asks a different question from shuffling across all
  races: a feature that is near-constant within a race (weather) can only
  matter across races, i.e. as a race-identity signal. Comparing the two
  separates "explains lap-to-lap variation" from "identifies the race".
- **Partial dependence accepts a ``derive`` hook** so that when tyre age is
  varied, its derived columns are recomputed consistently instead of being
  held at impossible combinations (``tyre_age=30`` with ``tyre_age_sq=4``).
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

PredictFn = Callable[[pd.DataFrame], np.ndarray]


def mean_absolute_error(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


class FoldRoutedPredictor:
    """Predicts each row with the model of the outer fold that held that row
    out. ``fold_of_row[i]`` is the index into ``models`` for positional row
    ``i`` of the frame passed to ``predict`` - the frame must keep the same
    row order as the one the fold assignment was computed on."""

    def __init__(self, models: Sequence[object], fold_of_row: np.ndarray, feature_columns: Sequence[str]):
        self.models = list(models)
        self.fold_of_row = np.asarray(fold_of_row)
        self.feature_columns = list(feature_columns)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if len(X) != len(self.fold_of_row):
            raise ValueError("X must have exactly the rows the fold assignment was built for.")
        preds = np.full(len(X), np.nan)
        features = X[self.feature_columns]
        for fold, model in enumerate(self.models):
            rows = np.flatnonzero(self.fold_of_row == fold)
            if rows.size:
                preds[rows] = model.predict(features.iloc[rows])
        return preds

    __call__ = predict


def single_feature_groups(columns: Sequence[str]) -> dict[str, list[str]]:
    return {c: [c] for c in columns}


def _permutation_index(n: int, rng: np.random.Generator, strata) -> np.ndarray:
    if strata is None:
        return rng.permutation(n)
    strata = np.asarray(strata)
    index = np.arange(n)
    for value in pd.unique(strata):
        positions = np.flatnonzero(strata == value)
        index[positions] = rng.permutation(positions)
    return index


def permutation_importance(
    predict_fn: PredictFn,
    X: pd.DataFrame,
    y,
    feature_groups: Mapping[str, Sequence[str]],
    n_repeats: int = 5,
    random_state: int = 0,
    strata=None,
) -> pd.DataFrame:
    """Increase in MAE when each feature group's columns are jointly shuffled
    (same row permutation for every column in the group), averaged over
    ``n_repeats``. ``strata`` restricts shuffling to within each stratum.

    Returns columns ``feature, importance_mean, importance_std,
    baseline_mae, n_repeats`` sorted by importance, largest first. An
    importance near zero (or slightly negative, from shuffle noise) means
    the model does not rely on that group for held-out accuracy.
    """
    X = X.reset_index(drop=True)
    y = np.asarray(y, dtype=float)
    baseline = mean_absolute_error(y, predict_fn(X))
    rng = np.random.default_rng(random_state)
    rows = []
    for name, columns in feature_groups.items():
        deltas = []
        for _ in range(n_repeats):
            index = _permutation_index(len(X), rng, strata)
            permuted = X.copy()
            for col in columns:
                permuted[col] = X[col].iloc[index].set_axis(X.index)
            deltas.append(mean_absolute_error(y, predict_fn(permuted)) - baseline)
        rows.append(
            {
                "feature": name,
                "importance_mean": float(np.mean(deltas)),
                "importance_std": float(np.std(deltas)),
                "baseline_mae": baseline,
                "n_repeats": n_repeats,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def model_feature_importance(pipeline) -> pd.DataFrame:
    """Model-native importance for a fitted ``Pipeline([("preprocess", ...),
    ("model", ...)])``, on the *transformed* (post-encoding) feature names:

    - linear models -> ``coefficient`` (seconds per one standard deviation
      for standardized numerics; seconds of offset for a one-hot level) and
      its absolute value as ``importance``;
    - tree ensembles -> impurity-based ``feature_importances_`` (biased
      toward high-cardinality / continuous features, and computed on
      training data - read alongside permutation importance, never alone).
    """
    names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
    model = pipeline.named_steps["model"]
    if hasattr(model, "coef_"):
        coef = np.ravel(model.coef_)
        out = pd.DataFrame({"feature": names, "coefficient": coef, "importance": np.abs(coef)})
        out["kind"] = "abs_standardized_coefficient"
    elif hasattr(model, "feature_importances_"):
        out = pd.DataFrame({"feature": names, "importance": model.feature_importances_})
        out["kind"] = "impurity_importance"
    else:
        raise TypeError(f"{type(model).__name__} exposes neither coef_ nor feature_importances_.")
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def quantile_grid(series: pd.Series, n_points: int = 20, lower: float = 0.05, upper: float = 0.95) -> np.ndarray:
    """Evenly spaced quantiles of the observed values, deduplicated - a grid
    that stays inside the data's support instead of extrapolating into
    regions with no training rows."""
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError("Cannot build a grid from an all-null series.")
    return np.unique(np.quantile(values, np.linspace(lower, upper, n_points)))


def partial_dependence(
    predict_fn: PredictFn,
    X: pd.DataFrame,
    feature: str,
    grid: Sequence[float],
    derive: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Average prediction over every row of `X` with `feature` set to each
    grid value (``derive`` then recomputes any columns that are functions of
    it). Returns columns ``value, mean_prediction``.

    Partial dependence assumes the varied feature can change independently
    of the others; for correlated inputs it averages over combinations that
    may never occur on track. Interpret the *shape*, not absolute levels.
    """
    rows = []
    for value in grid:
        varied = X.copy()
        varied[feature] = value
        if derive is not None:
            varied = derive(varied)
        rows.append({"value": float(value), "mean_prediction": float(np.mean(predict_fn(varied)))})
    return pd.DataFrame(rows)
