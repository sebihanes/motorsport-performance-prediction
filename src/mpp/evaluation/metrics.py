"""Regression evaluation metrics shared by every lap-time candidate
(architecture report SS12).

Reported together, each with a distinct engineering purpose - not because
more numbers look more rigorous:

- **MAE** - "predictions are off by X seconds on average." The primary
  number to quote to a non-ML audience.
- **RMSE** - penalizes large misses more heavily; a gap between RMSE and MAE
  flags that a few laps (incidents, safety-car restarts) are dominating the
  error rather than the model being uniformly imprecise.
- **R^2** - variance explained, computed on whatever split it is given
  (grouped-CV fold or the outer holdout) - never conflated with an in-sample
  number.
- **Median absolute error** - a robust companion to MAE, unaffected by the
  handful of incident-affected laps any real session contains.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE, RMSE, R^2 and median absolute error for one set of predictions.

    R^2 is undefined for a single sample (no variance to explain against);
    that case returns NaN rather than raising, since a race-grouped fold can
    legitimately be evaluated on very few rows in a narrow scope.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape[0] == 0:
        raise ValueError("Cannot compute regression metrics on zero rows.")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    medae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if y_true.shape[0] > 1 else float("nan")
    return {"mae": float(mae), "rmse": rmse, "r2": float(r2), "median_ae": float(medae)}


def interpret_mae(mae: float, unit: str = "second") -> str:
    """The architecture report's own worked example, generalized: 'An MAE of
    0.18 seconds means the model's typical absolute lap-time error is
    approximately 0.18 seconds.' Deliberately does not claim the model is
    race-ready - that judgement needs error analysis beyond this one number.
    """
    plural = "" if abs(mae - 1.0) < 1e-9 else "s"
    return (
        f"An MAE of {mae:.3f} {unit}{plural} means the model's typical absolute "
        f"lap-time error is approximately {mae:.3f} {unit}{plural}."
    )


def aggregate_fold_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Mean and standard deviation of each metric across CV folds, as
    ``{metric}_mean`` / ``{metric}_std`` - the shape the model comparison
    table (mpp.models.train_lap_time) reports for cross-validated results.
    """
    if not fold_metrics:
        raise ValueError("Cannot aggregate an empty list of fold metrics.")
    keys = fold_metrics[0].keys()
    result: dict[str, float] = {}
    for key in keys:
        values = np.array([m[key] for m in fold_metrics], dtype=float)
        result[f"{key}_mean"] = float(np.mean(values))
        result[f"{key}_std"] = float(np.std(values))
    return result
