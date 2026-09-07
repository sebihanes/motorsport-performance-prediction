"""Cross-validated evaluation harness shared by every lap-time candidate
(architecture SS8/SS12): race-grouped CV, never random
(docs/05_split_strategy.md), producing MAE/RMSE/R^2/median-AE per fold plus
their mean/std, and a matching single-fit holdout evaluator for the outer
test set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import pandas as pd

from mpp.evaluation.metrics import aggregate_fold_metrics, regression_metrics
from mpp.splits.group_split import race_group_kfold


@dataclass
class CVResult:
    fold_metrics: list[dict[str, float]]
    aggregated: dict[str, float]


def cross_validate(
    build_estimator: Callable[[], object],
    dataset: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    n_splits: int = 5,
) -> CVResult:
    """Race-grouped cross-validation: for each ``race_group_kfold`` fold of
    ``dataset``, ``build_estimator()`` is called to get a fresh, unfitted
    estimator, fit only on that fold's training rows and scored on that
    fold's test rows. Calling the factory fresh per fold (rather than
    cloning a single fitted instance) is what guarantees no fold's
    preprocessing statistics (e.g. an imputer's median, a scaler's mean) are
    ever computed using another fold's rows.
    """
    X = dataset[list(feature_columns)]
    y = dataset[target_column].to_numpy()
    fold_metrics = []
    for train_idx, test_idx in race_group_kfold(dataset, n_splits=n_splits):
        estimator = build_estimator()
        estimator.fit(X.iloc[train_idx], y[train_idx])
        preds = estimator.predict(X.iloc[test_idx])
        fold_metrics.append(regression_metrics(y[test_idx], preds))
    return CVResult(fold_metrics=fold_metrics, aggregated=aggregate_fold_metrics(fold_metrics))


def holdout_evaluate(
    fitted_estimator, dataset: pd.DataFrame, feature_columns: Sequence[str], target_column: str
) -> dict[str, float]:
    """Score an already-fitted estimator on a held-out set (the outer test
    split) - no fitting happens here, so this is safe to call on the outer
    test set without any leakage risk."""
    X = dataset[list(feature_columns)]
    y = dataset[target_column].to_numpy()
    preds = fitted_estimator.predict(X)
    return regression_metrics(y, preds)
