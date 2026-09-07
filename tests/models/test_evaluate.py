from __future__ import annotations

import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from mpp.models.evaluate import cross_validate, holdout_evaluate


def _grouped_dataset() -> pd.DataFrame:
    # 4 races, 10 laps each - mirrors tests/splits/test_group_split.py's fixture shape.
    rows = []
    races = [(1, "bahrain"), (2, "jeddah"), (3, "monaco"), (4, "baku")]
    for round_, circuit in races:
        for lap in range(1, 11):
            rows.append(
                {
                    "season": 2023,
                    "round": round_,
                    "circuit_id": circuit,
                    "x": float(lap),
                    "lap_time_s": 90.0 + round_ + 0.1 * lap,
                }
            )
    return pd.DataFrame(rows)


def test_cross_validate_produces_one_fold_metric_set_per_race():
    dataset = _grouped_dataset()
    result = cross_validate(
        lambda: DummyRegressor(strategy="mean"), dataset, ["x"], "lap_time_s", n_splits=4
    )
    assert len(result.fold_metrics) == 4
    for metrics in result.fold_metrics:
        assert set(metrics) == {"mae", "rmse", "r2", "median_ae"}


def test_cross_validate_aggregated_has_mean_and_std_keys():
    dataset = _grouped_dataset()
    result = cross_validate(
        lambda: DummyRegressor(strategy="median"), dataset, ["x"], "lap_time_s", n_splits=4
    )
    for key in ("mae_mean", "mae_std", "rmse_mean", "r2_mean", "median_ae_mean"):
        assert key in result.aggregated


def test_cross_validate_calls_the_factory_fresh_for_every_fold():
    dataset = _grouped_dataset()
    call_count = 0

    def factory():
        nonlocal call_count
        call_count += 1
        return DummyRegressor(strategy="mean")

    cross_validate(factory, dataset, ["x"], "lap_time_s", n_splits=4)
    assert call_count == 4  # one fresh, unfitted estimator per fold - no cross-fold leakage


def test_cross_validate_train_and_test_never_share_a_race():
    # Regression guard: cross_validate must delegate to race_group_kfold, not
    # a random split, for every fold it evaluates.
    dataset = _grouped_dataset()
    seen_folds = []

    class RecordingRegressor(DummyRegressor):
        def fit(self, X, y):
            seen_folds.append(set(dataset.loc[X.index, "round"]))
            return super().fit(X, y)

    cross_validate(RecordingRegressor, dataset, ["x"], "lap_time_s", n_splits=4)
    all_rounds = set(dataset["round"])
    for train_rounds in seen_folds:
        assert len(train_rounds) == 3  # 3 of the 4 races in every leave-one-race-out fold
        assert train_rounds < all_rounds


def test_holdout_evaluate_scores_an_already_fitted_estimator():
    dataset = _grouped_dataset()
    model = DummyRegressor(strategy="mean").fit(dataset[["x"]], dataset["lap_time_s"])
    metrics = holdout_evaluate(model, dataset, ["x"], "lap_time_s")
    assert metrics["mae"] >= 0.0
    assert set(metrics) == {"mae", "rmse", "r2", "median_ae"}
