from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mpp.models.baselines import (
    PerGroupLinearTrendRegressor,
    PerGroupMedianRegressor,
    StintPacePersistenceRegressor,
    mean_baseline,
    median_baseline,
)


def test_mean_baseline_predicts_the_training_mean_for_every_row():
    X = pd.DataFrame({"circuit_id": ["bahrain", "jeddah", "monaco"]})
    y = [90.0, 100.0, 110.0]
    model = mean_baseline().fit(X, y)
    preds = model.predict(pd.DataFrame({"circuit_id": ["baku", "miami"]}))
    assert preds == pytest.approx([100.0, 100.0])


def test_median_baseline_predicts_the_training_median_for_every_row():
    X = pd.DataFrame({"circuit_id": ["bahrain", "jeddah", "monaco", "baku", "miami"]})
    y = [90.0, 100.0, 105.0, 110.0, 1000.0]  # outlier should not move the median
    model = median_baseline().fit(X, y)
    preds = model.predict(pd.DataFrame({"circuit_id": ["monza"]}))
    assert preds == pytest.approx([105.0])


def test_per_group_median_regressor_predicts_the_seen_groups_own_median():
    X = pd.DataFrame({"circuit_id": ["bahrain", "bahrain", "monaco", "monaco"]})
    y = [90.0, 92.0, 70.0, 74.0]
    model = PerGroupMedianRegressor(group_column="circuit_id").fit(X, y)
    preds = model.predict(pd.DataFrame({"circuit_id": ["bahrain", "monaco"]}))
    assert preds == pytest.approx([91.0, 72.0])


def test_per_group_median_regressor_falls_back_to_global_median_for_unseen_group():
    X = pd.DataFrame({"circuit_id": ["bahrain", "bahrain", "monaco", "monaco", "jeddah"]})
    y = [90.0, 92.0, 70.0, 74.0, 100.0]
    model = PerGroupMedianRegressor(group_column="circuit_id").fit(X, y)
    preds = model.predict(pd.DataFrame({"circuit_id": ["baku"]}))  # never seen at fit time
    assert preds == pytest.approx([float(np.median(y))])


def test_per_group_median_regressor_rejects_mismatched_lengths():
    X = pd.DataFrame({"circuit_id": ["bahrain", "monaco"]})
    with pytest.raises(ValueError):
        PerGroupMedianRegressor().fit(X, [1.0])


def test_per_group_median_regressor_rejects_zero_rows():
    X = pd.DataFrame({"circuit_id": []})
    with pytest.raises(ValueError):
        PerGroupMedianRegressor().fit(X, [])


def test_per_group_median_regressor_output_is_a_numpy_array_of_matching_length():
    X = pd.DataFrame({"circuit_id": ["bahrain", "monaco", "baku"]})
    y = [90.0, 70.0, 80.0]
    model = PerGroupMedianRegressor().fit(X, y)
    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == 3


def test_stint_pace_persistence_predicts_stint_pace_and_falls_back_to_training_median():
    X_train = pd.DataFrame({"stint_prior_mean_lap_time_s": [90.0, np.nan, 95.0]})
    model = StintPacePersistenceRegressor().fit(X_train, [91.0, 100.0, 96.0])
    preds = model.predict(pd.DataFrame({"stint_prior_mean_lap_time_s": [88.5, np.nan]}))
    assert preds == pytest.approx([88.5, 96.0])  # fallback = median(91, 100, 96)


def test_stint_pace_persistence_rejects_empty_fit():
    with pytest.raises(ValueError):
        StintPacePersistenceRegressor().fit(pd.DataFrame({"stint_prior_mean_lap_time_s": []}), [])


def test_per_group_linear_trend_fits_one_line_per_group():
    X = pd.DataFrame({"compound": ["SOFT"] * 5 + ["HARD"] * 5, "tyre_age_delta": list(range(5)) * 2})
    y = [0.1 * a for a in range(5)] + [1.0 + 0.02 * a for a in range(5)]
    model = PerGroupLinearTrendRegressor(group_column="compound", x_column="tyre_age_delta").fit(X, y)
    preds = model.predict(pd.DataFrame({"compound": ["SOFT", "HARD"], "tyre_age_delta": [10.0, 10.0]}))
    assert preds == pytest.approx([1.0, 1.2])


def test_per_group_linear_trend_falls_back_for_unseen_small_or_missing():
    X = pd.DataFrame({"compound": ["SOFT"] * 5 + ["WET"] * 2, "tyre_age_delta": [0, 1, 2, 3, 4, 0, 1]})
    y = [0.0, 1.0, 2.0, 3.0, 4.0, 50.0, 60.0]
    model = PerGroupLinearTrendRegressor(group_column="compound", x_column="tyre_age_delta").fit(X, y)
    preds = model.predict(
        pd.DataFrame({"compound": ["WET", "HARD", "SOFT"], "tyre_age_delta": [1.0, 1.0, np.nan]})
    )
    assert preds == pytest.approx([3.0, 3.0, 3.0])  # global median of y
