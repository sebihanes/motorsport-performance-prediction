from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mpp.models.baselines import PerGroupMedianRegressor, mean_baseline, median_baseline


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
