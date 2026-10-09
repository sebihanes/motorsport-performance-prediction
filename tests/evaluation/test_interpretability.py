from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from mpp.evaluation.interpretability import (
    FoldRoutedPredictor,
    _permutation_index,
    model_feature_importance,
    partial_dependence,
    permutation_importance,
    quantile_grid,
    single_feature_groups,
)
from mpp.models.generic_pipelines import build_pipeline


def _data(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "signal": rng.normal(size=n),
            "noise": rng.normal(size=n),
            "signal_sq": np.zeros(n),
            "cat": rng.choice(["a", "b"], size=n),
        }
    )
    X["signal_sq"] = X["signal"] ** 2
    y = 3.0 * X["signal"] + 0.01 * rng.normal(size=n)
    return X, y.to_numpy()


def _linear_fn(X: pd.DataFrame) -> np.ndarray:
    # a known "model": uses `signal` only
    return 3.0 * X["signal"].to_numpy(dtype=float)


def test_permutation_importance_ranks_used_feature_first_and_unused_near_zero():
    X, y = _data()
    table = permutation_importance(_linear_fn, X, y, single_feature_groups(["signal", "noise"]), n_repeats=3)
    assert table.iloc[0]["feature"] == "signal"
    signal = table.set_index("feature").loc["signal", "importance_mean"]
    noise = table.set_index("feature").loc["noise", "importance_mean"]
    assert signal > 1.0
    assert noise == pytest.approx(0.0, abs=1e-12)
    # baseline MAE is the unpermuted error
    assert table["baseline_mae"].iloc[0] == pytest.approx(np.mean(np.abs(y - _linear_fn(X))))


def test_permutation_importance_is_deterministic_for_a_seed():
    X, y = _data()
    groups = single_feature_groups(["signal", "noise"])
    a = permutation_importance(_linear_fn, X, y, groups, n_repeats=3, random_state=7)
    b = permutation_importance(_linear_fn, X, y, groups, n_repeats=3, random_state=7)
    pd.testing.assert_frame_equal(a, b)


def test_grouped_permutation_moves_columns_together():
    X, y = _data()

    def uses_square_consistency(frame):
        # error grows only when signal and signal_sq disagree
        return 3.0 * frame["signal"].to_numpy() + 10.0 * np.abs(frame["signal_sq"] - frame["signal"] ** 2)

    joint = permutation_importance(uses_square_consistency, X, y, {"g": ["signal", "signal_sq"]}, n_repeats=2)
    split = permutation_importance(uses_square_consistency, X, y, {"g": ["signal"]}, n_repeats=2)
    # permuting jointly keeps signal_sq == signal**2, so no consistency penalty
    assert joint["importance_mean"].iloc[0] < split["importance_mean"].iloc[0]


def test_stratified_permutation_only_shuffles_within_strata():
    rng = np.random.default_rng(0)
    strata = np.array(["r1"] * 5 + ["r2"] * 7)
    index = _permutation_index(len(strata), rng, strata)
    assert sorted(index) == list(range(12))
    assert set(index[:5]) == set(range(5))
    assert set(index[5:]) == set(range(5, 12))


def test_within_race_permutation_of_a_race_constant_feature_has_zero_importance():
    X, y = _data()
    race = np.where(np.arange(len(X)) < 200, "r1", "r2")
    X["race_level"] = np.where(race == "r1", 1.0, 5.0)
    y = y + X["race_level"].to_numpy()

    def fn(frame):
        return 3.0 * frame["signal"].to_numpy() + frame["race_level"].to_numpy()

    across = permutation_importance(fn, X, y, {"race_level": ["race_level"]}, n_repeats=3)
    within = permutation_importance(fn, X, y, {"race_level": ["race_level"]}, n_repeats=3, strata=race)
    assert across["importance_mean"].iloc[0] > 1.0
    assert within["importance_mean"].iloc[0] == pytest.approx(0.0, abs=1e-12)


def test_fold_routed_predictor_uses_each_rows_own_fold_model():
    class Constant:
        def __init__(self, value):
            self.value = value

        def predict(self, X):
            return np.full(len(X), self.value)

    X = pd.DataFrame({"f": range(5)})
    predictor = FoldRoutedPredictor([Constant(1.0), Constant(2.0)], np.array([0, 1, 0, 1, 1]), ["f"])
    np.testing.assert_array_equal(predictor.predict(X), [1.0, 2.0, 1.0, 2.0, 2.0])
    with pytest.raises(ValueError):
        predictor.predict(X.iloc[:3])


def test_model_feature_importance_reads_linear_coefficients_with_names():
    X, y = _data()
    pipe = build_pipeline(LinearRegression(), ["signal", "noise"], ["cat"], scale_numeric=True, impute_numeric=True)
    pipe.fit(X, y)
    table = model_feature_importance(pipe)
    assert table.iloc[0]["feature"] == "numeric__signal"
    assert table.iloc[0]["kind"] == "abs_standardized_coefficient"
    assert (table["importance"] == table["coefficient"].abs()).all()
    # standardized coefficient ~ 3 * std(signal)
    assert table.iloc[0]["coefficient"] == pytest.approx(3.0 * X["signal"].std(ddof=0), rel=0.01)


def test_model_feature_importance_reads_tree_importances():
    X, y = _data()
    pipe = build_pipeline(
        RandomForestRegressor(n_estimators=20, random_state=0), ["signal", "noise"], ["cat"], impute_numeric=True
    )
    pipe.fit(X, y)
    table = model_feature_importance(pipe)
    assert table.iloc[0]["feature"] == "numeric__signal"
    assert table["importance"].sum() == pytest.approx(1.0)


def test_partial_dependence_matches_known_linear_effect_and_applies_derive():
    X, y = _data()
    curve = partial_dependence(_linear_fn, X, "signal", [0.0, 1.0, 2.0])
    np.testing.assert_allclose(curve["mean_prediction"], [0.0, 3.0, 6.0])

    def fn(frame):
        return frame["signal_sq"].to_numpy()

    def derive(frame):
        frame = frame.copy()
        frame["signal_sq"] = frame["signal"] ** 2
        return frame

    derived = partial_dependence(fn, X, "signal", [1.0, 3.0], derive=derive)
    np.testing.assert_allclose(derived["mean_prediction"], [1.0, 9.0])


def test_quantile_grid_stays_inside_observed_range():
    series = pd.Series([1, 2, 3, 4, 100, None])
    grid = quantile_grid(series, n_points=5)
    assert grid.min() >= 1 and grid.max() <= 100
    assert len(grid) == len(np.unique(grid))
    with pytest.raises(ValueError):
        quantile_grid(pd.Series([None, None]))
