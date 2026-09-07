from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mpp.datasets.lap_time import NUMERIC_FEATURE_COLUMNS
from mpp.models.regressors import (
    MODEL_FEATURE_COLUMNS,
    MODEL_NOMINAL_COLUMNS,
    MODEL_NUMERIC_COLUMNS,
    gradient_boosting_pipeline,
    hist_gradient_boosting_pipeline,
    linear_regression_pipeline,
    random_forest_pipeline,
    ridge_pipeline,
)


def _tiny_dataset(n: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {col: rng.normal(size=n) for col in MODEL_NUMERIC_COLUMNS}
    data["compound"] = ["SOFT", "MEDIUM"] * (n // 2)
    data["session_type"] = ["R"] * n
    frame = pd.DataFrame(data)
    # Introduce real missing values, like driver_prior_relative_pace_s at round 1.
    frame.loc[0, "driver_prior_relative_pace_s"] = np.nan
    frame.loc[1, "stint_prior_mean_lap_time_s"] = np.nan
    target = pd.Series(rng.normal(loc=90.0, scale=1.0, size=n))
    return frame, target


def test_degenerate_all_null_column_is_excluded_from_model_features():
    # driver_circuit_prior_relative_pace_s is 100% null in the current
    # 6-round scope (docs/03_dataset_definitions.md) - a regression test
    # guarding against silently reintroducing an uninformative, unimputable
    # column into the model feature set.
    assert "driver_circuit_prior_relative_pace_s" in NUMERIC_FEATURE_COLUMNS
    assert "driver_circuit_prior_relative_pace_s" not in MODEL_NUMERIC_COLUMNS
    assert "driver_circuit_prior_relative_pace_s" not in MODEL_FEATURE_COLUMNS


@pytest.mark.parametrize(
    "build_pipeline",
    [linear_regression_pipeline, ridge_pipeline, random_forest_pipeline, gradient_boosting_pipeline],
)
def test_imputed_pipelines_fit_and_predict_despite_missing_numeric_values(build_pipeline):
    X, y = _tiny_dataset()
    pipeline = build_pipeline()
    pipeline.fit(X[MODEL_FEATURE_COLUMNS], y)
    preds = pipeline.predict(X[MODEL_FEATURE_COLUMNS])
    assert len(preds) == len(y)
    assert np.isfinite(preds).all()


def test_hist_gradient_boosting_fits_natively_without_imputation():
    X, y = _tiny_dataset()
    pipeline = hist_gradient_boosting_pipeline()
    pipeline.fit(X[MODEL_FEATURE_COLUMNS], y)
    preds = pipeline.predict(X[MODEL_FEATURE_COLUMNS])
    assert len(preds) == len(y)
    assert np.isfinite(preds).all()
    # No imputer step in this pipeline - NaN really does reach the estimator.
    preprocessor = pipeline.named_steps["preprocess"]
    numeric_transform = next(t for name, t, _ in preprocessor.transformers if name == "numeric")
    assert numeric_transform == "passthrough"


def test_linear_and_ridge_pipelines_scale_numeric_features():
    X, y = _tiny_dataset()
    for build_pipeline in (linear_regression_pipeline, ridge_pipeline):
        pipeline = build_pipeline()
        transformed = pipeline.named_steps["preprocess"].fit_transform(X[MODEL_FEATURE_COLUMNS], y)
        numeric_block = transformed[:, : len(MODEL_NUMERIC_COLUMNS)]
        assert numeric_block.mean(axis=0) == pytest.approx(0.0, abs=1e-8)


def test_unseen_category_at_predict_time_does_not_raise():
    X, y = _tiny_dataset()
    pipeline = random_forest_pipeline(n_estimators=10)
    pipeline.fit(X[MODEL_FEATURE_COLUMNS], y)
    unseen = X[MODEL_FEATURE_COLUMNS].iloc[[0]].copy()
    unseen["compound"] = "WET"  # never seen at fit time in this tiny dataset
    preds = pipeline.predict(unseen)
    assert len(preds) == 1
    assert np.isfinite(preds).all()


def test_model_feature_columns_is_numeric_plus_nominal():
    assert MODEL_FEATURE_COLUMNS == MODEL_NUMERIC_COLUMNS + MODEL_NOMINAL_COLUMNS


def test_pipeline_kwargs_reach_the_underlying_estimator():
    pipeline = random_forest_pipeline(n_estimators=17, max_depth=3)
    model = pipeline.named_steps["model"]
    assert model.n_estimators == 17
    assert model.max_depth == 3
