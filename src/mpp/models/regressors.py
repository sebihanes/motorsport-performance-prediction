"""Candidate model pipelines for lap-time prediction (architecture report
SS11): each one a scikit-learn ``Pipeline`` wrapping the shared
ColumnTransformer preprocessing (``mpp.features.encoding.build_feature_pipeline``)
around an estimator, so preprocessing is always fit fresh on a given split's
training fold only - never on the full dataset ahead of splitting, which
would leak test-fold information into train (architecture SS9).

Model-specific preprocessing choices, not a one-size-fits-all default:

- **Linear Regression / Ridge** - numeric features imputed (median) then
  standardized. A linear model cannot consume NaN, and the raw feature
  scales differ by orders of magnitude (``tyre_age`` vs. ``air_temp`` vs.
  relative-pace seconds).
- **Random Forest / Gradient Boosting** (the plain scikit-learn
  implementations) - numeric features imputed (median) but left unscaled:
  tree splits are scale-invariant, but these two implementations (unlike
  HistGradientBoostingRegressor) cannot consume NaN directly.
- **HistGradientBoostingRegressor** - neither imputed nor scaled. Native
  missing-value handling is the reason it is in this candidate list at all
  (architecture SS11); imputing ahead of it would discard that advantage.

``driver_circuit_prior_relative_pace_s`` is excluded from every pipeline
here: per ``docs/03_dataset_definitions.md`` it is 100% null in the current
6-round scope (no repeated circuits yet), so it carries zero information and,
worse, has no observed value for a median imputer to compute from. It
becomes usable the moment the season scope widens to include a circuit
repeat (``mpp.features.circuit`` already handles that case); dropping it here
is a training-time decision, not a change to the dataset schema.
"""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline

from mpp.datasets.lap_time import NOMINAL_FEATURE_COLUMNS, NUMERIC_FEATURE_COLUMNS
from mpp.features.encoding import build_feature_pipeline

DEGENERATE_NUMERIC_COLUMNS = ["driver_circuit_prior_relative_pace_s"]
MODEL_NUMERIC_COLUMNS = [c for c in NUMERIC_FEATURE_COLUMNS if c not in DEGENERATE_NUMERIC_COLUMNS]
MODEL_NOMINAL_COLUMNS = list(NOMINAL_FEATURE_COLUMNS)
MODEL_FEATURE_COLUMNS = MODEL_NUMERIC_COLUMNS + MODEL_NOMINAL_COLUMNS


def _pipeline(estimator, *, scale_numeric: bool, impute_numeric: bool) -> Pipeline:
    preprocessor = build_feature_pipeline(
        MODEL_NUMERIC_COLUMNS,
        MODEL_NOMINAL_COLUMNS,
        scale_numeric=scale_numeric,
        impute_numeric=impute_numeric,
    )
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def linear_regression_pipeline() -> Pipeline:
    return _pipeline(LinearRegression(), scale_numeric=True, impute_numeric=True)


def ridge_pipeline(**params) -> Pipeline:
    return _pipeline(Ridge(random_state=0, **params), scale_numeric=True, impute_numeric=True)


def random_forest_pipeline(**params) -> Pipeline:
    return _pipeline(
        RandomForestRegressor(random_state=0, n_jobs=-1, **params),
        scale_numeric=False,
        impute_numeric=True,
    )


def gradient_boosting_pipeline(**params) -> Pipeline:
    return _pipeline(
        GradientBoostingRegressor(random_state=0, **params), scale_numeric=False, impute_numeric=True
    )


def hist_gradient_boosting_pipeline(**params) -> Pipeline:
    return _pipeline(
        HistGradientBoostingRegressor(random_state=0, **params),
        scale_numeric=False,
        impute_numeric=False,
    )
