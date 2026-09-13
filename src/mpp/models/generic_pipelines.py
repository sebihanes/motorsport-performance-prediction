"""Generic scikit-learn Pipeline builder shared across every Phase 5 training
script (tyre degradation, race pace, pit stop) - the same
``preprocess -> estimator`` shape ``mpp.models.regressors`` established for
lap time, generalized over a dataset's own feature columns instead of being
hard-wired to lap_time's. Kept separate from ``regressors.py`` so that
module (and every test pinned to it) is untouched.
"""

from __future__ import annotations

from sklearn.pipeline import Pipeline

from mpp.features.encoding import build_feature_pipeline


def build_pipeline(
    estimator,
    numeric_columns: list[str],
    nominal_columns: list[str],
    *,
    scale_numeric: bool = False,
    impute_numeric: bool = False,
) -> Pipeline:
    preprocessor = build_feature_pipeline(
        numeric_columns, nominal_columns, scale_numeric=scale_numeric, impute_numeric=impute_numeric
    )
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])
