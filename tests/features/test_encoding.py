from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mpp.features.encoding import build_feature_pipeline


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tyre_age": [1.0, 5.0, 10.0],
            "compound": ["SOFT", "MEDIUM", "SOFT"],
            "team": ["Ferrari", "Mercedes", "Ferrari"],
        }
    )


def test_nominal_columns_are_one_hot_not_ordinal():
    pipeline = build_feature_pipeline(numeric_columns=["tyre_age"], nominal_columns=["compound"])
    transformed = pipeline.fit_transform(_df())
    # tyre_age passthrough (1 col) + 2 compound categories one-hot = 3 columns.
    assert transformed.shape == (3, 3)
    # Every one-hot block sums to exactly 1 per row - a real one-hot encoding,
    # not a single ordinal integer column standing in for the category.
    one_hot_block = transformed[:, 1:]
    assert np.allclose(one_hot_block.sum(axis=1), 1.0)


def test_numeric_columns_pass_through_unscaled_by_default():
    pipeline = build_feature_pipeline(numeric_columns=["tyre_age"], nominal_columns=["compound"])
    transformed = pipeline.fit_transform(_df())
    assert list(transformed[:, 0]) == [1.0, 5.0, 10.0]


def test_scale_numeric_standardizes_numeric_columns():
    pipeline = build_feature_pipeline(
        numeric_columns=["tyre_age"], nominal_columns=["compound"], scale_numeric=True
    )
    transformed = pipeline.fit_transform(_df())
    assert transformed[:, 0].mean() == pytest.approx(0.0, abs=1e-9)


def test_unseen_category_at_transform_time_does_not_raise():
    pipeline = build_feature_pipeline(numeric_columns=["tyre_age"], nominal_columns=["compound"])
    pipeline.fit(_df())
    unseen = pd.DataFrame({"tyre_age": [3.0], "compound": ["INTERMEDIATE"]})
    transformed = pipeline.transform(unseen)
    # handle_unknown="ignore": an unseen category becomes an all-zero one-hot
    # row rather than raising - required because a grouped split can put a
    # category in test that the encoder never saw during fit.
    assert transformed[0, 1:].sum() == 0.0


def test_columns_not_listed_are_dropped_not_silently_passed_through():
    pipeline = build_feature_pipeline(numeric_columns=["tyre_age"], nominal_columns=[])
    transformed = pipeline.fit_transform(_df())
    assert transformed.shape == (3, 1)
