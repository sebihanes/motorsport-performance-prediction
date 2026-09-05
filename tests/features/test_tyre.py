from __future__ import annotations

import pandas as pd

from mpp.features.tyre import add_tyre_age_squared


def test_tyre_age_squared_is_correct():
    df = pd.DataFrame({"tyre_age": pd.array([1, 4, 10], dtype="Int64")})
    result = add_tyre_age_squared(df)
    assert list(result["tyre_age_sq"]) == [1.0, 16.0, 100.0]


def test_tyre_age_squared_preserves_missing_values():
    df = pd.DataFrame({"tyre_age": pd.array([1, None], dtype="Int64")})
    result = add_tyre_age_squared(df)
    assert result["tyre_age_sq"].iloc[0] == 1.0
    assert pd.isna(result["tyre_age_sq"].iloc[1])
