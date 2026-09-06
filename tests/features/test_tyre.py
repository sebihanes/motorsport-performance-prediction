from __future__ import annotations

import pandas as pd

from mpp.features.tyre import add_compound_tyre_age_interaction, add_tyre_age_squared

COMPOUNDS = ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN")


def test_tyre_age_squared_is_correct():
    df = pd.DataFrame({"tyre_age": pd.array([1, 4, 10], dtype="Int64")})
    result = add_tyre_age_squared(df)
    assert list(result["tyre_age_sq"]) == [1.0, 16.0, 100.0]


def test_tyre_age_squared_preserves_missing_values():
    df = pd.DataFrame({"tyre_age": pd.array([1, None], dtype="Int64")})
    result = add_tyre_age_squared(df)
    assert result["tyre_age_sq"].iloc[0] == 1.0
    assert pd.isna(result["tyre_age_sq"].iloc[1])


def test_interaction_column_carries_tyre_age_only_for_its_own_compound():
    df = pd.DataFrame(
        {
            "compound": ["SOFT", "MEDIUM", "HARD"],
            "tyre_age": pd.array([4, 10, 20], dtype="Int64"),
        }
    )
    result = add_compound_tyre_age_interaction(df, COMPOUNDS)
    assert list(result["tyre_age_x_soft"]) == [4.0, 0.0, 0.0]
    assert list(result["tyre_age_x_medium"]) == [0.0, 10.0, 0.0]
    assert list(result["tyre_age_x_hard"]) == [0.0, 0.0, 20.0]


def test_interaction_skips_unknown_compound_column():
    df = pd.DataFrame({"compound": ["UNKNOWN"], "tyre_age": pd.array([5], dtype="Int64")})
    result = add_compound_tyre_age_interaction(df, COMPOUNDS)
    assert "tyre_age_x_unknown" not in result.columns
    # UNKNOWN gets no credit in any real compound's interaction column either.
    for compound in ("soft", "medium", "hard", "intermediate", "wet"):
        assert result[f"tyre_age_x_{compound}"].iloc[0] == 0.0


def test_interaction_preserves_missing_tyre_age():
    df = pd.DataFrame({"compound": ["SOFT"], "tyre_age": pd.array([None], dtype="Int64")})
    result = add_compound_tyre_age_interaction(df, COMPOUNDS)
    assert pd.isna(result["tyre_age_x_soft"].iloc[0])
