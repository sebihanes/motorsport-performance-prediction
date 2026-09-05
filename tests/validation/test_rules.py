from __future__ import annotations

import pandas as pd

from mpp.validation.rules import (
    duplicate_rows,
    non_negative,
    not_null,
    numeric_bounds,
    positive_integer,
    required_columns_present,
    value_in_set,
)


def test_required_columns_present_ok():
    df = pd.DataFrame({"a": [1], "b": [2]})
    result = required_columns_present(df, ["a", "b"], "t")
    assert result.status == "ok"


def test_required_columns_present_missing():
    df = pd.DataFrame({"a": [1]})
    result = required_columns_present(df, ["a", "b"], "t")
    assert result.status == "error"
    assert "b" in result.details["missing_columns"]


def test_not_null_detects_nulls():
    df = pd.DataFrame({"driver": ["VER", None, "LEC"]})
    result = not_null(df, "driver", "t", severity="error")
    assert result.status == "error"
    assert result.details["n_null"] == 1


def test_not_null_passes_when_no_nulls():
    df = pd.DataFrame({"driver": ["VER", "LEC"]})
    result = not_null(df, "driver", "t")
    assert result.status == "ok"


def test_duplicate_rows_detects_duplicate_key():
    df = pd.DataFrame(
        {"season": [2023, 2023, 2023], "round": [1, 1, 1], "driver": ["VER", "VER", "LEC"]}
    )
    result = duplicate_rows(df, ["season", "round", "driver"], "t")
    assert result.status == "error"
    assert result.details["n_duplicate_rows"] == 2


def test_duplicate_rows_ok_when_unique():
    df = pd.DataFrame({"season": [2023, 2023], "round": [1, 1], "driver": ["VER", "LEC"]})
    result = duplicate_rows(df, ["season", "round", "driver"], "t")
    assert result.status == "ok"


def test_positive_integer_flags_zero_and_negative():
    df = pd.DataFrame({"lap_number": [1, 0, -1, 5]})
    result = positive_integer(df, "lap_number", "t")
    assert result.status == "error"
    assert result.details["n_invalid"] == 2


def test_non_negative_flags_negative_values():
    df = pd.DataFrame({"tyre_age": [0, 3, -2]})
    result = non_negative(df, "tyre_age", "t")
    assert result.status == "warning"
    assert result.details["n_negative"] == 1


def test_value_in_set_flags_unknown_values():
    df = pd.DataFrame({"compound": ["SOFT", "medium", "C1", None]})
    result = value_in_set(df, "compound", {"SOFT", "MEDIUM", "HARD"}, "t")
    assert result.status == "warning"
    assert result.details["unknown_values"] == ["C1"]
    assert result.details["n_null"] == 1


def test_value_in_set_ok_when_all_recognized():
    df = pd.DataFrame({"compound": ["SOFT", "medium", None]})
    result = value_in_set(df, "compound", {"SOFT", "MEDIUM", "HARD"}, "t")
    assert result.status == "ok"


def test_numeric_bounds_flags_out_of_range_timedelta():
    df = pd.DataFrame({"LapTime": pd.to_timedelta(["0 days 00:01:30", "0 days 00:10:00"])})
    result = numeric_bounds(df, "LapTime", 55.0, 160.0, "t", as_timedelta_seconds=True)
    assert result.status == "warning"
    assert result.details["n_out_of_bounds"] == 1


def test_numeric_bounds_ok_when_within_range():
    df = pd.DataFrame({"LapTime": pd.to_timedelta(["0 days 00:01:30", "0 days 00:01:40"])})
    result = numeric_bounds(df, "LapTime", 55.0, 160.0, "t", as_timedelta_seconds=True)
    assert result.status == "ok"
