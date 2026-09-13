from __future__ import annotations

import pandas as pd

from mpp.evaluation.error_analysis import bucket_numeric, error_breakdown, error_by_group, largest_absolute_errors


def _frame():
    return pd.DataFrame(
        {
            "driver": ["VER", "VER", "HAM", "HAM"],
            "session_type": ["R", "R", "R", "R"],  # constant - degenerate
            "y_true": [100.0, 102.0, 90.0, 91.0],
            "y_pred": [101.0, 101.0, 95.0, 91.0],  # HAM rows carry the bigger error
        }
    )


def test_error_by_group_computes_mae_rmse_and_bias_per_group():
    table = error_by_group(_frame(), "y_true", "y_pred", "driver")
    ver = table.loc[table["driver"] == "VER"].iloc[0]
    ham = table.loc[table["driver"] == "HAM"].iloc[0]
    assert ver["n"] == 2 and ham["n"] == 2
    assert ham["mae"] > ver["mae"]  # HAM's 5.0 error dominates
    assert table.iloc[0]["driver"] == "HAM"  # sorted worst-first


def test_error_by_group_returns_none_for_missing_column():
    assert error_by_group(_frame(), "y_true", "y_pred", "team") is None


def test_error_by_group_returns_none_for_degenerate_column():
    assert error_by_group(_frame(), "y_true", "y_pred", "session_type") is None


def test_error_breakdown_reports_skipped_dimensions_with_a_reason():
    result = error_breakdown(_frame(), "y_true", "y_pred", ["driver", "session_type", "compound"])
    assert set(result["tables"]) == {"driver"}
    assert "distinct values" in result["skipped"]["session_type"]
    assert "not present" in result["skipped"]["compound"]


def test_bucket_numeric_assigns_labeled_ranges():
    series = pd.Series([0, 2, 5, 9])
    buckets = bucket_numeric(series, edges=[0, 3, 6, 10], labels=["low", "mid", "high"])
    assert list(buckets) == ["low", "low", "mid", "high"]


def test_largest_absolute_errors_sorted_descending_and_limited():
    table = largest_absolute_errors(_frame(), "y_true", "y_pred", n=2, extra_columns=["driver"])
    assert len(table) == 2
    assert table.iloc[0]["driver"] == "HAM"
    assert list(table["_abs_error"]) == sorted(table["_abs_error"], reverse=True)
