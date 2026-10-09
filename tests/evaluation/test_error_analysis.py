from __future__ import annotations

import pandas as pd

import numpy as np
import pytest

from mpp.evaluation.error_analysis import (
    add_residuals,
    between_group_error_share,
    bucket_numeric,
    classify_track_status,
    error_breakdown,
    error_by_group,
    error_vs_predicted,
    flag_unusual_observations,
    largest_absolute_errors,
    residual_summary,
    robust_z,
    systematic_bias_table,
)


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


# --- Phase 6 residual analysis ---


def test_add_residuals_uses_prediction_minus_actual():
    out = add_residuals(_frame(), "y_true", "y_pred")
    assert list(out["residual"]) == [1.0, -1.0, 5.0, 0.0]
    assert list(out["abs_error"]) == [1.0, 1.0, 5.0, 0.0]
    assert "residual" not in _frame().columns  # input not mutated


def test_residual_summary_values():
    summary = residual_summary([1.0, -1.0, 5.0, 0.0])
    assert summary["n"] == 4
    assert summary["mean_residual"] == pytest.approx(1.25)
    assert summary["median_residual"] == pytest.approx(0.5)
    assert summary["mae"] == pytest.approx(7.0 / 4)
    assert summary["share_overpredicted"] == pytest.approx(0.5)
    assert summary["share_underpredicted"] == pytest.approx(0.25)
    assert summary["max_overprediction"] == 5.0 and summary["max_underprediction"] == -1.0
    assert summary["skewness"] > 0  # one large positive residual
    with pytest.raises(ValueError):
        residual_summary([])


def test_error_vs_predicted_bins_cover_every_row():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"pred": rng.normal(90, 5, 200)})
    frame["residual"] = np.where(frame["pred"] > 90, 2.0, -2.0)
    table = error_vs_predicted(frame, "pred", n_bins=4)
    assert table["n"].sum() == 200
    assert len(table) == 4
    assert table["mean_bias"].iloc[0] == pytest.approx(-2.0)
    assert table["mean_bias"].iloc[-1] == pytest.approx(2.0)
    assert (table["mae"] == 2.0).all()


def test_between_group_error_share_is_an_exact_decomposition():
    residual = np.array([2.0, 2.0, 2.0, -1.0, 1.0, 0.0])
    groups = np.array(["a", "a", "a", "b", "b", "b"])
    out = between_group_error_share(residual, groups)
    assert out["mse_total"] == pytest.approx(np.mean(residual**2))
    assert out["mse_between"] + out["mse_within"] == pytest.approx(out["mse_total"])
    # group a is a pure constant offset; group b has zero mean
    assert out["mse_between"] == pytest.approx(0.5 * 4.0)
    assert out["mse_within"] == pytest.approx(0.5 * (2.0 / 3.0))


def test_between_group_share_is_one_for_pure_offsets():
    out = between_group_error_share([3.0, 3.0, -2.0, -2.0], ["a", "a", "b", "b"])
    assert out["share_between"] == pytest.approx(1.0)


def test_systematic_bias_table_labels_direction():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        {
            "g": ["over"] * 100 + ["under"] * 100 + ["none"] * 100 + ["tiny"] * 5,
            "residual": np.concatenate(
                [3 + rng.normal(0, 0.5, 100), -3 + rng.normal(0, 0.5, 100), rng.normal(0, 0.5, 100), [9.0] * 5]
            ),
        }
    )
    table = systematic_bias_table(frame, "g").set_index("g")
    assert table.loc["over", "direction"] == "overprediction"
    assert table.loc["under", "direction"] == "underprediction"
    assert table.loc["none", "direction"] == "no clear bias"
    assert table.loc["tiny", "direction"] == "too few rows"
    assert table.loc["over", "ci_low"] < table.loc["over", "mean_bias"] < table.loc["over", "ci_high"]
    assert table["n"].sum() == len(frame)


def test_systematic_bias_table_none_for_degenerate_column():
    frame = add_residuals(_frame(), "y_true", "y_pred")
    assert systematic_bias_table(frame, "session_type") is None
    assert systematic_bias_table(frame, "missing") is None


def test_robust_z_flags_outlier_and_handles_zero_mad():
    z = robust_z([0.0, 1.0, -1.0, 0.5, -0.5, 50.0])
    assert abs(z[-1]) > 3.5 and np.all(np.abs(z[:-1]) < 3.5)
    assert np.all(robust_z([2.0, 2.0, 2.0]) == 0)


def test_flag_unusual_observations_removes_group_offset_first():
    frame = pd.DataFrame(
        {"race": ["a"] * 6 + ["b"] * 6, "residual": [10, 10.5, 9.5, 10, 10.2, 9.8, 0, 0.5, -0.5, 0, 0.2, 12.0]}
    )
    centered = flag_unusual_observations(frame, center_by="race")
    assert centered.loc[centered["race"] == "a", "centered_residual"].abs().max() < 1.0
    assert centered["is_unusual"].sum() == 1
    assert bool(centered["is_unusual"].iloc[-1])


def test_classify_track_status_priority():
    assert classify_track_status("1") == "green"
    assert classify_track_status("12") == "yellow_flag"
    assert classify_track_status("124") == "safety_car"
    assert classify_track_status("671") == "vsc"
    assert classify_track_status("2671") == "vsc"
    assert classify_track_status("45") == "red_flag"
    assert classify_track_status(None) == "unknown"
    assert classify_track_status(pd.NA) == "unknown"
    assert classify_track_status(float("nan")) == "unknown"
