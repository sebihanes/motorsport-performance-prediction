from __future__ import annotations

import numpy as np
import pytest

from mpp.evaluation.metrics import aggregate_fold_metrics, interpret_mae, regression_metrics


def test_regression_metrics_hand_computed_values():
    # y_true - y_pred = [1, -1, 2] -> |errors| = [1, 1, 2]
    y_true = [10.0, 20.0, 30.0]
    y_pred = [9.0, 21.0, 28.0]
    metrics = regression_metrics(y_true, y_pred)
    assert metrics["mae"] == pytest.approx(4 / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt((1 + 1 + 4) / 3))
    assert metrics["median_ae"] == pytest.approx(1.0)
    # R^2 against hand-computed total/residual sum of squares.
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - y_true_arr.mean()) ** 2)
    assert metrics["r2"] == pytest.approx(1 - ss_res / ss_tot)


def test_regression_metrics_perfect_prediction_is_zero_error():
    metrics = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["median_ae"] == 0.0
    assert metrics["r2"] == pytest.approx(1.0)


def test_regression_metrics_single_row_r2_is_nan_not_an_error():
    metrics = regression_metrics([5.0], [4.0])
    assert metrics["mae"] == pytest.approx(1.0)
    assert np.isnan(metrics["r2"])


def test_regression_metrics_rejects_zero_rows():
    with pytest.raises(ValueError):
        regression_metrics([], [])


def test_regression_metrics_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        regression_metrics([1.0, 2.0], [1.0])


def test_interpret_mae_reads_as_engineering_sentence():
    text = interpret_mae(0.18)
    assert "0.18" in text
    assert "typical absolute" in text


def test_interpret_mae_singular_unit_for_exactly_one():
    text = interpret_mae(1.0)
    assert "1.000 second " in text  # no trailing 's' before the next word
    assert "seconds" not in text.split("means")[0]


def test_aggregate_fold_metrics_computes_mean_and_std():
    fold_metrics = [
        {"mae": 1.0, "rmse": 2.0},
        {"mae": 3.0, "rmse": 4.0},
    ]
    aggregated = aggregate_fold_metrics(fold_metrics)
    assert aggregated["mae_mean"] == pytest.approx(2.0)
    assert aggregated["mae_std"] == pytest.approx(1.0)
    assert aggregated["rmse_mean"] == pytest.approx(3.0)
    assert aggregated["rmse_std"] == pytest.approx(1.0)


def test_aggregate_fold_metrics_rejects_empty_list():
    with pytest.raises(ValueError):
        aggregate_fold_metrics([])
