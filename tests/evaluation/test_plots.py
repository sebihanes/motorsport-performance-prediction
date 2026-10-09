from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from mpp.evaluation import plots
from mpp.evaluation.error_analysis import add_residuals, error_vs_predicted, systematic_bias_table


def _frame(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "race": np.repeat(["r1", "r2", "r3", "r4"], n // 4),
            "y_true": rng.normal(90, 5, n),
        }
    )
    frame["y_pred"] = frame["y_true"] + rng.normal(0, 2, n)
    return add_residuals(frame, "y_true", "y_pred")


def test_plot_feature_importance_draws_one_bar_per_feature_up_to_top_n():
    table = pd.DataFrame({"feature": list("abcdef"), "importance_mean": [5, -1, 3, 0, 2, 1], "importance_std": 0.1})
    fig = plots.plot_feature_importance(table, "importance_mean", "t", "x", err_col="importance_std", top_n=4)
    assert isinstance(fig, Figure)
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert labels == ["f", "e", "c", "a"]  # top 4 by value, largest at the top
    signed = plots.plot_feature_importance(table, "importance_mean", "t", "x", signed=True, top_n=6)
    assert [t.get_text() for t in signed.axes[0].get_yticklabels()][-1] == "a"
    plots.plt.close("all")


def test_plot_predicted_vs_actual_one_panel_per_facet():
    fig = plots.plot_predicted_vs_actual(_frame(), "y_true", "y_pred", facet_col="race")
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 4
    single = plots.plot_predicted_vs_actual(_frame(), "y_true", "y_pred")
    assert len([ax for ax in single.axes if ax.get_visible()]) == 1
    plots.plt.close("all")


def test_residual_distribution_and_grouped_plots_build():
    frame = _frame()
    fig = plots.plot_residuals_vs_predicted(frame, "y_pred", "residual", error_vs_predicted(frame, "y_pred", n_bins=4))
    assert len(fig.axes) == 1
    fig = plots.plot_error_distributions(frame["residual"], frame["abs_error"])
    assert len(fig.axes) == 2
    fig = plots.plot_grouped_errors(systematic_bias_table(frame, "race", min_n=5), "race", "by race")
    assert len(fig.axes) == 2
    assert len(fig.axes[0].patches) == 4  # one MAE bar per race
    plots.plt.close("all")


def test_partial_dependence_and_comparison_plots_build():
    curves = {"SOFT": pd.DataFrame({"value": [1, 2, 3], "mean_prediction": [90, 90.5, 91]})}
    fig = plots.plot_partial_dependence(curves, "age", "pd")
    assert len(fig.axes[0].get_lines()) == 1
    table = pd.DataFrame({"feature": ["a", "b"], "across": [1.0, 0.5], "within": [0.2, 0.4]})
    fig = plots.plot_importance_comparison(table, "feature", ["across", "within"], ["A", "W"], "t", "x")
    assert len(fig.axes[0].patches) == 4
    plots.plt.close("all")


def test_save_figure_writes_png(tmp_path):
    fig = plots.plot_error_distributions([1.0, -1.0, 0.5], [1.0, 1.0, 0.5])
    path = plots.save_figure(fig, tmp_path / "nested" / "f.png")
    assert path.exists() and path.stat().st_size > 0
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
