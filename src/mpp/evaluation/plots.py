"""Static matplotlib figures for interpretability and error analysis
(Phase 6). Every function takes plain DataFrames/arrays, returns a
``matplotlib.figure.Figure`` and draws nothing to screen - callers save
with ``save_figure``. The Agg backend is forced so this works headless
(CI, tests).

Visual conventions, kept deliberately plain: one neutral series color;
overprediction vs. underprediction is the only two-color encoding (orange =
model predicted too slow, blue = model predicted too fast), always labelled
in the legend or axis title rather than by color alone; recessive grid; a
zero/identity reference line wherever "no error" has a location on the
axis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

PRIMARY = "#2a78d6"  # neutral series / underprediction
SECONDARY = "#eb6834"  # overprediction
REFERENCE = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def _style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(REFERENCE)
    ax.tick_params(colors=REFERENCE, labelsize=9)


def save_figure(fig: Figure, path: Path, dpi: int = 130) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_importance(
    table: pd.DataFrame,
    value_col: str,
    title: str,
    xlabel: str,
    label_col: str = "feature",
    err_col: str | None = None,
    top_n: int = 15,
    signed: bool = False,
) -> Figure:
    """Horizontal bars, largest at the top. ``signed=True`` colors bars by
    sign (for coefficients) and draws a zero line."""
    data = table.copy()
    order_key = data[value_col].abs() if signed else data[value_col]
    data = data.assign(_key=order_key).sort_values("_key", ascending=False).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(data) + 1.4))
    colors = [SECONDARY if v > 0 else PRIMARY for v in data[value_col]] if signed else PRIMARY
    xerr = data[err_col] if err_col else None
    ax.barh(data[label_col].astype(str), data[value_col], xerr=xerr, color=colors, height=0.6,
            error_kw={"ecolor": REFERENCE, "elinewidth": 1, "capsize": 2})
    ax.axvline(0, color=REFERENCE, linewidth=1)
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, fontsize=9)
    _style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_predicted_vs_actual(
    frame: pd.DataFrame, y_true_col: str, y_pred_col: str, facet_col: str | None = None, title: str = ""
) -> Figure:
    """Scatter of prediction vs. actual with the y = x identity line; one
    panel per `facet_col` value (shared axes) so per-group offsets show up
    as a cloud displaced from the diagonal."""
    groups = [(None, frame)] if facet_col is None else list(frame.groupby(facet_col, sort=True))
    n = len(groups)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.3 * nrows), sharex=True, sharey=True,
                             squeeze=False)
    lo = float(min(frame[y_true_col].min(), frame[y_pred_col].min()))
    hi = float(max(frame[y_true_col].max(), frame[y_pred_col].max()))
    for ax, (name, group) in zip(axes.flat, groups):
        ax.scatter(group[y_true_col], group[y_pred_col], s=6, alpha=0.35, color=PRIMARY, linewidths=0)
        ax.plot([lo, hi], [lo, hi], color=REFERENCE, linewidth=1, linestyle="--")
        if name is not None:
            ax.set_title(str(name), fontsize=10, loc="left")
        _style(ax)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.supxlabel("Actual lap time (s)", fontsize=9)
    fig.supylabel("Predicted lap time (s)", fontsize=9)
    if title:
        fig.suptitle(title, fontsize=11, x=0.02, ha="left")
    fig.tight_layout()
    return fig


def plot_residuals_vs_predicted(
    frame: pd.DataFrame, y_pred_col: str, residual_col: str, binned: pd.DataFrame | None = None, title: str = ""
) -> Figure:
    """Residual (prediction - actual) against prediction, with an optional
    binned mean-bias line from ``error_vs_predicted``."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.scatter(frame[y_pred_col], frame[residual_col], s=6, alpha=0.3, color=PRIMARY, linewidths=0,
               label="lap (out-of-fold)")
    ax.axhline(0, color=REFERENCE, linewidth=1, linestyle="--")
    if binned is not None:
        ax.plot(binned["pred_mean"], binned["mean_bias"], color=SECONDARY, linewidth=2, marker="o", markersize=5,
                label="mean residual per predicted-value decile")
    ax.set_xlabel("Predicted lap time (s)", fontsize=9)
    ax.set_ylabel("Residual (s)  [+ = predicted too slow]", fontsize=9)
    ax.set_title(title, fontsize=11, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    _style(ax)
    fig.tight_layout()
    return fig


def plot_error_distributions(residual, abs_error, title: str = "") -> Figure:
    """Two panels: residual histogram (signed, zero line) and absolute-error
    histogram with MAE and median absolute error marked."""
    residual = np.asarray(residual, dtype=float)
    abs_error = np.asarray(abs_error, dtype=float)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax1.hist(residual, bins=60, color=PRIMARY, edgecolor="white", linewidth=0.4)
    ax1.axvline(0, color=REFERENCE, linewidth=1, linestyle="--")
    ax1.axvline(np.mean(residual), color=SECONDARY, linewidth=1.5, label=f"mean {np.mean(residual):+.2f}s")
    ax1.set_xlabel("Residual (s)  [+ = predicted too slow]", fontsize=9)
    ax1.set_ylabel("Laps", fontsize=9)
    ax1.set_title("Residual distribution", fontsize=10, loc="left")
    ax1.legend(fontsize=8, frameon=False)
    ax2.hist(abs_error, bins=60, color=PRIMARY, edgecolor="white", linewidth=0.4)
    ax2.axvline(np.mean(abs_error), color=SECONDARY, linewidth=1.5, label=f"MAE {np.mean(abs_error):.2f}s")
    ax2.axvline(np.median(abs_error), color=REFERENCE, linewidth=1.5, linestyle=":",
                label=f"median {np.median(abs_error):.2f}s")
    ax2.set_xlabel("Absolute error (s)", fontsize=9)
    ax2.set_title("Absolute error distribution", fontsize=10, loc="left")
    ax2.legend(fontsize=8, frameon=False)
    for ax in (ax1, ax2):
        _style(ax)
    if title:
        fig.suptitle(title, fontsize=11, x=0.02, ha="left")
    fig.tight_layout()
    return fig


def plot_grouped_errors(table: pd.DataFrame, group_col: str, title: str, top_n: int = 25) -> Figure:
    """Two aligned panels per group: MAE (bars) and mean bias with its
    confidence interval (orange = overprediction, blue = underprediction).
    Expects ``systematic_bias_table`` output (mae, mean_bias, ci_low,
    ci_high)."""
    data = table.sort_values("mae", ascending=False).head(top_n).iloc[::-1]
    labels = data[group_col].astype(str)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 0.32 * len(data) + 1.6), sharey=True)
    ax1.barh(labels, data["mae"], color=PRIMARY, height=0.6)
    ax1.set_xlabel("MAE (s)", fontsize=9)
    ax1.set_title("Mean absolute error", fontsize=10, loc="left")
    colors = [SECONDARY if b > 0 else PRIMARY for b in data["mean_bias"]]
    ax2.barh(labels, data["mean_bias"], color=colors, height=0.6)
    if {"ci_low", "ci_high"} <= set(data.columns):
        ax2.errorbar(data["mean_bias"], labels,
                     xerr=[data["mean_bias"] - data["ci_low"], data["ci_high"] - data["mean_bias"]],
                     fmt="none", ecolor=REFERENCE, elinewidth=1, capsize=2)
    ax2.axvline(0, color=REFERENCE, linewidth=1)
    ax2.set_xlabel("Mean residual (s)  [+ predicted too slow / - too fast]", fontsize=9)
    ax2.set_title("Systematic bias", fontsize=10, loc="left")
    for ax in (ax1, ax2):
        _style(ax)
        ax.grid(axis="y", visible=False)
    fig.suptitle(title, fontsize=11, x=0.02, ha="left")
    fig.tight_layout()
    return fig


def plot_partial_dependence(
    curves: Mapping[str, pd.DataFrame], xlabel: str, title: str, ylabel: str = "Mean predicted lap time (s)"
) -> Figure:
    """One line per labelled ``partial_dependence`` table (e.g. one per
    compound), with a legend when there is more than one curve."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, (label, curve) in enumerate(curves.items()):
        color = SERIES[i % len(SERIES)]
        ax.plot(curve["value"], curve["mean_prediction"], color=color, linewidth=2, label=label)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=11, loc="left")
    if len(curves) > 1:
        ax.legend(fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_importance_comparison(
    table: pd.DataFrame, label_col: str, value_cols: Sequence[str], value_labels: Sequence[str], title: str,
    xlabel: str,
) -> Figure:
    """Grouped horizontal bars comparing two (or more) importance measures
    per feature - e.g. across-race vs. within-race permutation importance."""
    data = table.iloc[::-1]
    y = np.arange(len(data))
    height = 0.8 / len(value_cols)
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(data) + 1.6))
    for i, (col, lab) in enumerate(zip(value_cols, value_labels)):
        ax.barh(y + (i - (len(value_cols) - 1) / 2) * height, data[col], height=height * 0.92,
                color=SERIES[i % len(SERIES)], label=lab)
    ax.set_yticks(y, data[label_col].astype(str))
    ax.axvline(0, color=REFERENCE, linewidth=1)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=11, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    _style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig
