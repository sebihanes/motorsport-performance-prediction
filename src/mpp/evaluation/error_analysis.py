"""Residual slicing shared by every Phase 5 model (architecture report SS14
"error analysis strategy"): breaks out-of-fold prediction error down by
whatever identity/context columns a given dataset actually carries.

A requested slicing dimension that is absent from the frame, or present but
degenerate (fewer than two distinct non-null values in the current scope -
e.g. `session_type` is constant "R" everywhere, since config/settings.toml
scopes this project to race sessions only), is skipped rather than silently
producing a meaningless one-row table - callers get that back explicitly in
``skipped`` so a report can say *why* a dimension is missing instead of just
omitting it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def error_by_group(frame: pd.DataFrame, y_true_col: str, y_pred_col: str, group_col: str) -> pd.DataFrame | None:
    """Per-distinct-value-of-`group_col` MAE/RMSE/mean bias/count, sorted by
    MAE descending (worst-explained groups first). Returns None if
    `group_col` is missing or has fewer than two distinct non-null values -
    a single-group breakdown carries no information.
    """
    if group_col not in frame.columns:
        return None
    if frame[group_col].nunique(dropna=True) < 2:
        return None

    residual = frame[y_pred_col].to_numpy(dtype=float) - frame[y_true_col].to_numpy(dtype=float)
    working = frame[[group_col]].copy()
    working["_abs_error"] = np.abs(residual)
    working["_sq_error"] = residual**2
    working["_bias"] = residual

    grouped = working.groupby(group_col, dropna=False, observed=True)
    out = grouped.agg(
        n=("_abs_error", "size"),
        mae=("_abs_error", "mean"),
        mean_bias=("_bias", "mean"),
    )
    out["rmse"] = np.sqrt(grouped["_sq_error"].mean())
    out = out.reset_index()[[group_col, "n", "mae", "rmse", "mean_bias"]]
    return out.sort_values("mae", ascending=False).reset_index(drop=True)


def error_breakdown(
    frame: pd.DataFrame, y_true_col: str, y_pred_col: str, group_cols: list[str]
) -> dict[str, object]:
    """Error tables for every column in `group_cols` that actually supports
    one, plus the list of columns that were skipped and why - so a report
    can state "session_type: N/A (constant in this scope)" rather than
    leaving a silent gap.
    """
    tables: dict[str, pd.DataFrame] = {}
    skipped: dict[str, str] = {}
    for col in group_cols:
        if col not in frame.columns:
            skipped[col] = "column not present in this dataset"
            continue
        if frame[col].nunique(dropna=True) < 2:
            skipped[col] = "fewer than two distinct values in the current scope"
            continue
        tables[col] = error_by_group(frame, y_true_col, y_pred_col, col)
    return {"tables": tables, "skipped": skipped}


def bucket_numeric(series: pd.Series, edges: list[float], labels: list[str] | None = None) -> pd.Series:
    """Bucket a numeric column (e.g. tyre age) into labeled ranges for a
    group breakdown - pd.cut with sensible defaults for this project's use."""
    return pd.cut(series, bins=edges, labels=labels, include_lowest=True)


def _format_cell(value) -> str:
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:.4f}"
    return str(value)


def to_markdown_table(frame: pd.DataFrame) -> list[str]:
    """A dependency-free Markdown table (the project does not depend on
    ``tabulate``, which ``DataFrame.to_markdown`` requires) - used by every
    Phase 5 training script's generated comparison report."""
    columns = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for _, row in frame.iterrows():
        cells = [_format_cell(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def largest_absolute_errors(
    frame: pd.DataFrame, y_true_col: str, y_pred_col: str, n: int = 10, extra_columns: list[str] | None = None
) -> pd.DataFrame:
    """The n rows with the largest |prediction - actual| - for manual
    inspection (architecture SS14: separate genuine model failure from
    unmodelable events no feature set could have predicted)."""
    working = frame.copy()
    working["_abs_error"] = (working[y_pred_col] - working[y_true_col]).abs()
    columns = [y_true_col, y_pred_col, "_abs_error"] + [c for c in (extra_columns or []) if c in working.columns]
    return working.sort_values("_abs_error", ascending=False).head(n)[columns].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Phase 6 - residual analysis.
#
# Sign convention (shared with ``error_by_group``'s ``mean_bias`` above):
# ``residual = prediction - actual``. A positive residual is an
# *overprediction* (the model predicted a slower lap than was actually
# driven); a negative residual is an *underprediction* (the model predicted
# a faster lap than was driven).
# ---------------------------------------------------------------------------

RESIDUAL_COLUMN = "residual"
ABS_ERROR_COLUMN = "abs_error"


def add_residuals(frame: pd.DataFrame, y_true_col: str, y_pred_col: str) -> pd.DataFrame:
    """A copy of `frame` with ``residual`` (prediction - actual) and
    ``abs_error`` columns appended."""
    out = frame.copy()
    out[RESIDUAL_COLUMN] = out[y_pred_col].to_numpy(dtype=float) - out[y_true_col].to_numpy(dtype=float)
    out[ABS_ERROR_COLUMN] = np.abs(out[RESIDUAL_COLUMN])
    return out


def residual_summary(residual) -> dict[str, float]:
    """Distribution summary of a residual vector: location, spread, shape,
    tails, and the over/underprediction split."""
    r = np.asarray(residual, dtype=float)
    r = r[~np.isnan(r)]
    if r.size == 0:
        raise ValueError("Cannot summarize an empty residual vector.")
    std = float(np.std(r))
    skew = float(np.mean(((r - r.mean()) / std) ** 3)) if std > 0 else 0.0
    abs_r = np.abs(r)
    return {
        "n": int(r.size),
        "mean_residual": float(np.mean(r)),
        "median_residual": float(np.median(r)),
        "std_residual": std,
        "skewness": skew,
        "p05": float(np.percentile(r, 5)),
        "p25": float(np.percentile(r, 25)),
        "p75": float(np.percentile(r, 75)),
        "p95": float(np.percentile(r, 95)),
        "mae": float(np.mean(abs_r)),
        "median_abs_error": float(np.median(abs_r)),
        "p90_abs_error": float(np.percentile(abs_r, 90)),
        "max_abs_error": float(np.max(abs_r)),
        "share_overpredicted": float(np.mean(r > 0)),
        "share_underpredicted": float(np.mean(r < 0)),
        "max_overprediction": float(np.max(r)),
        "max_underprediction": float(np.min(r)),
    }


def error_vs_predicted(
    frame: pd.DataFrame, y_pred_col: str, residual_col: str = RESIDUAL_COLUMN, n_bins: int = 10
) -> pd.DataFrame:
    """Residuals binned by predicted-value quantile - a trend in mean bias
    across bins means the error depends on the prediction's own magnitude
    (e.g. slow-circuit predictions systematically too fast)."""
    bins = pd.qcut(frame[y_pred_col], q=n_bins, duplicates="drop")
    working = pd.DataFrame({"bin": bins, "pred": frame[y_pred_col], "r": frame[residual_col]})
    working["abs_r"] = working["r"].abs()
    out = working.groupby("bin", observed=True).agg(
        pred_low=("pred", "min"),
        pred_high=("pred", "max"),
        pred_mean=("pred", "mean"),
        n=("r", "size"),
        mean_bias=("r", "mean"),
        mae=("abs_r", "mean"),
    )
    return out.reset_index(drop=True)


def between_group_error_share(residual, groups) -> dict[str, float]:
    """Exact decomposition of mean squared error into a between-group part
    (each group's constant mean residual - a per-group *offset*) and a
    within-group part (scatter around that offset):

        mean(r^2) = sum_g (n_g/N) * mean_g^2  +  sum_g (n_g/N) * var_g

    A large ``share_between`` means most of the error is a systematic
    per-group bias rather than lap-to-lap noise.
    """
    r = np.asarray(residual, dtype=float)
    if r.size == 0:
        raise ValueError("Cannot decompose an empty residual vector.")
    frame = pd.DataFrame({"r": r, "g": np.asarray(groups)})
    grouped = frame.groupby("g")["r"]
    n = grouped.size()
    means = grouped.mean()
    variances = grouped.var(ddof=0)
    weights = n / r.size
    mse_between = float(np.sum(weights * means**2))
    mse_within = float(np.sum(weights * variances))
    mse_total = float(np.mean(r**2))
    return {
        "mse_total": mse_total,
        "mse_between": mse_between,
        "mse_within": mse_within,
        "share_between": mse_between / mse_total if mse_total > 0 else 0.0,
    }


def systematic_bias_table(
    frame: pd.DataFrame,
    group_col: str,
    residual_col: str = RESIDUAL_COLUMN,
    min_n: int = 30,
    z: float = 1.96,
    min_abs_bias_s: float = 0.5,
) -> pd.DataFrame | None:
    """Per-group mean residual with a normal-approximation confidence
    interval, and a direction label: ``overprediction`` / ``underprediction``
    when the interval excludes zero *and* the bias is at least
    ``min_abs_bias_s`` in magnitude, otherwise ``no clear bias``. Groups
    with fewer than ``min_n`` rows are labelled ``too few rows``.

    The interval treats laps as independent, which they are not (laps in
    one race/stint share conditions), so it is optimistic - it is a
    screening device for "consistently one-sided", not a significance test.
    Returns None for a missing or single-valued column.
    """
    if group_col not in frame.columns or frame[group_col].nunique(dropna=True) < 2:
        return None
    working = frame[[group_col, residual_col]].copy()
    working["_abs"] = working[residual_col].abs()
    out = (
        working.groupby(group_col, observed=True, dropna=False)
        .agg(n=(residual_col, "size"), mean_bias=(residual_col, "mean"), std=(residual_col, "std"), mae=("_abs", "mean"))
        .reset_index()
    )
    out["se"] = out["std"].fillna(0.0) / np.sqrt(out["n"])
    out["ci_low"] = out["mean_bias"] - z * out["se"]
    out["ci_high"] = out["mean_bias"] + z * out["se"]

    def _direction(row) -> str:
        if row["n"] < min_n:
            return "too few rows"
        if row["ci_low"] > 0 and row["mean_bias"] >= min_abs_bias_s:
            return "overprediction"
        if row["ci_high"] < 0 and row["mean_bias"] <= -min_abs_bias_s:
            return "underprediction"
        return "no clear bias"

    out["direction"] = out.apply(_direction, axis=1)
    out = out.drop(columns=["std"])
    return out.sort_values("mae", ascending=False).reset_index(drop=True)


def robust_z(values) -> np.ndarray:
    """Median/MAD-based z-score (0.6745 * (x - median) / MAD) - unlike a
    mean/std z-score it is not itself inflated by the outliers it is meant
    to find. Returns zeros if MAD is 0."""
    x = np.asarray(values, dtype=float)
    median = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - median))
    if mad == 0 or np.isnan(mad):
        return np.zeros_like(x)
    return 0.6745 * (x - median) / mad


def flag_unusual_observations(
    frame: pd.DataFrame, residual_col: str = RESIDUAL_COLUMN, center_by: str | None = None, threshold: float = 3.5
) -> pd.DataFrame:
    """Copy of `frame` with ``centered_residual`` (residual minus its
    `center_by` group's mean residual - what remains once a per-race offset
    is removed; equal to the residual when `center_by` is None),
    ``robust_z`` of that centered residual, and ``is_unusual``
    (|robust_z| > threshold, the conventional 3.5 cut-off)."""
    out = frame.copy()
    if center_by is not None:
        group_mean = out.groupby(center_by, observed=True)[residual_col].transform("mean")
        out["centered_residual"] = out[residual_col] - group_mean
    else:
        out["centered_residual"] = out[residual_col]
    out["robust_z"] = robust_z(out["centered_residual"])
    out["is_unusual"] = np.abs(out["robust_z"]) > threshold
    return out


# FastF1 TrackStatus codes: a lap's status string concatenates every status
# seen during that lap (e.g. "124" = green -> yellow -> safety car).
_TRACK_STATUS_PRIORITY = [("5", "red_flag"), ("4", "safety_car"), ("6", "vsc"), ("7", "vsc"), ("2", "yellow_flag")]


def classify_track_status(status) -> str:
    """Most severe neutralisation seen during a lap: red_flag > safety_car >
    vsc > yellow_flag > green. Missing -> unknown."""
    if status is None or status is pd.NA or (isinstance(status, float) and np.isnan(status)):
        return "unknown"
    text = str(status)
    for code, label in _TRACK_STATUS_PRIORITY:
        if code in text:
            return label
    return "green" if "1" in text else "unknown"
