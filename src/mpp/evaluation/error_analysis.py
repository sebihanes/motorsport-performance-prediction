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
