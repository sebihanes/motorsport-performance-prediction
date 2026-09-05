"""Reusable validation rule functions.

Every rule takes a DataFrame (and sometimes the pipeline config) and returns a
RuleResult. Rules only *report* - they never modify or drop data. Row-level
decisions about what to do with an issue belong to mpp.cleaning, which reads
these reports for its own bookkeeping but re-derives conditions itself so
cleaning behaviour does not depend on validation's internal representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class RuleResult:
    name: str
    status: str  # "ok" | "warning" | "error"
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def required_columns_present(df: pd.DataFrame, required: list[str], table_name: str) -> RuleResult:
    missing = [c for c in required if c not in df.columns]
    if missing:
        return RuleResult(
            name=f"{table_name}.required_columns_present",
            status="error",
            message=f"Missing required columns: {missing}",
            details={"missing_columns": missing},
        )
    return RuleResult(
        name=f"{table_name}.required_columns_present",
        status="ok",
        message="All required columns present.",
    )


def not_null(df: pd.DataFrame, column: str, table_name: str, severity: str = "error") -> RuleResult:
    if column not in df.columns:
        return RuleResult(
            name=f"{table_name}.{column}.not_null",
            status="error",
            message=f"Column '{column}' does not exist.",
        )
    n_null = int(df[column].isna().sum())
    if n_null > 0:
        return RuleResult(
            name=f"{table_name}.{column}.not_null",
            status=severity,
            message=f"{n_null} of {len(df)} rows have a null '{column}'.",
            details={"n_null": n_null, "n_rows": len(df)},
        )
    return RuleResult(
        name=f"{table_name}.{column}.not_null",
        status="ok",
        message=f"No nulls in '{column}'.",
    )


def duplicate_rows(df: pd.DataFrame, key: list[str], table_name: str) -> RuleResult:
    missing_keys = [k for k in key if k not in df.columns]
    if missing_keys:
        return RuleResult(
            name=f"{table_name}.duplicate_rows",
            status="error",
            message=f"Cannot check duplicates - missing key columns: {missing_keys}",
        )
    dup_mask = df.duplicated(subset=key, keep=False)
    n_dup = int(dup_mask.sum())
    if n_dup > 0:
        examples = df.loc[dup_mask, key].drop_duplicates().head(5).to_dict("records")
        return RuleResult(
            name=f"{table_name}.duplicate_rows",
            status="error",
            message=f"{n_dup} rows share a duplicate key {key}.",
            details={"n_duplicate_rows": n_dup, "example_keys": examples},
        )
    return RuleResult(
        name=f"{table_name}.duplicate_rows",
        status="ok",
        message=f"No duplicate rows on key {key}.",
    )


def positive_integer(df: pd.DataFrame, column: str, table_name: str) -> RuleResult:
    if column not in df.columns:
        return RuleResult(
            name=f"{table_name}.{column}.positive_integer",
            status="error",
            message=f"Column '{column}' does not exist.",
        )
    values = pd.to_numeric(df[column], errors="coerce")
    invalid_mask = values.isna() | (values <= 0)
    n_invalid = int(invalid_mask.sum())
    if n_invalid > 0:
        return RuleResult(
            name=f"{table_name}.{column}.positive_integer",
            status="error",
            message=f"{n_invalid} of {len(df)} rows have a non-positive or missing '{column}'.",
            details={"n_invalid": n_invalid},
        )
    return RuleResult(
        name=f"{table_name}.{column}.positive_integer",
        status="ok",
        message=f"All '{column}' values are positive.",
    )


def non_negative(df: pd.DataFrame, column: str, table_name: str, severity: str = "warning") -> RuleResult:
    if column not in df.columns:
        return RuleResult(
            name=f"{table_name}.{column}.non_negative",
            status="error",
            message=f"Column '{column}' does not exist.",
        )
    values = pd.to_numeric(df[column], errors="coerce")
    n_negative = int((values < 0).sum())
    if n_negative > 0:
        return RuleResult(
            name=f"{table_name}.{column}.non_negative",
            status=severity,
            message=f"{n_negative} of {len(df)} rows have a negative '{column}'.",
            details={"n_negative": n_negative},
        )
    return RuleResult(
        name=f"{table_name}.{column}.non_negative",
        status="ok",
        message=f"No negative '{column}' values.",
    )


def value_in_set(
    df: pd.DataFrame,
    column: str,
    allowed: set[str],
    table_name: str,
    case_insensitive: bool = True,
    allow_null: bool = True,
) -> RuleResult:
    if column not in df.columns:
        return RuleResult(
            name=f"{table_name}.{column}.value_in_set",
            status="error",
            message=f"Column '{column}' does not exist.",
        )
    series = df[column]
    non_null_series = series.dropna()
    check_values = non_null_series.astype(str).str.upper() if case_insensitive else non_null_series.astype(str)
    allowed_check = {v.upper() for v in allowed} if case_insensitive else allowed
    unknown_mask = ~check_values.isin(allowed_check)
    unknown_values = sorted(set(non_null_series[unknown_mask.values].astype(str)))
    n_null = int(series.isna().sum())
    if unknown_values:
        return RuleResult(
            name=f"{table_name}.{column}.value_in_set",
            status="warning",
            message=f"{len(unknown_values)} unrecognized '{column}' value(s) found: {unknown_values}",
            details={"unknown_values": unknown_values, "n_null": n_null},
        )
    if n_null and not allow_null:
        return RuleResult(
            name=f"{table_name}.{column}.value_in_set",
            status="warning",
            message=f"{n_null} rows have a null '{column}'.",
            details={"n_null": n_null},
        )
    return RuleResult(
        name=f"{table_name}.{column}.value_in_set",
        status="ok",
        message=f"All non-null '{column}' values are recognized.",
        details={"n_null": n_null},
    )


def numeric_bounds(
    df: pd.DataFrame,
    column: str,
    lower: float,
    upper: float,
    table_name: str,
    as_timedelta_seconds: bool = False,
) -> RuleResult:
    if column not in df.columns:
        return RuleResult(
            name=f"{table_name}.{column}.numeric_bounds",
            status="error",
            message=f"Column '{column}' does not exist.",
        )
    values = df[column]
    if as_timedelta_seconds:
        values = pd.to_timedelta(values, errors="coerce").dt.total_seconds()
    else:
        values = pd.to_numeric(values, errors="coerce")
    non_null = values.dropna()
    out_of_bounds = non_null[(non_null < lower) | (non_null > upper)]
    if len(out_of_bounds) > 0:
        return RuleResult(
            name=f"{table_name}.{column}.numeric_bounds",
            status="warning",
            message=f"{len(out_of_bounds)} of {len(non_null)} non-null '{column}' values fall "
            f"outside [{lower}, {upper}].",
            details={
                "n_out_of_bounds": int(len(out_of_bounds)),
                "n_non_null": int(len(non_null)),
                "min_found": float(non_null.min()) if len(non_null) else None,
                "max_found": float(non_null.max()) if len(non_null) else None,
            },
        )
    return RuleResult(
        name=f"{table_name}.{column}.numeric_bounds",
        status="ok",
        message=f"All non-null '{column}' values fall within [{lower}, {upper}].",
    )
