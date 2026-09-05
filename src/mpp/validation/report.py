"""Assemble per-table rule checks into a validation report (JSON + Markdown).

Reports are written, never silently swallowed - see docs/02_data_pipeline.md
for how "error"-status rules are expected to be handled by cleaning.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from mpp.config import PipelineConfig
from mpp.validation.rules import (
    RuleResult,
    duplicate_rows,
    non_negative,
    not_null,
    numeric_bounds,
    positive_integer,
    required_columns_present,
    value_in_set,
)

REQUIRED_LAP_COLUMNS = [
    "season",
    "round",
    "Driver",
    "Team",
    "LapNumber",
    "Stint",
    "LapTime",
    "Compound",
    "TyreLife",
]
REQUIRED_PITSTOP_COLUMNS = ["season", "round", "driver_id", "lap", "stop_number", "duration_raw"]
REQUIRED_SESSION_COLUMNS = ["season", "round", "circuit_id", "total_laps"]


_INTERMEDIATE_LAP_COLUMN_NAMES = {"driver": "Driver", "lap_number": "LapNumber"}


def validate_laps(df: pd.DataFrame, config: PipelineConfig) -> list[RuleResult]:
    results = [required_columns_present(df, REQUIRED_LAP_COLUMNS, "laps")]
    if results[0].status == "error":
        return results  # can't meaningfully run column-specific rules
    duplicate_key = [
        _INTERMEDIATE_LAP_COLUMN_NAMES.get(k, k) for k in config.cleaning.duplicate_key
    ]
    results += [
        not_null(df, "Driver", "laps", severity="error"),
        not_null(df, "Team", "laps", severity="error"),
        not_null(df, "season", "laps", severity="error"),
        not_null(df, "round", "laps", severity="error"),
        positive_integer(df, "LapNumber", "laps"),
        duplicate_rows(df, duplicate_key, "laps"),
        not_null(df, "LapTime", "laps", severity="warning"),
        numeric_bounds(
            df,
            "LapTime",
            config.validation.lap_time_min_s,
            config.validation.lap_time_max_s,
            "laps",
            as_timedelta_seconds=True,
        ),
        non_negative(df, "TyreLife", "laps"),
        value_in_set(df, "Compound", set(config.validation.valid_compounds), "laps"),
    ]
    return results


def validate_pitstops(df: pd.DataFrame, config: PipelineConfig) -> list[RuleResult]:
    if df.empty:
        return [
            RuleResult(
                name="pitstops.required_columns_present",
                status="warning",
                message="Pitstops table is empty for the configured rounds.",
            )
        ]
    results = [required_columns_present(df, REQUIRED_PITSTOP_COLUMNS, "pitstops")]
    if results[0].status == "error":
        return results
    results += [
        not_null(df, "driver_id", "pitstops", severity="error"),
        positive_integer(df, "lap", "pitstops"),
        positive_integer(df, "stop_number", "pitstops"),
        duplicate_rows(df, ["season", "round", "driver_id", "stop_number"], "pitstops"),
        not_null(df, "duration_raw", "pitstops", severity="warning"),
    ]
    return results


def validate_sessions(df: pd.DataFrame, config: PipelineConfig) -> list[RuleResult]:
    results = [required_columns_present(df, REQUIRED_SESSION_COLUMNS, "sessions")]
    if results[0].status == "error":
        return results
    results += [
        not_null(df, "circuit_id", "sessions", severity="error"),
        not_null(df, "season", "sessions", severity="error"),
        not_null(df, "round", "sessions", severity="error"),
        duplicate_rows(df, ["season", "round"], "sessions"),
    ]
    return results


def _to_markdown(all_results: dict[str, list[RuleResult]]) -> str:
    lines = ["# Data Validation Report", ""]
    icons = {"ok": "PASS", "warning": "WARN", "error": "FAIL"}
    n_error = sum(1 for rs in all_results.values() for r in rs if r.status == "error")
    n_warning = sum(1 for rs in all_results.values() for r in rs if r.status == "warning")
    lines.append(f"**{n_error} error(s), {n_warning} warning(s).**\n")
    for table_name, results in all_results.items():
        lines.append(f"## {table_name}")
        lines.append("")
        lines.append("| Status | Rule | Message |")
        lines.append("|---|---|---|")
        for r in results:
            lines.append(f"| {icons[r.status]} | {r.name} | {r.message} |")
        lines.append("")
    return "\n".join(lines)


def run_validation(
    intermediate: dict[str, pd.DataFrame], config: PipelineConfig
) -> dict[str, list[RuleResult]]:
    all_results = {
        "laps": validate_laps(intermediate["laps"], config),
        "pitstops": validate_pitstops(intermediate["pitstops"], config),
        "sessions": validate_sessions(intermediate["sessions"], config),
    }

    quality_dir = config.paths.quality_dir
    quality_dir.mkdir(parents=True, exist_ok=True)

    json_payload = {
        table: [asdict(r) for r in results] for table, results in all_results.items()
    }
    with open(quality_dir / "validation_report.json", "w", encoding="utf-8") as fh:
        json.dump(json_payload, fh, indent=2)

    with open(quality_dir / "validation_report.md", "w", encoding="utf-8") as fh:
        fh.write(_to_markdown(all_results))

    return all_results


def has_errors(all_results: dict[str, list[RuleResult]]) -> bool:
    return any(r.status == "error" for results in all_results.values() for r in results)
