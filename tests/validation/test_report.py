from __future__ import annotations

import pandas as pd

from mpp.validation.report import has_errors, run_validation, validate_laps, validate_pitstops


def _minimal_laps_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2023, 2023],
            "round": [1, 1],
            "Driver": ["VER", "LEC"],
            "Team": ["Red Bull Racing", "Ferrari"],
            "LapNumber": [1, 2],
            "Stint": [1, 1],
            "LapTime": pd.to_timedelta(["0 days 00:01:39", "0 days 00:01:38"]),
            "Compound": ["SOFT", "SOFT"],
            "TyreLife": [4, 5],
        }
    )


def test_validate_laps_clean_data_has_no_errors(test_config):
    results = validate_laps(_minimal_laps_df(), test_config)
    assert not any(r.status == "error" for r in results)


def test_validate_laps_flags_duplicate_key(test_config):
    df = pd.concat([_minimal_laps_df(), _minimal_laps_df().iloc[[0]]], ignore_index=True)
    results = validate_laps(df, test_config)
    dup_result = next(r for r in results if r.name == "laps.duplicate_rows")
    assert dup_result.status == "error"


def test_validate_laps_flags_missing_driver(test_config):
    df = _minimal_laps_df()
    df.loc[0, "Driver"] = None
    results = validate_laps(df, test_config)
    driver_result = next(r for r in results if r.name == "laps.Driver.not_null")
    assert driver_result.status == "error"


def test_validate_pitstops_empty_table_is_a_warning_not_error(test_config):
    results = validate_pitstops(pd.DataFrame(), test_config)
    assert not any(r.status == "error" for r in results)


def test_run_validation_writes_reports_and_detects_errors(test_config):
    laps = _minimal_laps_df()
    laps.loc[1, "Driver"] = None  # force an error
    intermediate = {
        "laps": laps,
        "pitstops": pd.DataFrame(
            columns=["season", "round", "driver_id", "lap", "stop_number", "duration_raw"]
        ),
        "sessions": pd.DataFrame(
            {"season": [2023], "round": [1], "circuit_id": ["bahrain"], "total_laps": [57]}
        ),
    }
    results = run_validation(intermediate, test_config)
    assert has_errors(results)
    assert (test_config.paths.quality_dir / "validation_report.json").exists()
    assert (test_config.paths.quality_dir / "validation_report.md").exists()
