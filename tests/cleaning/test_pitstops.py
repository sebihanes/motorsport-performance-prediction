from __future__ import annotations

import json

import pandas as pd

from mpp.cleaning.pitstops import clean_pitstops, parse_duration_to_seconds


def test_parse_plain_seconds():
    assert parse_duration_to_seconds("25.885") == 25.885


def test_parse_minutes_seconds():
    assert parse_duration_to_seconds("1:05.123") == 65.123


def test_parse_none_returns_none():
    assert parse_duration_to_seconds(None) is None


def test_parse_empty_string_returns_none():
    assert parse_duration_to_seconds("") is None


def test_parse_garbage_returns_none():
    assert parse_duration_to_seconds("not-a-duration") is None


def _drivers_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"season": 2023, "round": 1, "driver_id": "max_verstappen", "driver_code": "VER"},
            {"season": 2023, "round": 1, "driver_id": "leclerc", "driver_code": "LEC"},
        ]
    )


def test_clean_pitstops_parses_duration_and_maps_driver_code(test_config):
    pitstops = pd.DataFrame(
        [
            {
                "season": 2023,
                "round": 1,
                "driver_id": "max_verstappen",
                "lap": 14,
                "stop_number": 1,
                "duration_raw": "24.289",
            }
        ]
    )
    cleaned, log = clean_pitstops(pitstops, _drivers_df(), test_config)
    assert len(cleaned) == 1
    assert cleaned.loc[0, "driver"] == "VER"
    assert cleaned.loc[0, "duration_s"] == 24.289
    assert cleaned.loc[0, "lap_number"] == 14


def test_clean_pitstops_drops_unparseable_duration(test_config):
    pitstops = pd.DataFrame(
        [
            {
                "season": 2023,
                "round": 1,
                "driver_id": "max_verstappen",
                "lap": 14,
                "stop_number": 1,
                "duration_raw": "n/a",
            }
        ]
    )
    cleaned, log = clean_pitstops(pitstops, _drivers_df(), test_config)
    assert len(cleaned) == 0
    assert log["unparseable_duration_rows_removed"] == 1


def test_clean_pitstops_drops_unmapped_driver(test_config):
    pitstops = pd.DataFrame(
        [
            {
                "season": 2023,
                "round": 1,
                "driver_id": "unknown_driver",
                "lap": 14,
                "stop_number": 1,
                "duration_raw": "24.289",
            }
        ]
    )
    cleaned, log = clean_pitstops(pitstops, _drivers_df(), test_config)
    assert len(cleaned) == 0
    assert log["unmapped_driver_rows_removed"] == 1


def test_clean_pitstops_handles_empty_input(test_config):
    cleaned, log = clean_pitstops(pd.DataFrame(), _drivers_df(), test_config)
    assert len(cleaned) == 0
    assert log["n_rows_out"] == 0


def test_clean_pitstops_on_real_data_slice(test_config, fixtures_dir):
    pitstops = pd.DataFrame(json.loads((fixtures_dir / "real_pitstops_slice_2023_r1.json").read_text()))
    drivers = pd.DataFrame(json.loads((fixtures_dir / "real_drivers_slice_2023_r1.json").read_text()))
    cleaned, log = clean_pitstops(pitstops, drivers, test_config)
    assert len(cleaned) == 2
    assert set(cleaned["driver"]) == {"VER", "LEC"}
    assert log["unparseable_duration_rows_removed"] == 0
    assert log["unmapped_driver_rows_removed"] == 0
