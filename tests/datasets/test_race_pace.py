from __future__ import annotations

import pandas as pd

from mpp.datasets.race_pace import FEATURE_COLUMNS, TARGET_COLUMN, build_race_pace_dataset


def _lap(driver, lap_number, lap_time_s, rainfall=False, is_valid=True):
    return {
        "season": 2023,
        "round": 1,
        "circuit_id": "bahrain",
        "driver": driver,
        "team": "Red Bull Racing",
        "lap_number": lap_number,
        "lap_time_s": lap_time_s,
        "air_temp": 29.0,
        "track_temp": 40.0,
        "rainfall": rainfall,
        "driver_prior_relative_pace_s": -0.5,
        "driver_prior_relative_pace_std_s": 0.3,
        "team_prior_relative_pace_s": -0.4,
        "driver_circuit_prior_relative_pace_s": None,
        "is_valid_lap": is_valid,
    }


def test_session_pace_is_the_median_of_valid_laps():
    laps = pd.DataFrame(
        [
            _lap("VER", 1, 99.0),
            _lap("VER", 2, 98.0),
            _lap("VER", 3, 200.0),  # outlier - median should not be dragged by it
        ]
    )
    dataset = build_race_pace_dataset(laps)
    assert dataset.iloc[0]["session_pace_s"] == 99.0


def test_invalid_laps_are_excluded_from_the_pace_calculation():
    laps = pd.DataFrame(
        [
            _lap("VER", 1, 99.0),
            _lap("VER", 2, None, is_valid=False),
        ]
    )
    dataset = build_race_pace_dataset(laps)
    assert dataset.iloc[0]["session_pace_s"] == 99.0


def test_rainfall_any_is_true_if_any_valid_lap_was_wet():
    laps = pd.DataFrame(
        [
            _lap("VER", 1, 99.0, rainfall=False),
            _lap("VER", 2, 98.0, rainfall=True),
        ]
    )
    dataset = build_race_pace_dataset(laps)
    assert bool(dataset.iloc[0]["rainfall_any"])


def test_one_row_per_driver_session():
    laps = pd.DataFrame(
        [
            _lap("VER", 1, 99.0),
            _lap("VER", 2, 98.0),
            _lap("LEC", 1, 101.0),
        ]
    )
    dataset = build_race_pace_dataset(laps)
    assert len(dataset) == 2
    assert set(dataset["driver"]) == {"VER", "LEC"}


def test_output_columns_match_definition():
    laps = pd.DataFrame([_lap("VER", 1, 99.0)])
    dataset = build_race_pace_dataset(laps)
    expected = {"season", "round", "circuit_id", "driver", "team", TARGET_COLUMN, *FEATURE_COLUMNS}
    assert set(dataset.columns) == expected
