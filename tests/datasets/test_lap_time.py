from __future__ import annotations

import pandas as pd

from mpp.datasets.lap_time import FEATURE_COLUMNS, TARGET_COLUMN, build_lap_time_dataset


def _base_row(**overrides) -> dict:
    row = {
        "season": 2023,
        "round": 1,
        "circuit_id": "bahrain",
        "driver": "VER",
        "team": "Red Bull Racing",
        "lap_number": 1,
        "race_fraction": 1 / 57,
        "stint": 1,
        "tyre_age": 4,
        "tyre_age_sq": 16.0,
        "laps_into_stint": 1,
        "compound": "SOFT",
        "session_type": "R",
        "air_temp": 29.0,
        "track_temp": 40.0,
        "rainfall": False,
        "driver_prior_relative_pace_s": None,
        "driver_prior_relative_pace_std_s": None,
        "team_prior_relative_pace_s": None,
        "driver_circuit_prior_relative_pace_s": None,
        "stint_prior_mean_lap_time_s": None,
        "tyre_age_x_soft": 4.0,
        "tyre_age_x_medium": 0.0,
        "tyre_age_x_hard": 0.0,
        "tyre_age_x_intermediate": 0.0,
        "tyre_age_x_wet": 0.0,
        "lap_time_s": 99.0,
        "is_valid_lap": True,
    }
    row.update(overrides)
    return row


def _laps_features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _base_row(),
            _base_row(
                lap_number=2,
                race_fraction=2 / 57,
                tyre_age=5,
                tyre_age_sq=25.0,
                laps_into_stint=2,
                tyre_age_x_soft=5.0,
                lap_time_s=None,
                is_valid_lap=False,  # e.g. a pit-in lap
            ),
        ]
    )


def test_only_valid_laps_are_kept():
    dataset = build_lap_time_dataset(_laps_features())
    assert len(dataset) == 1
    assert dataset.iloc[0]["lap_number"] == 1


def test_output_columns_match_definition():
    dataset = build_lap_time_dataset(_laps_features())
    expected = {"season", "round", "circuit_id", "driver", "team", TARGET_COLUMN, *FEATURE_COLUMNS}
    assert set(dataset.columns) == expected
