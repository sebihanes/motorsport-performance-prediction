from __future__ import annotations

import pandas as pd

from mpp.datasets.lap_time import FEATURE_COLUMNS, TARGET_COLUMN, build_lap_time_dataset


def _laps_features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
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
                "compound": "SOFT",
                "air_temp": 29.0,
                "track_temp": 40.0,
                "rainfall": False,
                "driver_prior_relative_pace_s": None,
                "lap_time_s": 99.0,
                "is_valid_lap": True,
            },
            {
                "season": 2023,
                "round": 1,
                "circuit_id": "bahrain",
                "driver": "VER",
                "team": "Red Bull Racing",
                "lap_number": 2,
                "race_fraction": 2 / 57,
                "stint": 1,
                "tyre_age": 5,
                "tyre_age_sq": 25.0,
                "compound": "SOFT",
                "air_temp": 29.0,
                "track_temp": 40.0,
                "rainfall": False,
                "driver_prior_relative_pace_s": None,
                "lap_time_s": None,
                "is_valid_lap": False,  # e.g. a pit-in lap
            },
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
