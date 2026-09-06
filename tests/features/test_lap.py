from __future__ import annotations

import pandas as pd

from mpp.features.lap import add_stint_relative_lap_number


def test_stint_relative_lap_number_starts_at_one():
    df = pd.DataFrame(
        [
            {"season": 2023, "round": 1, "driver": "VER", "stint": 1, "lap_number": 12},
            {"season": 2023, "round": 1, "driver": "VER", "stint": 1, "lap_number": 13},
            {"season": 2023, "round": 1, "driver": "VER", "stint": 1, "lap_number": 14},
        ]
    )
    result = add_stint_relative_lap_number(df)
    assert list(result["laps_into_stint"]) == [1, 2, 3]


def test_stint_relative_lap_number_differs_from_tyre_age_style_offset():
    # A stint that starts on tyres already worn from pre-race laps (tyre_age
    # starts at, say, 4) must still count "laps into this stint" from 1 - the
    # two concepts are independent (see mpp.features.lap module docstring).
    df = pd.DataFrame(
        [
            {"season": 2023, "round": 1, "driver": "VER", "stint": 1, "lap_number": 1, "tyre_age": 4},
            {"season": 2023, "round": 1, "driver": "VER", "stint": 1, "lap_number": 2, "tyre_age": 5},
        ]
    )
    result = add_stint_relative_lap_number(df)
    assert list(result["laps_into_stint"]) == [1, 2]
    assert list(result["tyre_age"]) == [4, 5]


def test_each_stint_restarts_the_count():
    df = pd.DataFrame(
        [
            {"season": 2023, "round": 1, "driver": "VER", "stint": 1, "lap_number": 1},
            {"season": 2023, "round": 1, "driver": "VER", "stint": 1, "lap_number": 2},
            {"season": 2023, "round": 1, "driver": "VER", "stint": 2, "lap_number": 3},
            {"season": 2023, "round": 1, "driver": "VER", "stint": 2, "lap_number": 4},
        ]
    )
    result = add_stint_relative_lap_number(df)
    assert list(result["laps_into_stint"]) == [1, 2, 1, 2]


def test_different_drivers_and_rounds_do_not_interfere():
    df = pd.DataFrame(
        [
            {"season": 2023, "round": 1, "driver": "VER", "stint": 1, "lap_number": 5},
            {"season": 2023, "round": 1, "driver": "LEC", "stint": 1, "lap_number": 5},
        ]
    )
    result = add_stint_relative_lap_number(df)
    assert list(result["laps_into_stint"]) == [1, 1]
