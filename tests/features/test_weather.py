from __future__ import annotations

import pandas as pd

from mpp.features.weather import add_weather


def test_weather_matches_nearest_sample():
    laps = pd.DataFrame(
        {
            "season": [2023, 2023],
            "round": [1, 1],
            "time_s": [100.0, 400.0],
        }
    )
    weather = pd.DataFrame(
        {
            "season": [2023, 2023],
            "round": [1, 1],
            "Time": pd.to_timedelta([90, 390], unit="s"),
            "AirTemp": [29.0, 31.0],
            "TrackTemp": [40.0, 44.0],
            "Rainfall": [False, False],
        }
    )
    result = add_weather(laps, weather)
    assert result.loc[0, "air_temp"] == 29.0
    assert result.loc[1, "track_temp"] == 44.0


def test_weather_does_not_leak_across_rounds():
    # Round 2's weather sample is at time_s=100 too (clocks reset each round);
    # round 1's lap at time_s=100 must not match round 2's weather.
    laps = pd.DataFrame({"season": [2023], "round": [1], "time_s": [100.0]})
    weather = pd.DataFrame(
        {
            "season": [2023, 2023],
            "round": [1, 2],
            "Time": pd.to_timedelta([95, 100], unit="s"),
            "AirTemp": [25.0, 999.0],
            "TrackTemp": [35.0, 999.0],
            "Rainfall": [False, False],
        }
    )
    result = add_weather(laps, weather)
    assert result.loc[0, "air_temp"] == 25.0


def test_weather_beyond_tolerance_is_null():
    laps = pd.DataFrame({"season": [2023], "round": [1], "time_s": [10000.0]})
    weather = pd.DataFrame(
        {
            "season": [2023],
            "round": [1],
            "Time": pd.to_timedelta([0], unit="s"),
            "AirTemp": [25.0],
            "TrackTemp": [35.0],
            "Rainfall": [False],
        }
    )
    result = add_weather(laps, weather)
    assert pd.isna(result.loc[0, "air_temp"])
