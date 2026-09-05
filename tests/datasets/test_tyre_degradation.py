from __future__ import annotations

import pandas as pd

from mpp.datasets.tyre_degradation import build_tyre_degradation_dataset


def _lap(driver, stint, lap_number, tyre_age, lap_time_s, compound="SOFT", is_valid=True):
    return {
        "season": 2023,
        "round": 1,
        "circuit_id": "bahrain",
        "driver": driver,
        "team": "Red Bull Racing",
        "stint": stint,
        "lap_number": lap_number,
        "tyre_age": tyre_age,
        "compound": compound,
        "race_fraction": lap_number / 57,
        "air_temp": 29.0,
        "track_temp": 40.0,
        "lap_time_s": lap_time_s,
        "is_valid_lap": is_valid,
    }


def test_baseline_lap_has_zero_delta_and_subsequent_laps_are_relative():
    laps = pd.DataFrame(
        [
            _lap("VER", 1, 1, 4, 99.0),
            _lap("VER", 1, 2, 5, 98.0),
            _lap("VER", 1, 3, 6, 97.5),
        ]
    )
    dataset, summary = build_tyre_degradation_dataset(laps)
    dataset = dataset.sort_values("tyre_age_delta").reset_index(drop=True)
    assert list(dataset["lap_time_delta_s"]) == [0.0, -1.0, -1.5]
    assert list(dataset["tyre_age_delta"]) == [0.0, 1.0, 2.0]
    assert list(dataset["tyre_age_delta_sq"]) == [0.0, 1.0, 4.0]


def test_short_stints_are_excluded_and_counted():
    laps = pd.DataFrame(
        [
            _lap("VER", 1, 1, 4, 99.0),
            _lap("VER", 1, 2, 5, 98.0),  # only 2 valid laps - below MIN_VALID_LAPS_PER_STINT
        ]
    )
    dataset, summary = build_tyre_degradation_dataset(laps)
    assert len(dataset) == 0
    assert summary["n_laps_excluded_short_stint"] == 2


def test_unknown_compound_laps_are_excluded_and_counted():
    laps = pd.DataFrame(
        [
            _lap("VER", 1, 1, 4, 99.0, compound="UNKNOWN"),
            _lap("VER", 1, 2, 5, 98.0, compound="UNKNOWN"),
            _lap("VER", 1, 3, 6, 97.5, compound="UNKNOWN"),
        ]
    )
    dataset, summary = build_tyre_degradation_dataset(laps)
    assert len(dataset) == 0
    assert summary["n_laps_excluded_unknown_compound"] == 3


def test_invalid_laps_never_reach_the_dataset():
    laps = pd.DataFrame(
        [
            _lap("VER", 1, 1, 4, 99.0),
            _lap("VER", 1, 2, 5, None, is_valid=False),
            _lap("VER", 1, 3, 6, 97.5),
            _lap("VER", 1, 4, 7, 97.0),
        ]
    )
    dataset, summary = build_tyre_degradation_dataset(laps)
    assert len(dataset) == 3  # the invalid lap is dropped, not imputed
