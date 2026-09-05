from __future__ import annotations

import pandas as pd

from mpp.datasets.pit_stop import build_pit_stop_dataset


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
            }
        ]
    )


def test_pit_stop_dataset_joins_team_and_circuit():
    pitstops = pd.DataFrame(
        [{"season": 2023, "round": 1, "driver": "VER", "lap_number": 14, "stop_number": 1, "duration_s": 24.289}]
    )
    dataset, summary = build_pit_stop_dataset(pitstops, _laps_features())
    assert len(dataset) == 1
    assert dataset.iloc[0]["team"] == "Red Bull Racing"
    assert dataset.iloc[0]["circuit_id"] == "bahrain"
    assert summary["n_unmatched_context_removed"] == 0


def test_pit_stop_without_matching_lap_context_is_dropped_and_counted():
    pitstops = pd.DataFrame(
        [{"season": 2023, "round": 1, "driver": "UNKNOWN_DRIVER", "lap_number": 14, "stop_number": 1, "duration_s": 24.289}]
    )
    dataset, summary = build_pit_stop_dataset(pitstops, _laps_features())
    assert len(dataset) == 0
    assert summary["n_unmatched_context_removed"] == 1


def test_empty_pitstops_returns_empty_dataset():
    dataset, summary = build_pit_stop_dataset(pd.DataFrame(), _laps_features())
    assert len(dataset) == 0
    assert summary["n_rows_out"] == 0
