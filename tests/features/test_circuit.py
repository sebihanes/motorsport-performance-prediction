from __future__ import annotations

import pandas as pd

from mpp.features.circuit import compute_driver_circuit_prior_relative_pace


def _sessions() -> pd.DataFrame:
    # Circuit "monza" repeats at round 3; every other round is a fresh circuit.
    return pd.DataFrame(
        [
            {"season": 2023, "round": 1, "circuit_id": "bahrain"},
            {"season": 2023, "round": 2, "circuit_id": "monza"},
            {"season": 2023, "round": 3, "circuit_id": "monza"},
        ]
    )


def _valid_laps() -> pd.DataFrame:
    rows = []
    for round_, field_pace in [(1, 90.0), (2, 100.0), (3, 100.0)]:
        rows.append({"season": 2023, "round": round_, "driver": "VER", "lap_time_s": field_pace - 1.0})
        rows.append({"season": 2023, "round": round_, "driver": "LEC", "lap_time_s": field_pace + 1.0})
    return pd.DataFrame(rows)


def test_no_history_on_a_circuit_visited_for_the_first_time():
    result = compute_driver_circuit_prior_relative_pace(_valid_laps(), _sessions())
    row = result[(result["round"] == 2) & (result["driver"] == "VER")].iloc[0]
    assert pd.isna(row["driver_circuit_prior_relative_pace_s"])


def test_history_populates_on_a_repeat_visit_to_the_same_circuit():
    result = compute_driver_circuit_prior_relative_pace(_valid_laps(), _sessions())
    row = result[(result["round"] == 3) & (result["driver"] == "VER")].iloc[0]
    # VER's only prior visit to "monza" was round 2, relative pace -1.0 there.
    assert row["driver_circuit_prior_relative_pace_s"] == -1.0


def test_a_different_circuit_never_contributes_to_the_history():
    # Round 1 (bahrain) must not feed into round 3's monza-specific history,
    # even though it is an earlier round for the same driver.
    laps = _valid_laps()
    laps.loc[laps["round"] == 1, "lap_time_s"] += 1000.0
    result = compute_driver_circuit_prior_relative_pace(laps, _sessions())
    row = result[(result["round"] == 3) & (result["driver"] == "VER")].iloc[0]
    assert row["driver_circuit_prior_relative_pace_s"] == -1.0


def test_current_scope_with_no_repeated_circuits_is_all_null():
    # Regression check for the documented current-scope limitation: with
    # season 2023 rounds 1-6 (each a different circuit), this feature is null
    # for every row - simulated here with three distinct circuits.
    sessions = pd.DataFrame(
        [
            {"season": 2023, "round": 1, "circuit_id": "bahrain"},
            {"season": 2023, "round": 2, "circuit_id": "jeddah"},
            {"season": 2023, "round": 3, "circuit_id": "albert_park"},
        ]
    )
    result = compute_driver_circuit_prior_relative_pace(_valid_laps(), sessions)
    assert result["driver_circuit_prior_relative_pace_s"].isna().all()
