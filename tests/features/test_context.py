from __future__ import annotations

import pandas as pd
import pytest

from mpp.features.context import add_race_fraction


def test_race_fraction_is_lap_over_total_laps():
    laps = pd.DataFrame(
        {"season": [2023, 2023], "round": [1, 1], "lap_number": pd.array([1, 57], dtype="Int64")}
    )
    sessions = pd.DataFrame({"season": [2023], "round": [1], "total_laps": [57]})
    result = add_race_fraction(laps, sessions)
    assert result.loc[0, "race_fraction"] == pytest.approx(1 / 57)
    assert result.loc[1, "race_fraction"] == 1.0


def test_race_fraction_uses_correct_total_laps_per_round():
    laps = pd.DataFrame(
        {"season": [2023, 2023], "round": [1, 2], "lap_number": pd.array([10, 10], dtype="Int64")}
    )
    sessions = pd.DataFrame({"season": [2023, 2023], "round": [1, 2], "total_laps": [50, 20]})
    result = add_race_fraction(laps, sessions)
    round1 = result[result["round"] == 1].iloc[0]
    round2 = result[result["round"] == 2].iloc[0]
    assert round1["race_fraction"] == 0.2
    assert round2["race_fraction"] == 0.5
