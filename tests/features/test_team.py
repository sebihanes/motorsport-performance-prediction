from __future__ import annotations

import pandas as pd

from mpp.features.team import compute_team_prior_relative_pace


def _valid_laps() -> pd.DataFrame:
    # Two teams, three rounds. Red Bull is 1s faster than the field median
    # each round (regardless of which of its two drivers set the lap);
    # Ferrari is 1s slower.
    rows = []
    for round_, field_pace in [(1, 90.0), (2, 100.0), (3, 80.0)]:
        for driver in ("VER", "PER"):  # both Red Bull cars
            for lap in range(1, 3):
                rows.append(
                    {
                        "season": 2023,
                        "round": round_,
                        "team": "Red Bull Racing",
                        "driver": driver,
                        "lap_time_s": field_pace - 1.0,
                    }
                )
        for driver in ("LEC", "SAI"):  # both Ferrari cars
            for lap in range(1, 3):
                rows.append(
                    {
                        "season": 2023,
                        "round": round_,
                        "team": "Ferrari",
                        "driver": driver,
                        "lap_time_s": field_pace + 1.0,
                    }
                )
    return pd.DataFrame(rows)


def test_round_one_has_no_team_history():
    result = compute_team_prior_relative_pace(_valid_laps())
    r1 = result[result["round"] == 1].set_index("team")
    assert pd.isna(r1.loc["Red Bull Racing", "team_prior_relative_pace_s"])
    assert pd.isna(r1.loc["Ferrari", "team_prior_relative_pace_s"])


def test_round_two_uses_only_round_one():
    result = compute_team_prior_relative_pace(_valid_laps())
    r2 = result[result["round"] == 2].set_index("team")
    assert r2.loc["Red Bull Racing", "team_prior_relative_pace_s"] == -1.0
    assert r2.loc["Ferrari", "team_prior_relative_pace_s"] == 1.0


def test_team_history_survives_a_driver_change():
    # A mid-season driver swap at Red Bull must not reset the team's history -
    # the feature is keyed on team, not on which driver happened to be in the car.
    laps = _valid_laps()
    laps.loc[(laps["team"] == "Red Bull Racing") & (laps["round"] == 2), "driver"] = "NEWDRIVER"
    result = compute_team_prior_relative_pace(laps)
    r3 = result[(result["round"] == 3) & (result["team"] == "Red Bull Racing")]
    assert r3.iloc[0]["team_prior_relative_pace_s"] == -1.0


def test_future_round_never_leaks_into_an_earlier_rounds_team_feature():
    baseline = compute_team_prior_relative_pace(_valid_laps())
    modified_laps = _valid_laps()
    modified_laps.loc[modified_laps["round"] == 3, "lap_time_s"] += 1000.0
    modified = compute_team_prior_relative_pace(modified_laps)

    for round_ in (1, 2):
        before = baseline[baseline["round"] == round_].set_index("team")["team_prior_relative_pace_s"]
        after = modified[modified["round"] == round_].set_index("team")["team_prior_relative_pace_s"]
        pd.testing.assert_series_equal(before, after)
