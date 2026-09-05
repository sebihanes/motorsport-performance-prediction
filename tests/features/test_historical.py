from __future__ import annotations

import pandas as pd

from mpp.features.historical import compute_driver_prior_relative_pace


def _valid_laps() -> pd.DataFrame:
    # Two drivers, three rounds. VER is 1s faster than the field median each
    # round; LEC is 1s slower. Field median lap time also changes per round
    # (different circuits), which the relative-pace feature must normalize away.
    rows = []
    for round_, field_pace in [(1, 90.0), (2, 100.0), (3, 80.0)]:
        for lap in range(1, 4):
            rows.append(
                {"season": 2023, "round": round_, "driver": "VER", "lap_time_s": field_pace - 1.0}
            )
            rows.append(
                {"season": 2023, "round": round_, "driver": "LEC", "lap_time_s": field_pace + 1.0}
            )
    return pd.DataFrame(rows)


def test_round_one_has_no_history():
    result = compute_driver_prior_relative_pace(_valid_laps())
    r1 = result[result["round"] == 1].set_index("driver")
    assert pd.isna(r1.loc["VER", "driver_prior_relative_pace_s"])
    assert pd.isna(r1.loc["LEC", "driver_prior_relative_pace_s"])


def test_round_two_uses_only_round_one():
    result = compute_driver_prior_relative_pace(_valid_laps())
    r2 = result[result["round"] == 2].set_index("driver")
    assert r2.loc["VER", "driver_prior_relative_pace_s"] == -1.0
    assert r2.loc["LEC", "driver_prior_relative_pace_s"] == 1.0


def test_round_three_averages_rounds_one_and_two():
    result = compute_driver_prior_relative_pace(_valid_laps())
    r3 = result[result["round"] == 3].set_index("driver")
    # Both prior rounds gave VER a relative pace of -1.0, so the expanding mean is -1.0.
    assert r3.loc["VER", "driver_prior_relative_pace_s"] == -1.0


def test_feature_normalizes_across_different_field_paces():
    # Field pace differs a lot between rounds (90s vs 100s vs 80s circuits),
    # but VER's relative pace is a constant -1.0 in every round - the feature
    # must reflect that constant, not the swinging absolute lap time.
    result = compute_driver_prior_relative_pace(_valid_laps())
    ver_history = result[(result["driver"] == "VER") & (result["round"] > 1)]
    assert (ver_history["driver_prior_relative_pace_s"] == -1.0).all()


def test_future_round_never_leaks_into_an_earlier_rounds_feature():
    baseline = compute_driver_prior_relative_pace(_valid_laps())
    modified_laps = _valid_laps()
    # Blow up round 3's lap times - if this changes round 1 or 2's feature,
    # there is a leakage bug.
    modified_laps.loc[modified_laps["round"] == 3, "lap_time_s"] += 1000.0
    modified = compute_driver_prior_relative_pace(modified_laps)

    for round_ in (1, 2):
        before = baseline[baseline["round"] == round_].set_index("driver")[
            "driver_prior_relative_pace_s"
        ]
        after = modified[modified["round"] == round_].set_index("driver")[
            "driver_prior_relative_pace_s"
        ]
        pd.testing.assert_series_equal(before, after)
