from __future__ import annotations

import pandas as pd
import pytest

from mpp.features.performance import add_stint_prior_pace_trend

STINT_KEY = ["season", "round", "driver", "stint"]


def _stint_laps(lap_times: list[float], valid: list[bool] | None = None) -> pd.DataFrame:
    valid = valid or [True] * len(lap_times)
    return pd.DataFrame(
        [
            {
                "season": 2023,
                "round": 1,
                "driver": "VER",
                "stint": 1,
                "lap_number": i + 1,
                "lap_time_s": lt,
                "is_valid_lap": v,
            }
            for i, (lt, v) in enumerate(zip(lap_times, valid))
        ]
    )


def test_first_lap_of_a_stint_has_no_prior_trend():
    result = add_stint_prior_pace_trend(_stint_laps([99.0, 98.0, 97.5]))
    assert pd.isna(result.iloc[0]["stint_prior_mean_lap_time_s"])


def test_trend_is_expanding_mean_of_strictly_prior_laps():
    result = add_stint_prior_pace_trend(_stint_laps([99.0, 98.0, 97.5]))
    assert result.iloc[1]["stint_prior_mean_lap_time_s"] == 99.0
    assert result.iloc[2]["stint_prior_mean_lap_time_s"] == pytest.approx((99.0 + 98.0) / 2)


def test_a_later_lap_never_leaks_into_an_earlier_rows_trend():
    baseline = add_stint_prior_pace_trend(_stint_laps([99.0, 98.0, 97.5]))
    modified = add_stint_prior_pace_trend(_stint_laps([99.0, 98.0, 5000.0]))
    pd.testing.assert_series_equal(
        baseline.iloc[:2]["stint_prior_mean_lap_time_s"],
        modified.iloc[:2]["stint_prior_mean_lap_time_s"],
    )


def test_invalid_laps_are_excluded_from_the_trend_and_receive_no_value():
    result = add_stint_prior_pace_trend(
        _stint_laps([99.0, None, 97.5], valid=[True, False, True])
    )
    # lap 2 is invalid: it gets no trend value of its own...
    assert pd.isna(result.iloc[1]["stint_prior_mean_lap_time_s"])
    # ...and lap 3's trend is based only on lap 1 (the only valid prior lap), not lap 2.
    assert result.iloc[2]["stint_prior_mean_lap_time_s"] == 99.0


def test_different_stints_do_not_share_a_trend():
    df = pd.concat(
        [
            _stint_laps([99.0, 98.0]),
            _stint_laps([50.0, 49.0]).assign(stint=2, lap_number=lambda d: d["lap_number"] + 2),
        ],
        ignore_index=True,
    )
    result = add_stint_prior_pace_trend(df)
    stint2_first_lap = result[(result["stint"] == 2)].iloc[0]
    assert pd.isna(stint2_first_lap["stint_prior_mean_lap_time_s"])
