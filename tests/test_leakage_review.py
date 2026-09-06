"""Leakage review: explicit, automated checks that every engineered feature
only ever uses information available at prediction time, and that the
recommended split strategy actually enforces train/test separation the way
docs/05_split_strategy.md claims it does.

This file is deliberately separate from the per-module unit tests
(tests/features/test_*.py already cover each feature's own leakage-safety in
isolation): here the point is to review the *assembled* feature table built
by mpp.features.build.build_laps_features end-to-end, the way a reviewer
would look at the finished product rather than its parts, plus a concrete,
empirical comparison of why a random split is the wrong choice for this data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from mpp.features.build import build_laps_features
from mpp.splits.group_split import race_group_shuffle_split

SESSION_CODE = "R"
VALID_COMPOUNDS = ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN")


def _synthetic_pipeline_inputs(round3_lap_time_shift: float = 0.0) -> tuple[dict, dict]:
    """Three rounds, two drivers (one per team), two circuits (round 3 repeats
    round 1's circuit) - enough structure to exercise every historical
    feature (driver, team, driver-circuit) and the within-stint trend at once.
    """
    rows = []
    circuits = {1: "bahrain", 2: "jeddah", 3: "bahrain"}
    base_pace = {1: 90.0, 2: 100.0, 3: 95.0}
    for round_ in (1, 2, 3):
        pace = base_pace[round_]
        shift = round3_lap_time_shift if round_ == 3 else 0.0
        for driver, team, offset in (("VER", "Red Bull Racing", -1.0), ("LEC", "Ferrari", 1.0)):
            for lap in range(1, 6):
                rows.append(
                    {
                        "season": 2023,
                        "round": round_,
                        "Driver": driver,
                        "DriverNumber": "1",
                        "team": team,
                        "driver": driver,
                        "lap_number": lap,
                        "stint": 1,
                        "lap_time_s": pace + offset + shift,
                        "time_s": lap * 100.0,
                        "compound": "SOFT",
                        "tyre_age": lap,
                        "fresh_tyre": True,
                        "pit_in_lap": False,
                        "pit_out_lap": False,
                        "track_status": "1",
                        "deleted": False,
                        "is_accurate": True,
                        "is_valid_lap": True,
                        "exclusion_reason": None,
                    }
                )
    laps = pd.DataFrame(rows)

    sessions = pd.DataFrame(
        [
            {"season": 2023, "round": r, "circuit_id": c, "total_laps": 5}
            for r, c in circuits.items()
        ]
    )
    weather = pd.DataFrame(
        {
            "season": [2023] * 3,
            "round": [1, 2, 3],
            "Time": pd.to_timedelta([0, 0, 0], unit="s"),
            "AirTemp": [25.0, 25.0, 25.0],
            "TrackTemp": [35.0, 35.0, 35.0],
            "Rainfall": [False, False, False],
        }
    )

    processed = {"laps": laps}
    intermediate = {"sessions": sessions, "weather": weather}
    return processed, intermediate


def test_no_historical_feature_in_an_earlier_round_changes_when_a_later_round_does():
    """The end-to-end regression test: perturb round 3 dramatically and check
    that every historical column mpp.features.build produces is bit-for-bit
    identical for rounds 1 and 2. If any future-information leak existed in
    how build_laps_features wires the driver/team/circuit/performance
    features together, this would catch it even if each feature function
    passed its own isolated unit test.
    """
    processed, intermediate = _synthetic_pipeline_inputs(round3_lap_time_shift=0.0)
    baseline = build_laps_features(processed, intermediate, SESSION_CODE, VALID_COMPOUNDS)

    processed_perturbed, intermediate_perturbed = _synthetic_pipeline_inputs(
        round3_lap_time_shift=10_000.0
    )
    perturbed = build_laps_features(
        processed_perturbed, intermediate_perturbed, SESSION_CODE, VALID_COMPOUNDS
    )

    historical_columns = [
        "driver_prior_relative_pace_s",
        "driver_prior_relative_pace_std_s",
        "team_prior_relative_pace_s",
        "driver_circuit_prior_relative_pace_s",
    ]
    for column in historical_columns:
        before = baseline.loc[baseline["round"] < 3, ["round", "driver", column]].sort_values(
            ["round", "driver"]
        )
        after = perturbed.loc[perturbed["round"] < 3, ["round", "driver", column]].sort_values(
            ["round", "driver"]
        )
        pd.testing.assert_frame_equal(
            before.reset_index(drop=True), after.reset_index(drop=True), check_dtype=False
        )


def test_stint_prior_pace_trend_is_unaffected_by_a_later_lap_in_the_same_stint():
    processed, intermediate = _synthetic_pipeline_inputs()
    baseline = build_laps_features(processed, intermediate, SESSION_CODE, VALID_COMPOUNDS)

    processed_perturbed, intermediate_perturbed = _synthetic_pipeline_inputs()
    # Blow up round 1 VER's final lap - must not affect that same stint's
    # earlier laps' trend values.
    mask = (
        (processed_perturbed["laps"]["round"] == 1)
        & (processed_perturbed["laps"]["driver"] == "VER")
        & (processed_perturbed["laps"]["lap_number"] == 5)
    )
    processed_perturbed["laps"].loc[mask, "lap_time_s"] += 10_000.0
    perturbed = build_laps_features(
        processed_perturbed, intermediate_perturbed, SESSION_CODE, VALID_COMPOUNDS
    )

    earlier_laps = (baseline["round"] == 1) & (baseline["driver"] == "VER") & (baseline["lap_number"] < 5)
    before = baseline.loc[earlier_laps, "stint_prior_mean_lap_time_s"].reset_index(drop=True)
    after = perturbed.loc[earlier_laps, "stint_prior_mean_lap_time_s"].reset_index(drop=True)
    pd.testing.assert_series_equal(before, after)


def test_no_feature_column_is_a_disguised_copy_of_the_target():
    """A cheap but real check: none of the engineered feature columns should
    be perfectly, trivially derived from the row's own lap_time_s (the
    target) - e.g. accidentally joining a per-row aggregate that includes the
    row itself. driver_prior_relative_pace_s etc. are round-level and must
    not vary lap-to-lap within the same driver-round; stint_prior_mean is
    expected to vary (and does, since it is a *different* row's average).
    """
    processed, intermediate = _synthetic_pipeline_inputs()
    laps_features = build_laps_features(processed, intermediate, SESSION_CODE, VALID_COMPOUNDS)

    round_level_columns = [
        "driver_prior_relative_pace_s",
        "driver_prior_relative_pace_std_s",
        "team_prior_relative_pace_s",
        "driver_circuit_prior_relative_pace_s",
    ]
    for column in round_level_columns:
        per_driver_round_nunique = laps_features.groupby(["season", "round", "driver"])[column].nunique(
            dropna=False
        )
        assert (per_driver_round_nunique <= 1).all(), f"{column} varies within a single driver-round"


# --- Split-strategy comparison: random vs. grouped -------------------------


def _lap_level_dataset() -> pd.DataFrame:
    rows = []
    for round_ in (1, 2, 3, 4):
        for lap in range(1, 21):  # 20 laps per race, like a real stint-heavy sample
            rows.append({"season": 2023, "round": round_, "lap_number": lap})
    return pd.DataFrame(rows)


def test_random_split_leaks_races_across_train_and_test():
    """The concrete demonstration of why random splitting is the wrong choice
    for this data (docs/05_split_strategy.md): with laps this correlated
    within a race, a naive random split puts most races on both sides of the
    train/test boundary.
    """
    dataset = _lap_level_dataset()
    train_idx, test_idx = train_test_split(
        np.arange(len(dataset)), test_size=0.25, random_state=0, shuffle=True
    )
    train_races = set(dataset.iloc[train_idx]["round"])
    test_races = set(dataset.iloc[test_idx]["round"])
    assert train_races & test_races, (
        "expected the naive random split to leak at least one race across "
        "train and test - if this fails, the demonstration dataset needs revisiting"
    )


def test_group_split_fixes_the_leak_the_random_split_demonstrated():
    dataset = _lap_level_dataset()
    train_idx, test_idx = race_group_shuffle_split(dataset, test_size=0.25, random_state=0)
    train_races = set(dataset.iloc[train_idx]["round"])
    test_races = set(dataset.iloc[test_idx]["round"])
    assert train_races.isdisjoint(test_races)
