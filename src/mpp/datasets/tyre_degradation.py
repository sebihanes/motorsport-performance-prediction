"""Tyre degradation dataset (architecture SS3.2, SS7).

Row = one clean racing lap within a stint that has a usable baseline. Target
= lap_time_delta_s, the lap time relative to that stint's first clean lap -
this isolates within-stint degradation from cross-stint/cross-circuit
absolute pace differences, at the cost of conflating tyre wear with fuel burn
within the stint (see docs/ARCHITECTURE_REPORT.md SS18, the known
fuel-load/tyre-wear confound this project does not attempt to resolve).

A stint is excluded entirely (not silently truncated) if it has fewer than
MIN_VALID_LAPS_PER_STINT clean laps, or an UNKNOWN compound - both counted
in the returned exclusion summary for traceability.
"""

from __future__ import annotations

import pandas as pd

MIN_VALID_LAPS_PER_STINT = 3

GROUP_COLUMNS = ["season", "round", "circuit_id"]
STINT_KEY = ["season", "round", "driver", "stint"]
IDENTITY_COLUMNS = ["driver", "team"]
NUMERIC_FEATURE_COLUMNS = [
    "tyre_age_delta",
    "tyre_age_delta_sq",
    "laps_into_stint",
    "race_fraction",
    "air_temp",
    "track_temp",
    "driver_prior_relative_pace_s",
    "driver_prior_relative_pace_std_s",
    "team_prior_relative_pace_s",
    "driver_circuit_prior_relative_pace_s",
]
NOMINAL_FEATURE_COLUMNS = ["compound"]
FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + NOMINAL_FEATURE_COLUMNS
TARGET_COLUMN = "lap_time_delta_s"


def build_tyre_degradation_dataset(laps_features: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    valid_laps_only = laps_features.loc[laps_features["is_valid_lap"]].copy()
    n_excluded_unknown_compound = int((valid_laps_only["compound"] == "UNKNOWN").sum())
    valid = valid_laps_only.loc[valid_laps_only["compound"] != "UNKNOWN"].copy()

    stint_sizes = valid.groupby(STINT_KEY)["lap_number"].transform("size")
    n_excluded_short_stint = int((stint_sizes < MIN_VALID_LAPS_PER_STINT).sum())
    valid = valid.loc[stint_sizes >= MIN_VALID_LAPS_PER_STINT].copy()

    baseline_idx = valid.groupby(STINT_KEY)["lap_number"].idxmin()
    baseline = valid.loc[
        baseline_idx, STINT_KEY + ["lap_time_s", "tyre_age"]
    ].rename(columns={"lap_time_s": "baseline_lap_time_s", "tyre_age": "baseline_tyre_age"})

    valid = valid.merge(baseline, on=STINT_KEY, how="left")
    valid["lap_time_delta_s"] = valid["lap_time_s"] - valid["baseline_lap_time_s"]
    valid["tyre_age_delta"] = valid["tyre_age"].astype("Float64") - valid[
        "baseline_tyre_age"
    ].astype("Float64")
    valid["tyre_age_delta_sq"] = valid["tyre_age_delta"] ** 2

    columns = GROUP_COLUMNS + IDENTITY_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]
    dataset = valid[columns].reset_index(drop=True)

    summary = {
        "n_laps_excluded_unknown_compound": n_excluded_unknown_compound,
        "n_laps_excluded_short_stint": n_excluded_short_stint,
        "min_valid_laps_per_stint": MIN_VALID_LAPS_PER_STINT,
        "n_rows_out": len(dataset),
    }
    return dataset, summary
