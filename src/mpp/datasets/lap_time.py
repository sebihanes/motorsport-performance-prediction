"""Lap-time prediction dataset (architecture SS3.1, SS7).

Row = one clean racing lap. Target = lap_time_s.
"""

from __future__ import annotations

import pandas as pd

GROUP_COLUMNS = ["season", "round", "circuit_id"]
IDENTITY_COLUMNS = ["driver", "team"]

TYRE_AGE_INTERACTION_COLUMNS = [
    "tyre_age_x_soft",
    "tyre_age_x_medium",
    "tyre_age_x_hard",
    "tyre_age_x_intermediate",
    "tyre_age_x_wet",
]
NUMERIC_FEATURE_COLUMNS = [
    "lap_number",
    "race_fraction",
    "stint",
    "tyre_age",
    "tyre_age_sq",
    "laps_into_stint",
    "air_temp",
    "track_temp",
    "rainfall",
    "driver_prior_relative_pace_s",
    "driver_prior_relative_pace_std_s",
    "team_prior_relative_pace_s",
    "driver_circuit_prior_relative_pace_s",
    "stint_prior_mean_lap_time_s",
] + TYRE_AGE_INTERACTION_COLUMNS
NOMINAL_FEATURE_COLUMNS = ["compound", "session_type"]
FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + NOMINAL_FEATURE_COLUMNS
TARGET_COLUMN = "lap_time_s"


def build_lap_time_dataset(laps_features: pd.DataFrame) -> pd.DataFrame:
    valid = laps_features.loc[laps_features["is_valid_lap"]].copy()
    columns = GROUP_COLUMNS + IDENTITY_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]
    return valid[columns].reset_index(drop=True)
