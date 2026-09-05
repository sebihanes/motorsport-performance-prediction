"""Lap-time prediction dataset (architecture SS3.1, SS7).

Row = one clean racing lap. Target = lap_time_s.
"""

from __future__ import annotations

import pandas as pd

GROUP_COLUMNS = ["season", "round", "circuit_id"]
IDENTITY_COLUMNS = ["driver", "team"]
FEATURE_COLUMNS = [
    "lap_number",
    "race_fraction",
    "stint",
    "tyre_age",
    "tyre_age_sq",
    "compound",
    "air_temp",
    "track_temp",
    "rainfall",
    "driver_prior_relative_pace_s",
]
TARGET_COLUMN = "lap_time_s"


def build_lap_time_dataset(laps_features: pd.DataFrame) -> pd.DataFrame:
    valid = laps_features.loc[laps_features["is_valid_lap"]].copy()
    columns = GROUP_COLUMNS + IDENTITY_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]
    return valid[columns].reset_index(drop=True)
