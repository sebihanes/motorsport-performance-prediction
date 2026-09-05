"""Race pace dataset (architecture SS3.3, SS7).

Row = one driver's session. Target = session_pace_s, the median of that
driver's clean racing laps - a median, not a mean, so a handful of outlier
laps (traffic, minor incidents) don't distort the pace estimate.

Round 1 of the configured scope has no prior-round history for any driver,
so driver_prior_relative_pace_s is null there by construction (see
mpp.features.historical) - documented, not imputed away.
"""

from __future__ import annotations

import pandas as pd

GROUP_COLUMNS = ["season", "round", "circuit_id"]
IDENTITY_COLUMNS = ["driver", "team"]
FEATURE_COLUMNS = ["driver_prior_relative_pace_s", "air_temp_mean", "track_temp_mean", "rainfall_any"]
TARGET_COLUMN = "session_pace_s"


def build_race_pace_dataset(laps_features: pd.DataFrame) -> pd.DataFrame:
    valid = laps_features.loc[laps_features["is_valid_lap"]].copy()

    grouped = valid.groupby(GROUP_COLUMNS + IDENTITY_COLUMNS, as_index=False).agg(
        session_pace_s=("lap_time_s", "median"),
        driver_prior_relative_pace_s=("driver_prior_relative_pace_s", "first"),
        air_temp_mean=("air_temp", "mean"),
        track_temp_mean=("track_temp", "mean"),
        rainfall_any=("rainfall", "max"),
    )

    columns = GROUP_COLUMNS + IDENTITY_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]
    return grouped[columns].reset_index(drop=True)
