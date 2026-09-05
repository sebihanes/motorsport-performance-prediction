"""Leakage-safe historical driver-performance feature.

driver_prior_relative_pace_s: how far above/below the field's median race
pace a driver has been, averaged over that driver's rounds *strictly before*
the round being featured (an expanding mean, never including the current or
a future round). Expressed as a delta (seconds), not an absolute lap time, so
it is comparable across circuits with very different absolute lap times.

Round 1 of the configured scope has no prior rounds for any driver, so the
feature is NaN there by construction - this is a real limitation of a
narrow season scope, not a bug (see docs/02_data_pipeline.md "Limitations").
"""

from __future__ import annotations

import pandas as pd


def compute_driver_prior_relative_pace(valid_laps: pd.DataFrame) -> pd.DataFrame:
    """valid_laps must contain only rows that should count towards pace (is_valid_lap==True).

    Returns one row per (season, round, driver) with driver_prior_relative_pace_s.
    """
    driver_round_pace = (
        valid_laps.groupby(["season", "round", "driver"])["lap_time_s"]
        .median()
        .reset_index(name="driver_pace_s")
    )
    field_round_pace = (
        driver_round_pace.groupby(["season", "round"])["driver_pace_s"]
        .median()
        .reset_index(name="field_pace_s")
    )
    merged = driver_round_pace.merge(field_round_pace, on=["season", "round"])
    merged["relative_pace_s"] = merged["driver_pace_s"] - merged["field_pace_s"]
    merged = merged.sort_values(["driver", "round"]).reset_index(drop=True)

    merged["driver_prior_relative_pace_s"] = merged.groupby("driver")["relative_pace_s"].transform(
        lambda s: s.shift(1).expanding().mean()
    )

    return merged[["season", "round", "driver", "driver_prior_relative_pace_s"]]
