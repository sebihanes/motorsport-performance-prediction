"""Leakage-safe historical driver-performance features.

driver_prior_relative_pace_s: how far above/below the field's median race
pace a driver has been, averaged over that driver's rounds *strictly before*
the round being featured (an expanding mean, never including the current or
a future round). Expressed as a delta (seconds), not an absolute lap time, so
it is comparable across circuits with very different absolute lap times.

driver_prior_relative_pace_std_s: how *consistent* that same relative pace
has been historically (an expanding standard deviation, same strictly-prior
rule) - the "historical consistency" feature architecture SS2/Phase 3 calls
for. Needs at least two prior rounds to be defined, so it is NaN for rounds 1
and 2 of any scope, one round later than the mean feature.

Round 1 of the configured scope has no prior rounds for any driver, so both
features are NaN there by construction - this is a real limitation of a
narrow season scope, not a bug (see docs/02_data_pipeline.md "Limitations").
"""

from __future__ import annotations

import pandas as pd

from mpp.features.common import add_expanding_prior_stat, driver_round_relative_pace


def compute_driver_prior_relative_pace(valid_laps: pd.DataFrame) -> pd.DataFrame:
    """valid_laps must contain only rows that should count towards pace (is_valid_lap==True).

    Returns one row per (season, round, driver) with driver_prior_relative_pace_s.
    """
    pace = driver_round_relative_pace(valid_laps)
    pace = add_expanding_prior_stat(
        pace, ["driver"], "round", "relative_pace_s", "driver_prior_relative_pace_s", stat="mean"
    )
    return pace[["season", "round", "driver", "driver_prior_relative_pace_s"]]


def compute_driver_prior_relative_pace_std(valid_laps: pd.DataFrame) -> pd.DataFrame:
    """Returns one row per (season, round, driver) with driver_prior_relative_pace_std_s."""
    pace = driver_round_relative_pace(valid_laps)
    pace = add_expanding_prior_stat(
        pace,
        ["driver"],
        "round",
        "relative_pace_s",
        "driver_prior_relative_pace_std_s",
        stat="std",
    )
    return pace[["season", "round", "driver", "driver_prior_relative_pace_std_s"]]
