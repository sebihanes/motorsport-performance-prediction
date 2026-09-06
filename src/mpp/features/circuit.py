"""Leakage-safe historical driver-at-this-circuit feature.

driver_circuit_prior_relative_pace_s: a driver's historical relative pace
specifically at this circuit, averaged over that driver's *prior visits to
this same circuit* (strictly earlier rounds, same construction as
mpp.features.historical, grouped by (driver, circuit) instead of driver
alone). This is "normalized historical performance" for the CIRCUIT feature
group in the architecture report / Phase 3 brief: normalized because it is a
relative-pace delta, comparable across circuits, not a raw lap time.

Circuit static characteristics (length, corner count, elevation) are not
included here: neither FastF1 nor Jolpica-F1 provides them, and inventing
hand-typed numbers would violate the "real data only" rule (see
docs/01_data_sources.md). circuit_id itself is joined onto the feature table
separately in mpp.features.build, from the ingested session/circuit
reference - that identifier is the CIRCUIT group's other feature.

Known limitation: in the current 2023 rounds 1-6 scope, every round is a
different circuit, so this feature is NaN for every row (no driver has a
prior visit to any of these circuits within the scope). It is still
implemented and tested against synthetic data with repeated circuits, so it
activates correctly the moment the season scope is widened to include a
circuit visited more than once.
"""

from __future__ import annotations

import pandas as pd

from mpp.features.common import add_expanding_prior_stat, driver_round_relative_pace


def compute_driver_circuit_prior_relative_pace(
    valid_laps: pd.DataFrame, sessions: pd.DataFrame
) -> pd.DataFrame:
    """valid_laps: is_valid_lap==True rows with season, round, driver, lap_time_s.
    sessions: season, round, circuit_id (the round -> circuit reference table).

    Returns one row per (season, round, driver, circuit_id) with
    driver_circuit_prior_relative_pace_s.
    """
    pace = driver_round_relative_pace(valid_laps)
    pace = pace.merge(sessions[["season", "round", "circuit_id"]], on=["season", "round"], how="left")
    pace = add_expanding_prior_stat(
        pace,
        ["driver", "circuit_id"],
        "round",
        "relative_pace_s",
        "driver_circuit_prior_relative_pace_s",
        stat="mean",
    )
    return pace[["season", "round", "driver", "circuit_id", "driver_circuit_prior_relative_pace_s"]]
