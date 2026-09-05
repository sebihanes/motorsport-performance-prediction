"""Race-context feature derivations."""

from __future__ import annotations

import pandas as pd


def add_race_fraction(laps: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    """race_fraction = lap_number / total_laps - a fuel-load proxy (architecture SS6).

    Safe: total_laps for a round is known from the event itself, not from any
    outcome, so this carries no leakage.
    """
    laps = laps.merge(sessions[["season", "round", "total_laps"]], on=["season", "round"], how="left")
    laps["race_fraction"] = laps["lap_number"].astype("Float64") / laps["total_laps"].astype("Float64")
    return laps.drop(columns=["total_laps"])
