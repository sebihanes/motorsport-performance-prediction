"""Leakage-safe historical team-performance feature.

team_prior_relative_pace_s: how far above/below the field's median race pace
a team's cars have been, averaged over that team's rounds *strictly before*
the round being featured. Same expanding-mean, strictly-prior construction as
mpp.features.historical's driver feature (see mpp.features.common), grouped
by team instead of driver - a team's historical form is a genuine, distinct
signal from a single driver's (car/setup performance vs. driver skill), and
carries over correctly across a mid-season driver change since it only
depends on which car scored the lap times, not who was driving it.
"""

from __future__ import annotations

import pandas as pd

from mpp.features.common import add_expanding_prior_stat, team_round_relative_pace


def compute_team_prior_relative_pace(valid_laps: pd.DataFrame) -> pd.DataFrame:
    """valid_laps must contain only rows that should count towards pace (is_valid_lap==True)
    and have columns season, round, team, lap_time_s.

    Returns one row per (season, round, team) with team_prior_relative_pace_s.
    """
    pace = team_round_relative_pace(valid_laps)
    pace = add_expanding_prior_stat(
        pace, ["team"], "round", "relative_pace_s", "team_prior_relative_pace_s", stat="mean"
    )
    return pace[["season", "round", "team", "team_prior_relative_pace_s"]]
