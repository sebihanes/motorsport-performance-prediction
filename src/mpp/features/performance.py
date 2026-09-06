"""Within-stint performance-trend feature.

stint_prior_mean_lap_time_s: the expanding mean of this same stint's own
*prior* clean laps' lap time - "how has pace trended so far this stint,"
computed the way a race engineer actually has it available live: only laps
already completed in the current stint, never a lap still to come.

This is deliberately built from lap_number order *within the stint*, not
tyre_age - a lap's own tyre_age is a fixed input already available as a
feature; this is a realized-outcome signal (how the laps actually went so
far), which is why it is kept in mpp.features.performance rather than
mpp.features.tyre.
"""

from __future__ import annotations

import pandas as pd

STINT_KEY = ["season", "round", "driver", "stint"]


def add_stint_prior_pace_trend(laps: pd.DataFrame) -> pd.DataFrame:
    """laps must have is_valid_lap, lap_number, lap_time_s and the STINT_KEY columns.

    Only valid laps contribute to (and receive) the trend value: an invalid
    lap (pit in/out, deleted, missing time) has no representative pace of its
    own to fold into the running mean, so it is left null here rather than
    given a value derived from an excluded observation.
    """
    valid = laps.loc[laps["is_valid_lap"], STINT_KEY + ["lap_number", "lap_time_s"]].copy()
    valid = valid.sort_values(STINT_KEY + ["lap_number"])
    valid["stint_prior_mean_lap_time_s"] = valid.groupby(STINT_KEY)["lap_time_s"].transform(
        lambda s: s.shift(1).expanding().mean()
    )

    merge_key = STINT_KEY + ["lap_number"]
    return laps.merge(valid[merge_key + ["stint_prior_mean_lap_time_s"]], on=merge_key, how="left")
