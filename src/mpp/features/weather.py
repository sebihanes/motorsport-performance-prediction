"""Join each lap to its nearest weather sample.

FastF1 weather is sampled periodically (roughly once a minute), not per lap,
so this is a nearest-time-in-session join, not an exact one. Weather is an
optional feature (architecture SS4/SS6): a lap with no weather sample within
tolerance gets nulls, never a dropped row.
"""

from __future__ import annotations

import pandas as pd

_TOLERANCE_S = 5 * 60.0  # 5 minutes - generous enough to cover the weather sampling cadence


def add_weather(laps: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    weather = weather.copy()
    weather["weather_time_s"] = weather["Time"].dt.total_seconds()

    # merge_asof requires the "on" key sorted *globally*, but time_s resets to
    # zero every round - so join one (season, round) group at a time instead
    # of trying to satisfy a single global sort across resetting clocks.
    merged_groups = []
    for (season, round_), laps_group in laps.groupby(["season", "round"], sort=False):
        weather_group = weather.loc[
            (weather["season"] == season) & (weather["round"] == round_)
        ].sort_values("weather_time_s")
        laps_group = laps_group.sort_values("time_s")
        merged_group = pd.merge_asof(
            laps_group,
            weather_group[["weather_time_s", "AirTemp", "TrackTemp", "Rainfall"]],
            left_on="time_s",
            right_on="weather_time_s",
            direction="nearest",
            tolerance=_TOLERANCE_S,
        )
        merged_groups.append(merged_group)

    merged = pd.concat(merged_groups, ignore_index=True)
    merged = merged.rename(
        columns={"AirTemp": "air_temp", "TrackTemp": "track_temp", "Rainfall": "rainfall"}
    )
    return merged.drop(columns=["weather_time_s"])
