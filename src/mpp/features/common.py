"""Shared leakage-safe building blocks used by the driver/team/circuit
historical-performance features (mpp.features.historical/team/circuit).

The pattern all three follow is the same: aggregate lap times up to some
per-round pace figure, express it relative to the field so it is comparable
across circuits, then average that figure *strictly before* the round being
featured. Factoring it once here means the leakage-safety guarantee (see
add_expanding_prior_stat) only has to be reasoned about, and tested, in one
place.
"""

from __future__ import annotations

import pandas as pd


def add_expanding_prior_stat(
    df: pd.DataFrame,
    group_cols: list[str],
    order_col: str,
    value_col: str,
    out_col: str,
    stat: str = "mean",
) -> pd.DataFrame:
    """out_col = expanding `stat` of value_col over strictly-prior order_col values, within each group.

    "Strictly prior" is what makes this leakage-safe: shift(1) moves every
    value one row later *before* the expanding window is taken, so the value
    attached to a given (group, order) row only ever summarizes rows with a
    smaller order_col value in that same group - never the row itself, never
    a later one. The first occurrence in every group is NaN by construction
    (there is nothing prior to average yet), which is the correct answer, not
    a missing-data defect.

    Returns a new frame sorted by group_cols + [order_col] with out_col added
    - callers merge it back onto their own frame by key rather than relying
    on row order.
    """
    df = df.sort_values(group_cols + [order_col]).reset_index(drop=True)
    grouped = df.groupby(group_cols)[value_col]
    if stat == "mean":
        df[out_col] = grouped.transform(lambda s: s.shift(1).expanding().mean())
    elif stat == "std":
        df[out_col] = grouped.transform(lambda s: s.shift(1).expanding().std())
    else:
        raise ValueError(f"Unsupported stat: {stat!r} (expected 'mean' or 'std')")
    return df


def driver_round_relative_pace(valid_laps: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, round, driver): that driver's relative_pace_s for
    the round - their median lap time minus the field's median lap time.

    valid_laps must be pre-filtered to laps that should count towards pace
    (is_valid_lap == True) and have columns season, round, driver, lap_time_s.
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
    return merged[["season", "round", "driver", "relative_pace_s"]]


def team_round_relative_pace(valid_laps: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, round, team): the team's relative_pace_s for the
    round - the median lap time across *both* of the team's cars, minus the
    field's median lap time. Same normalization idea as driver_round_relative_pace,
    grouped one level up so a team's historical form reflects whichever driver
    was in the car each round.

    valid_laps must have columns season, round, team, lap_time_s.
    """
    team_round_pace = (
        valid_laps.groupby(["season", "round", "team"])["lap_time_s"]
        .median()
        .reset_index(name="team_pace_s")
    )
    field_round_pace = (
        team_round_pace.groupby(["season", "round"])["team_pace_s"]
        .median()
        .reset_index(name="field_pace_s")
    )
    merged = team_round_pace.merge(field_round_pace, on=["season", "round"])
    merged["relative_pace_s"] = merged["team_pace_s"] - merged["field_pace_s"]
    return merged[["season", "round", "team", "relative_pace_s"]]
