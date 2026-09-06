"""Lap-context feature derivations.

laps_into_stint (stint-relative lap number) is deliberately a different
concept from tyre_age: FastF1's TyreLife (our tyre_age) can start above 1 on
the very first race lap of a stint, because pre-race reconnaissance/formation
laps on the same tyre set count towards it (observed directly in the ingested
2023 data - see docs/01_data_sources.md). laps_into_stint always starts at 1
on the first race lap of the stint, which is what "how far into this pit
strategy phase are we" means in race-engineering terms, independent of
whatever tyre-life clock the tyres carried into the stint.
"""

from __future__ import annotations

import pandas as pd

STINT_KEY = ["season", "round", "driver", "stint"]


def add_stint_relative_lap_number(df: pd.DataFrame) -> pd.DataFrame:
    """laps_into_stint = lap_number - (first lap_number of this stint) + 1.

    Not leakage: a stint's first lap number is fixed the moment the stint
    begins and never changes as the stint progresses, so every lap in the
    stint already "knows" it at the time it is driven - unlike, say, the
    stint's *last* lap number, which is only known once the stint ends and
    must never be used as a per-row feature.
    """
    df = df.copy()
    stint_start_lap = df.groupby(STINT_KEY)["lap_number"].transform("min")
    df["laps_into_stint"] = df["lap_number"].astype("Int64") - stint_start_lap.astype("Int64") + 1
    return df
