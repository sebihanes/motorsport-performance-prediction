"""Pit stop duration dataset (architecture SS3.4, SS7) - built only because the
ingested Jolpica-F1 data actually supports it at per-stop granularity for the
configured scope (the feasibility check the architecture report calls for).

Row = one recorded pit stop. Target = duration_s, the full stationary time
Jolpica-F1 reports; it is not decomposed into tyre-change vs. other causes
(damage repair, drive-through), which is a known limitation, not an error.
"""

from __future__ import annotations

import pandas as pd

GROUP_COLUMNS = ["season", "round", "circuit_id"]
IDENTITY_COLUMNS = ["driver", "team"]
FEATURE_COLUMNS = ["lap_number", "stop_number"]
TARGET_COLUMN = "duration_s"


def build_pit_stop_dataset(
    pitstops: pd.DataFrame, laps_features: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    if pitstops.empty:
        empty = pd.DataFrame(columns=GROUP_COLUMNS + IDENTITY_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN])
        return empty, {"n_rows_out": 0, "n_unmatched_context_removed": 0}

    context = laps_features.groupby(["season", "round", "driver"], as_index=False).agg(
        team=("team", "first"), circuit_id=("circuit_id", "first")
    )
    merged = pitstops.merge(context, on=["season", "round", "driver"], how="left")
    unmatched_mask = merged["team"].isna() | merged["circuit_id"].isna()
    n_unmatched = int(unmatched_mask.sum())
    merged = merged.loc[~unmatched_mask].copy()

    columns = GROUP_COLUMNS + IDENTITY_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]
    dataset = merged[columns].reset_index(drop=True)
    return dataset, {"n_rows_out": len(dataset), "n_unmatched_context_removed": n_unmatched}
