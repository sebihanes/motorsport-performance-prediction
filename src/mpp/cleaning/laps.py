"""Deterministic cleaning of the intermediate laps table.

Exclusion reasons are assigned in a fixed priority order so the decision for
any given row is reproducible and explainable:

    1. missing_lap_time    - LapTime is null (retirement, red flag, no time set)
    2. deleted_lap         - stewards deleted the lap (e.g. track limits)
    3. pit_in_out_lap      - the lap entered or left the pit lane
    4. lap_time_out_of_bounds - outside the plausible green-flag range (config)
    5. (none)              - the lap is valid racing pace

A lap that is *not* valid racing pace is kept in data/processed/laps.parquet
(is_valid_lap=False, exclusion_reason set) rather than removed, so it remains
available for error analysis and for problems that legitimately want it
(e.g. a future in/out-lap-aware feature). Only genuine data defects - exact
duplicate rows and corrupt lap numbers - are actually dropped.
"""

from __future__ import annotations

import pandas as pd

from mpp.config import PipelineConfig

PROCESSED_COLUMNS = [
    "season",
    "round",
    "driver",
    "driver_number",
    "team",
    "lap_number",
    "stint",
    "lap_time_s",
    "time_s",
    "compound",
    "tyre_age",
    "fresh_tyre",
    "pit_in_lap",
    "pit_out_lap",
    "track_status",
    "deleted",
    "is_accurate",
    "is_valid_lap",
    "exclusion_reason",
]


def _normalize_compound(raw: pd.Series, valid_compounds: tuple[str, ...]) -> tuple[pd.Series, int]:
    normalized = raw.astype("string").str.strip().str.upper()
    valid_upper = {c.upper() for c in valid_compounds}
    unknown_mask = normalized.isna() | ~normalized.isin(valid_upper)
    n_remapped = int(unknown_mask.sum())
    normalized = normalized.where(~unknown_mask, "UNKNOWN")
    return normalized, n_remapped


def clean_laps(df: pd.DataFrame, config: PipelineConfig) -> tuple[pd.DataFrame, dict]:
    log: dict = {"stage": "clean_laps", "n_rows_in": len(df)}
    df = df.copy()

    # --- corrupt rows: cannot be salvaged, dropped and counted ---
    lap_number_numeric = pd.to_numeric(df["LapNumber"], errors="coerce")
    corrupt_mask = lap_number_numeric.isna() | (lap_number_numeric <= 0)
    log["corrupt_lap_number_rows_removed"] = int(corrupt_mask.sum())
    df = df.loc[~corrupt_mask].copy()

    # --- exact duplicates on the configured key ---
    key = list(config.cleaning.duplicate_key)
    key_fastf1_names = {"driver": "Driver", "lap_number": "LapNumber"}
    dup_key = [key_fastf1_names.get(k, k) for k in key]
    n_before = len(df)
    df = df.drop_duplicates(subset=dup_key, keep="first")
    log["duplicate_rows_removed"] = n_before - len(df)

    # --- numeric conversion ---
    df["lap_time_s"] = df["LapTime"].dt.total_seconds()
    df["time_s"] = df["Time"].dt.total_seconds()

    # --- categorical normalization ---
    df["compound"], log["compound_rows_remapped_to_unknown"] = _normalize_compound(
        df["Compound"], config.validation.valid_compounds
    )

    # --- boolean normalization ---
    df["deleted"] = df["Deleted"].astype("boolean").fillna(False).astype(bool)
    df["is_accurate"] = df["IsAccurate"].astype("boolean").fillna(False).astype(bool)
    df["fresh_tyre"] = df["FreshTyre"].astype("boolean").fillna(False).astype(bool)
    df["pit_in_lap"] = df["PitInTime"].notna()
    df["pit_out_lap"] = df["PitOutTime"].notna()

    # --- identity / rename ---
    df["driver"] = df["Driver"]
    df["driver_number"] = df["DriverNumber"]
    df["team"] = df["Team"]
    df["lap_number"] = df["LapNumber"]
    df["stint"] = df["Stint"]
    df["tyre_age"] = df["TyreLife"]
    df["track_status"] = df["TrackStatus"].astype("string")

    # --- validity classification (fixed priority order) ---
    missing_lap_time = df["lap_time_s"].isna()
    within_bounds = df["lap_time_s"].between(
        config.validation.lap_time_min_s, config.validation.lap_time_max_s
    )
    reason = pd.Series(pd.NA, index=df.index, dtype="string")
    reason = reason.where(~missing_lap_time, "missing_lap_time")
    reason = reason.where(~(reason.isna() & df["deleted"]), "deleted_lap")
    reason = reason.where(
        ~(reason.isna() & (df["pit_in_lap"] | df["pit_out_lap"])), "pit_in_out_lap"
    )
    reason = reason.where(
        ~(reason.isna() & ~missing_lap_time & ~within_bounds), "lap_time_out_of_bounds"
    )
    df["exclusion_reason"] = reason
    df["is_valid_lap"] = reason.isna()

    log["n_rows_out"] = len(df)
    log["n_valid_laps"] = int(df["is_valid_lap"].sum())
    log["exclusion_reason_counts"] = (
        df["exclusion_reason"].value_counts(dropna=True).to_dict()
    )

    # --- incomplete-session flag (informational, no rows removed) ---
    max_lap_by_round = df.groupby("round")["lap_number"].max()
    log["max_lap_number_by_round"] = {int(k): int(v) for k, v in max_lap_by_round.items()}

    return df[PROCESSED_COLUMNS].reset_index(drop=True), log
