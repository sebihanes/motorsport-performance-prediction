"""Deterministic cleaning of the intermediate pit stops table.

Jolpica-F1 reports pit stop duration as a string, either plain seconds
("25.885") or minutes:seconds ("1:05.123") for long stops (damage, penalties).
Both are parsed to float seconds here (numeric conversion); the mapping from
Jolpica's driver_id to the FastF1 driver code used everywhere else in the
pipeline also happens here, via the intermediate drivers reference table.
"""

from __future__ import annotations

import pandas as pd

from mpp.config import PipelineConfig

PROCESSED_COLUMNS = [
    "season",
    "round",
    "driver",
    "driver_id",
    "lap_number",
    "stop_number",
    "duration_s",
]


def parse_duration_to_seconds(raw: str | float | None) -> float | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = str(raw).strip()
    if not text:
        return None
    if ":" in text:
        minutes_str, seconds_str = text.split(":", 1)
        try:
            return float(minutes_str) * 60.0 + float(seconds_str)
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def clean_pitstops(
    df: pd.DataFrame, drivers: pd.DataFrame, config: PipelineConfig
) -> tuple[pd.DataFrame, dict]:
    log: dict = {"stage": "clean_pitstops", "n_rows_in": len(df)}
    if df.empty:
        log["n_rows_out"] = 0
        return pd.DataFrame(columns=PROCESSED_COLUMNS), log

    df = df.copy()

    # --- numeric conversion ---
    df["duration_s"] = df["duration_raw"].apply(parse_duration_to_seconds)
    unparseable_mask = df["duration_s"].isna()
    log["unparseable_duration_rows_removed"] = int(unparseable_mask.sum())
    df = df.loc[~unparseable_mask].copy()

    # --- driver_id -> FastF1 driver code, via the season/round driver reference ---
    driver_map = drivers.drop_duplicates(subset=["season", "round", "driver_id"])[
        ["season", "round", "driver_id", "driver_code"]
    ]
    n_before = len(df)
    df = df.merge(driver_map, on=["season", "round", "driver_id"], how="left")
    unmapped_mask = df["driver_code"].isna()
    log["unmapped_driver_rows_removed"] = int(unmapped_mask.sum())
    df = df.loc[~unmapped_mask].copy()
    assert len(df) == n_before - log["unmapped_driver_rows_removed"]

    # --- duplicates ---
    n_before = len(df)
    df = df.drop_duplicates(subset=["season", "round", "driver_id", "stop_number"], keep="first")
    log["duplicate_rows_removed"] = n_before - len(df)

    df["driver"] = df["driver_code"]
    df["lap_number"] = pd.to_numeric(df["lap"], errors="coerce").astype("Int64")

    log["n_rows_out"] = len(df)
    return df[PROCESSED_COLUMNS].reset_index(drop=True), log
