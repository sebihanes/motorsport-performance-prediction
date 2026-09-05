"""Feature dataset orchestrator: PROCESSED -> FEATURE DATASET.

Assembles one wide table, one row per lap, combining the cleaned lap record
with weather, race-progress, tyre, and leakage-safe historical features. This
is the single feature table every per-problem model-ready dataset in
mpp.datasets is built from.
"""

from __future__ import annotations

import pandas as pd

from mpp.config import PipelineConfig
from mpp.features.context import add_race_fraction
from mpp.features.historical import compute_driver_prior_relative_pace
from mpp.features.tyre import add_tyre_age_squared
from mpp.features.weather import add_weather


def build_laps_features(
    processed: dict[str, pd.DataFrame], intermediate: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    laps = processed["laps"]
    sessions = intermediate["sessions"]

    laps = add_weather(laps, intermediate["weather"])
    laps = add_race_fraction(laps, sessions)
    laps = add_tyre_age_squared(laps)

    valid_laps = laps.loc[laps["is_valid_lap"], ["season", "round", "driver", "lap_time_s"]]
    history = compute_driver_prior_relative_pace(valid_laps)
    laps = laps.merge(history, on=["season", "round", "driver"], how="left")

    laps = laps.merge(sessions[["season", "round", "circuit_id"]], on=["season", "round"], how="left")

    return laps


def run_feature_build(
    processed: dict[str, pd.DataFrame], intermediate: dict[str, pd.DataFrame], config: PipelineConfig
) -> pd.DataFrame:
    laps_features = build_laps_features(processed, intermediate)
    out_dir = config.paths.features_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    laps_features.to_parquet(out_dir / "laps_features.parquet", index=False)
    return laps_features
