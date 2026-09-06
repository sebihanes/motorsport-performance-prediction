"""Feature dataset orchestrator: PROCESSED -> FEATURE DATASET.

Assembles one wide table, one row per lap, combining the cleaned lap record
with weather, race-progress, tyre, lap-context, session, and leakage-safe
historical (driver/team/circuit) and within-stint performance-trend features.
This is the single feature table every per-problem model-ready dataset in
mpp.datasets is built from. See docs/04_feature_dictionary.md for the meaning
of every column this produces, and docs/02_data_pipeline.md /
docs/05_split_strategy.md for the leakage-prevention and split reasoning.
"""

from __future__ import annotations

import pandas as pd

from mpp.config import PipelineConfig
from mpp.features.circuit import compute_driver_circuit_prior_relative_pace
from mpp.features.context import add_race_fraction
from mpp.features.historical import (
    compute_driver_prior_relative_pace,
    compute_driver_prior_relative_pace_std,
)
from mpp.features.lap import add_stint_relative_lap_number
from mpp.features.performance import add_stint_prior_pace_trend
from mpp.features.session import add_session_type
from mpp.features.team import compute_team_prior_relative_pace
from mpp.features.tyre import add_compound_tyre_age_interaction, add_tyre_age_squared
from mpp.features.weather import add_weather


def build_laps_features(
    processed: dict[str, pd.DataFrame],
    intermediate: dict[str, pd.DataFrame],
    session_code: str,
    valid_compounds: tuple[str, ...],
) -> pd.DataFrame:
    laps = processed["laps"]
    sessions = intermediate["sessions"]

    # circuit_id first - every historical feature below keys off it.
    laps = laps.merge(sessions[["season", "round", "circuit_id"]], on=["season", "round"], how="left")

    laps = add_weather(laps, intermediate["weather"])
    laps = add_race_fraction(laps, sessions)
    laps = add_tyre_age_squared(laps)
    laps = add_stint_relative_lap_number(laps)
    laps = add_session_type(laps, session_code)
    laps = add_compound_tyre_age_interaction(laps, valid_compounds)
    laps = add_stint_prior_pace_trend(laps)

    valid_laps = laps.loc[laps["is_valid_lap"], ["season", "round", "driver", "team", "lap_time_s"]]

    driver_mean = compute_driver_prior_relative_pace(valid_laps)
    laps = laps.merge(driver_mean, on=["season", "round", "driver"], how="left")

    driver_std = compute_driver_prior_relative_pace_std(valid_laps)
    laps = laps.merge(driver_std, on=["season", "round", "driver"], how="left")

    team_mean = compute_team_prior_relative_pace(valid_laps)
    laps = laps.merge(team_mean, on=["season", "round", "team"], how="left")

    circuit_mean = compute_driver_circuit_prior_relative_pace(valid_laps, sessions)
    laps = laps.merge(circuit_mean, on=["season", "round", "driver", "circuit_id"], how="left")

    return laps


def run_feature_build(
    processed: dict[str, pd.DataFrame], intermediate: dict[str, pd.DataFrame], config: PipelineConfig
) -> pd.DataFrame:
    laps_features = build_laps_features(
        processed, intermediate, config.session, config.validation.valid_compounds
    )
    out_dir = config.paths.features_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    laps_features.to_parquet(out_dir / "laps_features.parquet", index=False)
    return laps_features
