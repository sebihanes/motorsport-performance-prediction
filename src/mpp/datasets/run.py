"""Model dataset orchestrator: FEATURE DATASET -> MODEL DATASET."""

from __future__ import annotations

import json

import pandas as pd

from mpp.config import PipelineConfig
from mpp.datasets.lap_time import build_lap_time_dataset
from mpp.datasets.pit_stop import build_pit_stop_dataset
from mpp.datasets.race_pace import build_race_pace_dataset
from mpp.datasets.tyre_degradation import build_tyre_degradation_dataset


def run_dataset_build(
    laps_features: pd.DataFrame, pitstops_processed: pd.DataFrame, config: PipelineConfig
) -> dict[str, pd.DataFrame]:
    lap_time_ds = build_lap_time_dataset(laps_features)
    degradation_ds, degradation_summary = build_tyre_degradation_dataset(laps_features)
    race_pace_ds = build_race_pace_dataset(laps_features)
    pit_stop_ds, pit_stop_summary = build_pit_stop_dataset(pitstops_processed, laps_features)

    datasets = {
        "lap_time": lap_time_ds,
        "tyre_degradation": degradation_ds,
        "race_pace": race_pace_ds,
        "pit_stop": pit_stop_ds,
    }

    out_dir = config.paths.model_ready_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in datasets.items():
        frame.to_parquet(out_dir / f"{name}.parquet", index=False)

    summary = {
        "lap_time": {"n_rows": len(lap_time_ds)},
        "tyre_degradation": degradation_summary,
        "race_pace": {"n_rows": len(race_pace_ds)},
        "pit_stop": pit_stop_summary,
    }
    quality_dir = config.paths.quality_dir
    quality_dir.mkdir(parents=True, exist_ok=True)
    with open(quality_dir / "dataset_build_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return datasets
