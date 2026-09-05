"""Cleaning orchestrator: INTERMEDIATE -> PROCESSED."""

from __future__ import annotations

import json

import pandas as pd

from mpp.cleaning.laps import clean_laps
from mpp.cleaning.pitstops import clean_pitstops
from mpp.config import PipelineConfig


def run_cleaning(
    intermediate: dict[str, pd.DataFrame], config: PipelineConfig
) -> tuple[dict[str, pd.DataFrame], dict]:
    laps_clean, laps_log = clean_laps(intermediate["laps"], config)
    pitstops_clean, pitstops_log = clean_pitstops(
        intermediate["pitstops"], intermediate["drivers"], config
    )

    processed = {"laps": laps_clean, "pitstops": pitstops_clean}
    combined_log = {"laps": laps_log, "pitstops": pitstops_log}

    out_dir = config.paths.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in processed.items():
        frame.to_parquet(out_dir / f"{name}.parquet", index=False)

    quality_dir = config.paths.quality_dir
    quality_dir.mkdir(parents=True, exist_ok=True)
    with open(quality_dir / "cleaning_log.json", "w", encoding="utf-8") as fh:
        json.dump(combined_log, fh, indent=2, default=str)

    return processed, combined_log
