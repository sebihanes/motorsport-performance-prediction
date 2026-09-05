"""End-to-end integration test: INTERMEDIATE -> CLEANING -> FEATURES -> MODEL DATASETS.

Fully offline (no FastF1/Jolpica network calls) - uses the small real-data
fixture slice laid out as if fetch_raw() had already run. This checks the
stages are wired together correctly (shapes, columns, no exceptions), not
statistical correctness, which is covered by the per-stage unit tests.
"""

from __future__ import annotations

from mpp.cleaning.run import run_cleaning
from mpp.datasets.run import run_dataset_build
from mpp.features.build import run_feature_build
from mpp.ingest.run import build_intermediate
from mpp.validation.report import run_validation
from tests.conftest import populate_raw_dir_from_fixture


def test_full_pipeline_runs_end_to_end_on_a_tiny_real_slice(test_config, fixtures_dir):
    populate_raw_dir_from_fixture(test_config.paths.raw_dir, fixtures_dir)

    intermediate = build_intermediate(test_config)
    validation_results = run_validation(intermediate, test_config)
    assert not any(r.status == "error" for rs in validation_results.values() for r in rs)

    processed, cleaning_log = run_cleaning(intermediate, test_config)
    assert cleaning_log["laps"]["n_valid_laps"] == 9
    assert cleaning_log["pitstops"]["n_rows_out"] == 2

    laps_features = run_feature_build(processed, intermediate, test_config)
    assert len(laps_features) == 9
    assert "driver_prior_relative_pace_s" in laps_features.columns
    assert "circuit_id" in laps_features.columns

    datasets = run_dataset_build(laps_features, processed["pitstops"], test_config)
    assert set(datasets) == {"lap_time", "tyre_degradation", "race_pace", "pit_stop"}
    assert len(datasets["lap_time"]) == 9  # all 9 fixture laps are valid racing laps
    assert len(datasets["race_pace"]) == 2  # VER and LEC each get one row
    assert len(datasets["pit_stop"]) == 2

    for name in ("lap_time", "tyre_degradation", "race_pace", "pit_stop"):
        assert (test_config.paths.model_ready_dir / f"{name}.parquet").exists()
