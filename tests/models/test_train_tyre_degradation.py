"""Integration test for the Phase 5 tyre-degradation training protocol, on a
small synthetic tyre_degradation-shaped dataset - same philosophy as
tests/models/test_train_lap_time.py: checked for shape/wiring correctness,
not exact numbers, with small monkeypatched search spaces so nested tuning
stays fast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mpp.models.train_common as train_common
from mpp.config import CleaningSettings, PipelineConfig, PipelinePaths, ValidationSettings
from mpp.models.train_tyre_degradation import run_training

N_SYNTHETIC_RACES = 6
EXPECTED_MODEL_NAMES = {"mean_baseline", "median_baseline", "strong_baseline", "linear_regression", "ridge", "random_forest"}


def _synthetic_tyre_degradation_dataset(n_races: int = N_SYNTHETIC_RACES, laps_per_stint: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    circuits = ["bahrain", "jeddah", "albert_park", "baku", "miami", "monaco"]
    rows = []
    for round_ in range(1, n_races + 1):
        circuit = circuits[(round_ - 1) % len(circuits)]
        for driver_idx in range(4):
            compound = ["SOFT", "MEDIUM", "HARD"][driver_idx % 3]
            degradation_rate = {"SOFT": 0.08, "MEDIUM": 0.05, "HARD": 0.02}[compound]
            for lap in range(laps_per_stint):
                rows.append(
                    {
                        "season": 2023,
                        "round": round_,
                        "circuit_id": circuit,
                        "driver": f"D{driver_idx}",
                        "team": f"T{driver_idx % 2}",
                        "tyre_age_delta": float(lap),
                        "tyre_age_delta_sq": float(lap**2),
                        "laps_into_stint": lap,
                        "race_fraction": lap / laps_per_stint,
                        "air_temp": 25.0 + rng.normal(),
                        "track_temp": 35.0 + rng.normal(),
                        "driver_prior_relative_pace_s": np.nan if round_ == 1 else rng.normal(),
                        "driver_prior_relative_pace_std_s": np.nan if round_ == 1 else abs(rng.normal()),
                        "team_prior_relative_pace_s": np.nan if round_ == 1 else rng.normal(),
                        "driver_circuit_prior_relative_pace_s": np.nan,
                        "compound": compound,
                        "lap_time_delta_s": degradation_rate * lap + rng.normal(scale=0.05),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def small_search_grids(monkeypatch):
    monkeypatch.setattr(train_common, "N_ITER_RANDOM_SEARCH", 2)
    monkeypatch.setattr(train_common, "RIDGE_PARAM_GRID", {"model__alpha": [1.0, 10.0]})
    monkeypatch.setattr(train_common, "RANDOM_FOREST_PARAM_DISTRIBUTIONS", {"model__n_estimators": [20, 40]})


@pytest.fixture
def synthetic_config(tmp_path):
    model_ready_dir = tmp_path / "model_ready"
    model_ready_dir.mkdir()
    dataset = _synthetic_tyre_degradation_dataset()
    dataset.to_parquet(model_ready_dir / "tyre_degradation.parquet", index=False)

    paths = PipelinePaths(
        raw_dir=tmp_path / "raw",
        intermediate_dir=tmp_path / "intermediate",
        processed_dir=tmp_path / "processed",
        features_dir=tmp_path / "features",
        model_ready_dir=model_ready_dir,
        quality_dir=tmp_path / "quality",
        models_dir=tmp_path / "models",
    )
    return PipelineConfig(
        season=2023,
        rounds=tuple(range(1, N_SYNTHETIC_RACES + 1)),
        session="R",
        paths=paths,
        validation=ValidationSettings(
            lap_time_min_s=55.0, lap_time_max_s=160.0, valid_compounds=("SOFT", "MEDIUM", "HARD", "UNKNOWN")
        ),
        cleaning=CleaningSettings(duplicate_key=("season", "round", "driver", "lap_number")),
        repo_root=tmp_path,
    ), dataset


def test_run_training_evaluates_every_baseline_and_candidate(small_search_grids, synthetic_config):
    config, _dataset = synthetic_config
    outcome = run_training(config, n_jobs=1)
    assert {row["model"] for row in outcome["results"]} == EXPECTED_MODEL_NAMES
    for row in outcome["results"]:
        assert np.isfinite(row["mae_mean"])


def test_run_training_uses_leave_one_race_out_outer_folds(small_search_grids, synthetic_config):
    config, _dataset = synthetic_config
    outcome = run_training(config, n_jobs=1)
    assert outcome["n_outer_folds"] == N_SYNTHETIC_RACES


def test_run_training_best_model_beats_baseline_on_this_clean_synthetic_signal(small_search_grids, synthetic_config):
    # The synthetic target is a near-deterministic function of compound + tyre age -
    # a real learned model should comfortably beat the strong (per-compound) baseline here.
    config, _dataset = synthetic_config
    outcome = run_training(config, n_jobs=1)
    assert outcome["beats_strong_baseline"]
    assert outcome["final_pipeline"] is not None


def test_run_training_error_report_breaks_down_by_compound_and_skips_absent_dims(small_search_grids, synthetic_config):
    config, _dataset = synthetic_config
    outcome = run_training(config, n_jobs=1)
    assert "compound" in outcome["error_report"]["tables"]
    assert "session_type" in outcome["error_report"]["skipped"]  # not a column in this dataset


def test_run_training_metadata_documents_known_limitations(small_search_grids, synthetic_config):
    config, dataset = synthetic_config
    outcome = run_training(config, n_jobs=1)
    metadata = outcome["metadata"]
    assert metadata["target_column"] == "lap_time_delta_s"
    assert metadata["training_dataset_n_rows"] == len(dataset)
    assert len(metadata["known_limitations"]) >= 1
    assert "causation_caveat" in metadata


def test_run_training_serializable_final_pipeline_predicts(small_search_grids, synthetic_config):
    config, dataset = synthetic_config
    outcome = run_training(config, n_jobs=1)
    if not outcome["beats_strong_baseline"]:
        pytest.skip("no candidate beat the strong baseline on this synthetic draw")
    from mpp.models.train_tyre_degradation import MODEL_NOMINAL_COLUMNS, MODEL_NUMERIC_COLUMNS

    preds = outcome["final_pipeline"].predict(dataset[MODEL_NUMERIC_COLUMNS + MODEL_NOMINAL_COLUMNS].head(5))
    assert len(preds) == 5
    assert np.isfinite(preds).all()
