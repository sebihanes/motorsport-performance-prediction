"""Integration test for the Phase 5/7 pit-stop training protocol, on a small
synthetic pit_stop-shaped dataset - same nested race-grouped CV
(mpp.models.train_common) as tyre_degradation, checked for wiring/shape
correctness rather than exact numbers, plus a dedicated edge-case test for
what happens when the dataset is too small/empty to train on at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mpp.models.train_common as train_common
from mpp.config import CleaningSettings, PipelineConfig, PipelinePaths, ValidationSettings
from mpp.models.train_pit_stop import load_pit_stop_dataset, run_training

N_SYNTHETIC_RACES = 6
EXPECTED_MODEL_NAMES = {"mean_baseline", "median_baseline", "strong_baseline", "linear_regression", "ridge", "random_forest"}


def _synthetic_pit_stop_dataset(n_races: int = N_SYNTHETIC_RACES, stops_per_race: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    circuits = ["bahrain", "jeddah", "monaco"]  # repeated across rounds, unlike the real 6-round scope
    team_speed = {f"T{i}": 20.0 + 2.0 * i for i in range(5)}
    circuit_effect = {c: 8.0 * i for i, c in enumerate(circuits)}
    rows = []
    for round_ in range(1, n_races + 1):
        circuit = circuits[(round_ - 1) % len(circuits)]
        for stop in range(stops_per_race):
            team = f"T{stop % 5}"
            rows.append(
                {
                    "season": 2023,
                    "round": round_,
                    "circuit_id": circuit,
                    "driver": f"D{stop}",
                    "team": team,
                    "lap_number": 10 + stop,
                    "stop_number": 1 + stop % 2,
                    "duration_s": team_speed[team] + circuit_effect[circuit] + rng.normal(scale=0.2),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def small_search_grids(monkeypatch):
    monkeypatch.setattr(train_common, "N_ITER_RANDOM_SEARCH", 2)
    monkeypatch.setattr(train_common, "RIDGE_PARAM_GRID", {"model__alpha": [1.0, 10.0]})
    monkeypatch.setattr(train_common, "RANDOM_FOREST_PARAM_DISTRIBUTIONS", {"model__n_estimators": [20, 40]})


def _config(tmp_path, dataset: pd.DataFrame) -> PipelineConfig:
    model_ready_dir = tmp_path / "model_ready"
    model_ready_dir.mkdir(exist_ok=True)
    dataset.to_parquet(model_ready_dir / "pit_stop.parquet", index=False)
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
        validation=ValidationSettings(lap_time_min_s=55.0, lap_time_max_s=160.0, valid_compounds=("SOFT",)),
        cleaning=CleaningSettings(duplicate_key=("season", "round", "driver", "lap_number")),
        repo_root=tmp_path,
    )


@pytest.fixture
def synthetic_config(tmp_path):
    dataset = _synthetic_pit_stop_dataset()
    return _config(tmp_path, dataset), dataset


def test_run_training_evaluates_every_baseline_and_candidate(small_search_grids, synthetic_config):
    config, _dataset = synthetic_config
    outcome = run_training(config, n_jobs=1)
    assert {row["model"] for row in outcome["results"]} == EXPECTED_MODEL_NAMES


def test_run_training_best_model_beats_baseline_on_clean_synthetic_signal(small_search_grids, synthetic_config):
    config, _dataset = synthetic_config
    outcome = run_training(config, n_jobs=1)
    assert outcome["beats_strong_baseline"]
    assert outcome["final_pipeline"] is not None


def test_run_training_feasibility_note_and_outlier_limitation_present(small_search_grids, synthetic_config):
    config, _dataset = synthetic_config
    outcome = run_training(config, n_jobs=1)
    metadata = outcome["metadata"]
    assert "feasibility_note" in metadata
    assert any("60 seconds" in item for item in metadata["known_limitations"])


def test_run_training_error_report_breaks_down_by_round_and_skips_absent_compound(small_search_grids, synthetic_config):
    config, _dataset = synthetic_config
    outcome = run_training(config, n_jobs=1)
    assert "round" in outcome["error_report"]["tables"]
    assert "compound" in outcome["error_report"]["skipped"]


def test_load_pit_stop_dataset_drops_rows_with_null_target(tmp_path):
    dataset = _synthetic_pit_stop_dataset(n_races=2, stops_per_race=4)
    dataset.loc[0, "duration_s"] = None
    config = _config(tmp_path, dataset)
    loaded = load_pit_stop_dataset(config)
    assert len(loaded) == len(dataset) - 1
    assert loaded["duration_s"].notna().all()


def test_run_training_raises_a_clear_error_on_a_single_race_dataset(tmp_path):
    # Edge case: leave-one-race-out CV is meaningless with only one race -
    # race_group_kfold cannot produce more than one fold, so there is no
    # out-of-fold test set at all. This should fail loudly, not silently
    # produce a degenerate "evaluated on its own training data" result.
    dataset = _synthetic_pit_stop_dataset(n_races=1, stops_per_race=6)
    config = _config(tmp_path, dataset)
    with pytest.raises(ValueError):
        run_training(config, n_jobs=1)
