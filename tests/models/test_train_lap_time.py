"""End-to-end integration test for the Phase 4 training protocol
(mpp.models.train_lap_time.run_training), on a small synthetic lap_time-shaped
dataset rather than the real 6-round data - fast, and, per the project's
existing integration-test philosophy (tests/test_pipeline_integration.py),
checked for shape/type/wiring correctness rather than exact numbers.

Hyperparameter grids are monkeypatched down to a couple of cheap candidates
so the nested RandomizedSearchCV/GridSearchCV calls stay fast; this does not
change production behavior (mpp.models.train_lap_time.main uses the real
grids).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mpp.models.tuning as tuning
from mpp.config import CleaningSettings, PipelineConfig, PipelinePaths, ValidationSettings
from mpp.models.regressors import MODEL_FEATURE_COLUMNS
from mpp.models.train_lap_time import run_training

EXPECTED_MODEL_NAMES = {
    "mean_baseline",
    "median_baseline",
    "per_circuit_median_baseline",
    "stint_pace_persistence_baseline",
    "linear_regression",
    "ridge",
    "random_forest",
    "gradient_boosting",
    "hist_gradient_boosting",
}
N_SYNTHETIC_RACES = 6


def _synthetic_lap_time_dataset(n_races: int = N_SYNTHETIC_RACES, laps_per_race: int = 24, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    circuits = ["bahrain", "jeddah", "albert_park", "baku", "miami", "monaco"]
    rows = []
    for round_ in range(1, n_races + 1):
        circuit = circuits[(round_ - 1) % len(circuits)]
        base_pace = 85.0 + 3.0 * round_
        for lap in range(1, laps_per_race + 1):
            tyre_age = (lap % 10) + 1
            compound = ["SOFT", "MEDIUM", "HARD"][lap % 3]
            rows.append(
                {
                    "season": 2023,
                    "round": round_,
                    "circuit_id": circuit,
                    "driver": f"D{lap % 4}",
                    "team": f"T{lap % 2}",
                    "lap_number": lap,
                    "race_fraction": lap / laps_per_race,
                    "stint": 1 + lap // 10,
                    "tyre_age": tyre_age,
                    "tyre_age_sq": float(tyre_age**2),
                    "laps_into_stint": tyre_age,
                    "air_temp": 25.0 + rng.normal(),
                    "track_temp": 35.0 + rng.normal(),
                    "rainfall": False,
                    "driver_prior_relative_pace_s": np.nan if round_ == 1 else rng.normal(),
                    "driver_prior_relative_pace_std_s": np.nan if round_ == 1 else abs(rng.normal()),
                    "team_prior_relative_pace_s": np.nan if round_ == 1 else rng.normal(),
                    "driver_circuit_prior_relative_pace_s": np.nan,  # degenerate in this scope, like the real data
                    "stint_prior_mean_lap_time_s": np.nan if tyre_age == 1 else base_pace,
                    "tyre_age_x_soft": float(tyre_age) if compound == "SOFT" else 0.0,
                    "tyre_age_x_medium": float(tyre_age) if compound == "MEDIUM" else 0.0,
                    "tyre_age_x_hard": float(tyre_age) if compound == "HARD" else 0.0,
                    "tyre_age_x_intermediate": 0.0,
                    "tyre_age_x_wet": 0.0,
                    "compound": compound,
                    "session_type": "R",
                    "lap_time_s": base_pace + rng.normal(scale=0.3),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def small_search_grids(monkeypatch):
    """Shrink every tuning search to a couple of cheap candidates - keeps this
    integration test fast without touching the real grids used in production
    (mpp.models.tuning module constants, looked up by name at call time)."""
    monkeypatch.setattr(tuning, "N_ITER_RANDOM_SEARCH", 2)
    monkeypatch.setattr(tuning, "RIDGE_PARAM_GRID", {"model__alpha": [1.0, 10.0]})
    monkeypatch.setattr(tuning, "RANDOM_FOREST_PARAM_DISTRIBUTIONS", {"model__n_estimators": [20, 40]})
    monkeypatch.setattr(tuning, "GRADIENT_BOOSTING_PARAM_DISTRIBUTIONS", {"model__n_estimators": [20, 40]})
    monkeypatch.setattr(tuning, "HIST_GRADIENT_BOOSTING_PARAM_DISTRIBUTIONS", {"model__max_iter": [20, 40]})


@pytest.fixture
def synthetic_config(tmp_path):
    model_ready_dir = tmp_path / "model_ready"
    model_ready_dir.mkdir()
    dataset = _synthetic_lap_time_dataset()
    dataset.to_parquet(model_ready_dir / "lap_time.parquet", index=False)

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
            lap_time_min_s=55.0,
            lap_time_max_s=160.0,
            valid_compounds=("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN"),
        ),
        cleaning=CleaningSettings(duplicate_key=("season", "round", "driver", "lap_number")),
        repo_root=tmp_path,
    ), dataset


def test_run_training_evaluates_every_baseline_and_candidate_model(small_search_grids, synthetic_config):
    config, _dataset = synthetic_config
    outcome = run_training(config, n_jobs=1)
    assert {row["model"] for row in outcome["results"]} == EXPECTED_MODEL_NAMES
    for row in outcome["results"]:
        for key in ("mae_mean", "mae_std", "rmse_mean", "r2_mean", "median_ae_mean"):
            assert np.isfinite(row[key])


def test_run_training_uses_leave_one_race_out_outer_folds(small_search_grids, synthetic_config):
    config, _dataset = synthetic_config
    outcome = run_training(config, n_jobs=1)
    assert outcome["n_outer_folds"] == N_SYNTHETIC_RACES
    assert len(outcome["all_races"]) == N_SYNTHETIC_RACES
    for row in outcome["results"]:
        assert row["n_outer_folds"] == N_SYNTHETIC_RACES


def test_run_training_tuned_models_carry_best_params_per_fold(small_search_grids, synthetic_config):
    config, _dataset = synthetic_config
    outcome = run_training(config, n_jobs=1)
    tuned = {row["model"]: row for row in outcome["results"] if row["category"] == "model"}
    assert set(tuned) == {"ridge", "random_forest", "gradient_boosting", "hist_gradient_boosting"}
    for row in tuned.values():
        assert len(row["best_params_per_fold"]) == N_SYNTHETIC_RACES
        assert all(row["best_params_per_fold"])  # every fold's dict is non-empty


def test_run_training_best_model_is_a_tuned_model_not_a_baseline(small_search_grids, synthetic_config):
    config, _dataset = synthetic_config
    outcome = run_training(config, n_jobs=1)
    assert outcome["best_row"]["category"] == "model"


def test_run_training_metadata_documents_dataset_and_split(small_search_grids, synthetic_config):
    config, dataset = synthetic_config
    outcome = run_training(config, n_jobs=1)
    metadata = outcome["metadata"]
    assert metadata["target_column"] == "lap_time_s"
    assert metadata["training_dataset_n_rows"] == len(dataset)
    assert metadata["feature_columns"] == MODEL_FEATURE_COLUMNS
    assert metadata["all_races"] == outcome["all_races"]
    assert metadata["n_outer_folds"] == N_SYNTHETIC_RACES
    assert "caveat" in metadata


def test_run_training_serializable_final_pipeline_predicts_on_full_feature_set(
    small_search_grids, synthetic_config
):
    config, dataset = synthetic_config
    outcome = run_training(config, n_jobs=1)
    if not outcome["beats_strong_baseline"]:
        pytest.skip("no candidate beat the strong baseline on this synthetic draw - nothing to serialize")
    preds = outcome["final_pipeline"].predict(dataset[MODEL_FEATURE_COLUMNS].head(5))
    assert len(preds) == 5
    assert np.isfinite(preds).all()


def test_run_training_gates_on_the_strongest_baseline(small_search_grids, synthetic_config):
    # Synthetic stint_prior_mean_lap_time_s equals each race's base pace and laps are
    # base pace + 0.3s noise, so pace persistence is near-perfect and must be the bar.
    config, _dataset = synthetic_config
    outcome = run_training(config, n_jobs=1)
    by_name = {r["model"]: r["mae_mean"] for r in outcome["results"]}
    expected = min(by_name["per_circuit_median_baseline"], by_name["stint_pace_persistence_baseline"])
    metadata = outcome["metadata"]
    assert metadata["strong_baseline_mae_mean"] == pytest.approx(expected)
    assert metadata["best_candidate_model"] == outcome["best_row"]["model"]
    assert outcome["beats_strong_baseline"] == (outcome["best_row"]["mae_mean"] < expected)
    if not outcome["beats_strong_baseline"]:
        assert outcome["final_pipeline"] is None
        assert metadata["best_model"] is None
