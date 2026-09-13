"""Integration test for the Phase 5/6 race-pace training protocol, on a
small synthetic race_pace-shaped dataset. Unlike lap_time/tyre_degradation,
this uses a forward-chaining temporal split and no hyperparameter tuning
(see mpp.models.train_race_pace's module docstring) - checked here for
wiring/shape correctness, not exact numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mpp.config import CleaningSettings, PipelineConfig, PipelinePaths, ValidationSettings
from mpp.models.train_race_pace import run_training

N_SYNTHETIC_ROUNDS = 6
EXPECTED_MODEL_NAMES = {"mean_baseline", "median_baseline", "strong_baseline", "linear_regression", "ridge", "random_forest"}


def _synthetic_race_pace_dataset(n_rounds: int = N_SYNTHETIC_ROUNDS, n_drivers: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    circuits = ["bahrain", "jeddah", "albert_park", "baku", "miami", "monaco"]
    driver_skill = {f"D{i}": rng.normal(scale=0.5) for i in range(n_drivers)}
    rows = []
    for round_ in range(1, n_rounds + 1):
        circuit = circuits[(round_ - 1) % len(circuits)]
        base_pace = 90.0 + 2.0 * round_
        for i in range(n_drivers):
            driver = f"D{i}"
            rows.append(
                {
                    "season": 2023,
                    "round": round_,
                    "circuit_id": circuit,
                    "driver": driver,
                    "team": f"T{i % 3}",
                    "driver_prior_relative_pace_s": np.nan if round_ == 1 else driver_skill[driver] + rng.normal(scale=0.1),
                    "driver_prior_relative_pace_std_s": np.nan if round_ == 1 else abs(rng.normal(scale=0.2)),
                    "team_prior_relative_pace_s": np.nan if round_ == 1 else rng.normal(scale=0.2),
                    "driver_circuit_prior_relative_pace_s": np.nan,
                    "air_temp_mean": 25.0 + rng.normal(),
                    "track_temp_mean": 35.0 + rng.normal(),
                    "rainfall_any": False,
                    "session_pace_s": base_pace + driver_skill[driver] + rng.normal(scale=0.3),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_config(tmp_path):
    model_ready_dir = tmp_path / "model_ready"
    model_ready_dir.mkdir()
    dataset = _synthetic_race_pace_dataset()
    dataset.to_parquet(model_ready_dir / "race_pace.parquet", index=False)

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
        rounds=tuple(range(1, N_SYNTHETIC_ROUNDS + 1)),
        session="R",
        paths=paths,
        validation=ValidationSettings(lap_time_min_s=55.0, lap_time_max_s=160.0, valid_compounds=("SOFT",)),
        cleaning=CleaningSettings(duplicate_key=("season", "round", "driver", "lap_number")),
        repo_root=tmp_path,
    ), dataset


def test_run_training_evaluates_every_baseline_and_candidate(synthetic_config):
    config, _dataset = synthetic_config
    outcome = run_training(config)
    assert {row["model"] for row in outcome["results"]} == EXPECTED_MODEL_NAMES
    for row in outcome["results"]:
        assert np.isfinite(row["mae_mean"])


def test_run_training_never_scores_round_one_out_of_fold(synthetic_config):
    config, dataset = synthetic_config
    outcome = run_training(config)
    assert outcome["n_outer_folds"] == N_SYNTHETIC_ROUNDS - 1
    round_one_rows = dataset.loc[dataset["round"] == 1]
    assert not round_one_rows.empty  # sanity: round 1 exists in the dataset...
    # ...but never appears in the error-analysis frame (never a test fold).
    for table in outcome["error_report"]["tables"].values():
        assert table["n"].sum() <= len(dataset) - len(round_one_rows)


def test_run_training_metadata_flags_small_sample_and_no_tuning(synthetic_config):
    config, _dataset = synthetic_config
    outcome = run_training(config)
    limitations = " ".join(outcome["metadata"]["known_limitations"])
    assert "tuning" in limitations.lower()
    assert "causation_caveat" in outcome["metadata"]


def test_run_training_best_model_beats_baseline_on_clean_synthetic_signal(synthetic_config):
    config, _dataset = synthetic_config
    outcome = run_training(config)
    assert outcome["beats_strong_baseline"]
    assert outcome["final_pipeline"] is not None


def test_run_training_serializable_final_pipeline_predicts(synthetic_config):
    config, dataset = synthetic_config
    outcome = run_training(config)
    if not outcome["beats_strong_baseline"]:
        pytest.skip("no candidate beat the strong baseline on this synthetic draw")
    from mpp.models.train_race_pace import MODEL_NUMERIC_COLUMNS

    preds = outcome["final_pipeline"].predict(dataset[MODEL_NUMERIC_COLUMNS].head(5))
    assert len(preds) == 5
    assert np.isfinite(preds).all()
