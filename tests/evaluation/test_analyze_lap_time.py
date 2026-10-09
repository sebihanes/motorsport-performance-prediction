"""Phase 6 analysis wiring, on a small synthetic lap_time-shaped dataset
(same philosophy as tests/models/test_train_lap_time.py: shape/wiring
correctness, not real-data numbers)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from mpp.evaluation.analyze_lap_time import (
    FEATURE_GROUPS,
    add_analysis_columns,
    analyze,
    annotate_failure,
    attach_context,
    derive_tyre_age_columns,
    write_figures,
    write_reports,
)
from mpp.models.regressors import MODEL_FEATURE_COLUMNS

N_RACES = 4


def _dataset(laps: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for rnd in range(1, N_RACES + 1):
        base = 85.0 + 4.0 * rnd
        for driver in ["VER", "HAM"]:
            for lap in range(1, laps + 1):
                stint = 1 if lap <= laps // 2 else 2
                age = lap if stint == 1 else lap - laps // 2
                compound = "MEDIUM" if stint == 1 else "HARD"
                status = "4" if lap == 10 else "1"
                rows.append(
                    {
                        "season": 2023,
                        "round": rnd,
                        "circuit_id": f"c{rnd}",
                        "driver": driver,
                        "team": "T1" if driver == "VER" else "T2",
                        "lap_number": lap,
                        "race_fraction": lap / laps,
                        "stint": stint,
                        "tyre_age": age,
                        "tyre_age_sq": float(age**2),
                        "laps_into_stint": age,
                        "air_temp": 20.0 + rnd + rng.normal(0, 0.1),
                        "track_temp": 30.0 + rnd + rng.normal(0, 0.1),
                        "rainfall": False,
                        "driver_prior_relative_pace_s": np.nan if rnd == 1 else rng.normal(),
                        "driver_prior_relative_pace_std_s": np.nan if rnd == 1 else abs(rng.normal()),
                        "team_prior_relative_pace_s": np.nan if rnd == 1 else rng.normal(),
                        "driver_circuit_prior_relative_pace_s": np.nan,
                        "stint_prior_mean_lap_time_s": np.nan if age == 1 else base + rng.normal(),
                        "tyre_age_x_soft": 0.0,
                        "tyre_age_x_medium": float(age) if compound == "MEDIUM" else 0.0,
                        "tyre_age_x_hard": float(age) if compound == "HARD" else 0.0,
                        "tyre_age_x_intermediate": 0.0,
                        "tyre_age_x_wet": 0.0,
                        "compound": compound,
                        "session_type": "R",
                        "lap_time_s": base + 0.05 * age + (25.0 if status == "4" else 0.0) + rng.normal(0, 0.3),
                        "track_status": status,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def outcome():
    return analyze(_dataset(), "ridge", [{"alpha": 1.0}] * N_RACES, n_repeats=2)


def test_analyze_produces_out_of_fold_predictions_for_every_row(outcome):
    frame = outcome["frame"]
    assert frame["predicted_lap_time_s"].notna().all()
    assert np.allclose(frame["residual"], frame["predicted_lap_time_s"] - frame["lap_time_s"])
    assert len(outcome["per_fold_metrics"]) == N_RACES
    assert sorted(r[0] for r in outcome["held_out_per_fold"]) == [f"2023_{i}" for i in range(1, N_RACES + 1)]
    assert outcome["mean_fold_mae"] == pytest.approx(np.mean([m["mae"] for m in outcome["per_fold_metrics"]]))


def test_analyze_interpretability_tables_cover_every_feature(outcome):
    assert set(outcome["permutation_grouped"]["feature"]) == set(FEATURE_GROUPS)
    assert set(outcome["permutation_single"]["feature"]) == set(MODEL_FEATURE_COLUMNS)
    assert {"importance_mean_across_races", "importance_mean_within_race"} <= set(outcome["permutation_grouped"])
    assert set(outcome["partial_dependence_tyre_age"]) == {"MEDIUM", "HARD"}


def test_analyze_error_tables_and_skips(outcome):
    assert {"driver", "team", "circuit", "compound", "tyre_age", "track_status", "lap_phase"} <= set(
        outcome["group_tables"]
    )
    assert "session" in outcome["skipped_dimensions"]  # constant "R"
    status = outcome["group_tables"]["track_status"].set_index("track_status_class")
    # safety-car laps are 25s slower and track status is not a feature -> underpredicted
    assert status.loc["safety_car", "mean_bias"] < -10
    share = outcome["between_race_error_share"]
    assert share["mse_between"] + share["mse_within"] == pytest.approx(share["mse_total"])


def test_failure_examples_are_annotated_with_evidence(outcome):
    examples = outcome["failure_examples"]
    assert examples
    assert all(ex["evidence"] for ex in examples if abs(ex["residual_s"]) > 2)
    worst = max(examples, key=lambda ex: abs(ex["residual_s"]))
    assert any("safety car" in item for item in worst["evidence"])


def test_annotate_failure_marks_unexplained_when_no_recorded_cause():
    row = pd.Series(
        {
            "residual": 8.0,
            "track_status_class": "green",
            "track_status": "1",
            "lap_number": 20,
            "rainfall": False,
            "compound": "HARD",
            "stint_prior_mean_lap_time_s": 90.0,
            "race_median_lap_time_s": 90.5,
            "prior_neutralised_laps_in_stint": 0,
            "driver_prior_relative_pace_s": 0.1,
        }
    )
    note = annotate_failure(row, race_bias=0.2)
    assert note["unexplained_by_data"] is True
    assert "unexplained" in note["evidence"][-1]


def test_annotate_failure_flags_sign_inconsistent_cause_and_race_offset():
    row = pd.Series(
        {
            "residual": 10.0,  # predicted too slow
            "track_status_class": "safety_car",  # SC predicts *under*prediction
            "track_status": "4",
            "lap_number": 5,
            "rainfall": False,
            "compound": "HARD",
            "stint_prior_mean_lap_time_s": 90.0,
            "race_median_lap_time_s": 90.0,
            "prior_neutralised_laps_in_stint": 0,
            "driver_prior_relative_pace_s": 0.1,
        }
    )
    note = annotate_failure(row, race_bias=4.0)
    assert note["residual_after_race_offset_s"] == pytest.approx(6.0)
    assert note["evidence"][0].startswith("race-level offset")
    assert "NOT consistent" in note["evidence"][1]
    assert note["unexplained_by_data"] is True


def test_annotate_failure_explains_inflated_stint_history():
    row = pd.Series(
        {
            "residual": 30.0,
            "track_status_class": "green",
            "track_status": "1",
            "lap_number": 21,
            "rainfall": False,
            "compound": "MEDIUM",
            "stint_prior_mean_lap_time_s": 140.0,
            "race_median_lap_time_s": 95.0,
            "prior_neutralised_laps_in_stint": 1,
            "driver_prior_relative_pace_s": 0.1,
        }
    )
    note = annotate_failure(row, race_bias=0.0)
    assert note["unexplained_by_data"] is False
    assert "out of band" in note["evidence"][0] and "consistent with" in note["evidence"][0]


def test_add_analysis_columns_counts_prior_neutralised_laps_in_stint():
    data = add_analysis_columns(_dataset(laps=30))
    one_stint = data[(data["round"] == 1) & (data["driver"] == "VER") & (data["stint"] == 1)].sort_values("lap_number")
    counts = one_stint.set_index("lap_number")["prior_neutralised_laps_in_stint"]
    assert counts.loc[10] == 0  # the SC lap itself is not "prior"
    assert counts.loc[11] == 1
    assert (data["lap_phase"].isin(["race lap 1", "first lap of stint", "later stint lap"])).all()


def test_derive_tyre_age_columns_recomputes_interactions():
    frame = pd.DataFrame({"tyre_age": [7.0, 7.0], "compound": ["SOFT", "HARD"]})
    out = derive_tyre_age_columns(frame)
    assert list(out["tyre_age_sq"]) == [49.0, 49.0]
    assert list(out["tyre_age_x_soft"]) == [7.0, 0.0]
    assert list(out["tyre_age_x_hard"]) == [0.0, 7.0]


def test_attach_context_preserves_rows_and_rejects_duplicates():
    data = _dataset(laps=5).drop(columns=["track_status"])
    context = data[["season", "round", "driver", "lap_number"]].assign(track_status="1")
    merged = attach_context(data, context)
    assert len(merged) == len(data) and (merged["track_status"] == "1").all()
    with pytest.raises(Exception):
        attach_context(data, pd.concat([context, context.iloc[:1]]))


def test_write_outputs(outcome, tmp_path):
    figures = write_figures(outcome, tmp_path / "figs")
    assert figures and all(p.exists() for p in figures)
    write_reports(outcome, tmp_path / "quality", phase4_mae=None)
    payload = json.loads((tmp_path / "quality" / "interpretability_lap_time.json").read_text(encoding="utf-8"))
    assert payload["model_name"] == "ridge"
    assert "unobservable_causes_note" in payload
    assert (tmp_path / "quality" / "interpretability_lap_time.md").read_text(encoding="utf-8").startswith("# ")
