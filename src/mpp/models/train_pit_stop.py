"""Phase 5/7 training entry point for the pit-stop duration model
(architecture report SS17: Phase 7 is "gated on a feasibility check of the
ingested data's pit-stop granularity/volume").

**Feasibility verdict: attempted, not deferred, with the limitation stated
plainly rather than hidden.** Jolpica-F1 provides 219 per-stop rows across
the 6-round scope - enough rows for a race-grouped CV - but
``mpp.datasets.pit_stop`` only carries ``lap_number``/``stop_number`` as
features; ``team`` and ``circuit_id`` (pit-crew identity and pit-lane
layout - both genuinely known before a stop happens, not future
information) are added here at training time since they are already present
columns on the persisted dataset (its ``IDENTITY_COLUMNS``/``GROUP_COLUMNS``),
not fabricated. No tyre-compound-change or pit-lane-traffic signal is
available at all (not ingested by this project's sources), which is the real
ceiling on this problem - stated in ``known_limitations`` below, not papered
over with an invented feature.

A second, load-bearing limitation, found by inspection (not assumed): 49 of
219 stops (22%) exceed 60 seconds, and 10 of those - all from round 3 (the
2023 Australian Grand Prix, a race with multiple red flags) - exceed 1800
seconds. These are not slow tyre changes; Jolpica-F1's recorded "pit stop"
duration includes the red-flag stoppage time for cars parked in the pit lane
when the race was suspended. This project has no way to separate that from
a genuine pit-lane service time (docs/03_dataset_definitions.md already
documents that `duration_s` is not decomposed by cause) and does **not**
filter these rows out - the brief's "do not cherry-pick data to improve
metrics" applies directly. Instead, round 3 is expected to - and, stated
below, does - dominate this model's error when it is the held-out race,
which the by-race error breakdown makes visible rather than averaging away.

Usage: ``python -m mpp.models.train_pit_stop``
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from mpp.config import PipelineConfig, load_config
from mpp.datasets.pit_stop import FEATURE_COLUMNS
from mpp.evaluation.error_analysis import error_breakdown, largest_absolute_errors, to_markdown_table
from mpp.evaluation.metrics import interpret_mae
from mpp.models.train_common import NestedTrainingSpec, fit_final_pipeline, run_nested_race_grouped_training, sha256_of_file

TARGET_COLUMN = "duration_s"
MODEL_VERSION = "pit_stop-v1"

MODEL_NUMERIC_COLUMNS = list(FEATURE_COLUMNS)  # lap_number, stop_number
MODEL_NOMINAL_COLUMNS = ["team", "circuit_id"]

TRAINING_NOTES: dict[str, dict[str, str]] = {
    "mean_baseline": {
        "strengths": "Trivial floor.",
        "weaknesses": "Extremely sensitive to the red-flag-affected outlier stops (up to 1871s) - pulled far above the typical ~25s stop.",
        "training_considerations": "No fitting cost.",
    },
    "median_baseline": {
        "strengths": "Robust to the outlier stops - close to the typical clean tyre-change duration.",
        "weaknesses": "Ignores team/circuit/lap-number context entirely.",
        "training_considerations": "No fitting cost.",
    },
    "strong_baseline": {
        "strengths": "Captures team-level (pit crew) differences in stop duration - the most plausible dominant factor - with zero learned parameters.",
        "weaknesses": "Still produces one team-wide number regardless of lap/circuit/whether the stop happened during a red flag.",
        "training_considerations": "No fitting cost beyond a group-by, grouped on `team`.",
    },
    "linear_regression": {
        "strengths": "Interpretable.",
        "weaknesses": "A linear model cannot represent 'this one race's stops are structurally different' without an explicit circuit/race feature - which it has (circuit_id), but one-hot circuit dummies cannot separate 'Albert Park is a hard pit lane' from 'this specific race had a red flag'.",
        "training_considerations": "Median-imputed, standardized numeric features.",
    },
    "ridge": {
        "strengths": "Regularized against the team/circuit one-hot collinearity.",
        "weaknesses": "Same ceiling as linear regression for the round-3 anomaly.",
        "training_considerations": "Tuned over alpha via nested GridSearchCV (5 candidates) per outer fold.",
    },
    "random_forest": {
        "strengths": "Can isolate the circuit_id=albert_park leaf and assign it a distinct (if still averaged) prediction.",
        "weaknesses": "Still cannot distinguish a red-flag-affected stop from a normal one at that circuit using only lap_number/stop_number/team/circuit_id - no red-flag/track-status feature is wired into this dataset.",
        "training_considerations": "Tuned via nested RandomizedSearchCV (8 sampled candidates) per outer fold; numeric features median-imputed.",
    },
}

ERROR_ANALYSIS_GROUP_COLUMNS = ["driver", "team", "circuit_id", "round", "compound", "tyre_age", "session_type"]


def load_pit_stop_dataset(config: PipelineConfig) -> pd.DataFrame:
    path = config.paths.model_ready_dir / "pit_stop.parquet"
    dataset = pd.read_parquet(path)
    return dataset.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)


def run_training(config: PipelineConfig | None = None, n_jobs: int = 1) -> dict:
    config = config or load_config()
    dataset_path = config.paths.model_ready_dir / "pit_stop.parquet"
    dataset = load_pit_stop_dataset(config)

    n_outlier_stops = int((dataset[TARGET_COLUMN] > 60).sum())
    outlier_rounds = sorted(dataset.loc[dataset[TARGET_COLUMN] > 1800, "round"].unique().tolist())

    spec = NestedTrainingSpec(
        target_column=TARGET_COLUMN,
        numeric_columns=MODEL_NUMERIC_COLUMNS,
        nominal_columns=MODEL_NOMINAL_COLUMNS,
        strong_baseline_group_column="team",
        training_notes=TRAINING_NOTES,
    )
    outcome = run_nested_race_grouped_training(dataset, spec, n_jobs=n_jobs)

    final_pipeline = None
    final_best_params = None
    if outcome["beats_strong_baseline"]:
        final_pipeline, final_best_params = fit_final_pipeline(dataset, spec, outcome, n_jobs=n_jobs)

    best_name = outcome["best_row"]["model"]
    error_frame = dataset.copy()
    error_frame["_y_pred"] = outcome["oof_predictions"][best_name]
    error_frame = error_frame.dropna(subset=["_y_pred"])
    error_report = error_breakdown(error_frame, TARGET_COLUMN, "_y_pred", ERROR_ANALYSIS_GROUP_COLUMNS)
    worst_rows = largest_absolute_errors(
        error_frame, TARGET_COLUMN, "_y_pred", n=10, extra_columns=["season", "round", "driver", "team"]
    )

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "season": config.season,
        "rounds": list(config.rounds),
        "training_dataset_path": str(dataset_path),
        "training_dataset_sha256": sha256_of_file(dataset_path) if dataset_path.exists() else None,
        "training_dataset_n_rows": len(dataset),
        "target_column": TARGET_COLUMN,
        "feature_columns": spec.feature_columns,
        "validation_strategy": outcome["validation_strategy"],
        "all_races": outcome["all_races"],
        "n_outer_folds": outcome["n_outer_folds"],
        "best_model": outcome["best_row"]["model"] if outcome["beats_strong_baseline"] else None,
        "best_model_params": final_best_params,
        "strong_baseline_mae_mean": outcome["strong_baseline_mae_mean"],
        "beats_strong_baseline": outcome["beats_strong_baseline"],
        "feasibility_note": (
            "Attempted, not deferred (architecture report Phase 7 gate): 219 rows across 6 races is enough "
            "for a race-grouped CV, but the only genuinely available features are lap_number, stop_number, "
            "team and circuit_id - no tyre-compound-change or pit-lane-traffic signal is ingested by this "
            "project. See module docstring."
        ),
        "known_limitations": [
            f"{n_outlier_stops} of {len(dataset)} recorded stops exceed 60 seconds; stops from round(s) "
            f"{outlier_rounds} exceed 1800 seconds - these are red-flag stoppage time, not a slow tyre "
            "change (2023 round 3, the Australian GP, had multiple red flags). duration_s is not "
            "decomposed by cause (docs/03_dataset_definitions.md) and these rows are not filtered out - "
            "see the by-round error breakdown for how much they dominate that race's fold.",
            "No tyre-compound, weather, or pit-lane-traffic feature is available for this problem.",
            "duration_s is the full recorded stationary time, not isolated tyre-change time.",
        ],
        "causation_caveat": (
            "A team coefficient/importance here describes predictive association between team identity and "
            "recorded stop duration (pit crew skill is one plausible explanation, but round 3's red-flag "
            "stoppages are folded into the same column and are not a crew-skill signal at all)."
        ),
    }

    return {
        "results": outcome["results"],
        "best_row": outcome["best_row"],
        "beats_strong_baseline": outcome["beats_strong_baseline"],
        "final_pipeline": final_pipeline,
        "metadata": metadata,
        "all_races": outcome["all_races"],
        "n_outer_folds": outcome["n_outer_folds"],
        "error_report": error_report,
        "worst_rows": worst_rows,
    }


def _write_comparison_report(results: list[dict], error_report: dict, worst_rows: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model_comparison_pit_stop.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "results": results,
                "error_breakdown": {k: v.to_dict(orient="records") for k, v in error_report["tables"].items()},
                "error_breakdown_skipped": error_report["skipped"],
                "largest_absolute_errors": worst_rows.to_dict(orient="records"),
            },
            fh,
            indent=2,
            default=str,
        )

    columns = ["model", "category", "n_outer_folds", "mae_mean", "mae_std", "rmse_mean", "r2_mean", "median_ae_mean"]
    lines = ["# Pit Stop Duration Model Comparison", ""]
    lines.append(
        "Nested, race-grouped, leave-one-race-out cross-validation (docs/05_split_strategy.md). Target: "
        "full recorded pit-stop stationary time (seconds). See module docstring "
        "(mpp.models.train_pit_stop) for the feasibility assessment and the round-3 red-flag-stoppage "
        "caveat - not filtered out, visible in the by-round breakdown below."
    )
    lines.append("")
    lines.extend(to_markdown_table(pd.DataFrame(results)[columns]))
    lines.append("")
    for row in results:
        lines.append(f"### {row['model']}")
        lines.append("")
        lines.append(interpret_mae(row["mae_mean"]))
        lines.append("")
        lines.append(f"- Strengths: {row.get('strengths', 'n/a')}")
        lines.append(f"- Weaknesses: {row.get('weaknesses', 'n/a')}")
        lines.append(f"- Training considerations: {row.get('training_considerations', 'n/a')}")
        lines.append("")
    lines.append("## Error breakdown (out-of-fold predictions, best model)")
    lines.append("")
    for dim, table in error_report["tables"].items():
        lines.append(f"**By {dim}** (worst 5 of {len(table)}):")
        lines.append("")
        lines.extend(to_markdown_table(table.head(5)))
        lines.append("")
    for dim, reason in error_report["skipped"].items():
        lines.append(f"- `{dim}`: N/A - {reason}")
    lines.append("")
    lines.append(
        "**Caveat**: association, not causation - see `causation_caveat` in "
        "`model_comparison_pit_stop.json`'s metadata, and docs/PHASE_5_REPORT.md."
    )
    with open(out_dir / "model_comparison_pit_stop.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> int:
    config = load_config()
    outcome = run_training(config, n_jobs=-1)

    print(f"Races in scope ({outcome['n_outer_folds']} leave-one-race-out folds): {outcome['all_races']}")
    for row in outcome["results"]:
        print(
            f"{row['model']:<20} MAE={row['mae_mean']:.3f}+/-{row['mae_std']:.3f}s "
            f"RMSE={row['rmse_mean']:.3f}s R2={row['r2_mean']:.3f}"
        )

    _write_comparison_report(outcome["results"], outcome["error_report"], outcome["worst_rows"], config.paths.quality_dir)
    print(f"Comparison report written to {config.paths.quality_dir}")

    if outcome["beats_strong_baseline"]:
        best_row = outcome["best_row"]
        print(f"\nBest model: {best_row['model']} ({interpret_mae(best_row['mae_mean'])})")
        models_dir = config.paths.models_dir
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "pit_stop_model.joblib"
        joblib.dump(outcome["final_pipeline"], model_path)
        with open(models_dir / "pit_stop_model_metadata.json", "w", encoding="utf-8") as fh:
            json.dump(outcome["metadata"], fh, indent=2)
        print(f"Serialized best model to {model_path}")
        return 0

    print("\nNo candidate model beat the strong (per-team-median) baseline - nothing serialized.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
