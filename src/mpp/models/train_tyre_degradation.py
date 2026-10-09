"""Phase 5 training entry point for the tyre-degradation model (architecture
report SS17: "reuses Phase 4 infrastructure with a different target and
per-compound slicing").

Target: ``lap_time_delta_s`` - lap time relative to the stint's own baseline
lap (``mpp.datasets.tyre_degradation``). Protocol: the same nested, race-
grouped leave-one-race-out CV ``mpp.models.train_lap_time`` established
(``mpp.models.train_common``) - laps within a stint/race are correlated, the
same reasoning lap_time uses (docs/05_split_strategy.md).

``driver_circuit_prior_relative_pace_s`` is excluded from the model feature
set for the same reason as lap_time: 100% null in the current 6-round,
zero-repeated-circuit scope, so it carries no information and a median
imputer has nothing to compute from.

Association vs. causation (architecture SS13, SS18): a compound or
tyre-age coefficient/importance here describes *predictive association*
with the lap-time delta, not an isolated physical tyre-wear effect - the
delta also embeds fuel-burn-driven speedup over the same laps, which this
project cannot separate without fuel-mass telemetry. This is stated
wherever a result could be misread as a wear-mechanism claim.

Usage: ``python -m mpp.models.train_tyre_degradation``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from mpp.config import PipelineConfig, load_config
from mpp.datasets.tyre_degradation import NOMINAL_FEATURE_COLUMNS, NUMERIC_FEATURE_COLUMNS
from mpp.evaluation.error_analysis import bucket_numeric, error_breakdown, largest_absolute_errors, to_markdown_table
from mpp.evaluation.metrics import interpret_mae
from mpp.models.baselines import PerGroupLinearTrendRegressor
from mpp.models.train_common import (
    NestedTrainingSpec,
    fit_final_pipeline,
    run_nested_race_grouped_training,
    sha256_of_file,
)
import platform
from datetime import datetime, timezone
import sklearn

TARGET_COLUMN = "lap_time_delta_s"
MODEL_VERSION = "tyre_degradation-v1"

# 100% null in the current 6-round, zero-repeated-circuit scope - same
# exclusion reasoning mpp.models.regressors applies to the lap_time model.
DEGENERATE_NUMERIC_COLUMNS = ["driver_circuit_prior_relative_pace_s"]
MODEL_NUMERIC_COLUMNS = [c for c in NUMERIC_FEATURE_COLUMNS if c not in DEGENERATE_NUMERIC_COLUMNS]
MODEL_NOMINAL_COLUMNS = list(NOMINAL_FEATURE_COLUMNS)

TRAINING_NOTES: dict[str, dict[str, str]] = {
    "mean_baseline": {
        "strengths": "Trivial sanity floor.",
        "weaknesses": "Ignores compound and tyre age entirely - the two features degradation is actually about.",
        "training_considerations": "No fitting cost.",
    },
    "median_baseline": {
        "strengths": "Robust to outlier laps (incidents, SC-affected laps not yet filtered from this dataset).",
        "weaknesses": "Same blindness to compound/tyre age as the mean baseline.",
        "training_considerations": "No fitting cost.",
    },
    "strong_baseline": {
        "strengths": "Captures the single largest compound-level difference in degradation behavior (soft vs. hard) with zero learned parameters - the bar this model must clear.",
        "weaknesses": "Blind to tyre age within a compound, circuit, and driver/team form - explains none of the within-compound degradation curve.",
        "training_considerations": "No fitting cost beyond a group-by; group column is `compound`, not `circuit_id` - the dominant source of degradation variance is which tyre, not which track.",
    },
    "per_compound_linear_trend_baseline": {
        "strengths": "Architecture SS10's specified strong baseline: a straight line of delta vs. tyre age per compound - the classic 'degradation slope per compound' an engineer would draw first, with two parameters per compound.",
        "weaknesses": "One slope per compound for every circuit, driver and fuel load; cannot see temperature, race progress or driver form.",
        "training_considerations": "np.polyfit per compound on the training fold; compounds with fewer than 5 training rows fall back to the global median. Implemented in the final audit (docs/FINAL_AUDIT.md) - it was specified in SS10 but missing until then.",
    },
    "linear_regression": {
        "strengths": "Interpretable coefficients - the tyre_age_delta coefficient is directly readable as an average seconds-per-lap delta trend.",
        "weaknesses": "Assumes a single linear degradation slope; cannot flex per compound or per circuit without interaction terms.",
        "training_considerations": "Median-imputed, standardized numeric features.",
    },
    "ridge": {
        "strengths": "Same interpretability as linear regression with regularization against the one-hot compound collinearity.",
        "weaknesses": "Same linear-slope ceiling.",
        "training_considerations": "Tuned over alpha via nested GridSearchCV (5 candidates) per outer fold.",
    },
    "random_forest": {
        "strengths": "Can fit a non-linear, per-compound degradation curve shape without manually specified interaction terms.",
        "weaknesses": "Cannot extrapolate to tyre ages beyond what a training race's stints actually reached; same small-scope, zero-circuit-repeat ceiling documented for lap_time.",
        "training_considerations": "Tuned via nested RandomizedSearchCV (8 sampled candidates) per outer fold; numeric features median-imputed.",
    },
}

ERROR_ANALYSIS_GROUP_COLUMNS = ["driver", "team", "circuit_id", "compound", "session_type"]
TYRE_AGE_BUCKET_EDGES = [0, 5, 10, 15, 100]
TYRE_AGE_BUCKET_LABELS = ["0-5", "6-10", "11-15", "16+"]


def load_tyre_degradation_dataset(config: PipelineConfig) -> pd.DataFrame:
    path = config.paths.model_ready_dir / "tyre_degradation.parquet"
    dataset = pd.read_parquet(path)
    return dataset.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)


def run_training(config: PipelineConfig | None = None, n_jobs: int = 1) -> dict:
    config = config or load_config()
    dataset_path = config.paths.model_ready_dir / "tyre_degradation.parquet"
    dataset = load_tyre_degradation_dataset(config)

    spec = NestedTrainingSpec(
        target_column=TARGET_COLUMN,
        numeric_columns=MODEL_NUMERIC_COLUMNS,
        nominal_columns=MODEL_NOMINAL_COLUMNS,
        strong_baseline_group_column="compound",
        training_notes=TRAINING_NOTES,
        extra_strong_baselines={
            "per_compound_linear_trend_baseline": lambda: PerGroupLinearTrendRegressor(
                group_column="compound", x_column="tyre_age_delta"
            )
        },
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
    error_frame["tyre_age_bucket"] = bucket_numeric(
        error_frame["laps_into_stint"], TYRE_AGE_BUCKET_EDGES, TYRE_AGE_BUCKET_LABELS
    )
    error_report = error_breakdown(
        error_frame, TARGET_COLUMN, "_y_pred", ERROR_ANALYSIS_GROUP_COLUMNS + ["tyre_age_bucket"]
    )
    worst_rows = largest_absolute_errors(
        error_frame, TARGET_COLUMN, "_y_pred", n=10, extra_columns=["season", "round", "driver", "compound"]
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
        "strong_baseline": outcome["strongest_baseline"],
        "strong_baseline_mae_mean": outcome["strong_baseline_mae_mean"],
        "beats_strong_baseline": outcome["beats_strong_baseline"],
        "best_candidate_model": outcome["best_row"]["model"],
        "best_candidate_mae_mean": outcome["best_row"]["mae_mean"],
        "known_limitations": [
            "Target conflates tyre wear with fuel burn within the stint - not separable without fuel-mass "
            "telemetry (architecture report SS18).",
            "Safety-car/VSC-affected laps are not filtered out of this dataset - their artificially slower "
            "pace is a speed-limit artifact, not degradation, and is not distinguished from it here.",
            "Zero repeated circuits in the current 6-round scope: driver_circuit_prior_relative_pace_s is "
            "100% null and excluded from every model's feature set.",
        ],
        "causation_caveat": (
            "Coefficients/importances here describe predictive association between compound/tyre age and "
            "the lap-time delta, not an isolated causal tyre-wear mechanism (architecture report SS13)."
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
    with open(out_dir / "model_comparison_tyre_degradation.json", "w", encoding="utf-8") as fh:
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
    lines = ["# Tyre Degradation Model Comparison", ""]
    lines.append(
        "Nested, race-grouped, leave-one-race-out cross-validation (docs/05_split_strategy.md) - each model "
        "scored once per race it never trained or tuned on. Target: lap-time delta relative to each stint's "
        "own baseline lap."
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
        "`model_comparison_tyre_degradation.json`'s metadata, and docs/PHASE_5_REPORT.md."
    )
    with open(out_dir / "model_comparison_tyre_degradation.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> int:
    config = load_config()
    outcome = run_training(config, n_jobs=-1)

    print(f"Races in scope ({outcome['n_outer_folds']} leave-one-race-out folds): {outcome['all_races']}")
    for row in outcome["results"]:
        print(
            f"{row['model']:<20} MAE={row['mae_mean']:.4f}+/-{row['mae_std']:.4f}s "
            f"RMSE={row['rmse_mean']:.4f}s R2={row['r2_mean']:.3f}"
        )

    _write_comparison_report(outcome["results"], outcome["error_report"], outcome["worst_rows"], config.paths.quality_dir)
    print(f"Comparison report written to {config.paths.quality_dir}")

    if outcome["beats_strong_baseline"]:
        best_row = outcome["best_row"]
        print(f"\nBest model: {best_row['model']} ({interpret_mae(best_row['mae_mean'])})")
        models_dir = config.paths.models_dir
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "tyre_degradation_model.joblib"
        joblib.dump(outcome["final_pipeline"], model_path)
        with open(models_dir / "tyre_degradation_model_metadata.json", "w", encoding="utf-8") as fh:
            json.dump(outcome["metadata"], fh, indent=2)
        print(f"Serialized best model to {model_path}")
        return 0

    models_dir = config.paths.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    stale = models_dir / "tyre_degradation_model.joblib"
    if stale.exists():
        stale.unlink()  # never leave an artifact that no longer clears the bar
    with open(models_dir / "tyre_degradation_model_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(outcome["metadata"], fh, indent=2)
    print(
        f"\nNo candidate model beat the strongest baseline ({outcome['metadata']['strong_baseline']}, "
        f"MAE {outcome['metadata']['strong_baseline_mae_mean']:.3f}s) - nothing serialized; metadata written."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
