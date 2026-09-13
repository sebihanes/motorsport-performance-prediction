"""Phase 5/6 training entry point for the race-pace model (architecture
report SS17: "reuses Phase 4 features, chronological split").

Target: ``session_pace_s`` - a driver's median clean-lap time for a session
(``mpp.datasets.race_pace``). Validation is **chronological**, not grouped:
docs/05_split_strategy.md is explicit that this problem's real deployment
question is forward-looking ("expected pace before the next race, given
history so far"), so a grouped/random split that let the model see a later
round during training would flatter it with hindsight a real forecast would
never have. ``mpp.splits.temporal_split.time_series_splits`` provides this -
round 1 can never be a test fold, since there is no earlier round to learn
from, which is a structural consequence of the question being asked, not a
bug.

No hyperparameter tuning is attempted here, unlike the lap_time/tyre
degradation/pit-stop problems. With only 6 rounds, the smallest outer
training fold has a single round (~20 rows); nesting a nested CV for tuning
inside would further shrink that to a handful of rows per inner fold - not
enough to choose hyperparameters meaningfully, and exactly the kind of
brute-force-on-too-little-data the brief warns against. Candidates use
fixed, conservative hyperparameters instead, documented inline.

Usage: ``python -m mpp.models.train_race_pace``
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

from mpp.config import PipelineConfig, load_config
from mpp.datasets.race_pace import NUMERIC_FEATURE_COLUMNS
from mpp.evaluation.error_analysis import error_breakdown, largest_absolute_errors, to_markdown_table
from mpp.evaluation.metrics import aggregate_fold_metrics, interpret_mae, regression_metrics
from mpp.models.baselines import PerGroupMedianRegressor, mean_baseline, median_baseline
from mpp.models.generic_pipelines import build_pipeline
from mpp.models.train_common import sha256_of_file
from mpp.splits.temporal_split import time_series_splits

TARGET_COLUMN = "session_pace_s"
MODEL_VERSION = "race_pace-v1"
STRONG_BASELINE_GROUP_COLUMN = "driver"

# 100% null in the current 6-round, zero-repeated-circuit scope - same
# exclusion reasoning as lap_time/tyre_degradation.
DEGENERATE_NUMERIC_COLUMNS = ["driver_circuit_prior_relative_pace_s"]
MODEL_NUMERIC_COLUMNS = [c for c in NUMERIC_FEATURE_COLUMNS if c not in DEGENERATE_NUMERIC_COLUMNS]

# Fixed, not tuned - see module docstring.
RIDGE_ALPHA = 1.0
RF_PARAMS = dict(n_estimators=200, max_depth=3, min_samples_leaf=5)

TRAINING_NOTES: dict[str, dict[str, str]] = {
    "mean_baseline": {"strengths": "Trivial floor.", "weaknesses": "Ignores driver skill entirely.", "training_considerations": "No fitting cost."},
    "median_baseline": {"strengths": "Robust to outlier sessions.", "weaknesses": "Same blindness to driver skill.", "training_considerations": "No fitting cost."},
    "strong_baseline": {
        "strengths": "Captures the dominant source of pace variance (driver skill/car package) using only that driver's own prior training-fold sessions - the bar this model must clear.",
        "weaknesses": "Blind to track/weather context and recent form trend; falls back to the global median for a driver unseen in training.",
        "training_considerations": "No fitting cost beyond a group-by, grouped on `driver`.",
    },
    "linear_regression": {
        "strengths": "Interpretable; uses prior-round relative pace directly, which is known before the session starts.",
        "weaknesses": "Assumes a linear relationship between prior relative pace and absolute session pace; only ~80-99 usable rows to fit on.",
        "training_considerations": "Median-imputed, standardized numeric features.",
    },
    "ridge": {
        "strengths": "Regularized against collinearity between the prior-pace features.",
        "weaknesses": "alpha fixed at 1.0, not tuned (see module docstring) - a deliberately conservative, not optimized, choice.",
        "training_considerations": "Median-imputed, standardized; fit once per outer fold, no inner CV.",
    },
    "random_forest": {
        "strengths": "Can capture non-linear effects of weather/prior pace without manual interaction terms.",
        "weaknesses": "Hyperparameters fixed and deliberately shallow (max_depth=3, min_samples_leaf=5) to avoid overfitting the smallest training folds (~20 rows); not tuned.",
        "training_considerations": "Numeric features median-imputed; no nominal features in this dataset.",
    },
}

ERROR_ANALYSIS_GROUP_COLUMNS = ["driver", "team", "circuit_id", "compound", "tyre_age", "session_type"]


def load_race_pace_dataset(config: PipelineConfig) -> pd.DataFrame:
    path = config.paths.model_ready_dir / "race_pace.parquet"
    dataset = pd.read_parquet(path)
    dataset = dataset.copy()
    dataset["rainfall_any"] = dataset["rainfall_any"].astype(float)
    return dataset.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)


def _fit_predict(builder, train: pd.DataFrame, test: pd.DataFrame, feature_columns: list[str]):
    model = builder().fit(train[feature_columns], train[TARGET_COLUMN])
    return model, model.predict(test[feature_columns])


def run_training(config: PipelineConfig | None = None) -> dict:
    config = config or load_config()
    dataset_path = config.paths.model_ready_dir / "race_pace.parquet"
    dataset = load_race_pace_dataset(config)

    n_rounds = dataset["round"].nunique()
    outer_folds = list(time_series_splits(dataset, n_splits=max(n_rounds - 1, 1)))
    validation_strategy = (
        f"chronological (time_series_splits), forward-chaining over {n_rounds} rounds: {len(outer_folds)} folds, "
        "each trained only on rounds strictly before its test round(s). Round 1 is never a test fold (no prior "
        "round exists to train on) - a structural consequence of this being a forward-looking prediction "
        "question (docs/05_split_strategy.md), not a bug. No hyperparameter tuning - see module docstring."
    )

    strong_baseline_feature_columns = MODEL_NUMERIC_COLUMNS + [STRONG_BASELINE_GROUP_COLUMN]
    baseline_builders = {
        "mean_baseline": (mean_baseline, MODEL_NUMERIC_COLUMNS),
        "median_baseline": (median_baseline, MODEL_NUMERIC_COLUMNS),
        "strong_baseline": (
            lambda: PerGroupMedianRegressor(group_column=STRONG_BASELINE_GROUP_COLUMN),
            strong_baseline_feature_columns,
        ),
    }
    candidate_builders = {
        "linear_regression": lambda: build_pipeline(
            LinearRegression(), MODEL_NUMERIC_COLUMNS, [], scale_numeric=True, impute_numeric=True
        ),
        "ridge": lambda: build_pipeline(
            Ridge(alpha=RIDGE_ALPHA, random_state=0), MODEL_NUMERIC_COLUMNS, [], scale_numeric=True, impute_numeric=True
        ),
        "random_forest": lambda: build_pipeline(
            RandomForestRegressor(random_state=0, n_jobs=-1, **RF_PARAMS),
            MODEL_NUMERIC_COLUMNS,
            [],
            scale_numeric=False,
            impute_numeric=True,
        ),
    }

    results: list[dict] = []
    oof_predictions: dict[str, np.ndarray] = {}

    def _run(name, category, builder, feature_columns):
        fold_metrics = []
        oof = np.full(len(dataset), np.nan)
        for train_idx, test_idx in outer_folds:
            train = dataset.iloc[train_idx]
            test = dataset.iloc[test_idx]
            _model, preds = _fit_predict(builder, train, test, feature_columns)
            fold_metrics.append(regression_metrics(test[TARGET_COLUMN], preds))
            oof[test_idx] = preds
        aggregated = aggregate_fold_metrics(fold_metrics)
        row = {
            "model": name,
            "category": category,
            "validation_strategy": validation_strategy,
            "n_outer_folds": len(fold_metrics),
            "mae_mean": aggregated["mae_mean"],
            "mae_std": aggregated["mae_std"],
            "rmse_mean": aggregated["rmse_mean"],
            "rmse_std": aggregated["rmse_std"],
            "r2_mean": aggregated["r2_mean"],
            "r2_std": aggregated["r2_std"],
            "median_ae_mean": aggregated["median_ae_mean"],
            "median_ae_std": aggregated["median_ae_std"],
            **TRAINING_NOTES.get(name, {}),
        }
        results.append(row)
        oof_predictions[name] = oof
        return row

    for name, (builder, cols) in baseline_builders.items():
        _run(name, "baseline", builder, cols)

    # linear_regression is categorized as a baseline (like lap_time/tyre_degradation/pit_stop):
    # it is the brief's "simple linear regression" reference point, not a tuned candidate eligible
    # to be selected as the best model - only ridge and random_forest are.
    _run("linear_regression", "baseline", candidate_builders["linear_regression"], MODEL_NUMERIC_COLUMNS)
    for name in ("ridge", "random_forest"):
        _run(name, "model", candidate_builders[name], MODEL_NUMERIC_COLUMNS)

    strong_baseline_mae = next(r["mae_mean"] for r in results if r["model"] == "strong_baseline")
    model_rows = [r for r in results if r["category"] == "model"]
    best_row = min(model_rows, key=lambda r: r["mae_mean"])
    beats_strong_baseline = best_row["mae_mean"] < strong_baseline_mae

    final_pipeline = None
    if beats_strong_baseline:
        final_pipeline = candidate_builders[best_row["model"]]()
        final_pipeline.fit(dataset[MODEL_NUMERIC_COLUMNS], dataset[TARGET_COLUMN])

    best_oof = oof_predictions[best_row["model"]]
    error_frame = dataset.copy()
    error_frame["_y_pred"] = best_oof
    n_untested = int(np.isnan(best_oof).sum())
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
        "feature_columns": MODEL_NUMERIC_COLUMNS,
        "validation_strategy": validation_strategy,
        "n_rounds": n_rounds,
        "n_outer_folds": len(outer_folds),
        "best_model": best_row["model"] if beats_strong_baseline else None,
        "strong_baseline_mae_mean": strong_baseline_mae,
        "beats_strong_baseline": beats_strong_baseline,
        "known_limitations": [
            "session_pace_s is an absolute lap time, dominated by which circuit the session was at (like "
            "lap_time - see docs/PHASE_4_REPORT.md), but this dataset carries no circuit-identity feature "
            "(only prior-pace/weather, per mpp.datasets.race_pace) - verified by inspection that a model "
            "trained on round 1 alone predicts round 2's pace band off by roughly 30 seconds. This is a "
            "feature-availability limitation of the current dataset design, not an implementation bug.",
            f"Only {len(dataset)} rows (one per driver-session) across {n_rounds} rounds - every metric here "
            "is a small-sample estimate.",
            "Round 1's rows are never scored out-of-fold (no prior round exists to train on) - "
            f"{n_untested} of {len(dataset)} rows are excluded from the error-analysis breakdown for this reason.",
            "No hyperparameter tuning (see module docstring): outer training folds are too small to support "
            "a further nested split without the tuning signal becoming noise.",
            "driver_circuit_prior_relative_pace_s is 100% null in this scope (zero repeated circuits) and is "
            "excluded from the feature set.",
        ],
        "causation_caveat": (
            "A prior-relative-pace coefficient describes predictive association with this session's pace, "
            "not a causal claim about car development or driver form (architecture report SS13)."
        ),
    }

    return {
        "results": results,
        "best_row": best_row,
        "beats_strong_baseline": beats_strong_baseline,
        "final_pipeline": final_pipeline,
        "metadata": metadata,
        "n_outer_folds": len(outer_folds),
        "n_rounds": n_rounds,
        "error_report": error_report,
        "worst_rows": worst_rows,
    }


def _write_comparison_report(results: list[dict], error_report: dict, worst_rows: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model_comparison_race_pace.json", "w", encoding="utf-8") as fh:
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
    lines = ["# Race Pace Model Comparison", ""]
    lines.append(
        "Chronological, forward-chaining cross-validation (docs/05_split_strategy.md) - each model trained "
        "only on rounds strictly before the round(s) it is scored on. Target: a driver's median clean-lap "
        "time for a session."
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
    lines.append("**Caveat**: association, not causation - see docs/PHASE_5_REPORT.md.")
    with open(out_dir / "model_comparison_race_pace.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> int:
    config = load_config()
    outcome = run_training(config)

    print(f"Rounds in scope ({outcome['n_outer_folds']} forward-chaining folds over {outcome['n_rounds']} rounds)")
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
        model_path = models_dir / "race_pace_model.joblib"
        joblib.dump(outcome["final_pipeline"], model_path)
        with open(models_dir / "race_pace_model_metadata.json", "w", encoding="utf-8") as fh:
            json.dump(outcome["metadata"], fh, indent=2)
        print(f"Serialized best model to {model_path}")
        return 0

    print("\nNo candidate model beat the strong (per-driver-median) baseline - nothing serialized.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
