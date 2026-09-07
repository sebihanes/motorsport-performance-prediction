"""Phase 4 training entry point for the lap-time model - the anchor problem
(architecture SS17): baselines, candidate ML models, grouped cross-validation,
nested hyperparameter tuning, a structured model comparison, and
serialization of the winning pipeline, if one earns it.

Protocol - **fully nested, leave-one-race-out CV**, per docs/05_split_strategy.md's
"Hyperparameter tuning" section, which specifies the split functions here are
"fold-generator-shaped ... so a Phase 4 GridSearchCV/RandomizedSearchCV cv=
argument can consume them directly, nested inside an outer loop over the same
functions":

1. **Outer loop** - ``race_group_kfold`` over *every* race in the current
   scope, requesting more folds than there are races so the function caps to
   exactly one fold per race (leave-one-race-out). Each outer fold's held-out
   race is scored only once, after training on every other race.
2. **Inner loop** - for each outer fold, hyperparameters are tuned with
   ``GridSearchCV``/``RandomizedSearchCV`` whose ``cv=`` is a second
   ``race_group_kfold`` computed *only* over that outer fold's training
   races - so no race is ever used both to choose hyperparameters and to
   score the result, even transitively.
3. Each candidate's reported metric is the **mean (+/- std) across all outer
   folds** - one generalization estimate per model, not a single arbitrarily
   -drawn train/test split. With only 6 races in the current scope, a single
   75/25 holdout was found (during development) to be highly sensitive to
   *which* 1-2 races happened to land in test; leave-one-race-out uses every
   race as a held-out test exactly once and removes that sensitivity.
4. Nested CV estimates the generalization performance of the "tune, then
   fit" *procedure* - not one specific fitted instance. Per standard nested-CV
   practice, the model that is actually serialized is produced by running
   that same tuning procedure **one more time**, with an inner CV over *all*
   races, then fitting on the full dataset - a real deployment should not
   permanently withhold any race's data once evaluation is complete.
5. The best model (lowest mean MAE across outer folds among the ML
   candidates) is only serialized if it beats the strong per-circuit-median
   baseline (architecture SS10).

An MAE/RMSE/R^2 number on its own is not a "race-ready" claim - see the
caveat written into the generated comparison report, and
docs/PHASE_4_REPORT.md for the full discussion of what this scope's results
do and do not support.

Usage: ``python -m mpp.models.train_lap_time``
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from mpp.config import PipelineConfig, load_config
from mpp.evaluation.metrics import aggregate_fold_metrics, interpret_mae, regression_metrics
from mpp.models.baselines import PerGroupMedianRegressor, mean_baseline, median_baseline
from mpp.models.regressors import (
    MODEL_FEATURE_COLUMNS,
    gradient_boosting_pipeline,
    hist_gradient_boosting_pipeline,
    linear_regression_pipeline,
    random_forest_pipeline,
    ridge_pipeline,
)
from mpp.models.tuning import (
    tune_gradient_boosting,
    tune_hist_gradient_boosting,
    tune_random_forest,
    tune_ridge,
)
from mpp.splits.group_split import race_group_kfold, race_groups

TARGET_COLUMN = "lap_time_s"
BASELINE_FEATURE_COLUMNS = MODEL_FEATURE_COLUMNS + ["circuit_id"]
MODEL_VERSION = "lap_time-v1"

# Requesting more folds than there are races - race_group_kfold caps this to
# the number of distinct races actually present, giving leave-one-race-out at
# both the outer and (per outer fold) inner level automatically, however many
# rounds config/settings.toml's scope ends up covering.
MAX_FOLDS_REQUESTED = 1000

TRAINING_NOTES: dict[str, dict[str, str]] = {
    "mean_baseline": {
        "strengths": "Trivial to compute; sanity-checks that any learned model beats doing nothing.",
        "weaknesses": "Ignores every feature - always wrong by the full circuit-to-circuit pace spread.",
        "training_considerations": "No fitting cost; no hyperparameters.",
    },
    "median_baseline": {
        "strengths": "Robust to the outlier laps (incidents, safety cars) that skew the mean.",
        "weaknesses": "Still ignores circuit, tyre, weather and driver context entirely.",
        "training_considerations": "No fitting cost; no hyperparameters.",
    },
    "linear_regression": {
        "strengths": "Fast, fully interpretable coefficients; a genuine step up from the naive baselines once circuit/tyre context is usable as features.",
        "weaknesses": "Cannot model the tyre-age x compound x circuit interactions the tree models capture natively; sensitive to the one-hot dummy collinearity from circuit/compound; extrapolates linearly (and therefore badly) outside the training races' weather range.",
        "training_considerations": "Requires imputing and standardizing numeric features inside the pipeline (no closed-form NaN handling).",
    },
    "per_circuit_median_baseline": {
        "strengths": "Captures the single largest source of lap-time variance (which circuit) with zero learned parameters - the bar architecture SS10 says a real model must clear.",
        "weaknesses": "Blind to tyre age, compound, weather, and driver/team form - explains none of the within-circuit variation. In the current 6-round scope, every circuit appears exactly once, so this baseline is mathematically identical to the global median baseline for every leave-one-race-out fold - it will only start to differ once a circuit repeats within the ingested scope.",
        "training_considerations": "No fitting cost beyond a group-by; falls back to the global median for a circuit unseen in training (always true here).",
    },
    "ridge": {
        "strengths": "Regularization tames the collinearity from one-hot circuit/compound dummies while keeping coefficients directly inspectable.",
        "weaknesses": "Same linear-interaction ceiling and extrapolation risk as plain linear regression.",
        "training_considerations": "Tuned over alpha via GridSearchCV (5 candidates) on the inner race-grouped CV folds, per outer fold.",
    },
    "random_forest": {
        "strengths": "Captures nonlinear tyre-age x compound x circuit interactions without manual interaction terms; permutation importance available for free.",
        "weaknesses": "Larger serialized footprint than boosting; cannot extrapolate outside the training feature range - a real limitation here, since the held-out race's weather/pace features can fall entirely outside the training races' observed range.",
        "training_considerations": "Tuned via RandomizedSearchCV (8 sampled candidates) on the inner CV folds, per outer fold; numeric features imputed (median) since this implementation cannot consume NaN.",
    },
    "gradient_boosting": {
        "strengths": "Often strong accuracy on tabular data of this size; sequential error-correction can pick up subtler patterns than a single Random Forest.",
        "weaknesses": "Slower to train than HistGradientBoosting at comparable accuracy; more sensitive to learning-rate/depth choices; same extrapolation ceiling as other tree ensembles.",
        "training_considerations": "Tuned via RandomizedSearchCV (8 sampled candidates), per outer fold; numeric features imputed (median).",
    },
    "hist_gradient_boosting": {
        "strengths": "Native missing-value handling (no imputation needed - relevant for the patchy historical-pace features); fast even as more rounds are added.",
        "weaknesses": "Less directly interpretable than Ridge; needs permutation importance rather than reading off coefficients; same extrapolation ceiling as other tree ensembles.",
        "training_considerations": "Tuned via RandomizedSearchCV (8 sampled candidates), per outer fold; the only candidate evaluated without numeric imputation, by design.",
    },
}

BASELINE_BUILDERS = {
    "mean_baseline": mean_baseline,
    "median_baseline": median_baseline,
    "per_circuit_median_baseline": PerGroupMedianRegressor,
}

_TUNING_SPECS = [
    ("ridge", tune_ridge, ridge_pipeline),
    ("random_forest", tune_random_forest, random_forest_pipeline),
    ("gradient_boosting", tune_gradient_boosting, gradient_boosting_pipeline),
    ("hist_gradient_boosting", tune_hist_gradient_boosting, hist_gradient_boosting_pipeline),
]
_BUILDER_BY_NAME = {name: build_fn for name, _, build_fn in _TUNING_SPECS}
_TUNER_BY_NAME = {name: tuner_fn for name, tuner_fn, _ in _TUNING_SPECS}


def _strip_model_prefix(params: dict) -> dict:
    return {key.split("__", 1)[1]: value for key, value in params.items()}


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_lap_time_dataset(config: PipelineConfig) -> pd.DataFrame:
    path = config.paths.model_ready_dir / "lap_time.parquet"
    dataset = pd.read_parquet(path)
    return dataset.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)


def _evaluate_baseline_nested(
    builder, dataset: pd.DataFrame, feature_columns: list[str], outer_folds
) -> list[dict[str, float]]:
    fold_metrics = []
    for train_idx, test_idx in outer_folds:
        train = dataset.iloc[train_idx]
        test = dataset.iloc[test_idx]
        model = builder().fit(train[feature_columns], train[TARGET_COLUMN])
        preds = model.predict(test[feature_columns])
        fold_metrics.append(regression_metrics(test[TARGET_COLUMN], preds))
    return fold_metrics


def _evaluate_tuned_model_nested(
    tuner_fn, build_fn, dataset: pd.DataFrame, outer_folds, n_jobs: int
) -> tuple[list[dict[str, float]], list[dict]]:
    fold_metrics = []
    fold_best_params = []
    for train_idx, test_idx in outer_folds:
        train = dataset.iloc[train_idx].reset_index(drop=True)
        test = dataset.iloc[test_idx]
        inner_folds = list(race_group_kfold(train, n_splits=MAX_FOLDS_REQUESTED))
        search = tuner_fn(inner_folds, n_jobs=n_jobs)
        search.fit(train[MODEL_FEATURE_COLUMNS], train[TARGET_COLUMN])
        best_params = _strip_model_prefix(search.best_params_)
        fold_best_params.append(best_params)
        fitted = build_fn(**best_params).fit(train[MODEL_FEATURE_COLUMNS], train[TARGET_COLUMN])
        preds = fitted.predict(test[MODEL_FEATURE_COLUMNS])
        fold_metrics.append(regression_metrics(test[TARGET_COLUMN], preds))
    return fold_metrics, fold_best_params


def run_training(config: PipelineConfig | None = None, n_jobs: int = 1) -> dict:
    """Run the full nested leave-one-race-out protocol described in this
    module's docstring and return everything a caller (CLI or test) needs:
    the comparison table, the fitted best model, and metadata about the run.
    Nothing is written to disk here except via the caller's explicit choice
    (see ``main`` below) - keeping this function pure and unit-testable on
    synthetic data.
    """
    config = config or load_config()
    dataset_path = config.paths.model_ready_dir / "lap_time.parquet"
    dataset = load_lap_time_dataset(config)

    all_races = sorted(set(race_groups(dataset)))
    outer_folds = list(race_group_kfold(dataset, n_splits=MAX_FOLDS_REQUESTED))
    validation_strategy = (
        f"nested race-grouped CV: {len(outer_folds)} outer leave-one-race-out folds over all "
        f"{len(all_races)} races; each candidate's hyperparameters are (re)tuned, per outer "
        f"fold, via an inner race-grouped CV computed only on that fold's training races - no "
        f"race is ever used both to tune and to score. Reported metrics are the mean +/- std "
        f"across all outer folds."
    )

    results: list[dict] = []

    def _finalize(name: str, category: str, fold_metrics: list[dict], fold_best_params=None) -> dict:
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
            "best_params_per_fold": fold_best_params,
            **TRAINING_NOTES[name],
        }
        results.append(row)
        return row

    # --- Baselines (mean, median, per-circuit-median) ---
    for name, builder in BASELINE_BUILDERS.items():
        fold_metrics = _evaluate_baseline_nested(builder, dataset, BASELINE_FEATURE_COLUMNS, outer_folds)
        _finalize(name, "baseline", fold_metrics)

    # --- Linear regression (baseline per the brief; uses the model feature set) ---
    fold_metrics = _evaluate_baseline_nested(
        linear_regression_pipeline, dataset, MODEL_FEATURE_COLUMNS, outer_folds
    )
    _finalize("linear_regression", "baseline", fold_metrics)

    strong_baseline_mae = next(r["mae_mean"] for r in results if r["model"] == "per_circuit_median_baseline")

    # --- Tuned ML models, nested CV ---
    for name, tuner_fn, build_fn in _TUNING_SPECS:
        fold_metrics, fold_best_params = _evaluate_tuned_model_nested(
            tuner_fn, build_fn, dataset, outer_folds, n_jobs
        )
        _finalize(name, "model", fold_metrics, fold_best_params)

    model_rows = [r for r in results if r["category"] == "model"]
    best_row = min(model_rows, key=lambda r: r["mae_mean"])
    beats_strong_baseline = best_row["mae_mean"] < strong_baseline_mae

    final_pipeline = None
    final_best_params = None
    if beats_strong_baseline:
        # Standard nested-CV practice: the loop above estimates the
        # generalization performance of the "tune, then fit" procedure for
        # the winning model; the deployed artifact is produced by running
        # that same procedure once more, tuned via CV over every race, fit
        # on the full dataset.
        tuner_fn = _TUNER_BY_NAME[best_row["model"]]
        build_fn = _BUILDER_BY_NAME[best_row["model"]]
        final_search = tuner_fn(outer_folds, n_jobs=n_jobs)
        final_search.fit(dataset[MODEL_FEATURE_COLUMNS], dataset[TARGET_COLUMN])
        final_best_params = _strip_model_prefix(final_search.best_params_)
        final_pipeline = build_fn(**final_best_params)
        final_pipeline.fit(dataset[MODEL_FEATURE_COLUMNS], dataset[TARGET_COLUMN])

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "season": config.season,
        "rounds": list(config.rounds),
        "training_dataset_path": str(dataset_path),
        "training_dataset_sha256": _sha256_of_file(dataset_path) if dataset_path.exists() else None,
        "training_dataset_n_rows": len(dataset),
        "target_column": TARGET_COLUMN,
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "all_races": all_races,
        "n_outer_folds": len(outer_folds),
        "validation_strategy": validation_strategy,
        "best_model": best_row["model"] if beats_strong_baseline else None,
        "best_model_params": final_best_params,
        "best_model_nested_cv_metrics": {
            "mae_mean": best_row["mae_mean"],
            "mae_std": best_row["mae_std"],
            "rmse_mean": best_row["rmse_mean"],
            "r2_mean": best_row["r2_mean"],
            "median_ae_mean": best_row["median_ae_mean"],
        }
        if beats_strong_baseline
        else None,
        "strong_baseline_mae_mean": strong_baseline_mae,
        "beats_strong_baseline": beats_strong_baseline,
        "final_pipeline_note": (
            "Nested CV above (best_model_nested_cv_metrics) estimates the generalization "
            "performance of this model's tune-then-fit procedure across all races; the "
            "serialized pipeline itself was tuned once more (CV over every race) and fit on "
            "the full dataset, per standard nested-CV practice, and was not itself scored on "
            "held-out data."
        )
        if beats_strong_baseline
        else None,
        "caveat": (
            "These metrics describe leave-one-race-out predictive accuracy on 2023 rounds 1-6 "
            "only. They are not a claim that this model is race-ready for live strategy use - "
            "see docs/PHASE_4_REPORT.md for scope, limitations and recommended next steps."
        ),
    }

    return {
        "results": results,
        "best_row": best_row,
        "beats_strong_baseline": beats_strong_baseline,
        "final_pipeline": final_pipeline,
        "metadata": metadata,
        "all_races": all_races,
        "n_outer_folds": len(outer_folds),
    }


def _write_comparison_report(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model_comparison_lap_time.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    columns = [
        "model",
        "category",
        "n_outer_folds",
        "mae_mean",
        "mae_std",
        "rmse_mean",
        "r2_mean",
        "median_ae_mean",
    ]
    lines = ["# Lap-Time Model Comparison", ""]
    lines.append(
        "Every number below comes from nested, race-grouped, leave-one-race-out cross-"
        "validation (never a random row split - docs/05_split_strategy.md): each model is "
        "evaluated once per race, having never seen that race during training or tuning, and "
        "`*_mean`/`*_std` are computed across those per-race folds. See "
        "`model_comparison_lap_time.json` for each fold's individual metrics and chosen "
        "hyperparameters."
    )
    lines.append("")
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "---|" * len(columns))
    for row in results:
        cells = []
        for col in columns:
            value = row[col]
            cells.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    for row in results:
        lines.append(f"### {row['model']}")
        lines.append("")
        lines.append(interpret_mae(row["mae_mean"]) + f" (std across folds: {row['mae_std']:.3f}s)")
        lines.append("")
        lines.append(f"- Strengths: {row['strengths']}")
        lines.append(f"- Weaknesses: {row['weaknesses']}")
        lines.append(f"- Training considerations: {row['training_considerations']}")
        lines.append("")
    lines.append(
        "**Caveat**: these metrics describe leave-one-race-out predictive accuracy on 2023 "
        "rounds 1-6 only and are not, on their own, a claim that any model here is race-ready "
        "for live strategy use. See `docs/PHASE_4_REPORT.md` for the full discussion."
    )
    with open(out_dir / "model_comparison_lap_time.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> int:
    config = load_config()
    outcome = run_training(config, n_jobs=-1)

    print(f"Races in scope ({outcome['n_outer_folds']} leave-one-race-out folds): {outcome['all_races']}")
    print()
    for row in outcome["results"]:
        print(
            f"{row['model']:<28} MAE={row['mae_mean']:.3f}+/-{row['mae_std']:.3f}s "
            f"RMSE={row['rmse_mean']:.3f}s R2={row['r2_mean']:.3f} MedAE={row['median_ae_mean']:.3f}s"
        )
    print()

    _write_comparison_report(outcome["results"], config.paths.quality_dir)
    print(f"Comparison report written to {config.paths.quality_dir}")

    if outcome["beats_strong_baseline"]:
        best_row = outcome["best_row"]
        print(f"\nBest model: {best_row['model']} ({interpret_mae(best_row['mae_mean'])})")
        models_dir = config.paths.models_dir
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "lap_time_model.joblib"
        joblib.dump(outcome["final_pipeline"], model_path)
        with open(models_dir / "lap_time_model_metadata.json", "w", encoding="utf-8") as fh:
            json.dump(outcome["metadata"], fh, indent=2)
        print(f"Serialized best model to {model_path}")
        return 0

    print("\nNo candidate model beat the per-circuit-median baseline - nothing serialized.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
