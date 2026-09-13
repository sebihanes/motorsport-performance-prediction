"""Shared nested, race-grouped leave-one-race-out training protocol
(architecture report SS17 - "Phase 5/6 reuse Phase 4's infrastructure with a
different target"). This is exactly the outer/inner nested-CV loop
``mpp.models.train_lap_time`` built for the lap-time anchor problem,
factored out so the tyre-degradation and pit-stop problems (both grouped by
(season, round) per docs/05_split_strategy.md - lap-level and stop-level
rows within a race are correlated, same reasoning as lap_time) reuse it
instead of re-implementing it. race_pace uses a different (temporal) split
and is trained separately in ``mpp.models.train_race_pace`` for that reason.

Candidates here are deliberately narrower than lap_time's five: a plain
Linear Regression (the brief's "simple linear regression"), a tuned Ridge,
and a tuned Random Forest (the brief's "tree-based model where appropriate")
- not the full five-model sweep, since the brief asks to *compare* the two
families, not to re-run every lap-time candidate on a smaller dataset.

Every outer fold's test predictions are also collected and returned
out-of-fold (each row scored exactly once, only by a model that never saw
its race during training or tuning) so callers can run error analysis
(``mpp.evaluation.error_analysis``) without any leakage risk.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from mpp.evaluation.metrics import aggregate_fold_metrics, regression_metrics
from mpp.models.baselines import PerGroupMedianRegressor, mean_baseline, median_baseline
from mpp.models.generic_pipelines import build_pipeline
from mpp.splits.group_split import race_group_kfold, race_groups

RANDOM_STATE = 0
N_ITER_RANDOM_SEARCH = 8
MAX_FOLDS_REQUESTED = 1000  # race_group_kfold caps this to one fold per race

RIDGE_PARAM_GRID = {"model__alpha": [0.1, 1.0, 10.0, 50.0, 100.0]}
RANDOM_FOREST_PARAM_DISTRIBUTIONS = {
    "model__n_estimators": [100, 200, 300, 400],
    "model__max_depth": [None, 4, 6, 10],
    "model__min_samples_leaf": [1, 2, 4, 8],
    "model__max_features": ["sqrt", 0.5, 1.0],
}


@dataclass(frozen=True)
class NestedTrainingSpec:
    target_column: str
    numeric_columns: list[str]
    nominal_columns: list[str]
    strong_baseline_group_column: str
    training_notes: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def feature_columns(self) -> list[str]:
        return self.numeric_columns + self.nominal_columns


def _linear_pipeline(spec: NestedTrainingSpec) -> object:
    return build_pipeline(
        LinearRegression(), spec.numeric_columns, spec.nominal_columns, scale_numeric=True, impute_numeric=True
    )


def _ridge_pipeline(spec: NestedTrainingSpec, **params) -> object:
    return build_pipeline(
        Ridge(random_state=RANDOM_STATE, **params),
        spec.numeric_columns,
        spec.nominal_columns,
        scale_numeric=True,
        impute_numeric=True,
    )


def _random_forest_pipeline(spec: NestedTrainingSpec, **params) -> object:
    return build_pipeline(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params),
        spec.numeric_columns,
        spec.nominal_columns,
        scale_numeric=False,
        impute_numeric=True,
    )


def _strip_model_prefix(params: dict) -> dict:
    return {key.split("__", 1)[1]: value for key, value in params.items()}


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluate_baseline_nested(builder, dataset, feature_columns, target_column, outer_folds):
    fold_metrics = []
    oof_pred = np.full(len(dataset), np.nan)
    for train_idx, test_idx in outer_folds:
        train = dataset.iloc[train_idx]
        test = dataset.iloc[test_idx]
        model = builder().fit(train[feature_columns], train[target_column])
        preds = model.predict(test[feature_columns])
        fold_metrics.append(regression_metrics(test[target_column], preds))
        oof_pred[test_idx] = preds
    return fold_metrics, oof_pred


def _evaluate_tuned_nested(build_fn, param_space, search_cls, search_kwargs, dataset, spec, outer_folds, n_jobs):
    fold_metrics = []
    fold_best_params = []
    oof_pred = np.full(len(dataset), np.nan)
    feature_columns = spec.feature_columns
    for train_idx, test_idx in outer_folds:
        train = dataset.iloc[train_idx].reset_index(drop=True)
        test = dataset.iloc[test_idx]
        inner_folds = list(race_group_kfold(train, n_splits=MAX_FOLDS_REQUESTED))
        search = search_cls(
            build_fn(spec), param_space, cv=inner_folds, scoring="neg_mean_absolute_error", n_jobs=n_jobs,
            **search_kwargs,
        )
        search.fit(train[feature_columns], train[spec.target_column])
        best_params = _strip_model_prefix(search.best_params_)
        fold_best_params.append(best_params)
        fitted = build_fn(spec, **best_params).fit(train[feature_columns], train[spec.target_column])
        preds = fitted.predict(test[feature_columns])
        fold_metrics.append(regression_metrics(test[spec.target_column], preds))
        oof_pred[test_idx] = preds
    return fold_metrics, fold_best_params, oof_pred


def run_nested_race_grouped_training(dataset: pd.DataFrame, spec: NestedTrainingSpec, n_jobs: int = 1) -> dict:
    """Runs mean/median/strong-group baselines, a plain linear regression, a
    tuned Ridge and a tuned Random Forest through nested leave-one-race-out
    CV, exactly mirroring mpp.models.train_lap_time's protocol. Returns the
    comparison rows, the best model's fitted-on-everything pipeline (only
    built by the caller, not here), and every model's out-of-fold
    predictions for error analysis.
    """
    all_races = sorted(set(race_groups(dataset)))
    outer_folds = list(race_group_kfold(dataset, n_splits=MAX_FOLDS_REQUESTED))
    validation_strategy = (
        f"nested race-grouped CV: {len(outer_folds)} outer leave-one-race-out folds over all "
        f"{len(all_races)} races; tuned candidates' hyperparameters are (re)chosen, per outer fold, via "
        f"an inner race-grouped CV computed only on that fold's training races - no race is ever used "
        f"both to tune and to score. Reported metrics are the mean +/- std across all outer folds."
    )

    results: list[dict] = []
    oof_predictions: dict[str, np.ndarray] = {}

    def _finalize(name, category, fold_metrics, fold_best_params=None) -> dict:
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
            **spec.training_notes.get(name, {}),
        }
        results.append(row)
        return row

    # The strong baseline's group column may already be one of the model's own
    # feature columns (e.g. tyre_degradation uses `compound`, which is also a
    # nominal model feature) - avoid selecting it twice, which would make
    # X[group_column] resolve to a 2-column frame instead of a Series.
    baseline_feature_columns = spec.feature_columns + (
        [] if spec.strong_baseline_group_column in spec.feature_columns else [spec.strong_baseline_group_column]
    )
    baseline_builders = {
        "mean_baseline": mean_baseline,
        "median_baseline": median_baseline,
        f"strong_baseline": lambda: PerGroupMedianRegressor(group_column=spec.strong_baseline_group_column),
    }
    for name, builder in baseline_builders.items():
        cols = baseline_feature_columns if name == "strong_baseline" else spec.feature_columns
        fold_metrics, oof = _evaluate_baseline_nested(builder, dataset, cols, spec.target_column, outer_folds)
        _finalize(name, "baseline", fold_metrics)
        oof_predictions[name] = oof

    lin_fold_metrics, lin_oof = _evaluate_baseline_nested(
        lambda: _linear_pipeline(spec), dataset, spec.feature_columns, spec.target_column, outer_folds
    )
    _finalize("linear_regression", "baseline", lin_fold_metrics)
    oof_predictions["linear_regression"] = lin_oof

    strong_baseline_mae = next(r["mae_mean"] for r in results if r["model"] == "strong_baseline")

    ridge_metrics, ridge_params, ridge_oof = _evaluate_tuned_nested(
        _ridge_pipeline, RIDGE_PARAM_GRID, GridSearchCV, {}, dataset, spec, outer_folds, n_jobs
    )
    _finalize("ridge", "model", ridge_metrics, ridge_params)
    oof_predictions["ridge"] = ridge_oof

    rf_metrics, rf_params, rf_oof = _evaluate_tuned_nested(
        _random_forest_pipeline,
        RANDOM_FOREST_PARAM_DISTRIBUTIONS,
        RandomizedSearchCV,
        {"n_iter": N_ITER_RANDOM_SEARCH, "random_state": RANDOM_STATE},
        dataset,
        spec,
        outer_folds,
        n_jobs,
    )
    _finalize("random_forest", "model", rf_metrics, rf_params)
    oof_predictions["random_forest"] = rf_oof

    model_rows = [r for r in results if r["category"] == "model"]
    best_row = min(model_rows, key=lambda r: r["mae_mean"])
    beats_strong_baseline = best_row["mae_mean"] < strong_baseline_mae

    return {
        "results": results,
        "best_row": best_row,
        "beats_strong_baseline": beats_strong_baseline,
        "strong_baseline_mae_mean": strong_baseline_mae,
        "all_races": all_races,
        "outer_folds": outer_folds,
        "n_outer_folds": len(outer_folds),
        "validation_strategy": validation_strategy,
        "oof_predictions": oof_predictions,
        "build_fn_by_name": {"ridge": _ridge_pipeline, "random_forest": _random_forest_pipeline},
    }


def fit_final_pipeline(dataset: pd.DataFrame, spec: NestedTrainingSpec, outcome: dict, n_jobs: int = 1):
    """Per standard nested-CV practice: the winning model is tuned once more
    (inner CV over *every* race) and fit on the full dataset, rather than
    reusing any one outer fold's fitted instance."""
    best_name = outcome["best_row"]["model"]
    build_fn = outcome["build_fn_by_name"][best_name]
    param_space = RIDGE_PARAM_GRID if best_name == "ridge" else RANDOM_FOREST_PARAM_DISTRIBUTIONS
    search_cls = GridSearchCV if best_name == "ridge" else RandomizedSearchCV
    search_kwargs = {} if best_name == "ridge" else {"n_iter": N_ITER_RANDOM_SEARCH, "random_state": RANDOM_STATE}
    search = search_cls(
        build_fn(spec), param_space, cv=outcome["outer_folds"], scoring="neg_mean_absolute_error", n_jobs=n_jobs,
        **search_kwargs,
    )
    search.fit(dataset[spec.feature_columns], dataset[spec.target_column])
    best_params = _strip_model_prefix(search.best_params_)
    pipeline = build_fn(spec, **best_params)
    pipeline.fit(dataset[spec.feature_columns], dataset[spec.target_column])
    return pipeline, best_params
