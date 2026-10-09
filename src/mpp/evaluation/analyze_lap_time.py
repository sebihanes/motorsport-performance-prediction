"""Phase 6 entry point: interpretability and error analysis for the
serialized lap-time model (the project's anchor problem and its best
model - see docs/PHASE_4_REPORT.md).

What it does, in order:

1. **Reproduces the Phase 4 out-of-fold predictions exactly** - the same
   leave-one-race-out outer folds (``race_group_kfold``, deterministic), the
   same model family (``best_model`` from the serialized metadata) and the
   same per-fold hyperparameters the nested CV selected
   (``best_params_per_fold`` in ``model_comparison_lap_time.json``). Each
   lap is therefore predicted by a model that never saw its race during
   training or tuning. The reproduced MAE is checked against the Phase 4
   number and the difference is reported.
2. **Interpretability** (``mpp.evaluation.interpretability``): grouped and
   single-feature permutation importance on those out-of-fold predictions -
   shuffled across all races and, separately, only within each race - plus
   the model's own coefficients across folds and partial dependence.
3. **Error analysis** (``mpp.evaluation.error_analysis``): residual
   distribution, error vs. predicted value, grouped error/bias tables
   (driver, team, circuit, compound, tyre age, session, track status, lap
   phase), the between-race share of squared error, and unusual laps.
4. **Failure analysis**: representative large-error laps, each annotated
   only with causes the data actually records (track status, lap 1, wet
   running, missing history features, the race-level offset). Laps with no
   recorded cause are labelled as unexplained by this dataset.

Outputs: ``data/quality/interpretability_lap_time.{json,md}`` and figures
under ``docs/figures/phase6/``.

Usage: ``python -m mpp.evaluation.analyze_lap_time``
(requires ``python -m mpp.models.train_lap_time`` to have been run first).
The best ML candidate is analysed whether or not it cleared the strongest
baseline; when it did not, no serialized pipeline exists and the analysis
uses only the per-fold models.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from mpp.config import PipelineConfig, load_config
from mpp.evaluation import plots
from mpp.evaluation.error_analysis import (
    ABS_ERROR_COLUMN,
    RESIDUAL_COLUMN,
    add_residuals,
    between_group_error_share,
    bucket_numeric,
    classify_track_status,
    error_vs_predicted,
    flag_unusual_observations,
    residual_summary,
    systematic_bias_table,
    to_markdown_table,
)
from mpp.evaluation.interpretability import (
    FoldRoutedPredictor,
    model_feature_importance,
    partial_dependence,
    permutation_importance,
    quantile_grid,
    single_feature_groups,
)
from mpp.evaluation.metrics import regression_metrics
from mpp.models.baselines import StintPacePersistenceRegressor
from mpp.models.regressors import MODEL_FEATURE_COLUMNS, MODEL_NUMERIC_COLUMNS
from mpp.models.train_lap_time import _BUILDER_BY_NAME, MAX_FOLDS_REQUESTED, TARGET_COLUMN, load_lap_time_dataset
from mpp.splits.group_split import race_group_kfold, race_groups

PRED_COLUMN = "predicted_lap_time_s"
LAP_KEY = ["season", "round", "driver", "lap_number"]
STINT_KEY = ["season", "round", "driver", "stint"]
DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
TYRE_AGE_EDGES = [0, 5, 10, 15, 20, 30, 80]
TYRE_AGE_LABELS = ["0-5", "6-10", "11-15", "16-20", "21-30", "31+"]
RACE_OFFSET_MIN_S = 1.0
UNEXPLAINED_MIN_S = 2.0
STINT_PRIOR_DEVIATION_S = 5.0

# Columns that are deterministic functions of one another are permuted
# together (see mpp.evaluation.interpretability's module docstring).
FEATURE_GROUPS: dict[str, list[str]] = {
    "tyre age (age, age^2, age x compound)": [
        "tyre_age",
        "tyre_age_sq",
        "tyre_age_x_soft",
        "tyre_age_x_medium",
        "tyre_age_x_hard",
        "tyre_age_x_intermediate",
        "tyre_age_x_wet",
    ],
    "compound": ["compound"],
    "laps_into_stint": ["laps_into_stint"],
    "stint": ["stint"],
    "race progress (lap_number, race_fraction)": ["lap_number", "race_fraction"],
    "temperature (air, track)": ["air_temp", "track_temp"],
    "rainfall": ["rainfall"],
    "driver prior pace (mean, std)": ["driver_prior_relative_pace_s", "driver_prior_relative_pace_std_s"],
    "team prior pace": ["team_prior_relative_pace_s"],
    "stint prior mean lap time": ["stint_prior_mean_lap_time_s"],
    "session_type": ["session_type"],
}

ERROR_GROUP_COLUMNS = {
    "driver": "driver",
    "team": "team",
    "circuit": "circuit_id",
    "compound": "compound",
    "tyre_age": "tyre_age_bucket",
    "session": "session_type",
    "track_status": "track_status_class",
    "lap_phase": "lap_phase",
}

UNOBSERVABLE_NOTE = (
    "Traffic (gaps to cars ahead), car damage, driver error, team orders and fuel/tyre-saving strategy are not "
    "recorded in this dataset, so they can neither be confirmed nor ruled out for any lap."
)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def load_lap_context(config: PipelineConfig) -> pd.DataFrame | None:
    """Per-lap context the model never sees as a feature (track status),
    from the feature table the lap_time dataset was built from."""
    path = config.paths.features_dir / "laps_features.parquet"
    if not path.exists():
        return None
    laps = pd.read_parquet(path, columns=LAP_KEY + ["stint", "track_status", "is_valid_lap"])
    return laps.loc[laps["is_valid_lap"], LAP_KEY + ["track_status"]].reset_index(drop=True)


def attach_context(dataset: pd.DataFrame, context: pd.DataFrame | None) -> pd.DataFrame:
    """Left-join context columns onto `dataset` by lap key, preserving row
    order and refusing to duplicate rows."""
    if context is None:
        out = dataset.copy()
        out["track_status"] = pd.NA
        return out
    merged = dataset.merge(context, on=LAP_KEY, how="left", validate="one_to_one")
    if len(merged) != len(dataset):
        raise ValueError("Context join changed the row count.")
    return merged


def add_analysis_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Derived slicing columns for error analysis - none of these are model
    inputs."""
    out = frame.copy()
    out["race"] = race_groups(out) + "_" + out["circuit_id"].astype(str)
    out["tyre_age_bucket"] = bucket_numeric(out["tyre_age"].astype(float), TYRE_AGE_EDGES, TYRE_AGE_LABELS).astype(str)
    out["track_status_class"] = [classify_track_status(s) for s in out["track_status"]]
    out["is_neutralised"] = out["track_status_class"].isin(["safety_car", "vsc", "red_flag", "yellow_flag"])

    lap_phase = np.where(out["stint_prior_mean_lap_time_s"].isna(), "first lap of stint", "later stint lap")
    lap_phase = np.where(out["lap_number"].astype(float) == 1, "race lap 1", lap_phase)
    out["lap_phase"] = lap_phase
    out["has_driver_history"] = out["driver_prior_relative_pace_s"].notna()

    # How many earlier clean laps in this same stint were neutralised - those
    # laps feed stint_prior_mean_lap_time_s (an expanding mean), so a stint
    # that contained a safety car carries a slow history into later laps.
    order = out.sort_values(STINT_KEY + ["lap_number"]).index
    neutral = out.loc[order, "is_neutralised"].astype(int)
    prior = neutral.groupby([out.loc[order, c] for c in STINT_KEY]).transform(lambda s: s.shift(1).fillna(0).cumsum())
    out["prior_neutralised_laps_in_stint"] = prior.reindex(out.index).astype(int)
    return out


def derive_tyre_age_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Recompute every column that is a function of tyre_age (used when
    partial dependence varies tyre_age)."""
    out = X.copy()
    age = out["tyre_age"].astype(float)
    out["tyre_age_sq"] = age**2
    compound = out["compound"].astype(str)
    for name in ["soft", "medium", "hard", "intermediate", "wet"]:
        out[f"tyre_age_x_{name}"] = np.where(compound == name.upper(), age, 0.0)
    return out


# ---------------------------------------------------------------------------
# Model reproduction
# ---------------------------------------------------------------------------


def imputed_value(model, column: str):
    """The median a fitted pipeline's numeric imputer substitutes for
    `column`, or None if the pipeline has no such imputer."""
    try:
        numeric = model.named_steps["preprocess"].named_transformers_["numeric"]
        return float(numeric.named_steps["impute"].statistics_[MODEL_NUMERIC_COLUMNS.index(column)])
    except (AttributeError, KeyError, ValueError, IndexError):
        return None


def fit_fold_models(dataset: pd.DataFrame, model_name: str, fold_params: list[dict]):
    """Refit the Phase 4 per-fold models: one per leave-one-race-out outer
    fold, with that fold's nested-CV-selected hyperparameters. Returns
    (models, fold_of_row, held_out_race_per_fold)."""
    if model_name not in _BUILDER_BY_NAME:
        raise ValueError(f"No pipeline builder registered for model {model_name!r}.")
    build_fn = _BUILDER_BY_NAME[model_name]
    outer_folds = list(race_group_kfold(dataset, n_splits=MAX_FOLDS_REQUESTED))
    if len(fold_params) != len(outer_folds):
        raise ValueError(f"Expected {len(outer_folds)} per-fold parameter sets, got {len(fold_params)}.")
    groups = race_groups(dataset)
    fold_of_row = np.full(len(dataset), -1)
    models, held_out = [], []
    for fold, ((train_idx, test_idx), params) in enumerate(zip(outer_folds, fold_params)):
        train = dataset.iloc[train_idx]
        models.append(build_fn(**(params or {})).fit(train[MODEL_FEATURE_COLUMNS], train[TARGET_COLUMN]))
        fold_of_row[test_idx] = fold
        held_out.append(sorted(set(groups[test_idx])))
    return models, fold_of_row, held_out


# ---------------------------------------------------------------------------
# Failure annotation - evidence only
# ---------------------------------------------------------------------------

_STATUS_TEXT = {
    "safety_car": "a safety car",
    "vsc": "a virtual safety car",
    "red_flag": "a red flag",
    "yellow_flag": "yellow flags",
}


def annotate_failure(row: pd.Series, race_bias: float) -> dict[str, object]:
    """Explain one lap's error using only recorded evidence.

    The race-level offset (this race's mean residual) is reported first and
    removed; every lap-level cause is then checked against the *sign* of the
    remaining error (neutralised laps are slow -> expected underprediction,
    an inflated stint history -> expected overprediction, ...). A cause
    whose direction contradicts the error is still listed but marked as
    unlikely, and does not count as an explanation. A lap with no
    sign-consistent recorded cause and a remaining error of at least
    ``UNEXPLAINED_MIN_S`` is explicitly labelled unexplained by the data.
    """
    residual = float(row[RESIDUAL_COLUMN])
    evidence: list[str] = []
    remainder = residual

    if abs(race_bias) >= RACE_OFFSET_MIN_S and np.sign(race_bias) == np.sign(residual):
        share = min(1.0, abs(race_bias) / abs(residual)) if residual else 0.0
        evidence.append(
            f"race-level offset: the model's mean residual for this race is {race_bias:+.2f}s "
            f"({share:.0%} of this lap's error); circuit identity is not a model feature."
        )
        remainder = residual - race_bias

    n_supporting = 0

    def _add(text: str, expected_sign: int | None) -> None:
        """expected_sign: -1 = cause predicts underprediction, +1 =
        overprediction, None = direction not determined by the cause."""
        nonlocal n_supporting
        if expected_sign is None or abs(remainder) < 0.5:
            n_supporting += 1
            evidence.append(text + ".")
        elif np.sign(remainder) == expected_sign:
            n_supporting += 1
            evidence.append(text + " (consistent with the error's sign).")
        else:
            evidence.append(text + " (NOT consistent with the error's sign - unlikely to be the cause).")

    status_class = row.get("track_status_class", "unknown")
    if status_class in _STATUS_TEXT:
        _add(
            f"lap ran under {_STATUS_TEXT[status_class]} (track_status={row.get('track_status')}); neutralised pace "
            f"is slower and track status is not a model feature",
            -1,
        )
    if float(row.get("lap_number", np.nan)) == 1:
        _add("race lap 1 (standing start, congested field)", -1)
    if bool(row.get("rainfall", False)) or str(row.get("compound")) in ("INTERMEDIATE", "WET"):
        _add(
            f"wet running (rainfall={bool(row.get('rainfall', False))}, compound={row.get('compound')}); few wet "
            f"laps exist in training",
            None,
        )

    race_median = row.get("race_median_lap_time_s")
    has_race_median = race_median is not None and not pd.isna(race_median)
    stint_prior = row.get("stint_prior_mean_lap_time_s")
    if pd.isna(stint_prior):
        imputed = row.get("imputed_stint_prior_s")
        if imputed is not None and not pd.isna(imputed) and has_race_median:
            gap = float(imputed) - float(race_median)
            _add(
                f"first lap of its stint: stint_prior_mean_lap_time_s is missing and was median-imputed from the "
                f"training races (other circuits) as {float(imputed):.2f}s, {gap:+.2f}s vs. this race's median "
                f"lap {float(race_median):.2f}s",
                int(np.sign(gap)),
            )
        else:
            _add("first lap of its stint: stint_prior_mean_lap_time_s is missing and was median-imputed", None)
    elif has_race_median and abs(float(stint_prior) - float(race_median)) >= STINT_PRIOR_DEVIATION_S:
        gap = float(stint_prior) - float(race_median)
        n_prior_neutral = int(row.get("prior_neutralised_laps_in_stint", 0) or 0)
        source = (
            f"{n_prior_neutral} earlier lap(s) in this stint were neutralised"
            if n_prior_neutral
            else "no earlier lap in this stint is flagged as neutralised"
        )
        _add(
            f"stint_prior_mean_lap_time_s = {float(stint_prior):.2f}s, {gap:+.2f}s vs. this race's median lap "
            f"{float(race_median):.2f}s - the stint's lap history is out of band ({source})",
            int(np.sign(gap)),
        )
    if pd.isna(row.get("driver_prior_relative_pace_s")):
        _add("no driver/team history features (first round in scope) - median-imputed", None)

    unexplained = n_supporting == 0 and abs(remainder) >= UNEXPLAINED_MIN_S
    if unexplained:
        evidence.append(
            f"no sign-consistent recorded cause for the remaining {remainder:+.2f}s: unexplained by the available "
            f"data."
        )
    return {"residual_after_race_offset_s": remainder, "evidence": evidence, "unexplained_by_data": unexplained}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze(
    dataset: pd.DataFrame,
    model_name: str,
    fold_params: list[dict],
    final_pipeline=None,
    n_repeats: int = 5,
    random_state: int = 0,
) -> dict:
    """Pure analysis on an in-memory dataset (context columns optional).
    Returns tables and summaries; writes nothing."""
    dataset = dataset.reset_index(drop=True)
    if "track_status" not in dataset.columns:
        dataset = attach_context(dataset, None)
    models, fold_of_row, held_out = fit_fold_models(dataset, model_name, fold_params)
    predictor = FoldRoutedPredictor(models, fold_of_row, MODEL_FEATURE_COLUMNS)
    y = dataset[TARGET_COLUMN].to_numpy(dtype=float)

    frame = add_analysis_columns(dataset)
    frame[PRED_COLUMN] = predictor.predict(dataset)
    frame = add_residuals(frame, TARGET_COLUMN, PRED_COLUMN)

    # Stint pace persistence - the strongest baseline (docs/FINAL_AUDIT.md) -
    # scored on exactly the same out-of-fold rows, its fallback median fit on
    # each fold's training races only.
    persistence = np.full(len(dataset), np.nan)
    for fold in range(len(models)):
        test_rows = fold_of_row == fold
        baseline = StintPacePersistenceRegressor().fit(dataset.loc[~test_rows], y[~test_rows])
        persistence[test_rows] = baseline.predict(dataset.loc[test_rows])
    frame["persistence_prediction_s"] = persistence

    per_fold_metrics = []
    for fold, races in enumerate(held_out):
        rows = fold_of_row == fold
        per_fold_metrics.append(
            {
                "held_out": races,
                **regression_metrics(y[rows], frame.loc[rows, PRED_COLUMN]),
                "persistence_mae": float(np.mean(np.abs(persistence[rows] - y[rows]))),
            }
        )
    mean_fold_mae = float(np.mean([m["mae"] for m in per_fold_metrics]))
    persistence_mean_fold_mae = float(np.mean([m["persistence_mae"] for m in per_fold_metrics]))

    # --- Interpretability ---
    races = race_groups(dataset)
    perm_grouped_across = permutation_importance(
        predictor, dataset, y, FEATURE_GROUPS, n_repeats=n_repeats, random_state=random_state
    )
    perm_grouped_within = permutation_importance(
        predictor, dataset, y, FEATURE_GROUPS, n_repeats=n_repeats, random_state=random_state, strata=races
    )
    perm_grouped = perm_grouped_across.merge(
        perm_grouped_within[["feature", "importance_mean", "importance_std"]],
        on="feature",
        suffixes=("_across_races", "_within_race"),
    )
    perm_single = permutation_importance(
        predictor, dataset, y, single_feature_groups(MODEL_FEATURE_COLUMNS), n_repeats=n_repeats,
        random_state=random_state,
    )

    fold_importances = pd.concat(
        [model_feature_importance(m).assign(fold=i) for i, m in enumerate(models)], ignore_index=True
    )
    value_col = "coefficient" if "coefficient" in fold_importances.columns else "importance"
    native = (
        fold_importances.groupby("feature")[value_col]
        .agg(["mean", "std", "min", "max"])
        .rename(columns=lambda c: f"{value_col}_{c}")
        .reset_index()
    )
    native["sign_stable_across_folds"] = (native[f"{value_col}_min"] > 0) | (native[f"{value_col}_max"] < 0)
    native = native.reindex(native[f"{value_col}_mean"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    final_native = model_feature_importance(final_pipeline) if final_pipeline is not None else None

    pd_tyre_age = {}
    for compound in DRY_COMPOUNDS:
        subset = dataset.loc[dataset["compound"].astype(str) == compound]
        if len(subset) < 20:
            continue
        grid = quantile_grid(subset["tyre_age"], n_points=15)
        sub_predictor = FoldRoutedPredictor(models, fold_of_row[subset.index.to_numpy()], MODEL_FEATURE_COLUMNS)
        pd_tyre_age[compound] = partial_dependence(
            sub_predictor, subset.reset_index(drop=True), "tyre_age", grid, derive=derive_tyre_age_columns
        )
    pd_ranges = []
    for feature in MODEL_NUMERIC_COLUMNS:
        if feature.startswith("tyre_age") or dataset[feature].nunique(dropna=True) < 2:
            continue
        grid = quantile_grid(dataset[feature].astype(float), n_points=2)
        if len(grid) < 2:
            continue
        curve = partial_dependence(predictor, dataset, feature, grid)
        pd_ranges.append(
            {
                "feature": feature,
                "p05_value": float(grid[0]),
                "p95_value": float(grid[-1]),
                "prediction_change_p05_to_p95_s": float(curve["mean_prediction"].iloc[-1] - curve["mean_prediction"].iloc[0]),
            }
        )
    pd_ranges = pd.DataFrame(pd_ranges)
    if not pd_ranges.empty:
        pd_ranges = pd_ranges.reindex(
            pd_ranges["prediction_change_p05_to_p95_s"].abs().sort_values(ascending=False).index
        ).reset_index(drop=True)

    # --- Error analysis ---
    summary = residual_summary(frame[RESIDUAL_COLUMN])
    by_predicted = error_vs_predicted(frame, PRED_COLUMN)
    race_share = between_group_error_share(frame[RESIDUAL_COLUMN], frame["race"])
    group_tables, skipped = {}, {}
    for label, col in ERROR_GROUP_COLUMNS.items():
        table = systematic_bias_table(frame, col)
        if table is None:
            skipped[label] = f"`{col}` has fewer than two distinct values in this scope"
        else:
            group_tables[label] = table
    race_table = systematic_bias_table(frame, "race")

    frame = flag_unusual_observations(frame, center_by="race")
    unusual = frame.loc[frame["is_unusual"]]
    unusual_by_status = (
        unusual["track_status_class"].value_counts().rename_axis("track_status_class").reset_index(name="n_unusual")
    )
    status_mix = frame["track_status_class"].value_counts(normalize=True).rename("share_of_all_laps")
    unusual_by_status = unusual_by_status.merge(status_mix, left_on="track_status_class", right_index=True, how="left")
    unusual_by_status["share_of_unusual_laps"] = unusual_by_status["n_unusual"] / max(len(unusual), 1)

    green = frame["track_status_class"] == "green"
    condition_mae = {
        "green_laps_mae": float(frame.loc[green, ABS_ERROR_COLUMN].mean()) if green.any() else None,
        "neutralised_laps_mae": float(frame.loc[frame["is_neutralised"], ABS_ERROR_COLUMN].mean())
        if frame["is_neutralised"].any()
        else None,
        "green_laps_mae_after_race_offset": float(frame.loc[green, "centered_residual"].abs().mean())
        if green.any()
        else None,
    }

    frame["_persistence_abs"] = np.abs(frame["persistence_prediction_s"] - frame[TARGET_COLUMN])
    model_vs_persistence = (
        frame.groupby(["lap_phase", "track_status_class"], observed=True)
        .agg(n=(ABS_ERROR_COLUMN, "size"), model_mae=(ABS_ERROR_COLUMN, "mean"), persistence_mae=("_persistence_abs", "mean"))
        .reset_index()
    )
    model_vs_persistence["model_minus_persistence_s"] = (
        model_vs_persistence["model_mae"] - model_vs_persistence["persistence_mae"]
    )
    frame = frame.drop(columns=["_persistence_abs"])

    # --- Failure analysis ---
    race_bias = frame.groupby("race")[RESIDUAL_COLUMN].mean()
    frame["imputed_stint_prior_s"] = [imputed_value(models[f], "stint_prior_mean_lap_time_s") for f in fold_of_row]
    frame["race_median_lap_time_s"] = frame.groupby("race")[TARGET_COLUMN].transform("median")
    top_overall = frame.sort_values(ABS_ERROR_COLUMN, ascending=False).head(6)
    remaining = frame.drop(index=top_overall.index)
    top_within = remaining.reindex(remaining["robust_z"].abs().sort_values(ascending=False).index).head(4)
    remaining = remaining.drop(index=top_within.index)
    clean = remaining.loc[(remaining["track_status_class"] == "green") & (remaining["lap_number"].astype(float) > 1)]
    top_green = clean.reindex(clean["robust_z"].abs().sort_values(ascending=False).index).head(6)
    failure_rows = []
    selections = [
        ("largest absolute error", top_overall),
        ("largest within-race (offset-removed) error", top_within),
        ("largest green-flag error, lap > 1 (offset-removed)", top_green),
    ]
    for kind, rows in selections:
        for idx, row in rows.iterrows():
            note = annotate_failure(row, float(race_bias[row["race"]]))
            failure_rows.append(
                {
                    "selection": kind,
                    "race": row["race"],
                    "driver": row["driver"],
                    "team": row["team"],
                    "lap_number": int(row["lap_number"]),
                    "compound": str(row["compound"]),
                    "tyre_age": int(row["tyre_age"]),
                    "track_status": None if pd.isna(row["track_status"]) else str(row["track_status"]),
                    "actual_s": float(row[TARGET_COLUMN]),
                    "predicted_s": float(row[PRED_COLUMN]),
                    "residual_s": float(row[RESIDUAL_COLUMN]),
                    "race_mean_residual_s": float(race_bias[row["race"]]),
                    **note,
                }
            )

    return {
        "model_name": model_name,
        "fold_params": fold_params,
        "held_out_per_fold": held_out,
        "per_fold_metrics": per_fold_metrics,
        "mean_fold_mae": mean_fold_mae,
        "persistence_mean_fold_mae": persistence_mean_fold_mae,
        "model_vs_persistence": model_vs_persistence,
        "pooled_metrics": regression_metrics(y, frame[PRED_COLUMN]),
        "frame": frame,
        "permutation_grouped": perm_grouped,
        "permutation_single": perm_single,
        "native_importance_across_folds": native,
        "native_importance_value": value_col,
        "native_importance_final": final_native,
        "partial_dependence_tyre_age": pd_tyre_age,
        "partial_dependence_ranges": pd_ranges,
        "residual_summary": summary,
        "error_vs_predicted": by_predicted,
        "between_race_error_share": race_share,
        "group_tables": group_tables,
        "race_table": race_table,
        "skipped_dimensions": skipped,
        "n_unusual": int(len(unusual)),
        "unusual_by_track_status": unusual_by_status,
        "condition_mae": condition_mae,
        "failure_examples": failure_rows,
    }


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def write_figures(outcome: dict, figures_dir: Path) -> list[Path]:
    frame = outcome["frame"]
    written = []
    perm = outcome["permutation_grouped"]
    written.append(plots.save_figure(
        plots.plot_importance_comparison(
            perm.sort_values("importance_mean_across_races", ascending=False),
            "feature",
            ["importance_mean_across_races", "importance_mean_within_race"],
            ["shuffled across all races", "shuffled within each race only"],
            "Grouped permutation importance (out-of-fold)",
            "Increase in MAE when the feature group is shuffled (s)",
        ),
        figures_dir / "permutation_importance_grouped.png",
    ))
    written.append(plots.save_figure(
        plots.plot_feature_importance(
            outcome["permutation_single"], "importance_mean", "Single-feature permutation importance (out-of-fold)",
            "Increase in MAE when shuffled across races (s)", err_col="importance_std",
        ),
        figures_dir / "permutation_importance_single.png",
    ))
    native = outcome["native_importance_across_folds"]
    value_col = outcome["native_importance_value"]
    written.append(plots.save_figure(
        plots.plot_feature_importance(
            native, f"{value_col}_mean",
            f"Model {value_col}s, mean across the 6 fold models (error bar = std)",
            "Seconds per 1 SD (numeric) / seconds offset (one-hot level)" if value_col == "coefficient" else value_col,
            err_col=f"{value_col}_std", signed=value_col == "coefficient",
        ),
        figures_dir / "model_coefficients.png",
    ))
    if outcome["partial_dependence_tyre_age"]:
        written.append(plots.save_figure(
            plots.plot_partial_dependence(
                outcome["partial_dependence_tyre_age"], "Tyre age (laps)",
                "Partial dependence on tyre age, by compound (derived tyre-age columns recomputed)",
            ),
            figures_dir / "partial_dependence_tyre_age.png",
        ))
    written.append(plots.save_figure(
        plots.plot_predicted_vs_actual(frame, TARGET_COLUMN, PRED_COLUMN, facet_col="race",
                                       title="Predicted vs. actual lap time, per held-out race"),
        figures_dir / "predicted_vs_actual.png",
    ))
    written.append(plots.save_figure(
        plots.plot_residuals_vs_predicted(frame, PRED_COLUMN, RESIDUAL_COLUMN, outcome["error_vs_predicted"],
                                          title="Residual vs. predicted value"),
        figures_dir / "residuals_vs_predicted.png",
    ))
    written.append(plots.save_figure(
        plots.plot_error_distributions(frame[RESIDUAL_COLUMN], frame[ABS_ERROR_COLUMN]),
        figures_dir / "error_distributions.png",
    ))
    tables = dict(outcome["group_tables"])
    if outcome["race_table"] is not None:
        tables["race"] = outcome["race_table"]
    for label, table in tables.items():
        col = ERROR_GROUP_COLUMNS.get(label, label)
        written.append(plots.save_figure(
            plots.plot_grouped_errors(table, col, f"Out-of-fold error by {label.replace('_', ' ')}"),
            figures_dir / f"error_by_{label}.png",
        ))
    return written


def _json_ready(outcome: dict) -> dict:
    def records(df):
        return None if df is None else df.to_dict(orient="records")

    return {
        "model_name": outcome["model_name"],
        "fold_params": outcome["fold_params"],
        "held_out_per_fold": outcome["held_out_per_fold"],
        "per_fold_metrics": outcome["per_fold_metrics"],
        "mean_fold_mae": outcome["mean_fold_mae"],
        "persistence_mean_fold_mae": outcome["persistence_mean_fold_mae"],
        "model_vs_persistence": records(outcome["model_vs_persistence"]),
        "pooled_metrics": outcome["pooled_metrics"],
        "permutation_grouped": records(outcome["permutation_grouped"]),
        "permutation_single": records(outcome["permutation_single"]),
        "native_importance_across_folds": records(outcome["native_importance_across_folds"]),
        "native_importance_final": records(outcome["native_importance_final"]),
        "partial_dependence_tyre_age": {k: records(v) for k, v in outcome["partial_dependence_tyre_age"].items()},
        "partial_dependence_ranges": records(outcome["partial_dependence_ranges"]),
        "residual_summary": outcome["residual_summary"],
        "error_vs_predicted": records(outcome["error_vs_predicted"]),
        "between_race_error_share": outcome["between_race_error_share"],
        "group_tables": {k: records(v) for k, v in outcome["group_tables"].items()},
        "race_table": records(outcome["race_table"]),
        "skipped_dimensions": outcome["skipped_dimensions"],
        "n_unusual": outcome["n_unusual"],
        "unusual_by_track_status": records(outcome["unusual_by_track_status"]),
        "condition_mae": outcome["condition_mae"],
        "failure_examples": outcome["failure_examples"],
        "unobservable_causes_note": UNOBSERVABLE_NOTE,
    }


def write_reports(outcome: dict, quality_dir: Path, phase4_mae: float | None) -> None:
    quality_dir.mkdir(parents=True, exist_ok=True)
    payload = _json_ready(outcome)
    payload["phase4_reported_mae_mean"] = phase4_mae
    with open(quality_dir / "interpretability_lap_time.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)

    rnd = lambda df: df.round(3)  # noqa: E731
    lines = ["# Lap-Time Model - Interpretability & Error Analysis (generated)", ""]
    lines.append(
        f"Model: `{outcome['model_name']}`; every number is computed on out-of-fold (leave-one-race-out) "
        f"predictions. Mean per-fold MAE {outcome['mean_fold_mae']:.3f}s"
        + (f" (Phase 4 reported {phase4_mae:.3f}s)." if phase4_mae is not None else ".")
    )
    lines += [
        "",
        f"Stint pace persistence baseline on the same rows: mean per-fold MAE "
        f"{outcome['persistence_mean_fold_mae']:.3f}s.",
        "",
        "Residual = prediction - actual (+ = predicted too slow, - = predicted too fast).",
        "",
        "## Model vs. stint pace persistence, per fold",
        "",
    ]
    lines += to_markdown_table(
        pd.DataFrame(
            [
                {"held_out": ",".join(m["held_out"]), "model_mae": m["mae"], "persistence_mae": m["persistence_mae"]}
                for m in outcome["per_fold_metrics"]
            ]
        ).round(3)
    )
    lines += ["", "## Model vs. stint pace persistence, by lap phase and track status", ""]
    lines += to_markdown_table(rnd(outcome["model_vs_persistence"]))
    lines += [""]
    lines += ["## Grouped permutation importance", ""] + to_markdown_table(rnd(outcome["permutation_grouped"]))
    lines += ["", "## Single-feature permutation importance", ""] + to_markdown_table(rnd(outcome["permutation_single"]))
    lines += ["", "## Model coefficients across folds", ""] + to_markdown_table(
        rnd(outcome["native_importance_across_folds"])
    )
    if not outcome["partial_dependence_ranges"].empty:
        lines += ["", "## Partial dependence range (p05 -> p95)", ""] + to_markdown_table(
            rnd(outcome["partial_dependence_ranges"])
        )
    lines += ["", "## Residual summary", ""]
    lines += [f"- {k}: {v:.3f}" if isinstance(v, float) else f"- {k}: {v}" for k, v in outcome["residual_summary"].items()]
    share = outcome["between_race_error_share"]
    lines += ["", f"Between-race share of MSE: {share['share_between']:.1%} (MSE {share['mse_total']:.2f}).", ""]
    lines += ["## Error vs. predicted value", ""] + to_markdown_table(rnd(outcome["error_vs_predicted"]))
    if outcome["race_table"] is not None:
        lines += ["", "## Error by race", ""] + to_markdown_table(rnd(outcome["race_table"]))
    for label, table in outcome["group_tables"].items():
        lines += ["", f"## Error by {label}", ""] + to_markdown_table(rnd(table))
    for label, reason in outcome["skipped_dimensions"].items():
        lines += ["", f"## Error by {label}", "", f"N/A - {reason}."]
    lines += ["", "## MAE by track condition", ""]
    lines += [f"- {k}: {v:.3f}" if isinstance(v, float) else f"- {k}: {v}" for k, v in outcome["condition_mae"].items()]
    lines += ["", "## Unusual laps by track status", "", f"{outcome['n_unusual']} laps flagged (|robust z| > 3.5 after "
              "removing each race's mean residual).", ""]
    lines += to_markdown_table(rnd(outcome["unusual_by_track_status"]))
    lines += ["", "## Representative failures", ""]
    for ex in outcome["failure_examples"]:
        lines.append(
            f"- **{ex['race']} {ex['driver']} lap {ex['lap_number']}** ({ex['selection']}): actual {ex['actual_s']:.2f}s, "
            f"predicted {ex['predicted_s']:.2f}s, residual {ex['residual_s']:+.2f}s; {ex['compound']} age {ex['tyre_age']}, "
            f"track_status={ex['track_status']}."
        )
        for item in ex["evidence"]:
            lines.append(f"  - {item}")
    lines += ["", UNOBSERVABLE_NOTE, ""]
    with open(quality_dir / "interpretability_lap_time.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> int:
    config = load_config()
    metadata_path = config.paths.models_dir / "lap_time_model_metadata.json"
    comparison_path = config.paths.quality_dir / "model_comparison_lap_time.json"
    if not metadata_path.exists() or not comparison_path.exists():
        print("Run `python -m mpp.models.train_lap_time` first (model metadata/comparison not found).")
        return 1
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    # Analyse the best ML candidate whether or not it cleared the strongest
    # baseline (and was therefore serialized) - explaining *why* it falls
    # short is the point of the analysis.
    candidates = [r for r in comparison if r["category"] == "model"]
    row = min(candidates, key=lambda r: r["mae_mean"])
    model_name = row["model"]
    fold_params = row["best_params_per_fold"]

    dataset = attach_context(load_lap_time_dataset(config), load_lap_context(config))
    model_path = config.paths.models_dir / "lap_time_model.joblib"
    final_pipeline = joblib.load(model_path) if metadata.get("best_model") == model_name and model_path.exists() else None
    outcome = analyze(dataset, model_name, fold_params, final_pipeline=final_pipeline)

    print(f"Model: {model_name}; reproduced mean per-fold MAE {outcome['mean_fold_mae']:.3f}s "
          f"(Phase 4: {row['mae_mean']:.3f}s); stint pace persistence {outcome['persistence_mean_fold_mae']:.3f}s; "
          f"serialized: {final_pipeline is not None}")
    figures_dir = config.repo_root / "docs" / "figures" / "phase6"
    written = write_figures(outcome, figures_dir)
    write_reports(outcome, config.paths.quality_dir, row["mae_mean"])
    print(f"Wrote {len(written)} figures to {figures_dir}")
    print(f"Wrote interpretability_lap_time.{{json,md}} to {config.paths.quality_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
