# Motorsport Performance Prediction

An engineering-focused machine learning platform for predicting Formula 1 lap
time, tyre degradation, race pace, and (where the data supports it) pit stop
duration — built around baseline-relative evaluation, leakage-aware validation,
and engineering interpretation rather than model complexity for its own sake.

```
RAW DATA → DATA VALIDATION → DATA CLEANING → FEATURE ENGINEERING
→ TRAIN/VALIDATION/TEST → BASELINE → MODEL TRAINING → CROSS-VALIDATION
→ HYPERPARAMETER TUNING → EVALUATION → ERROR ANALYSIS → FEATURE IMPORTANCE
→ ENGINEERING INTERPRETATION
```

**Status: Phase 4 complete — lap-time baselines (mean/median/per-circuit-median),
five candidate ML models (Linear Regression, Ridge, Random Forest, Gradient
Boosting, HistGradientBoostingRegressor), nested leave-one-race-out
cross-validation and hyperparameter tuning, a structured model comparison,
and serialization of the winning pipeline (Ridge).** See
[`docs/ARCHITECTURE_REPORT.md`](docs/ARCHITECTURE_REPORT.md) for the full
architecture, [`docs/PHASE_2_REPORT.md`](docs/PHASE_2_REPORT.md) /
[`docs/PHASE_3_REPORT.md`](docs/PHASE_3_REPORT.md) /
[`docs/PHASE_4_REPORT.md`](docs/PHASE_4_REPORT.md) for what's been delivered
in each phase, [`docs/04_feature_dictionary.md`](docs/04_feature_dictionary.md)
for every feature's definition, and
[`docs/05_split_strategy.md`](docs/05_split_strategy.md) for the
random-vs-grouped-vs-temporal split comparison.

**Headline result (see `docs/PHASE_4_REPORT.md` for the full, honest
discussion): Ridge is the only candidate that beats the strong baseline,
with a leave-one-race-out MAE of ~5.95s versus the baseline's ~8.96s — a
real improvement, but nowhere near "race-ready."** The current 6-round 2023
scope has zero repeated circuits, so no model or baseline has any
circuit-specific history to draw on for a held-out race; tree-based models
in particular cannot extrapolate past the training races' weather range and
underperform even the naive baseline as a result. This is a data-scope
limitation, not an implementation bug, and is expected to improve once the
season scope widens.

This is the predictive leg of a three-part motorsport analytics portfolio,
alongside `motorsport-data-analytics-bi` (descriptive) and
`motorsport-statistical-analysis` (inferential).

## Quickstart

```bash
pip install -e ".[dev]"
python -m mpp.pipeline               # ingest -> validate -> clean -> feature dataset -> model datasets
python -m mpp.models.train_lap_time  # baselines -> candidate models -> nested CV/tuning -> comparison -> serialize
pytest
```

Outputs land in `data/` (git-ignored, fully regenerable from `config/settings.toml`'s
season/round scope):

```
data/raw/          immutable FastF1 + Jolpica-F1 pulls + manifest.json
data/intermediate/ typed, combined tables (laps, weather, pitstops, sessions, drivers)
data/processed/    cleaned laps/pitstops (numeric conversion, categorical normalization,
                    validity classification - see docs/02_data_pipeline.md)
data/features/     laps_features.parquet - the full per-lap feature table
data/model_ready/  lap_time.parquet, tyre_degradation.parquet, race_pace.parquet,
                    pit_stop.parquet - see docs/03_dataset_definitions.md
data/quality/      validation_report.{json,md}, cleaning_log.json, dataset_build_summary.json,
                    model_comparison_lap_time.{json,md}
data/models/       lap_time_model.joblib, lap_time_model_metadata.json - see docs/PHASE_4_REPORT.md
```

## Data

Real Formula 1 timing data only — [FastF1](https://github.com/theOehrly/Fast-F1)
(primary: session/lap/stint/tyre data) and
[Jolpica-F1](https://github.com/jolpica/jolpica-f1) (secondary: results,
qualifying, pit stops). No data is invented, and no data files are committed to
this repository. Source, retrieval process, coverage and licensing:
[`docs/01_data_sources.md`](docs/01_data_sources.md).
