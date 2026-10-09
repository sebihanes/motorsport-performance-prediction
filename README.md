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

**Status: Phase 6 complete — lap time (Phase 4), tyre degradation, race pace,
and pit stop duration all trained and evaluated against the real 2023
rounds 1-6 data, and the lap-time model explained and error-analysed
(Phase 6)**, each with baselines, candidate ML models, a validation
strategy matched to that problem's own correlation structure (race-grouped
or chronological), a structured model comparison, and an error-analysis
breakdown by driver/team/circuit/compound/tyre-age where the dataset
supports it. See [`docs/ARCHITECTURE_REPORT.md`](docs/ARCHITECTURE_REPORT.md)
for the full architecture, [`docs/PHASE_2_REPORT.md`](docs/PHASE_2_REPORT.md) /
[`docs/PHASE_3_REPORT.md`](docs/PHASE_3_REPORT.md) /
[`docs/PHASE_4_REPORT.md`](docs/PHASE_4_REPORT.md) /
[`docs/PHASE_5_REPORT.md`](docs/PHASE_5_REPORT.md) /
[`docs/PHASE_6_REPORT.md`](docs/PHASE_6_REPORT.md) for what's been delivered
in each phase, [`docs/04_feature_dictionary.md`](docs/04_feature_dictionary.md)
for every feature's definition, and
[`docs/05_split_strategy.md`](docs/05_split_strategy.md) for the
random-vs-grouped-vs-temporal split comparison.

**Headline result: no learned model beats the strongest reasonable baseline
on any of the four problems in the current scope, so no model is
serialized.** This is a deliberate, verified negative result, not a
missing step. For lap time, Ridge (5.95s MAE) beats the per-circuit median
(8.96s) but loses to **stint pace persistence**, the zero-parameter "same
pace as this stint so far" predictor (3.67s), which the final audit added
(`docs/FINAL_AUDIT.md`). The lap-time models are next-lap predictors: they
use the current stint's already-completed laps, so they are not pre-race
forecasts. The full scored audit, including what was fixed and what is
still open, is in [`docs/FINAL_AUDIT.md`](docs/FINAL_AUDIT.md).
Tyre degradation, race pace, and pit stop duration (`docs/PHASE_5_REPORT.md`)
are each limited by a different, concretely identified cause: tyre
degradation's target is confounded with fuel burn and still-unfiltered
safety-car laps; race pace's dataset carries no circuit-identity feature at
all, so predicting an absolute, circuit-dominated pace from weather alone
generalizes badly; and pit-stop duration is dominated by one race's
red-flag stoppages being recorded as pit-stop time. The common thread across
every problem is the current 6-round, zero-repeated-circuit scope - expected
to improve, likely substantially, once the season scope widens
(config-only change, `config/settings.toml`).

**Why the lap-time model errs (Phase 6, `docs/PHASE_6_REPORT.md`):** about
half its squared error is a constant per-race offset (it under-reacts to a
new circuit's pace level and pulls predictions toward the training mean),
and safety-car/VSC laps - ~8% of laps, ~19% of absolute error - are
unflagged in its features. On green-flag laps with a clean stint history
and the race offset removed, the typical error is ~1.3s.

This is the predictive leg of a three-part motorsport analytics portfolio,
alongside `motorsport-data-analytics-bi` (descriptive) and
`motorsport-statistical-analysis` (inferential).

## Quickstart

```bash
pip install -e ".[dev]"
python -m mpp.pipeline                      # ingest -> validate -> clean -> feature dataset -> model datasets
python -m mpp.models.train_lap_time         # lap time: baselines -> 5 candidates -> nested CV/tuning -> serialize if it beats the strongest baseline
python -m mpp.models.train_tyre_degradation # tyre degradation: baselines -> linear/Ridge/RF -> nested CV -> (same gate)
python -m mpp.models.train_race_pace        # race pace: baselines -> linear/Ridge/RF -> chronological CV -> (same gate)
python -m mpp.models.train_pit_stop         # pit stop: baselines -> linear/Ridge/RF -> nested CV -> (same gate)
python -m mpp.evaluation.analyze_lap_time   # lap time: permutation importance, PDP, residual/failure analysis
pytest
```

Each trainer exits with status 1 when no candidate beats its strongest
baseline. In the current scope that is expected for all four, so run
the trainers as separate commands rather than chaining them with `&&`.

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
                    model_comparison_{lap_time,tyre_degradation,race_pace,pit_stop}.{json,md},
                    interpretability_lap_time.{json,md}
data/models/       <problem>_model.joblib only for a problem whose best candidate beats its
                    strongest baseline (none does in the current scope); lap_time and
                    tyre_degradation always write _metadata.json recording the best candidate
                    and the bar it missed
```

Phase 6 figures are written to `docs/figures/phase6/` (committed, since the
report embeds them).

## Data

Real Formula 1 timing data only — [FastF1](https://github.com/theOehrly/Fast-F1)
(primary: session/lap/stint/tyre data) and
[Jolpica-F1](https://github.com/jolpica/jolpica-f1) (secondary: results,
qualifying, pit stops). No data is invented, and no data files are committed to
this repository. Source, retrieval process, coverage and licensing:
[`docs/01_data_sources.md`](docs/01_data_sources.md).
