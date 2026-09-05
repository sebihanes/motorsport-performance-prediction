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

**Status: Phase 0 — architecture defined, no pipeline implemented yet.**
See [`docs/ARCHITECTURE_REPORT.md`](docs/ARCHITECTURE_REPORT.md) for the full
architecture: prediction problems, data sources, feature and target
definitions, split/leakage strategy, baselines, candidate models, evaluation
metrics, interpretability and error-analysis plans, repository structure, and
implementation phases.

This is the predictive leg of a three-part motorsport analytics portfolio,
alongside `motorsport-data-analytics-bi` (descriptive) and
`motorsport-statistical-analysis` (inferential).

## Data

Real Formula 1 timing data only — [FastF1](https://github.com/theOehrly/Fast-F1)
(primary: session/lap/stint/tyre data) and
[Jolpica-F1](https://github.com/jolpica/jolpica-f1) (secondary: results,
qualifying, pit stops). No data is invented, and no data files are committed to
this repository.
