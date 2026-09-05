# Phase 2 Report - Data Pipeline

**Status: complete.** A reproducible, tested, leakage-safe pipeline from real
FastF1/Jolpica-F1 data to four model-ready datasets.

## What was built

- **Ingestion** (`mpp.ingest`): idempotent FastF1 + Jolpica-F1 clients,
  immutable raw storage under `data/raw/`, a `manifest.json` recording
  retrieval time/source/coverage, and raw -> intermediate parsing with no
  business-rule decisions.
- **Validation** (`mpp.validation`): a rule engine (required columns, null
  identifiers, duplicates, bounds, categorical checks) that only reports -
  `data/quality/validation_report.{json,md}`.
- **Cleaning** (`mpp.cleaning`): deterministic, logged numeric conversion,
  categorical normalization, and a fixed-priority validity classification
  that flags rather than silently drops non-racing laps -
  `data/quality/cleaning_log.json`.
- **Feature dataset** (`mpp.features`): one wide per-lap table with weather
  (nearest-time, per-round, tolerance-bounded), race-context, tyre, and a
  leakage-safe expanding-mean historical driver-pace feature.
- **Model datasets** (`mpp.datasets`): `lap_time`, `tyre_degradation`,
  `race_pace`, `pit_stop` - each with its own row-selection rule and target
  definition, documented in `docs/03_dataset_definitions.md`. Performance
  classification was **not** built as a dataset, per the architecture report:
  it has no independent ground-truth label and belongs downstream of model
  residuals, not in the data pipeline.
- **Orchestrator**: `python -m mpp.pipeline` runs all five stages in order and
  prints a per-stage summary; `--force` re-fetches from the network.
- **Tests**: 72 pytest cases across ingestion parsing, every validation rule,
  every cleaning decision (including FastF1's `None`-means-not-deleted
  quirk and the `M:SS.mmm` pit-stop duration format), every feature function
  (including an explicit leakage-safety regression test for the historical
  pace feature), every dataset builder's row-selection logic, and one
  end-to-end integration test on a committed real-data slice. All offline -
  no test hits the network.

## Verification performed

- `python -m pytest -q` - **72 passed, 0 failed**.
- `python -m mpp.pipeline` run against the real, configured scope (2023,
  rounds 1-6): **0 validation errors, 1 warning** (expected - a fraction of
  laps genuinely have no lap time: pit in/out and retirement laps).
- Generated datasets manually inspected (schema, dtypes, and spot-checked
  values against the raw FastF1 output) - see `data/quality/dataset_build_summary.json`
  for row counts: `lap_time` 6,145, `tyre_degradation` 6,132, `race_pace` 119,
  `pit_stop` 219.

## Known limitations (see `docs/01-03` for detail)

- Circuit static characteristics (length, corner count) are not sourced - the
  ingested APIs don't provide them, and inventing hand-typed numbers would
  violate the "real data only" rule; deferred until a properly sourced
  reference table is added.
- Tyre-wear and fuel-burn effects are not separated in the degradation target
  (no fuel-mass telemetry available).
- Safety Car / VSC laps are not filtered out of the degradation dataset yet
  (`TrackStatus` is carried through but unused as a filter in Phase 2).
- `driver_prior_relative_pace_s` is null for round 1 of any scope, and for a
  driver in a round with zero clean laps - by construction, not a bug.

## Next: Phase 3+

Train/validation/test splitting (`GroupKFold`/`GroupShuffleSplit` on
`circuit_id`/`(season, round)`), baselines, model training, and evaluation -
architecture report SS8-SS14, phases 3-6.
