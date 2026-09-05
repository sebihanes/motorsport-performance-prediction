# Data Pipeline

```
RAW DATA -> INGESTION -> VALIDATION -> CLEANING -> FEATURE DATASET -> MODEL DATASET
data/raw/  mpp.ingest    mpp.validation  mpp.cleaning  mpp.features    mpp.datasets
           -> data/intermediate/         -> data/processed/ -> data/features/ -> data/model_ready/
```

Run the whole thing with `python -m mpp.pipeline` (add `--force` to re-fetch
from the network even if raw files already exist for a round). Every stage
writes its own artifacts to disk and can be re-run independently by reading
the previous stage's output back from disk - nothing is passed between stages
purely in memory except within a single `run_pipeline()` call.

## 1. Ingestion (`mpp.ingest`) - raw -> intermediate

`fetch_raw()` hits FastF1 and Jolpica-F1 (SS`docs/01_data_sources.md`) and
writes untouched artifacts under `data/raw/`. It is idempotent by round: if
`data/raw/fastf1/<season>/<round>/laps_raw.csv` already exists, that round is
not re-fetched unless `force=True`. A `manifest.json` records exactly what was
retrieved and when.

`build_intermediate()` does no network I/O. It reads the raw artifacts back,
concatenates them across the configured rounds, and applies only *structural*
typing (parsing timedelta strings back to `timedelta64`, `TrackStatus` kept as
text, `LapNumber`/`Stint`/`TyreLife` as nullable `Int64`) - no business-rule
decisions about validity happen here. Output: `data/intermediate/{laps,weather,drivers,pitstops,sessions}.parquet`.

## 2. Validation (`mpp.validation`) - reports, never modifies

Runs a fixed set of rules (`mpp.validation.rules`) against the intermediate
tables and writes `data/quality/validation_report.{json,md}`. Rules check:
required columns, non-null identifiers (driver, team, season, round, circuit),
positive lap numbers, exact-duplicate rows on the configured key, lap time
null-rate and bounds, non-negative tyre age, and recognized compound values.
**No row is ever dropped or changed by validation** - `has_errors()` tells the
pipeline whether to keep going, but the decision about what to actually do
with an issue belongs to cleaning.

On the current 2023 R1-6 scope: **0 errors, 1 warning** (a subset of laps have
a null lap time - expected, since pit in/out and retirement laps genuinely
have none; see `data/quality/validation_report.md` for the exact count).

## 3. Cleaning (`mpp.cleaning`) - intermediate -> processed

Deterministic, logged transformations - `data/quality/cleaning_log.json`
records exactly what changed and why, for every run.

**Laps** (`mpp.cleaning.laps.clean_laps`):
1. Corrupt rows (`LapNumber` null or <= 0) are dropped - a real data defect,
   not a modeling choice.
2. Exact duplicate rows (on `config.cleaning.duplicate_key`) are dropped,
   keeping the first occurrence.
3. Numeric conversion: `LapTime`/`Time` (timedelta) -> `lap_time_s`/`time_s`
   (float seconds).
4. Categorical normalization: `Compound` is upper-cased and stripped; any
   value outside `config.validation.valid_compounds` becomes `"UNKNOWN"`
   (counted in the log, never silently merged into a real compound).
5. Boolean normalization: `Deleted`/`IsAccurate`/`FreshTyre` nulls (FastF1's
   "not deleted" is `None`, not `False`) become `False`.
6. **Validity classification** (fixed priority order, first match wins):
   `missing_lap_time` -> `deleted_lap` -> `pit_in_out_lap` ->
   `lap_time_out_of_bounds` -> otherwise valid. A lap that fails this
   classification is **kept**, not dropped - `is_valid_lap=False` and
   `exclusion_reason` are added as columns, so the row stays available for
   error analysis and any future problem that wants it. Only genuine data
   defects (step 1-2) are actually removed from the table.
7. Incomplete-session check: the max recorded lap number per round is logged
   (not compared/enforced) - a red-flag-shortened race would show up here for
   manual review, not be silently patched.

**Pit stops** (`mpp.cleaning.pitstops.clean_pitstops`):
1. Numeric conversion: `duration_raw` (string, `"25.885"` or `"1:05.123"`)
   parsed to `duration_s` (float). Unparseable values are dropped and counted.
2. `driver_id` (Jolpica, e.g. `"max_verstappen"`) is mapped to the FastF1
   driver code (e.g. `"VER"`) via the intermediate drivers reference table
   (Jolpica's own `code` field already matches FastF1's driver abbreviation,
   confirmed against real data - no manual mapping table needed). Unmapped
   rows are dropped and counted.
3. Exact duplicates on `(season, round, driver_id, stop_number)` are dropped.

On the current scope: 0 rows dropped at any cleaning step (the ingested data
was already well-formed) - see `data/quality/cleaning_log.json` for the live
counts, which will change if the scope is widened to noisier rounds/seasons.

## 4. Feature dataset (`mpp.features`) - processed -> one wide table

`run_feature_build()` produces `data/features/laps_features.parquet`, one row
per lap, adding:

- **Weather** (`mpp.features.weather`): nearest-time join to the weather
  samples, done per `(season, round)` group so a lap in one race is never
  matched to another race's weather (session clocks reset every round).
  Optional feature - null beyond a 5-minute tolerance, never dropped.
- **Race context** (`mpp.features.context`): `race_fraction = lap_number / total_laps`,
  a fuel-load proxy that is safe to use (total race distance is known in
  advance, not an outcome).
- **Tyre** (`mpp.features.tyre`): `tyre_age_sq`, for the non-linear shape of
  degradation.
- **Historical driver pace** (`mpp.features.historical`): see
  "Leakage prevention" below - the one feature in this pipeline that requires
  careful ordering to avoid contaminating a prediction with future
  information.

## 5. Model datasets (`mpp.datasets`) - one table per prediction problem

See `docs/03_dataset_definitions.md` for the exact schema of each. Built by
`run_dataset_build()` into `data/model_ready/{lap_time,tyre_degradation,race_pace,pit_stop}.parquet`,
with a build summary at `data/quality/dataset_build_summary.json`.

## Leakage prevention in this phase

- **`driver_prior_relative_pace_s`** (`mpp.features.historical`) is an
  expanding mean computed *after* sorting by round and shifting by one - a
  round's value only ever reflects strictly earlier rounds. Round 1 is `NaN`
  for every driver by construction (no history exists yet). This is unit
  tested directly: `tests/features/test_historical.py::test_future_round_never_leaks_into_an_earlier_rounds_feature`
  perturbs a later round's lap times and asserts every earlier round's feature
  value is bit-for-bit unchanged.
- **Circuit identity** (`circuit_id`) is known before the session runs, so it
  carries no leakage as a feature; it is also the natural `GroupKFold` /
  `GroupShuffleSplit` key for Phase 3 model validation (not yet implemented).
- **Weather and race-context features** use only information available at (or
  before) the lap being predicted - no post-lap or post-race aggregate is
  ever joined onto a lap's own row.
- Feature engineering deliberately does **not** include finishing position,
  championship points, or any other race-outcome field.

## What Phase 2 does not do

Model training, cross-validation, hyperparameter tuning, and evaluation are
Phase 3+ (`docs/ARCHITECTURE_REPORT.md` SS17). This phase's job is to produce
correct, leakage-safe, well-documented model-ready tables - nothing here fits
a model or reports an accuracy number.
