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
per lap. Full column-by-column definitions (type, unit, source, availability
timing, prediction-safety, limitations) are in
`docs/04_feature_dictionary.md`; in brief, `mpp.features.build` adds:

- **Weather** (`mpp.features.weather`): nearest-time join to the weather
  samples, done per `(season, round)` group so a lap in one race is never
  matched to another race's weather (session clocks reset every round).
  Optional feature - null beyond a 5-minute tolerance, never dropped.
- **Race context** (`mpp.features.context`): `race_fraction = lap_number / total_laps`,
  a fuel-load proxy that is safe to use (total race distance is known in
  advance, not an outcome).
- **Lap context** (`mpp.features.lap`): `laps_into_stint`, deliberately
  independent of `tyre_age` (which can start above 1 due to pre-race laps -
  see `docs/01_data_sources.md`).
- **Session** (`mpp.features.session`): `session_type`, constant today
  (Race-only ingestion) but wired through as a real column.
- **Tyre** (`mpp.features.tyre`): `tyre_age_sq` for the non-linear shape of
  degradation, plus `tyre_age_x_<compound>` interaction columns (a real
  race-engineering interaction - softer compounds degrade faster - not an
  arbitrary cross term).
- **Historical driver/team/circuit pace and driver consistency**
  (`mpp.features.historical`, `.team`, `.circuit`): see "Leakage prevention"
  below - the features in this pipeline that require careful ordering to
  avoid contaminating a prediction with future information.
- **Within-stint performance trend** (`mpp.features.performance`):
  `stint_prior_mean_lap_time_s`, an expanding mean of the *same stint's own
  prior* clean laps only - see `docs/04_feature_dictionary.md` for why a
  full-stint (including future laps) version was rejected as leakage.

Categorical columns (`compound`, `session_type`, and any others a future
phase adds) are one-hot encoded via `mpp.features.encoding.build_feature_pipeline`
- never given an arbitrary ordinal integer, which would imply a false
distance between categories a linear model would wrongly learn from.

## 5. Model datasets (`mpp.datasets`) - one table per prediction problem

See `docs/03_dataset_definitions.md` for the exact schema of each. Built by
`run_dataset_build()` into `data/model_ready/{lap_time,tyre_degradation,race_pace,pit_stop}.parquet`,
with a build summary at `data/quality/dataset_build_summary.json`.

## Leakage prevention in this phase

- **`driver_prior_relative_pace_s`/`_std_s`, `team_prior_relative_pace_s`,
  `driver_circuit_prior_relative_pace_s`** (`mpp.features.historical`/`.team`/`.circuit`,
  sharing the leakage-safe `add_expanding_prior_stat` helper in
  `mpp.features.common`) are expanding statistics computed *after* sorting by
  round and shifting by one - a round's value only ever reflects strictly
  earlier rounds. Round 1 is `NaN` for every driver/team by construction (no
  history exists yet). Unit tested directly per feature
  (`tests/features/test_historical.py`, `test_team.py`, `test_circuit.py`,
  each with a `..._never_leaks_into_an_earlier_round...` test that perturbs a
  later round and asserts every earlier round's value is bit-for-bit
  unchanged), and again end-to-end across all of them together in
  `tests/test_leakage_review.py`.
- **`stint_prior_mean_lap_time_s`** (`mpp.features.performance`) is an
  expanding mean over a lap's *own stint's prior laps only* - shift-then-expand
  within the stint, the same pattern as the round-level features above but at
  lap grain. `tests/features/test_performance.py` and
  `tests/test_leakage_review.py` both verify a later lap in the same stint
  never changes an earlier lap's value.
- **`laps_into_stint`** (`mpp.features.lap`) looks like it could leak (it's a
  `groupby(...).transform("min")` over the whole stint), but is not: the
  minimum lap number of a stint is fixed and known the instant the stint
  begins, so every lap in it already "knows" that value in real time - unlike
  a `transform("max")` over the stint (the *last* lap), which is not known
  until the stint ends and is deliberately never computed as a per-row
  feature anywhere in this pipeline.
- **Circuit identity** (`circuit_id`) is known before the session runs, so it
  carries no leakage as a feature; it is also the grouping key for
  `mpp.splits.group_split`'s circuit-grouped split (`docs/05_split_strategy.md`).
- **Weather and race-context features** use only information available at (or
  before) the lap being predicted - no post-lap or post-race aggregate is
  ever joined onto a lap's own row.
- Feature engineering deliberately does **not** include finishing position,
  championship points, or any other race-outcome field, nor a same-session
  relative-pace feature (comparing a lap to other drivers' pace in the very
  race being predicted) - see `docs/04_feature_dictionary.md`
  "Deliberately not built" for the full reasoning.

## What this phase does not do

Model training, cross-validation, hyperparameter tuning, and evaluation are
Phase 4+ (`docs/ARCHITECTURE_REPORT.md` SS17). Phases 2-3's job is to produce
correct, leakage-safe, well-documented model-ready tables with a matched
split strategy already implemented - nothing here fits a model or reports an
accuracy number.
