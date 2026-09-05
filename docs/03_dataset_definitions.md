# Model Dataset Definitions

Four datasets, one per prediction problem that the ingested data actually
supports (architecture report SS3). Performance classification is not built
as a fifth dataset: per the architecture report, it is a derived, post-hoc
view of another problem's residuals, not an independently labeled target - it
belongs in Phase 4+ model evaluation, not in the data pipeline.

All four live under `data/model_ready/` as Parquet, and all share the same
group columns (`season`, `round`, `circuit_id`) so a future `GroupKFold` /
`GroupShuffleSplit` (architecture report SS8) can hold out whole races or
circuits consistently across problems.

## `lap_time.parquet` (`mpp.datasets.lap_time`)

One row per clean racing lap (`is_valid_lap == True`).

| Column | Meaning |
|---|---|
| `season`, `round`, `circuit_id` | grouping key |
| `driver`, `team` | identity |
| `lap_number`, `race_fraction`, `stint`, `tyre_age`, `tyre_age_sq`, `compound`, `air_temp`, `track_temp`, `rainfall`, `driver_prior_relative_pace_s` | features |
| `lap_time_s` | **target** - lap time in seconds |

## `tyre_degradation.parquet` (`mpp.datasets.tyre_degradation`)

One row per clean racing lap belonging to a stint with at least 3 clean laps
and a known compound. A stint's baseline is its first clean lap (by lap
number); the target and the tyre-age feature are both expressed *relative to
that baseline*, isolating within-stint change from cross-stint/cross-circuit
absolute pace differences.

| Column | Meaning |
|---|---|
| `season`, `round`, `circuit_id` | grouping key |
| `driver`, `team` | identity |
| `tyre_age_delta`, `tyre_age_delta_sq`, `compound`, `race_fraction`, `air_temp`, `track_temp` | features |
| `lap_time_delta_s` | **target** - lap time minus the stint's baseline lap time (0.0 at the baseline lap itself, by construction) |

**Known limitation**: this delta conflates tyre wear with fuel burn within
the stint - both make later laps faster or slower relative to the baseline,
and this pipeline cannot separate them without fuel-mass telemetry (see
architecture report SS18). Laps taken under Safety Car / VSC are not filtered
out of this dataset in Phase 2 (`TrackStatus` is passed through but not yet
used as a filter) - a widened scope with more SC-affected laps will show
larger positive deltas that are speed-limit artifacts, not degradation; this
is a documented Phase 2 limitation, not a bug (see
`docs/ARCHITECTURE_REPORT.md` SS14 error-analysis-by-session-status, planned
for Phase 4).

## `race_pace.parquet` (`mpp.datasets.race_pace`)

One row per driver's session (`season`, `round`, `driver`). Target is a
median, not a mean, of that driver's clean racing laps - a handful of outlier
laps (traffic, minor incidents) should not distort the pace estimate.

| Column | Meaning |
|---|---|
| `season`, `round`, `circuit_id` | grouping key |
| `driver`, `team` | identity |
| `driver_prior_relative_pace_s`, `air_temp_mean`, `track_temp_mean`, `rainfall_any` | features |
| `session_pace_s` | **target** - median lap time (s) among that driver's clean laps |

**Known limitation**: round 1 has no prior-round history for any driver, so
`driver_prior_relative_pace_s` is null there by construction - not imputed,
since there is genuinely no information yet.

## `pit_stop.parquet` (`mpp.datasets.pit_stop`)

One row per recorded pit stop, built only because Jolpica-F1 actually
provides per-stop duration at usable granularity for the configured scope -
the feasibility check the architecture report called for before committing to
this problem (SS3.4, SS17 Phase 7).

| Column | Meaning |
|---|---|
| `season`, `round`, `circuit_id` | grouping key |
| `driver`, `team` | identity |
| `lap_number`, `stop_number` | features |
| `duration_s` | **target** - full recorded stationary time (s) |

**Known limitation**: `duration_s` is not decomposed into tyre-change time vs.
other causes (damage repair, drive-through penalty served during the stop) -
a model trained on this target is predicting "how long was this driver
stationary," not "how long does a clean tyre change take."

## Current build (2023, rounds 1-6)

See `data/quality/dataset_build_summary.json` for live counts. As of the last
pipeline run: `lap_time` 6,145 rows, `tyre_degradation` 6,132 rows (13 laps
excluded for being in a stint shorter than 3 clean laps), `race_pace` 119
rows (one driver-round combination produced zero clean laps and is absent,
not imputed), `pit_stop` 219 rows.
