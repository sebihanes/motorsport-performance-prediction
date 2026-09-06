# Feature Dictionary

Every column produced by `mpp.features.build.build_laps_features` (the one
wide per-lap table every model-ready dataset in `mpp.datasets` is built
from), grouped the way the Phase 3 brief asked for. For where each column
ends up (which of the four model-ready datasets carries it), see
`docs/03_dataset_definitions.md`.

"Usable for prediction" means: safe to feed to a model with no leakage, given
this pipeline's realistic prediction-time setting (predicting a lap/session
before or as it happens, using only information already realized by then).
"Source" cites the module that computes it - see that module's own docstring
for the full leakage-safety reasoning.

## Driver

| Feature | Description | Type | Unit | Source | Availability timing | Usable for prediction | Limitations |
|---|---|---|---|---|---|---|---|
| `driver` | 3-letter driver code (e.g. `VER`) | nominal categorical | - | `mpp.cleaning.laps` (from FastF1) | pre-session | Identity/reference column - carried for error analysis (architecture SS14), **not** included in any dataset's `FEATURE_COLUMNS` by default. One-hot encoding it directly would let a model memorize per-driver averages instead of generalizing via the relative-pace features below; a future modeling phase may deliberately choose to add it. | High cardinality (~20 categories) relative to 6 rounds of data. |
| `driver_prior_relative_pace_s` | Driver's median relative pace (own median lap time minus the field's median that round), expanding-averaged over that driver's rounds strictly before the current one | numeric | seconds (negative = faster than field) | `mpp.features.historical` | known only after the prior round(s) complete | Yes | Null for round 1 of any scope (no history yet), and for any driver's own round if they had zero clean laps that specific round (e.g. a lap-1 crash) - both by construction, not imputed. |
| `driver_prior_relative_pace_std_s` | "Historical consistency": expanding standard deviation of the same relative-pace series | numeric | seconds | `mpp.features.historical` | known only after >= 2 prior rounds complete | Yes | Null until a driver has at least 2 prior rounds of data - one round later than the mean feature above. Reflects race-to-race variability, not single-lap variability. |

## Team

| Feature | Description | Type | Unit | Source | Availability timing | Usable for prediction | Limitations |
|---|---|---|---|---|---|---|---|
| `team` | Constructor name (e.g. `Red Bull Racing`) | nominal categorical | - | `mpp.cleaning.laps` (from FastF1) | pre-session | Identity/reference column, same reasoning as `driver` above - not in `FEATURE_COLUMNS` by default. | Lower cardinality than driver (~10), so a future phase adding it back is cheaper. |
| `team_prior_relative_pace_s` | Team's relative pace (median across *both* of its cars that round vs. the field), expanding-averaged over strictly prior rounds | numeric | seconds | `mpp.features.team` | known only after the prior round(s) complete | Yes | Same null-until-history caveat as the driver feature. Correctly survives a mid-season driver change (keyed on team, not driver) - verified in `tests/features/test_team.py`. |

## Circuit

| Feature | Description | Type | Unit | Source | Availability timing | Usable for prediction | Limitations |
|---|---|---|---|---|---|---|---|
| `circuit_id` | Jolpica-F1 circuit identifier (e.g. `bahrain`) | nominal categorical | - | `mpp.features.build` (from the Jolpica-F1 schedule) | known before the season starts | Used as the **grouping key** for train/test splitting (`docs/05_split_strategy.md`), not fed as a raw model feature - a model instead sees the circuit's effect through the weather/temperature and historical features that vary with it. | - |
| `driver_circuit_prior_relative_pace_s` | Driver's relative pace specifically at this circuit, expanding-averaged over that driver's *prior visits to this same circuit* only | numeric | seconds | `mpp.features.circuit` | known only after a prior visit to the same circuit | Yes | **Null for every row in the current 2023 rounds 1-6 scope** - every round is a different circuit, so no driver has a prior visit yet. Implemented and unit-tested against synthetic data with a repeated circuit (`tests/features/test_circuit.py`); activates automatically once the season scope (`config/settings.toml`) is widened to include a repeat. |
| *(circuit static characteristics)* | Length, corner count, elevation, pit-lane length, etc. | - | - | not sourced | - | **Not implemented.** Neither FastF1 nor Jolpica-F1 provides these, and hand-typing them would violate the "real data only" rule (`docs/01_data_sources.md`). Deferred until a properly sourced reference table is added (architecture SS20). |

## Tyre

| Feature | Description | Type | Unit | Source | Availability timing | Usable for prediction | Limitations |
|---|---|---|---|---|---|---|---|
| `compound` | Tyre compound (`SOFT`/`MEDIUM`/`HARD`/`INTERMEDIATE`/`WET`/`UNKNOWN`) | nominal categorical | - | `mpp.cleaning.laps` (from FastF1) | known at stint start | Yes - one-hot encoded (`mpp.features.encoding`), never ordinally encoded (there is no meaningful order across a dry/wet compound axis). | `UNKNOWN` covers a small number of laps where FastF1 didn't report a recognized compound - see `docs/02_data_pipeline.md`. |
| `tyre_age` | Laps completed on the current tyre set (FastF1 `TyreLife`) | numeric | laps | `mpp.cleaning.laps` (from FastF1) | known live, lap by lap | Yes | Can start above 1 on a stint's first *race* lap - pre-race reconnaissance/formation laps on the same set count towards it. This is real FastF1 behaviour (confirmed against 2023 data), not a defect - see `mpp.features.lap` for why `laps_into_stint` exists as the independent, always-starts-at-1 alternative. |
| `stint` | Stint number within the session (1st set of tyres, 2nd, ...) | numeric (ordinal-like index) | count | `mpp.cleaning.laps` (from FastF1) | known live | Yes | - |
| `tyre_age_sq` | `tyre_age^2` | numeric | laps^2 | `mpp.features.tyre` | same as `tyre_age` | Yes | Captures non-linear (accelerating/decelerating) degradation shape for a linear model; redundant for tree-based models, which learn non-linearity from `tyre_age` alone. |
| `tyre_age_x_soft`, `_medium`, `_hard`, `_intermediate`, `_wet` | `tyre_age` if the lap ran that compound, else 0 | numeric | laps | `mpp.features.tyre` | same as `tyre_age`/`compound` | Yes | An explicit compound x tyre-age interaction, because softer compounds are known to degrade faster than harder ones - real race-engineering reasoning, not an arbitrary cross term (see module docstring). Only meaningful for linear models; tree-based models already learn this interaction from `compound` + `tyre_age`. `UNKNOWN` compound gets no interaction column (there is no degradation rate to attribute to an unidentified tyre). |

## Session

| Feature | Description | Type | Unit | Source | Availability timing | Usable for prediction | Limitations |
|---|---|---|---|---|---|---|---|
| `session_type` | Session code (`R` = Race) | nominal categorical | - | `mpp.features.session` | known before the session | Yes (one-hot encoded) | **Constant** in the current scope - only Race sessions are ingested (`docs/01_data_sources.md`). Wired through as a real column now so ingesting a second session type later needs no schema change, only a wider `config/settings.toml` scope. |

## Lap

| Feature | Description | Type | Unit | Source | Availability timing | Usable for prediction | Limitations |
|---|---|---|---|---|---|---|---|
| `lap_number` | Lap number within the race | numeric | count | `mpp.cleaning.laps` (from FastF1) | known live | Yes | - |
| `race_fraction` | `lap_number / total_laps` - a fuel-load proxy | numeric | fraction, 0-1 | `mpp.features.context` | known live (`total_laps` is known before the race starts) | Yes | Only a proxy for fuel load, not a direct measurement - no fuel-mass telemetry is available from either source. |
| `laps_into_stint` | Lap number relative to this stint's own start (always starts at 1) | numeric | count | `mpp.features.lap` | known live | Yes | Deliberately independent of `tyre_age` (see Tyre section above) - a genuinely distinct signal, not a duplicate feature. |

## Weather (optional context, not a named group in the brief but present in the feature table)

| Feature | Description | Type | Unit | Source | Availability timing | Usable for prediction | Limitations |
|---|---|---|---|---|---|---|---|
| `air_temp` | Air temperature, nearest sample to the lap | numeric | degrees Celsius | `mpp.features.weather` (from FastF1) | known live | Yes | Null if no weather sample falls within a 5-minute tolerance of the lap. |
| `track_temp` | Track surface temperature, nearest sample to the lap | numeric | degrees Celsius | `mpp.features.weather` (from FastF1) | known live | Yes | Same tolerance caveat as `air_temp`. |
| `rainfall` | Whether it was raining at the nearest weather sample | boolean | - | `mpp.features.weather` (from FastF1) | known live | Yes | Same tolerance caveat; a boolean flag, not a rainfall intensity measurement. |

## Performance

| Feature | Description | Type | Unit | Source | Availability timing | Usable for prediction | Limitations |
|---|---|---|---|---|---|---|---|
| `stint_prior_mean_lap_time_s` | Expanding mean of this *same stint's own prior clean laps'* lap time - "how has pace trended so far this stint" | numeric | seconds | `mpp.features.performance` | known live, lap by lap, within the current stint only | Yes | Null on a stint's first lap (no prior lap yet) and on any invalid lap (no representative pace of its own to contribute or receive). Never reflects a different stint or a future lap in the same stint - see `tests/features/test_performance.py` and `tests/test_leakage_review.py`. |

## Targets (not features - never appear on the input side of a model)

| Column | Dataset | Description | Type | Unit |
|---|---|---|---|---|
| `lap_time_s` | `lap_time` | Lap time | numeric | seconds |
| `lap_time_delta_s` | `tyre_degradation` | Lap time minus the stint's baseline (first clean) lap time | numeric | seconds |
| `session_pace_s` | `race_pace` | Median lap time among a driver's clean laps in the session | numeric | seconds |
| `duration_s` | `pit_stop` | Recorded pit stop stationary time | numeric | seconds |

## Group / split keys (not features, not targets)

| Column | Description | Type |
|---|---|---|
| `season`, `round` | The race a row belongs to - the grouping key for `mpp.splits.group_split.race_group_*` | numeric |
| `circuit_id` | The circuit a round was held at - the grouping key for `mpp.splits.group_split.circuit_group_*` | nominal categorical |

## Deliberately not built

- **Same-session relative pace** (comparing a lap to other drivers' pace *in
  the same, currently-being-predicted round*): rejected during the leakage
  review. Unlike the historical features above, this would use other
  competitors' realized results from the very race being predicted -
  information a real forecast would not have before or during that race.
- **Full-stint degradation slope as a per-lap feature** (fitting a trend
  across *all* of a stint's laps, including ones after the row being
  predicted, then attaching that slope to every lap in the stint): rejected
  for the same reason - a mid-stint lap cannot know the stint's own future
  laps. `stint_prior_mean_lap_time_s` (strictly prior laps only) is the safe
  version actually implemented.
