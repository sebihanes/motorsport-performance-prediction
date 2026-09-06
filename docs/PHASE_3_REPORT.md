# Phase 3 Report - Feature Engineering

**Status: complete.** A reusable, leakage-reviewed feature engineering layer
covering every group the brief asked for, a categorical-encoding pipeline,
and a split-strategy implementation matched to each prediction problem.

## What was built

- **Driver**: identity (`driver`, kept as a reference/error-analysis column,
  not a raw model feature), historical pace (`driver_prior_relative_pace_s`,
  existing from Phase 2) and historical consistency
  (`driver_prior_relative_pace_std_s`, new).
- **Team**: identity (`team`) and historical performance
  (`team_prior_relative_pace_s`, new, `mpp.features.team`) - correctly
  keyed on the team, so it survives a mid-season driver change.
- **Circuit**: identity (`circuit_id`) and normalized historical performance
  (`driver_circuit_prior_relative_pace_s`, new, `mpp.features.circuit`) - null
  in the current 6-round scope (no repeated circuits) but implemented and
  tested against synthetic data with a repeat, so it activates the moment
  the scope widens. Static circuit characteristics (length, corners) remain
  deliberately unbuilt - not available from either data source, and
  inventing them would violate the "real data only" rule.
- **Tyre**: `compound`, `tyre_age`, `stint` (existing), plus an explicit
  compound x tyre-age interaction (`tyre_age_x_<compound>`, new,
  `mpp.features.tyre`) - motivated by the real fact that softer compounds
  degrade faster, not an arbitrary cross term.
- **Session**: `session_type` (new, `mpp.features.session`) - constant today
  (Race-only ingestion) but wired through as a real column for when a second
  session type is ingested later.
- **Lap**: `lap_number`, `race_fraction` (existing), plus
  `laps_into_stint` (new, `mpp.features.lap`) - deliberately independent of
  `tyre_age`, since tyre age can start above 1 due to pre-race laps (a real
  nuance confirmed against the ingested 2023 data).
- **Performance**: historical average pace, relative pace (existing), plus
  rolling within-stint pace trend (`stint_prior_mean_lap_time_s`, new,
  `mpp.features.performance`) and consistency (the driver std feature above).
- **Categorical encoding** (`mpp.features.encoding.build_feature_pipeline`):
  a `ColumnTransformer` that one-hot encodes nominal columns
  (`handle_unknown="ignore"`, required because the grouped split strategy can
  put an unseen driver/circuit/compound in a test fold) and passes numeric
  columns through unscaled by default (scikit-learn's tree-based candidate
  models don't need it), with an opt-in `scale_numeric` flag for linear
  models.
- **Split strategy** (`mpp.splits`): `group_split.py` (`GroupKFold`/
  `GroupShuffleSplit`, grouped by race or by circuit) for the three
  lap/stop-correlated datasets (`lap_time`, `tyre_degradation`, `pit_stop`);
  `temporal_split.py` (chronological holdout / `TimeSeriesSplit`, ordered by
  round) for `race_pace`, whose prediction question is explicitly
  forward-looking. See `docs/05_split_strategy.md` for the full random vs.
  grouped vs. temporal comparison and per-dataset reasoning.
- **Feature dictionary** (`docs/04_feature_dictionary.md`): every column in
  the feature table, with description, type, unit, source, availability
  timing, prediction-safety, and known limitations - including a section on
  features deliberately *not* built (same-session relative pace; a
  whole-stint degradation slope) and why they would have leaked.

## Verification performed

- `python -m pytest -q` - **116 passed, 0 failed** (72 from Phase 2 plus 44
  new: every feature calculation, missing-value behaviour, categorical
  handling case, leakage-sensitive feature, split-integrity property, and
  edge case listed in the Phase 3 brief).
- `python -m mpp.pipeline` re-run against the real configured scope: **0
  validation errors, 1 warning** (unchanged from Phase 2, expected); feature
  table grew from 26 to 37 columns; all four model-ready datasets rebuilt
  successfully (`lap_time` 6,145 rows, `tyre_degradation` 6,132,
  `race_pace` 119, `pit_stop` 219).
- Feature columns validated on the real generated `lap_time.parquet`:
  null rates for every historical feature checked against expectations
  (`driver_circuit_prior_relative_pace_s` 100% null, correctly, in this
  scope; `driver_prior_relative_pace_s` ~15.5% null, consistent with round 1
  plus one driver's zero-clean-lap round); `build_feature_pipeline` fit and
  transformed the real dataset into a one-hot-encoded matrix with no errors.
- Leakage review (`tests/test_leakage_review.py`): an end-to-end regression
  test perturbs a later round's lap times by 10,000 seconds and asserts
  every historical feature (driver, team, circuit, within-stint trend) is
  bit-for-bit unchanged for earlier rounds; a same-driver-round invariance
  check confirms round-level historical features never vary lap-to-lap
  within a single driver's session (which would indicate an accidental join
  on the row's own data); a concrete random-vs-grouped-split demonstration
  shows a naive random split leaking races across train/test while the
  grouped split does not.
- Split strategies validated on the real `lap_time` dataset: race-grouped
  split produced disjoint race sets (2,945/3,200 rows), circuit-grouped
  split produced disjoint circuit sets (2,858/3,287 rows), and
  `race_group_kfold` produced exactly 6 folds (one per configured round)
  covering every row exactly once as test.

## Known limitations (see `docs/04_feature_dictionary.md` for detail)

- `driver_circuit_prior_relative_pace_s` is null for every row in the
  current scope - no circuit repeats within 2023 rounds 1-6.
- Circuit static characteristics (length, corners) remain unbuilt - not
  available from either ingested source.
- `session_type` is constant (`"R"`) - only Race sessions are ingested.
- The compound x tyre-age interaction columns are only useful for linear
  models; tree-based models learn the same interaction natively.

## Next: Phase 4

Baselines, model training, cross-validation and hyperparameter tuning using
the split functions built here, evaluation metrics, interpretability, and
error analysis - architecture report SS10-SS14, phase 4-6.
