# Phase 5 Report - Tyre Degradation & Additional Prediction

**Status: complete.** All three prediction problems `docs/PHASE_4_REPORT.md`
deferred - tyre degradation, race pace, and pit stop - are trained and
evaluated against the real 2023 rounds 1-6 data, each reusing Phase 4's
infrastructure (baselines, metrics, evaluation harness) with the target,
feature set, and validation strategy its own dataset actually supports, per
`docs/ARCHITECTURE_REPORT.md` SS17. A generic error-analysis module
(`mpp.evaluation.error_analysis`) and a shared nested race-grouped training
protocol (`mpp.models.train_common`) are new infrastructure this phase adds,
used by two of the three problems.

None of the three beats its strong baseline by a margin this project is
willing to call "race-ready" - consistent with Phase 4's own lap-time
finding and this project's habit of reporting the honest negative result
rather than reaching for one. Where a candidate's mean MAE *numerically*
edges the baseline below, the fold-to-fold standard deviation is reported
alongside it specifically so that margin can be judged against the noise,
not read as a confident win on its own.

## Bug found and fixed: `time_series_splits` could leak a round across train/test

While wiring up the race-pace model (the one problem using
`mpp.splits.temporal_split.time_series_splits`), a direct check of real
fold contents found every fold's train and test sets sharing part of the
boundary round - e.g. `train_rounds=[1,2,3], test_rounds=[3,4,5]`. The
function's own docstring promises "never puts a later round in train than
in test"; this silently broke that promise whenever a dataset has more than
one row per round (every dataset in this project does - one row per driver
per round), because `TimeSeriesSplit` was being applied to raw row
*positions* rather than to the *distinct rounds*, and a fold boundary could
fall in the middle of a round's block of rows.

Fixed in `src/mpp/splits/temporal_split.py`: `TimeSeriesSplit` is now
applied to the array of **distinct rounds**, and every row belonging to a
round is mapped to whichever side of the split that round landed on - so no
round can ever appear on both sides of a fold. A regression test
(`test_time_series_splits_never_splits_a_round_across_train_and_test`) was
added with a dataset shaped exactly like the leaking case (3 rows/round).
The existing tests for this function still pass unchanged - the fix makes
a documented guarantee strictly true where it was previously true only by
accident of row ordering.

## What was built

- **`mpp.evaluation.error_analysis`** - residual slicing shared by all three
  new models (architecture SS14): `error_by_group` (MAE/RMSE/mean bias per
  distinct value of a column), `error_breakdown` (runs that over several
  columns at once and reports *why* a column was skipped - absent from the
  dataset, or degenerate with fewer than two distinct values in this
  scope - rather than silently omitting it), `bucket_numeric` (for tyre-age
  buckets), `largest_absolute_errors` (manual-inspection candidates), and
  `to_markdown_table` (this project has no `tabulate` dependency, so
  `DataFrame.to_markdown()` isn't available - a small dependency-free
  formatter instead).
- **`mpp.models.generic_pipelines.build_pipeline`** - the same
  `preprocess -> estimator` `Pipeline` shape `mpp.models.regressors`
  established for lap_time, generalized over a dataset's own feature
  columns instead of being hard-wired to lap_time's. `regressors.py` is
  untouched.
- **`mpp.models.train_common`** - the nested, race-grouped
  leave-one-race-out protocol `mpp.models.train_lap_time` built for lap
  time, factored out and parametrized (`NestedTrainingSpec`) so tyre
  degradation and pit stop - both grouped by `(season, round)` per
  `docs/05_split_strategy.md`, same lap/stop-within-a-race correlation
  reasoning as lap_time - reuse it instead of re-implementing it. Collects
  each candidate's out-of-fold predictions (every row scored exactly once,
  only by a model that never saw its race during training or tuning) for
  error analysis.
- **`mpp.models.train_tyre_degradation`** (Task 1) - mean/median/
  per-compound-median baselines, linear regression, tuned Ridge, tuned
  Random Forest; nested leave-one-race-out CV via `train_common`; per-
  compound and per-tyre-age-bucket error breakdown.
- **`mpp.models.train_race_pace`** (Task 2) - mean/median/per-driver-median
  baselines, linear regression, Ridge, Random Forest; **chronological**,
  forward-chaining CV (`mpp.splits.temporal_split.time_series_splits`, now
  fixed); no hyperparameter tuning, by deliberate choice (see below).
- **`mpp.models.train_pit_stop`** (Task 3) - mean/median/per-team-median
  baselines, linear regression, tuned Ridge, tuned Random Forest; same
  nested leave-one-race-out protocol as tyre degradation, via
  `train_common`.

## Task 1: Tyre degradation

**Target**: `lap_time_delta_s` - lap time relative to each stint's own
baseline (first clean) lap (`mpp.datasets.tyre_degradation`, built in Phase
3, unchanged this phase). **Features**: `compound` (nominal) plus tyre-age,
race-progress, weather, and prior-pace numeric features;
`driver_circuit_prior_relative_pace_s` excluded (100% null, zero repeated
circuits in this scope - same exclusion lap_time applies).
**Validation**: nested, race-grouped, leave-one-race-out CV (6 outer folds,
identical protocol to lap_time).

| Model | Category | MAE (s) | RMSE (s) | R² |
|---|---|---|---|---|
| mean_baseline | baseline | 7.94 ± 1.76 | 11.19 | -0.71 |
| median_baseline | baseline | 7.96 ± 2.08 | 11.37 | -0.51 |
| **strong_baseline (per-compound median)** | baseline | **7.89 ± 2.26** | 11.45 | -0.47 |
| linear_regression | baseline | 9.94 ± 5.06 | 13.08 | -1.06 |
| ridge | model | 8.76 ± 3.72 | 12.06 | -0.68 |
| **random_forest** | **model** | **7.77 ± 2.22** | 11.41 | -0.63 |

Random Forest's MAE (7.77s) is numerically the lowest, but by 0.12s against
a strong baseline whose own fold-to-fold standard deviation is 2.26s - well
inside the noise. **This is not a confident win**, and is reported as such:
the honest reading of Task 1 is that none of these candidates meaningfully
outperforms "the per-compound median" here.

**Why the baseline is so hard to beat, in motorsport terms - verified by
inspection, not assumed:**

- The target's own distribution is wide and *median-negative*
  (`lap_time_delta_s`: mean -6.6s, median -3.8s, std 11.2s, range
  [-50.5, +53.2]). Later laps in a stint are typically *faster* than the
  stint's first lap, not slower - the fuel-burn effect (lighter car as the
  race progresses) dominates tyre-wear slowdown within a single stint, the
  exact confound `docs/ARCHITECTURE_REPORT.md` SS18 and
  `docs/03_dataset_definitions.md` already flagged as a known limitation
  this project cannot resolve without fuel-mass telemetry.
- **Safety-car/VSC-affected laps are still not filtered out of this
  dataset** (the same Phase 2 limitation Phase 4 flagged as "planned for
  Phase 4" and still open here) - a lap taken behind the safety car is
  artificially slow for a reason that has nothing to do with tyre wear, and
  inflates exactly the kind of large-delta outliers (+53s) visible in the
  target's range. This phase did not attempt a fix: the persisted
  `tyre_degradation.parquet` schema carries no `lap_number`/`stint` key that
  would let a training-time filter re-join cleanly back to `track_status`
  in `laps_features.parquet` without risking a silent many-to-one mismatch,
  and the brief's "do not chery-pick data" instruction cuts against a rushed
  partial fix - this is named here as the clear next step instead (see
  "Next steps" below).
- Zero repeated circuits (as in lap_time/Phase 4):
  `driver_circuit_prior_relative_pace_s` is 100% null and excluded.

**Error breakdown** (out-of-fold predictions, Random Forest): MAE is worst
for `WET` compound laps (14.2s, n=40 - the smallest compound bucket, most
exposed to single-lap noise) and for a handful of drivers/teams
(ZHO/HAM/SAR, Alfa Romeo/Williams, all 8.6-12.6s); best for `SOFT` compound
(5.4s) and shorter stints (`0-5` laps into stint: 6.0s vs. 8.0s for `16+`).
The tyre-age-bucket gradient (worse error at higher tyre age) is consistent
with - but does not prove - degradation becoming harder to predict later in
a stint; it is equally consistent with the fuel-burn confound growing along
with stint length. `session_type` is N/A (constant `"R"` - this project's
current scope is race-only).

**Association vs. causation**: a compound or tyre-age coefficient/
importance here is a statement about *predictive association* with the
lap-time delta in this 6-race sample, not an isolated physical tyre-wear
effect - the delta is a mix of tyre wear, fuel burn, and (per the point
above) unflagged SC/VSC laps. No claim here should be read as "compound X
costs Y seconds/lap of wear."

## Task 2: Race pace

**Target**: `session_pace_s` - a driver's median clean-lap time for a
session (`mpp.datasets.race_pace`, Phase 3, unchanged). **Features**: prior
relative-pace (driver/team), prior-pace variability, session weather;
`driver_circuit_prior_relative_pace_s` excluded (same reason as Task 1).
**Validation**: chronological, forward-chaining (`time_series_splits`, the
bug described above now fixed) - round 1 is never a test fold, since there
is no earlier round to train a forecast on; this is the question's own
structure, not a bug. **No hyperparameter tuning** - see below.

| Model | Category | MAE (s) | RMSE (s) | R² |
|---|---|---|---|---|
| mean_baseline | baseline | 10.30 ± 5.63 | 10.40 | -145.70 |
| median_baseline | baseline | 9.95 ± 5.10 | 10.06 | -135.79 |
| **strong_baseline (per-driver median)** | baseline | **9.83 ± 5.07** | 9.92 | -130.78 |
| linear_regression | baseline | 20.77 ± 12.42 | 20.83 | -448.74 |
| ridge | model | 15.03 ± 11.82 | 15.11 | -254.07 |
| random_forest | model | 10.25 ± 5.70 | 10.37 | -164.09 |

**No candidate beats the strong baseline, and by a wide margin for the
linear models.** R² is catastrophically negative for every model (down to
-448.7) because `session_pace_s` is an *absolute* lap time, overwhelmingly
driven by which circuit the session was at (the same dominant effect
lap_time's Phase 4 report documented) - but `mpp.datasets.race_pace` carries
**no circuit-identity feature at all** (only prior-pace and weather). This
was verified directly, not assumed: a linear model trained on round 1 alone
predicts round 2's pace band off by roughly 30 seconds (round 1's weather-
only signal has nothing to anchor round 2's different circuit to). The
per-driver median baseline sidesteps this entirely by not trying to predict
an absolute number from context at all - it just repeats the driver's own
recent level, which is why it wins comfortably.

**Why no hyperparameter tuning, unlike Task 1/3**: with 6 rounds and a
forward-chaining split, the smallest outer training fold is a single round
(~20 rows). A further nested split for tuning would leave only a handful of
rows per inner fold - not enough to choose a hyperparameter meaningfully,
and exactly the brute-force-on-too-little-data pattern the brief warns
against. Ridge uses scikit-learn's default `alpha=1.0`; Random Forest uses
fixed, deliberately shallow hyperparameters (`max_depth=3,
min_samples_leaf=5`) chosen to resist overfitting the smallest folds, not
tuned to this data.

**Error breakdown** (out-of-fold predictions, strong baseline; round 1's 20
rows are structurally excluded - never a test fold): worst by circuit is
`monaco` (18.7s MAE) and `baku` (13.1s) - both street circuits with unusual
pace bands relative to the permanent circuits the model had seen by that
point; best is `jeddah` (3.4s) and `miami` (4.4s). By driver, the spread is
narrow (10.3-11.9s for the worst 5) - no single driver's rows are a
standout outlier, consistent with the error being dominated by the
circuit-identity gap described above rather than a per-driver data issue.
`compound`/`tyre_age`/`session_type` are N/A: this dataset is a per-session
aggregate with no per-lap compound/tyre granularity, and session scope is
race-only.

## Task 3: Pit stop duration

**Feasibility verdict: attempted, not deferred** (architecture SS17 Phase 7
gate). Jolpica-F1 provides 219 per-stop rows across the 6-round scope -
enough for a race-grouped CV - but `mpp.datasets.pit_stop` only persists
`lap_number`/`stop_number` as model features (Phase 3). `team` and
`circuit_id` (pit-crew identity, pit-lane layout - both genuinely known
before a stop happens) are added at training time from the dataset's own
already-present identity/group columns, not fabricated. No tyre-compound-
change or pit-lane-traffic signal is available at all - this project's
sources don't carry it - and that absence is the real ceiling on this
problem, stated plainly rather than worked around.

**Target**: `duration_s` - full recorded stationary time.
**Validation**: nested, race-grouped, leave-one-race-out CV (same protocol
as Task 1, via `train_common`).

| Model | Category | MAE (s) | RMSE (s) | R² |
|---|---|---|---|---|
| mean_baseline | baseline | 362.1 ± 194.3 | 397.5 | -5,493.5 |
| **median_baseline** | baseline | **135.6 ± 295.3** | 172.3 | -1.02 |
| strong_baseline (per-team median) | baseline | 135.9 ± 295.0 | 172.5 | -1.40 |
| linear_regression | baseline | 272.1 ± 241.0 | 333.0 | -2,339.7 |
| ridge | model | 276.2 ± 240.7 | 336.5 | -2,433.7 |
| random_forest | model | 164.0 ± 285.3 | 239.2 | -818.2 |

**No candidate beats the median baseline - by a wide margin for every
model.** This is the clearest negative result of the three problems, and it
has a concrete, verified cause: **49 of 219 recorded stops (22%) exceed 60
seconds, and 10 of those - all from round 3 (the 2023 Australian Grand
Prix) - exceed 1,800 seconds (up to 1,871s, over 31 minutes)**. Round 3 had
multiple red flags; Jolpica-F1's recorded "pit stop" duration for cars
parked in the pit lane during a race suspension is indistinguishable, in
this data, from a tyre-change stop - `docs/03_dataset_definitions.md`
already documents that `duration_s` is not decomposed by cause, and this is
that limitation showing up concretely. **These rows are not filtered out**
(per the brief's "do not cherry-pick data to improve metrics") - instead,
the by-round error breakdown below shows exactly how much round 3 dominates
when it is the held-out fold.

**Error breakdown** (out-of-fold predictions, Random Forest):

| round | circuit | n | MAE (s) |
|---|---|---|---|
| 3 | albert_park | 65 | 794.8 |
| 6 | monaco | 37 | 130.8 |
| 4 | baku | 23 | 24.6 |
| 2 | jeddah | 24 | 16.2 |
| 1 | bahrain | 50 | 12.2 |

Round 3's fold alone (794.8s MAE) accounts for the overwhelming majority of
the overall 164.0s MAE - the other five races average under 40s. Monaco's
130.8s MAE is a real, secondary finding: Monaco's famously narrow pit lane
is a plausible non-causal explanation, but with one race's worth of data
this cannot be distinguished from another anomaly, and is flagged rather
than asserted as fact. `compound`/`tyre_age`/`session_type` are N/A - no
column for any of them exists in this dataset.

**Association vs. causation**: a `team` coefficient/importance here
conflates genuine pit-crew-skill signal with round 3's red-flag stoppage
time, which is not a crew-skill signal at all. No claim here should be read
as "team X has a Y-second pit crew."

## Error analysis strategy (architecture SS14) - how it was applied

Every breakdown above is computed on **out-of-fold predictions only** - each
row scored exactly once, by whichever model never saw its own race (Tasks
1/3) or a later-than-its-own round (Task 2) during training or tuning - so
none of it is an in-sample artifact. `mpp.evaluation.error_analysis.
error_breakdown` reports, for each requested slicing dimension, either a
table or an explicit reason it was skipped (column absent, or fewer than
two distinct values in this scope) - every "N/A" above is a pointer into
that, not a silent omission.

## Testing performed

- `python -m pytest -q` - **178 passed, 0 failed** (153 from Phases 2-4 plus
  25 new): `error_analysis` (group breakdown correctness, missing/degenerate
  column handling, largest-error ranking), the `time_series_splits` fix (a
  dedicated regression test plus the existing suite, all still passing),
  and one integration test per new training script (every
  baseline/candidate evaluated, metadata/limitations documented, a
  serializable final pipeline when one beats its baseline) on small
  synthetic, dataset-shaped fixtures with monkeypatched search grids -
  same philosophy as `tests/models/test_train_lap_time.py`.
- Edge cases: `tests/models/test_train_pit_stop.py` additionally covers a
  null-target row being dropped before training, and a single-race dataset
  raising `ValueError` rather than silently producing a meaningless
  zero-fold "evaluation" (leave-one-race-out CV requires at least 2 races).
- Every real-data run above (`python -m mpp.models.train_tyre_degradation`,
  `train_race_pace`, `train_pit_stop`) executed against the actual 2023
  rounds 1-6 `data/model_ready/*.parquet` files - the numbers in this report
  are that run's actual output, not illustrative figures.
- Manual inspection, not just automated assertion, for every "why" claim
  above: the target distribution for Task 1 (`lap_time_delta_s.describe()`),
  the round-1-trained-on-round-2 prediction gap for Task 2, and the
  duration-by-round breakdown for Task 3 were each checked directly against
  the real data before being written up here.

## Known limitations (all three problems)

- **No problem here is "race-ready"** - this phase's headline honest finding,
  matching Phase 4's for lap_time.
- **Tyre degradation**: fuel-burn/tyre-wear confound unresolved;
  safety-car/VSC laps unfiltered (a documented, not fixed, limitation -
  see "Next steps").
- **Race pace**: no circuit-identity feature in the dataset at all - the
  single most valuable missing feature for this specific problem.
- **Pit stop**: no tyre-compound-change or pit-lane-traffic signal
  available; `duration_s` conflates red-flag stoppage time with genuine
  service time for round 3.
- **Small scope throughout**: 6 races, 0 repeated circuits - the same
  structural ceiling Phase 4 identified for lap_time applies to all three
  problems here, usually more severely (race_pace has 119 rows total;
  pit_stop 219).

## Next steps

- Widening the season scope (config-only change) remains the single
  highest-leverage fix available, per Phase 4 - it would let the per-
  compound/per-driver/per-team strong baselines differentiate from a global
  median, give race_pace's temporal split more than 6 points to forecast
  across, and dilute round 3's outsized influence on the pit-stop model.
- A safety-car/VSC-aware filter for `tyre_degradation` is the clearest
  single improvement specific to Task 1: it needs `mpp.datasets.
  tyre_degradation` to carry a join key (e.g. `lap_number`) back to
  `laps_features.track_status` that the current schema doesn't persist -
  a Phase 3 schema change, not a training-time fix, and therefore out of
  scope for this phase.
- A circuit-identity feature (e.g. a per-circuit historical pace offset, if
  and when circuits repeat) is the clearest single improvement for Task 2.
