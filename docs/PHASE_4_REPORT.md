# Phase 4 Report - Baselines & Machine Learning Models

> **Final-audit correction (docs/FINAL_AUDIT.md).** The "strong" baseline
> used in this report (per-circuit median) degenerates to the global median
> in this zero-repeated-circuit scope, so it was not the strongest
> reasonable baseline that architecture SS10 requires. A zero-parameter
> **stint pace persistence** baseline (predict the stint's own mean lap time
> so far) scores **3.669s MAE** under the same nested leave-one-race-out
> protocol, against Ridge's 5.953s. **No lap-time candidate beats it**, so
> `train_lap_time` now gates on the strongest baseline and serializes
> nothing. Every number below is still correct as a measurement; the claims
> that Ridge "beats the strong baseline" and was serialized are superseded.

**Status: complete.** A leakage-reviewed baseline + candidate-model + nested
cross-validation + tuning + evaluation framework for the lap-time model (the
anchor problem, architecture SS17), run against the real 2023 rounds 1-6
data, plus an honest, non-cherry-picked interpretation of what the current
data scope does and does not support.

Only lap-time prediction was trained in this phase. Per the brief ("do not
implement multiple prediction problems if the data quality does not support
them") and `docs/03_dataset_definitions.md`'s own limitations section,
`tyre_degradation` conflates wear with fuel burn and doesn't yet filter
Safety Car laps, `race_pace` has only 119 rows, and `pit_stop` has only 219 -
none of the three is ready for a defensible modeling exercise yet. Lap time
(6,145 rows, `0` validation errors) is the dataset that supports it, matching
the architecture report's Phase 4 scope.

## What was built

- **Baselines** (`mpp.models.baselines`): `mean_baseline` / `median_baseline`
  (`sklearn.dummy.DummyRegressor`), plus `PerGroupMedianRegressor` - the
  per-circuit median the architecture report calls the real bar to clear
  (SS10), with a documented fallback to the global median for a circuit
  unseen at fit time.
- **Candidate model pipelines** (`mpp.models.regressors`): Linear Regression,
  Ridge, Random Forest, Gradient Boosting, and HistGradientBoostingRegressor,
  each a `sklearn.Pipeline` wrapping the shared
  `mpp.features.encoding.build_feature_pipeline` ColumnTransformer -
  preprocessing is always fit fresh on a fold's training rows only, never on
  the whole dataset ahead of splitting. `build_feature_pipeline` gained an
  `impute_numeric` option (Phase 3's `encoding.py` is otherwise unchanged;
  default behavior and every existing test are preserved) for the models
  that cannot consume `NaN` directly. `driver_circuit_prior_relative_pace_s`
  is deliberately excluded from every model's feature set: it is 100% null
  in the current 6-round scope, so it carries zero information and a median
  imputer has no observed value to compute from.
- **Hyperparameter tuning** (`mpp.models.tuning`): `GridSearchCV` for Ridge
  (5 alpha values - cheap enough to search exhaustively), `RandomizedSearchCV`
  (8 sampled candidates) for the three tree-based models - controlled search
  spaces, not brute force.
- **Evaluation** (`mpp.evaluation.metrics`, `mpp.models.evaluate`): MAE,
  RMSE, R^2, and median absolute error, each with a stated engineering
  purpose (SS12); a reusable single-level grouped-CV harness
  (`mpp.models.evaluate.cross_validate` / `holdout_evaluate`) for simpler
  cases, and a project-specific nested leave-one-race-out loop (below) for
  the full lap-time evaluation.
- **Training orchestrator** (`mpp.models.train_lap_time`, run via
  `python -m mpp.models.train_lap_time`): runs every baseline and candidate
  model through the same validation strategy, writes a structured comparison
  (`data/quality/model_comparison_lap_time.{json,md}`), and serializes the
  winning pipeline with `joblib` plus a metadata file - only if it actually
  beats the strong baseline.

## Validation strategy actually used, and why it changed mid-phase

`docs/05_split_strategy.md` (Phase 3) specifies a race-grouped split for
`lap_time` and, separately, a **nested** split for tuning: "an outer
grouped/temporal split ... held out purely for the final reported metric,
and an inner grouped/temporal split ... for choosing hyperparameters," with
the `mpp.splits` functions built "fold-generator-shaped ... so a
GridSearchCV/RandomizedSearchCV `cv=` argument can consume them directly,
**nested inside an outer loop over the same functions**."

The first implementation of this phase read that as a single 75/25
`race_group_shuffle_split` (outer) with `race_group_kfold` (inner) for
tuning - a reasonable reading, and the design used throughout Phase 3's own
examples. Run against the real data, it produced results that changed
sharply depending on which 1-2 of the current 6 races happened to land in
the one-shot outer test set (round 3 + round 6, by the fixed `random_state=0`
used throughout this project): every candidate model scored **worse** than
the naive global-mean baseline, by a wide margin. That is a real result, not
a display bug (verified: predictions, feature values, and per-circuit means
were all inspected directly), but it is also a noisy one - a single 2-race
sample is a small, high-variance estimate of "generalization to an unseen
race" when there are only 6 races total.

The fix, still within what `docs/05_split_strategy.md` specifies (the
"outer loop over the same functions" phrasing), is a fully **nested
leave-one-race-out** design:

1. **Outer loop**: `race_group_kfold(dataset, n_splits=<more than there are races>)` -
   the function caps this to exactly one fold per race, so every race is
   held out and scored exactly once, after training on every other race.
2. **Inner loop**: for each outer fold, `GridSearchCV`/`RandomizedSearchCV`'s
   `cv=` is a second `race_group_kfold` computed **only** over that outer
   fold's training races - never the held-out one, so no race is ever used
   both to tune and to score, even transitively.
3. Each candidate's reported metric is the **mean and std across all 6
   outer folds**, not one arbitrary split - removing the small-sample
   sensitivity that produced the first, misleadingly bleak result.
4. Nested CV estimates the generalization performance of the "tune, then
   fit" *procedure*. The deployed artifact is produced by running that
   same procedure once more - tuned via CV over every race, then fit on the
   full dataset - per standard nested-CV practice, and is not itself scored
   on held-out data.

This is documented here rather than silently replaced because the discarded
first version is itself informative: it demonstrates concretely why a single
small-sample holdout is the wrong tool when there are only 6 groups
available, which is exactly the kind of validation-strategy judgment call
the brief asks to "clearly document."

## Results (real 2023 rounds 1-6 data, leave-one-race-out nested CV)

| Model | Category | MAE (s) | RMSE (s) | R^2 | Median AE (s) |
|---|---|---|---|---|---|
| mean_baseline | baseline | 9.16 +/- 5.76 | 10.72 | -2.55 | 9.12 |
| median_baseline | baseline | 8.96 +/- 4.92 | 10.47 | -2.54 | 8.89 |
| per_circuit_median_baseline | baseline | 8.96 +/- 4.92 | 10.47 | -2.54 | 8.89 |
| linear_regression | baseline | 6.28 +/- 4.10 | 8.19 | -1.31 | 5.13 |
| **ridge** | **model** | **5.95 +/- 4.34** | **8.00** | **-1.16** | **4.73** |
| random_forest | model | 9.25 +/- 6.67 | 10.83 | -3.20 | 8.80 |
| gradient_boosting | model | 7.16 +/- 3.80 | 8.86 | -2.19 | 6.61 |
| hist_gradient_boosting | model | 9.95 +/- 6.34 | 11.74 | -5.14 | 9.37 |

Full per-fold metrics and the winning hyperparameters for each fold are in
`data/quality/model_comparison_lap_time.json` (regenerated by every training
run; not committed, per this repo's data-is-regenerable convention).

**Ridge (alpha=100) is the only candidate that beats the per-circuit-median baseline** (superseded: it does *not* beat stint pace persistence, 3.669s - see the correction at the top),
by about 3 seconds of MAE (5.95s vs. 8.96s). An MAE of 5.95 seconds means
the model's typical absolute lap-time error, on a race it never trained or
tuned on, is about 5.95 seconds - two orders of magnitude worse than the
architecture report's own worked example (0.18s), and **not, on its own, a
claim that this model is race-ready** for any live strategy use.

## Why the baseline is so hard to beat here, in motorsport terms

- **`per_circuit_median_baseline` is mathematically identical to
  `median_baseline` for every fold.** The current scope has 6 races at 6
  distinct circuits with zero repeats (confirmed in
  `docs/03_dataset_definitions.md`), so the "which circuit does this lap
  belong to" lookup never has an answer for the held-out race and always
  falls back to the global median. The strong baseline architecture SS10
  calls for cannot actually do its job yet - it will differentiate itself
  the moment the ingested scope includes a circuit repeat (e.g. a second
  season, or a sprint weekend's second race at the same track).
- **The candidate models have no circuit-identity feature at all** -
  `circuit_id` itself was excluded from the model feature set back in Phase
  3 (`mpp.datasets.lap_time.FEATURE_COLUMNS`), and the one feature that would
  carry circuit-specific history (`driver_circuit_prior_relative_pace_s`) is
  100% null in this scope. So on a held-out race, nothing in the feature set
  directly says "this is a different, faster/slower track."
- **Weather features end up acting as an unintended race-identity proxy that
  doesn't extrapolate.** Inspecting a fitted Random Forest's feature
  importances showed `air_temp` as by far the single most important feature
  (0.51, more than triple the next-highest). That is not a causal claim
  about temperature and lap time - with only 6 races, each contributing one
  characteristic temperature reading, `air_temp` is functionally a proxy for
  "which race this is," not a physical effect the model has isolated. A
  proxy like that only works while the held-out race's temperature falls
  inside the range already seen in training; Albert Park and Monaco (the
  races held out in the earlier single-split run) do not, and tree-based
  models cannot extrapolate past the split boundaries they learned - they
  just return whichever leaf's average the out-of-range input lands in,
  which is what drove Random Forest, Gradient Boosting and
  HistGradientBoosting's outer-fold errors above the naive baseline.
  Ridge's linear extrapolation, imperfect as it is, degrades more gracefully
  outside the training range than a tree ensemble's leaf lookup - which is
  the most plausible reason it is the only candidate that wins here.
- **This is a data-scope limitation, not an implementation bug.** It was
  checked directly: `stint_prior_mean_lap_time_s` (a genuinely
  race-specific, leakage-safe in-race pace signal - see
  `docs/PHASE_3_REPORT.md`) correctly reflects each held-out race's true
  pace band when inspected by hand, so the informative signal *is* present
  in the feature table; the models simply do not weight it enough relative
  to the race-confounded weather features to generalize well from only 6
  races.

## Model comparison (architecture SS10-SS12 structure)

See `data/quality/model_comparison_lap_time.md` for the full table with
strengths/weaknesses/training-considerations per candidate, regenerated by
every training run. Summary:

| Model | Validation strategy | Training considerations | Strengths | Weaknesses |
|---|---|---|---|---|
| mean / median baseline | leave-one-race-out nested CV | none | trivial sanity floor | ignores every feature |
| per-circuit median baseline | leave-one-race-out nested CV | groupby only | the real bar per SS10 | degenerate to the global median in this scope (see above) |
| Linear Regression | leave-one-race-out nested CV | median-imputed, standardized | interpretable | can't model interactions; extrapolates (both a strength and a risk here) |
| Ridge | nested CV + inner GridSearchCV (5 candidates/fold) | median-imputed, standardized | regularized, still interpretable, **only candidate to beat the baseline** | same linear ceiling as above |
| Random Forest | nested CV + inner RandomizedSearchCV (8 candidates/fold) | median-imputed | captures nonlinear interactions in-distribution | cannot extrapolate - hurt badly by the small-race-count, no-circuit-repeat scope |
| Gradient Boosting | nested CV + inner RandomizedSearchCV (8 candidates/fold) | median-imputed | sequential error correction | slower than HistGB; same extrapolation ceiling |
| HistGradientBoostingRegressor | nested CV + inner RandomizedSearchCV (8 candidates/fold) | native NaN handling, no imputation | fast, handles missing weather natively | worst outer-fold MAE here - same extrapolation ceiling as the other trees, without even Random Forest's averaging |

## Serialization

`python -m mpp.models.train_lap_time` serializes the winning pipeline only
when it beats the strong baseline (it did at the time of this phase; since the final audit the bar is the strongest baseline, stint pace persistence, which no candidate beats, so nothing is serialized): `joblib.dump` to
`data/models/lap_time_model.joblib`, alongside
`data/models/lap_time_model_metadata.json` recording the model version
(`lap_time-v1`), training timestamp, Python/scikit-learn/joblib versions,
the training dataset's path/row-count/SHA-256, the full feature list, the
validation strategy string, the winning hyperparameters, and the nested-CV
metrics estimating that model's generalization. Both files are regenerable
(`data/` is git-ignored, matching every other pipeline output) - the
metadata file is what makes a specific `.joblib` artifact reproducible and
attributable to a specific dataset snapshot without committing the binary.

## Verification performed

- `python -m pytest -q` - **153 passed, 0 failed** (116 from Phases 2-3 plus
  37 new): baseline correctness and edge cases (empty input, unseen group,
  mismatched lengths), pipeline construction and missing-value handling per
  model type, the degenerate-all-null-column exclusion (a regression guard),
  metric computation against hand-computed values, the evaluation harness's
  fold-isolation guarantee, and a full nested-CV integration test on a
  synthetic dataset shaped like the real one (monkeypatched to small search
  grids so it runs in seconds, not minutes).
- `python -m mpp.models.train_lap_time` run against the real, configured
  scope (2023, rounds 1-6): all 8 candidates evaluated via 6-fold
  leave-one-race-out nested CV, comparison report written, Ridge (the
  winner) serialized. See Results above.
- Leakage checked directly, not just asserted: every outer and inner fold is
  produced by `mpp.splits.group_split.race_group_kfold`, whose own tests
  (`tests/splits/test_group_split.py`) already prove no group crosses a
  fold boundary; `mpp.models.evaluate.cross_validate` is additionally tested
  to call its estimator factory fresh once per fold (`test_evaluate.py`),
  so no fold's preprocessing statistics can leak from another fold; and this
  phase's own inspection (`stint_prior_mean_lap_time_s` correctly reflecting
  each held-out race's own pace band, `air_temp`'s importance flagged as a
  confound) is exactly the kind of manual review that catches a *subtler*
  leak than a group boundary check alone would - it found none, but the
  weather-as-race-proxy pattern was worth surfacing regardless.

## Known limitations

- **No candidate model is close to "race-ready."** Ridge's ~6s MAE is a
  meaningful step past the ~9s baseline, not a usable lap-time predictor for
  strategy decisions. This is the headline honest finding of this phase, not
  a caveat to bury.
- **The strong per-circuit-median baseline can't do its job yet** - it needs
  at least one repeated circuit in the ingested scope to differentiate from
  the global median.
- **Tree-based models need more, more-varied races to be worth their
  complexity here** - their inability to extrapolate is actively penalized
  by a 6-race, zero-repeat scope; this is expected to improve, likely
  substantially, once the season scope widens (config-only change per
  `config/settings.toml`).
- **`air_temp`/`track_temp` are confounded with race identity** at this
  scope size and should not be read as evidence of a causal
  temperature-to-lap-time effect (architecture SS13's non-causal framing
  applies directly here).
- Consistent with `docs/03_dataset_definitions.md`, `tyre_degradation`,
  `race_pace`, and `pit_stop` were not modeled in this phase - the brief's
  instruction not to build multiple prediction problems ahead of the data
  supporting them.

## Next: Phase 5+

Per architecture SS17: tyre degradation modeling reusing this phase's
infrastructure with a different target and per-compound slicing; race pace
modeling with the chronological split already implemented in
`mpp.splits.temporal_split`; permutation importance and partial dependence
(SS13) and residual/error-analysis slicing by circuit/driver/compound
(SS14) for the lap-time model built here; and, as the single highest-value
next step suggested by this phase's own findings, widening the season scope
to include at least one repeated circuit so the strong baseline and the
circuit-aware features Phase 3 already built
(`driver_circuit_prior_relative_pace_s`) can be evaluated as designed.
