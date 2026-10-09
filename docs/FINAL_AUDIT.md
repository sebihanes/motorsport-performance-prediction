# MOTORSPORT PERFORMANCE PREDICTION — FINAL AUDIT

Audit date: 2026-10-09.

Scope:
- Every module under `src/mpp/`, all tests, every document.
- A full from-scratch run of the supported pipeline:
  `python -m mpp.pipeline`, then all four trainers, then
  `python -m mpp.evaluation.analyze_lap_time`.
- The generated datasets, metrics, metadata and reports in `data/`.

Every number below was produced by that run, or by an audit check that is
reproducible from the code in this repository.

---

## 1. Executive summary

This is a carefully engineered, honestly documented ML pipeline built on
real F1 timing data. Its strongest assets are methodological:
- leakage-safe feature construction, with tests that prove it;
- nested leave-one-race-out cross-validation;
- baseline-first evaluation;
- out-of-fold interpretability and error analysis that traces failures to
  recorded race events.

**The audit found two critical integrity problems, and both are now fixed.**

1. **The serialized lap-time model lost to an obvious baseline.**
   - The "strong" baseline was the per-circuit median. With no circuit
     repeating in scope, it collapses to the global median (8.96s MAE).
   - The baseline a race engineer would actually use is **stint pace
     persistence**: "next lap ≈ this stint's mean so far". It scores
     **3.67s**. Ridge, which was serialized as the winner, scores
     **5.95s**.
   - The project's own architecture rule (§10: beat the *strongest*
     reasonable baseline) was therefore violated.
2. **The serialized tyre-degradation model lost to the baseline the
   architecture specified.**
   - §10 specifies a per-compound linear fit, but it was never
     implemented.
   - Implemented now, it scores **7.64s** against the Random Forest's
     **7.77s**.

What the fix changed:
- Both baselines are now implemented and tested.
- The trainers gate on the strongest baseline.
- Stale artifacts are removed, and every document that claimed a "win"
  carries a correction.

**The honest headline is now: in this 6-race, no-repeated-circuit scope,
no learned model beats the strongest reasonable baseline on any of the
four problems, and nothing is serialized.** This is a credible, defensible
result, *if presented as such*. The project's analysis explains exactly
why.

**Overall score: 73 / 100.**

**Verdict: portfolio-ready as a methodology and engineering-judgement
piece.** It is not a "my model predicts F1 lap times" piece, and must not
be presented as one.

---

## Scores (0-10)

| # | Area | Score | Basis |
|---|---|---|---|
| 1 | Architecture | 8 | Clean staged package (ingest → validate → clean → features → datasets → models → evaluation); config-driven scope; problems share infrastructure (`train_common`). Minor duplication between `train_lap_time` and `train_common`. |
| 2 | Data ingestion | 8 | FastF1 + Jolpica-F1, cached raw pulls, provenance `manifest.json`, real-data test fixtures. Depends on external APIs and the raw data is not committed (by design). |
| 3 | Data quality | 7 | Validation report, cleaning log, explicit exclusion reasons (315 pit in/out laps, 157 missing times), nothing silently dropped. But safety-car/VSC/yellow laps (8.1%) remain "valid", and pit-stop durations include red-flag time. |
| 4 | Feature engineering | 6 | Every historical feature is strictly prior and documented with availability timing. But: no track-status feature; the stint pace mean includes neutralised laps; temperature acts as a race-identity proxy that *hurts* (negative permutation importance); the circuit history feature is 100% null in this scope. |
| 5 | Leakage prevention | 8 | Strong and tested (prior-only features, in-pipeline preprocessing, grouped splits, a random-split leak demonstration). Residual minor issues: M1-M3 below, each checked and immaterial today. |
| 6 | Train/validation/test methodology | 8 | Nested leave-one-race-out for lap-level problems, forward-chaining for race pace; the final model is refit on all data after evaluation. Only 6 groups, so the variance across folds is large. |
| 7 | Baseline quality | 7 | Now includes the strongest reasonable baselines (persistence; per-compound linear trend), with gating on the best of them. Scored below 8 because both were missing until this audit and their absence produced false "wins". |
| 8 | Model selection | 6 | Sensible, justified candidates (Ridge, tree ensembles, HistGB for NaN handling). But predicting absolute lap time instead of the deviation from stint pace is the wrong target framing for this scope, and no candidate wins. |
| 9 | Cross-validation | 8 | Nested, grouped, deterministic. A Phase 5 bug where the temporal split leaked rounds was found and fixed with a regression test. |
| 10 | Hyperparameter tuning | 7 | Inner race-grouped CV per outer fold; small, sensible grids. RandomizedSearch with n_iter=8 is light for the tree models. |
| 11 | Evaluation metrics | 7 | MAE/RMSE/R²/MedAE with distinct stated purposes; fold mean ± std. Mean-of-within-race R² (often negative) is reported without enough explanation in the earlier phases. |
| 12 | Model interpretability | 8 | Out-of-fold, grouped and within-race permutation importance; coefficient stability across folds; partial dependence with consistent derived features; causation caveats throughout. |
| 13 | Error analysis | 9 | Exact between/within-race MSE decomposition; bias tables with direction labels; failure cases annotated only from recorded evidence, with a sign check; confounds caught and reported. |
| 14 | Motorsport domain realism | 6 | Compound, tyre age, stint, driver/team form, weather and race progress are all modelled; neutralisations, the fuel confound and red flags are analysed. But no fuel model, traffic, track status, circuit characteristics, or qualifying/practice sessions. |
| 15 | Python engineering | 7 | Typed, documented, small focused modules, sklearn-native estimators. No CI, linter or type-checker config, no lockfile; some long modules (`analyze_lap_time.py`). |
| 16 | Testing | 8 | 221 tests (220 pass, 1 skips by design: there is no lap-time model to serialize): unit, real-data fixture integration, explicit leakage tests, synthetic end-to-end training/analysis. No CI; 30 known sklearn warnings. |
| 17 | Reproducibility | 7 | Deterministic seeds; the full run reproduces every documented metric exactly (Ridge 5.953s, persistence 3.669s); model metadata records library versions and the dataset SHA-256. Dependencies are unpinned (`>=`) and the raw data must be re-fetched. |
| 18 | Documentation | 8 | Architecture, data sources, pipeline, dataset definitions, feature dictionary, split strategy, and a report per phase. Earlier reports overclaimed; they are now corrected in place. Verbose for a recruiter skim. |
| 19 | Visualization | 7 | Phase 6 figures are clean and purposeful (importance, residuals, faceted predicted vs actual, grouped bias). No figures for Phases 4-5, no interactive view. |
| 20 | Portfolio quality | 7 | Strong engineering story and honest results; the headline negative result needs framing (see the interview answers). |
| | **Overall** | **73 / 100** | Sum 147/200, scaled. |

### Report scores by heading

| Heading | Score |
|---|---|
| 2. Architecture | 8/10 |
| 3. Data quality | 7/10 |
| 4. Feature engineering | 6/10 |
| 5. Leakage prevention | 8/10 |
| 6. ML methodology (split, CV, tuning, baselines, metrics) | 7.4/10 |
| 7. Model performance | **3/10**: no model beats the strongest baseline on any problem |
| 8. Interpretability | 8/10 |
| 9. Error analysis | 9/10 |
| 10. Motorsport realism | 6/10 |
| 11. Testing | 8/10 |
| 12. Reproducibility | 7/10 |
| 13. Documentation | 8/10 |
| 14. Portfolio readiness | 7/10 |

Model performance (3/10) is scored separately and is not one of the 20
areas. It measures the *results*, not the methodology, and the results
are a verified negative.

---

## Data integrity

| Check | Result |
|---|---|
| Real data only | ✅ FastF1 + Jolpica-F1 for 2023 R1-R6; `manifest.json` records the source and retrieval time. No synthetic data in `src/`. Synthetic data appears only in tests, labelled as such. |
| No fabricated results | ✅ Every metric in the docs was regenerated in this audit and matches: lap time 5.953 / 8.957 / 3.669s; tyre degradation 7.767 / 7.887 / 7.643s; race pace and pit stop unchanged. |
| No hardcoded predictions or metrics | ✅ `grep` for the reported metric values in `src/` finds nothing; every report is generated from computed results. |
| Reproducible preprocessing | ✅ Cleaning is deterministic and logged (`cleaning_log.json`). The rerun reproduced 6,617 rows, 6,145 valid laps and 0 duplicates. |
| Reproducible features | ✅ `laps_features.parquet` is rebuilt from the cleaned tables. The lap-time dataset SHA-256 is recorded in the model metadata. |
| Documented exclusions | ✅ Exclusion reasons are recorded per lap (`pit_in_out_lap` 315, `missing_lap_time` 157). ⚠️ Neutralised laps are *not* excluded. This is documented (Phase 6) but remains an open issue (H2). |

## Leakage audit

| Risk | Finding |
|---|---|
| Target leakage | ✅ `test_no_feature_column_is_a_disguised_copy_of_the_target`. `stint_prior_mean_lap_time_s` is a lagged target (earlier laps only), which is legitimate for a next-lap predictor and tested by `test_stint_prior_pace_trend_is_unaffected_by_a_later_lap_in_the_same_stint`. |
| Future information | ✅ Driver/team/circuit history is strictly prior-round (`add_expanding_prior_stat`, shift-then-expand, tested). ⚠️ M2: `race_fraction` uses `total_laps` = maximum *observed* lap, not scheduled distance. It is identical for all six races here (all ran full distance). ⚠️ M3: weather uses a `nearest`-time join, so a lap can take a sample from up to a few minutes after it ends. Weather is slow-moving, and the effect is negligible. |
| Same-event leakage | ✅ All lap-level splits hold out whole races (`race_group_kfold`); a random split is demonstrated to leak (`test_random_split_leaks_races_across_train_and_test`). |
| Same-stint leakage | ✅ Stints never cross races, so race grouping also holds out stints. |
| Preprocessing leakage | ✅ Imputation, scaling and encoding live inside each sklearn `Pipeline` and are fit per fold. |
| Historical-feature leakage across folds | ⚠️ M1: under leave-one-race-out, a *later* training race's driver/team history includes the held-out race's relative pace. Measured: removing those features changes Ridge's MAE from 5.953s to 5.875s, i.e. it slightly *improves*. The overlap does not inflate any result. |
| Random splitting | ✅ None is used for evaluation. The temporal split's round-boundary bug (Phase 5) was fixed and a regression test added. |

**Verdict: no material leakage.** Three minor issues are documented,
measured or bounded.

## Model quality

| Check | Result |
|---|---|
| Baselines exist | ✅ For every problem; now including the strongest reasonable one. |
| ML beats baselines where expected | ❌ **Nowhere**, against the strongest baseline. Lap time: Ridge 5.95s vs persistence 3.67s. Tyre degradation: RF 7.77s vs linear trend 7.64s. Race pace: RF 10.25s vs per-driver median 9.83s. Pit stop: RF 164.0s vs per-team median 135.9s. The reasons are identified per problem (Phase 4-6 reports). |
| Appropriate metrics | ✅ |
| Realistic validation | ✅ Leave-one-race-out answers "a race the model has never seen" honestly. |
| Overfitting evaluated | ✅ Nested CV separates tuning from scoring. Tree models losing badly out-of-fold (RF 9.25s, HGB 9.95s) is identified as an extrapolation failure across circuit pace levels. |
| Complexity justified | ✅ The project explicitly favours the simplest adequate model. The persistence result reinforces that a zero-parameter model is the right deployment choice in this scope. |

## Motorsport realism

| Factor | Status |
|---|---|
| Tyre compound | ✅ Feature, with compound × tyre-age interactions. Analysis shows compound is confounded with circuit (84% of SOFT laps are Bahrain; all wet laps are Monaco). |
| Tyre age | ✅ Feature (linear and quadratic). Partial dependence shows the fuel-burn confound; the docs say so. |
| Driver / team | ✅ Through prior relative pace (no measurable effect with 1-5 prior rounds); identity is used only for error analysis. |
| Circuit | ⚠️ Only via baselines and error analysis. No circuit features (no real source for static characteristics); history is null in this scope. This is the root cause of most error. |
| Stint | ✅ Stint number, laps into stint, stint pace history. |
| Session | ⚠️ Races only (`session_type` is constant). |
| Race conditions | ⚠️ Weather is a feature; track status (SC/VSC/flags) is analysed but not a feature (H2). |
| Incomplete observations | ✅ Missing lap times excluded with a reason; history features NaN by construction; HistGB evaluated for native NaN handling. |
| Outcome claims | ✅ The project predicts lap time, degradation delta, driver pace and stop duration. It **does not predict race results, and claims nothing about them.** The README now states that the lap-time model is a next-lap predictor, not a pre-race forecast. |

---

## 15. Critical issues

| ID | Issue | Status |
|---|---|---|
| C1 | Lap-time Ridge (5.95s) was serialized and reported as beating its strong baseline, while a zero-parameter stint-pace-persistence baseline scores 3.67s. | **Fixed.** `StintPacePersistenceRegressor` added; `train_lap_time` gates on the strongest baseline, writes metadata recording the missed margin, and removes the stale artifact. Phase 6 analysis compares against persistence per race and per lap phase. Phase 4/6 reports and README corrected. Tests added. |
| C2 | The tyre-degradation strong baseline specified in architecture §10 (per-compound linear fit) was never implemented; the serialized RF (7.77s) loses to it (7.64s). | **Fixed.** `PerGroupLinearTrendRegressor` added; `train_common` accepts extra strong baselines and gates on the best; stale artifact removed; Phase 5 report corrected. The synthetic test signal was given a component the baseline cannot see, so the "model can win" test still exercises serialization. |

## 16. High-priority issues

| ID | Issue | Status |
|---|---|---|
| H1 | No learned model beats its strongest baseline on any problem. Root cause: 6 races at 6 distinct circuits, so nothing about pace *level* transfers. | **Open, deliberately.** Widening the season scope is a config change (`config/settings.toml`), but it re-fetches data and changes every reported result, so it is a new analysis, not an audit fix. It is the #1 recommendation. |
| H2 | Safety-car/VSC/yellow laps are "valid" laps and track status is not a feature. They are 8.1% of laps and 19.4% of lap-time absolute error, and also contaminate the stint pace mean and the tyre-degradation target. | **Open.** `track_status` is already in `laps_features.parquet`, so adding it is a feature change with full re-evaluation; documented in Phase 6 with evidence. |
| H3 | It was ambiguous whether the lap-time model is a pre-race or next-lap predictor (it uses the current stint's completed laps). | **Fixed** in README (next-lap predictor, not a pre-race forecast). |

## 17. Medium-priority issues

| ID | Issue |
|---|---|
| M1 | Leave-one-race-out cross-fold overlap in driver/team history features. Measured as immaterial (removing them improves MAE by 0.08s). |
| M2 | `total_laps` (hence `race_fraction`) is the maximum observed lap, not the scheduled distance; it would leak for a red-flag-shortened race. No effect in the current scope. |
| M3 | The weather join uses `direction="nearest"` (can look minutes ahead); `backward` would be strictly causal. Negligible for slow-moving weather. |
| M4 | First-lap-of-stint imputation uses the training races' median (other circuits): -5.0s bias, 11.0s MAE on those laps. |
| M5 | Pit-stop `duration_s` includes red-flag stoppage time (source limitation; up to 1871s at Albert Park). |
| M6 | No dependency lockfile, no CI, no lint/type-check configuration. |
| M7 | Trainers exit with status 1 when nothing beats the baseline (expected for all four now), so `&&` chaining stops early. Documented in the README. |

## 18. Low-priority issues

| ID | Issue |
|---|---|
| L1 | 30 sklearn warnings in the test suite (median imputer on all-NaN history columns in early folds or synthetic data). |
| L2 | The race-pace trainer emits the same imputer warnings on real data (round 1 has no history). |
| L3 | Long modules (`analyze_lap_time.py` ~770 lines); `train_lap_time` duplicates `train_common`'s nested-CV loop. |
| L4 | Phases 4-5 have no figures; there is no interactive or dashboard view. |
| L5 | Per-fold R² (mean of within-race R²) is often negative and confusing next to a pooled R² of 0.32. |
| L6 | Documentation is long; a recruiter needs the README headline and this audit, not ~2,500 lines. |

## Intentionally deferred (not defects)

- Performance classification (designed as a post-hoc view of residuals; architecture §3).
- `driver_circuit_prior_relative_pace_s`: implemented and tested; activates once a circuit repeats.
- Circuit static characteristics (length, corners): no real data source; inventing them would break the real-data rule.
- Qualifying/practice sessions, fuel modelling, traffic/gap modelling, live inference.

---

## 19. Strengths

1. **Leakage discipline proven by tests, not asserted.** Prior-only
   features, in-pipeline preprocessing, a demonstration that random
   splits leak, and a real temporal-split bug found and fixed.
2. **Nested leave-one-race-out CV.** Every reported number is for a race
   the model never saw during training or tuning.
3. **Baseline-first evaluation that now holds itself to the strongest
   baseline,** including one that beats every model.
4. **Interpretability done properly.** Out-of-fold, grouped, within- vs
   across-race permutation importance; it exposed temperature as a harmful
   race-identity proxy.
5. **Error analysis that connects to race events.**
   - 50.9% of MSE is a per-race offset (exact decomposition).
   - Safety-car laps are underpredicted by 24s.
   - Failure laps are explained only from recorded track status, with a
     sign check, and labelled unexplained when the data cannot explain
     them.
6. **Intellectual honesty throughout.** Negative results are reported,
   confounds are caught (the pooled "stint contamination" effect
   disappears within races), and causation disclaimers are given.

## 20. Weaknesses

1. **No model beats the strongest baseline on any problem.** The
   project's ML adds no predictive value in its current scope.
2. Scope is too narrow (6 races, no repeated circuits, races only).
3. Race conditions (track status) are analysed but not modelled.
4. Absolute lap time is the wrong target for cross-circuit
   generalization; the deviation from stint pace would be better.
5. Engineering hygiene gaps: no CI, no lockfile, no linting.

## 21. Recommended future improvements (in order of value)

1. **Widen scope** to full seasons (e.g. 2022-2024) so circuits repeat.
   This activates circuit history, makes the per-circuit baseline
   meaningful, and removes the race-identity proxy problem.
2. **Add track status.** Either as a feature, or by excluding neutralised
   laps from targets and from `stint_prior_mean_lap_time_s`.
3. **Reframe lap time as the deviation from stint pace** (persistence +
   learned correction) and require it to beat persistence. The audit's
   quick check (Ridge on the persistence residual, 4.24s) shows features
   from fixes 1-2 are needed first.
4. Impute first-lap-of-stint pace from the same race (the field's
   prior-lap median), not other circuits.
5. Pin dependencies, add GitHub Actions running `pytest` and a linter.
6. Use the scheduled race distance for `race_fraction`, and a backward
   weather join.

---

## 22. Final verdict

**Overall: 73 / 100.** A methodologically strong, honest, well-tested
project whose models do not yet beat the strongest simple baseline. It
says so, and explains why with evidence.

### Is this project ready to be shown on a Data Analyst CV?

**Yes.** It demonstrates:
- real-data ingestion, validation and cleaning with an audit trail;
- grouped evaluation and baseline-relative thinking;
- clear error analysis and visualizations;
- written communication of a nuanced result.

Describe it as an "F1 lap-time and tyre-performance analysis and
modelling pipeline". Do not describe it as "an F1 lap-time prediction
model".

### Is this project strong enough for a Motorsport Data Analyst / Performance Engineer interview?

**Yes, as a discussion piece, with one condition: lead with the
methodology and the persistence finding, not with model accuracy.** An
interviewer from a team will immediately ask "what's your baseline?" and
"how do you handle safety cars?". This project now has good answers to
both, backed by numbers.

It is not yet strong enough to claim predictive capability: a
performance engineer would use the persistence baseline over any model
here. That limitation, and the plan to fix it (scope, track status,
target reframing), is itself a strong interview answer.

### 5 strongest technical talking points

1. **Nested leave-one-race-out CV**, and why a random split leaks: there
   is a test that demonstrates it.
2. **Finding a bug in my own temporal split.** `TimeSeriesSplit` on row
   positions let one round straddle train and test; I fixed it to split
   on rounds and added a regression test.
3. **Holding models to the strongest baseline:** discovering that
   zero-parameter pace persistence (3.67s) beats every model, and changing
   the serialization gate instead of the narrative.
4. **Within- vs across-race permutation importance:** temperature has
   *negative* out-of-fold importance, which proves it is a race-identity
   proxy that hurts generalization.
5. **Exact error decomposition:** 50.9% of squared error is a constant
   per-race offset. The model's 0.46-0.77 s/s slope on stint pace shows
   it shrinks toward other circuits' pace, which is exactly why
   persistence (slope 1) wins.

### 5 strongest motorsport/engineering talking points

1. **Neutralisations dominate error.** Safety-car laps are underpredicted
   by 24s; they are 8% of laps but 19% of error. A safety-car lap left in
   a stint's running mean then makes the *next* green lap predicted
   ~30-35s too slow (Jeddah lap 21).
2. **The tyre-age effect is confounded with fuel burn.** Lap time falls
   with tyre age for 15-25 laps. Without fuel-mass data you cannot call
   that degradation, and the analysis says so.
3. **Compound effects are partly circuit effects.** 84% of soft laps are
   Bahrain and every wet lap is Monaco. A per-compound error is not a
   tyre finding.
4. **Pit-stop data quality.** At the 2023 Australian GP, red-flag
   stoppage was recorded as pit-stop duration (up to 1871s). You need
   domain knowledge to see that it is not a 31-minute tyre change.
5. **Ridge beats persistence exactly at the two races with full-course
   neutralisations** (Jeddah, Albert Park) and loses heavily at Baku and
   Monaco. That shows when cross-race pooling helps (corrupted recent
   history) and when it hurts (atypical circuit pace level).

### Limitations to mention honestly in an interview

- Six races, six different circuits, races only: nothing about a
  circuit's pace level can transfer, so **no model beats the
  persistence baseline**.
- The lap-time model is a **next-lap** predictor (it uses the current
  stint's completed laps), not a pre-race forecast.
- Safety-car/VSC laps are analysed but not yet modelled or filtered.
- No fuel, traffic, damage or strategy data, so those causes are never
  attributed.
- Feature importances are predictive associations, not causal effects.
- Small-sample validation (6 folds) gives large fold-to-fold variance:
  per-race MAE ranges from 1.07s (Miami) to 14.51s (Baku).
- Engineering gaps: no CI or dependency lockfile.
