# Performance Prediction Architecture Report

Status: **Phase 0 — architecture only. No pipeline code has been implemented yet.**

This document defines the engineering architecture for `motorsport-performance-prediction`
before any model is trained. It is the third project in a three-part motorsport
analytics portfolio, alongside `motorsport-data-analytics-bi` (descriptive: warehouse
+ KPIs + dashboard) and `motorsport-statistical-analysis` (inferential: hypothesis
testing on F1 timing data). This project is the **predictive** leg: given
engineering-meaningful features, predict lap time, tyre degradation, race pace, and
(where the data supports it) pit stop duration — with the validation discipline to
make the numbers trustworthy rather than merely accurate-looking.

---

## 1. Project objective

Build a machine-learning pipeline that answers concrete race-engineering questions
from real Formula 1 timing data — not a demonstration of algorithms for their own
sake. Every model must:

- answer a question a race engineer, strategist, or performance analyst would
  actually ask ("how much pace does this compound lose per lap?", "what lap time
  should we expect from this stint?"),
- beat a clearly stated, honest baseline,
- be validated in a way that respects the structure of motorsport data (grouped by
  race/driver/circuit, not i.i.d.), and
- be explained in engineering terms, including where it is wrong and why.

The deliverable is the full chain — raw data to engineering interpretation — not
just a trained model object.

## 2. Target roles

| Role | What the project demonstrates for that role |
|---|---|
| Data Analyst | Data validation, cleaning, EDA, metric definition, and communicating model performance in plain terms (MAE in seconds, not just R²). |
| Motorsport Data Analyst | Domain-correct handling of stints, compounds, pit stops, and session structure; features that mirror how race engineers actually reason about a session. |
| Performance Analyst | Race-pace and degradation quantification — turning lap-by-lap noise into a defensible pace/degradation number per driver/stint. |
| Motorsport Performance Engineer | Engineering interpretation of feature effects (tyre age, compound, track) with explicit caveats against over-claiming causality; degradation curves that are physically sane. |
| Junior Data Scientist | The complete lifecycle: baseline → CV → tuning → evaluation → error analysis → interpretability, with leakage prevention treated as a first-class concern. |
| Automotive Data Scientist | The methodology (grouped validation on time/entity-correlated tabular data, baseline-relative evaluation) transfers directly to fleet, durability, or ADAS datasets outside motorsport. |

## 3. Prediction problems

Five candidate problems, in priority order. Lap time is the anchor: it has the
strongest data support and its infrastructure (features, splits, baselines) is
reused by the others.

1. **Lap-time prediction** (primary). Row = one clean lap. Predict lap time in
   seconds from driver, team, circuit, compound, tyre age, stint, lap number, and
   (where available) weather. This is the foundation problem.
2. **Tyre degradation prediction** (primary). Row = one lap within a stint.
   Predict the lap-time delta attributable to tyre age, isolating it as best as
   possible from fuel burn and track evolution (see §18 for the confound this
   cannot fully resolve).
3. **Race pace prediction** (secondary, derived from #1). Row = one
   driver-session. Predict representative race pace (a robust central tendency of
   clean racing laps) from pre-session/historical features — usable before a
   session as an expectation, and checkable against what happened.
4. **Pit stop time prediction** (conditional). Row = one pit stop. Only pursued if
   the ingested data source exposes stationary/pit-lane time at sufficient
   granularity and volume; otherwise explicitly deferred (§20) rather than forced.
5. **Performance classification** (exploratory, optional). Bucketing observations
   (competitive / average / anomalous) is treated as a **derived, post-hoc view**
   of the regression residuals/pace distribribution — not an independently
   labeled target — because the source data has no ground-truth class label.
   Not pursued if regression already answers the question better, per the
   project brief.

## 4. Data source strategy

Consistent with the sibling projects, and non-negotiable: **real data only, nothing
invented or simulated.**

- **Primary: FastF1** — session, lap, stint, and tyre-compound data with full
  timing granularity, 2018 onward. Source of lap times, sector times, stint/compound
  history, and (session-dependent) weather channels (track temp, air temp, rainfall
  flag, wind).
- **Secondary: Jolpica-F1** (Ergast-compatible REST API) — results, qualifying,
  and pit stop records, with much longer historical coverage (1950 onward) but
  coarser granularity. Used to fill in pit-stop duration and season/result context
  FastF1 doesn't carry, and as a cross-check on driver/team/round identifiers.
- **Scope**: start with 1–2 recent seasons (mirroring the sibling projects'
  approach of starting narrow and widening later) to keep iteration fast and make
  manual error inspection (§14) tractable. Season/round scope is a one-line config
  change, not a redesign.
- **Caching**: raw pulls are cached to `data/raw/` (git-ignored) with a manifest,
  so the pipeline is re-runnable without re-hitting the API and every derived
  artifact is reproducible from a recorded raw snapshot.
- **Weather caveat**: not every session/circuit exposes reliable weather
  telemetry. Weather features are treated as optional/imputable, never as a
  required input — a model that only works when weather is present is not
  deployable across the full dataset.
- **Licensing**: both sources are public and used non-commercially, matching the
  usage terms already documented in the sibling repos.

## 5. Data model

Core entities and their relationships:

```
Season ─┬─ Event (round, circuit) ─┬─ Session (FP1/FP2/FP3/Q/R) ─┬─ Lap
        │                          │                              └─ (driver, lap_number)
        │                          └─ Circuit (static reference: length, corner count,
        │                                       pit-lane length, elevation profile,
        │                                       abrasiveness/tyre-stress category)
        └─ DriverSeasonEntry (driver × team × season, handles mid-season team changes)

Lap ──belongs to── Stint (driver × session × compound × stint_number)
Lap ──belongs to── Session
PitStop ──references── (driver, session, lap_number_in, lap_number_out)
WeatherSnapshot ──references── (session, timestamp) — joined to laps by nearest timestamp
```

Key design choices:

- **Circuit is a first-class, mostly-static reference entity.** Its attributes are
  known before a session happens, so they are safe features (no leakage) and are
  exactly what makes cross-circuit generalization testable (§8).
  I use "Circuit" here as a fixed reference table, not `Sector`; the same laps that
  populate the model also validate that no leakage crept in through it.
- **DriverSeasonEntry**, not a bare `driver` column, so a driver's team is
  correctly scoped to the season/point-in-time being modeled (mid-season driver
  changes are real and must not silently corrupt team-based features).
- **Stint is the unit that owns tyre age and compound**, not the lap — a lap's
  tyre age is derived from its position within its stint, not stored redundantly.

## 6. Feature engineering strategy

Grouped by category, with explicit leakage exclusions.

**Identity / context**
- driver, team (via `DriverSeasonEntry`), circuit, session type.
- High-cardinality categoricals (driver, circuit) are handled with either
  native categorical support (HistGradientBoosting) or careful target encoding
  computed **within CV folds only** — never fit on the full dataset before
  splitting.

**Tyre**
- compound, tyre age (laps), tyre age² (captures degradation nonlinearity),
  stint number, fresh-tyre flag.

**Race context**
- lap number, fraction of race distance completed (fuel-load proxy in the
  absence of actual fuel-mass telemetry), session type.

**Track characteristics** (static, pre-known — safe)
- circuit length, corner count, pit-lane length, categorical tyre-stress rating.

**Weather** (optional, imputed when missing)
- track temperature, air temperature, rainfall flag.

**Historical performance** (computed strictly from *past* sessions relative to
the row being predicted — ordering enforced by round/session timestamp, never
by a random shuffle)
- driver's rolling pace percentile over prior N races,
- driver-circuit historical average lap time excluding the *current* event.

**Explicitly excluded (leakage risk)**
- final race/session classification or points (outcome, not a predictor),
- any other driver's lap time in the *same* lap (encodes race outcome, not
  car/tyre/track state),
- the classification label used in problem 5 (it is derived *from* the
  regression output, so it cannot also feed the regression),
- any feature computed using a window that includes the target lap or a future
  lap in the same stint.

## 7. Target definitions

| Problem | Target | Definition |
|---|---|---|
| Lap time | `lap_time_s` | Recorded lap time in seconds, restricted to clean racing laps (in/out laps and safety-car/VSC laps excluded from the primary target, retained separately as a flagged subset for error analysis). |
| Tyre degradation | `lap_time_delta_s` | Per-lap lap time minus that stint's first clean-lap time (a per-lap regression target). A per-stint linear slope (s/lap) is also computed as a summary statistic for interpretation and error analysis, but is not itself the row-level ML target. |
| Race pace | `session_pace_s` | A robust central tendency (median) of a driver's clean racing laps in a session — chosen over the mean because it is far less sensitive to outlier laps (traffic, minor incidents) that don't reflect true pace. |
| Pit stop time | `pit_stop_duration_s` | Recorded stationary time from the pit-stop record, if the ingested source provides it at sufficient volume/granularity (§3.4, §20). |
| Performance classification | `pace_bucket` | Derived post-hoc from residuals or pace-percentile thresholds (e.g. competitive / average / anomalous); not an independently observed label. |

## 8. Train / validation / test strategy

Plain random row splitting is invalid here: laps within a stint are highly
autocorrelated, and rows from the same race/driver leak information about each
other. The split strategy is chosen per problem, not applied uniformly:

- **Lap time & tyre degradation**: `GroupKFold` / `GroupShuffleSplit` grouped by
  **(season, round)** — an entire race is held out together. This tests
  generalization to unseen races, not memorization of a specific session's
  conditions.
- **Cross-circuit generalization check** (secondary evaluation): an additional
  `GroupShuffleSplit` grouped by **circuit**, holding out whole circuits never
  seen in training. This is a harder, more honest test of whether track-characteristic
  features actually generalize, versus the model just having memorized each
  circuit's typical pace.
- **Race pace**: chronological holdout — train on earlier rounds, evaluate on
  the most recent held-out rounds — because this mimics the actual deployment
  question ("what pace do we expect *next* race, given history so far").
  `TimeSeriesSplit` on round order is used for cross-validation during
  development.
- **Hyperparameter tuning**: nested — an outer `GroupKFold`/`GroupShuffleSplit`
  for the reported evaluation metric, an inner grouped CV for tuning, so the
  same held-out group is never used both to tune and to score.

## 9. Leakage prevention

Treated as a first-class, tested concern (§15), not an afterthought:

- **Driver leakage**: rolling/historical driver features must be computed using
  only rounds strictly before the target row's round; assembling them requires
  round order, not a random shuffle.
- **Circuit leakage**: static circuit attributes (known in advance) are safe;
  circuit-level *historical average* features must exclude the current event's
  own laps from their own computation window.
- **Repeated laps from the same stint**: never split laps randomly across
  train/test — near-identical consecutive laps from one stint would leak across
  the split and inflate apparent accuracy. Splitting is always at the
  race/stint-group level (§8).
- **Future information**: any feature using "session so far" aggregates must be
  computed with a strict `< current_lap` (or `< current_round`) boundary,
  verified by an explicit unit test, not just code review.
- **Target leakage via derived features**: finishing position, points, and
  classification results are never inputs to lap-time/degradation/pace models —
  they are outcomes of pace, not causes of it.
- **Derived-target leakage**: the classification bucket (§7) is computed from
  the same regression's residuals, so it is never evaluated as if it were an
  independent ground truth.

## 10. Baseline models

Every problem gets a baseline before any learned model is trained, and the
learned model must beat the **strongest** reasonable baseline, not a straw one:

| Problem | Weak baseline | Strong baseline (the one to beat) |
|---|---|---|
| Lap time | global mean lap time | per-circuit median lap time |
| Tyre degradation | zero-degradation assumption (`delta = 0`) | per-compound linear fit of delta vs. tyre age |
| Race pace | global mean pace | driver's own rolling historical average pace |
| Pit stop | global median pit stop time | per-team median pit stop time |

A model that only beats the weak baseline is not reported as a success.

## 11. Candidate ML models

Chosen for fit to the problem, not for coverage of "every algorithm":

- **Ridge regression** — the first learned-model step above baseline. Coefficients
  are directly inspectable, which matters for an engineering audience; regularization
  handles the collinearity from one-hot circuit/driver dummies.
- **Lasso** — used only where feature selection itself is the useful output (e.g.
  confirming which historical-performance features actually carry signal amid many
  near-duplicate rolling-window variants), not applied by default.
- **Random Forest** — captures the tyre-age × compound × circuit interactions that
  a linear model needs manual interaction terms for, is robust to the outlier laps
  that inevitably remain, and gives permutation importance for free.
- **HistGradientBoostingRegressor** — the primary "best accuracy" candidate:
  native missing-value handling (relevant for patchy weather data), native
  categorical support (driver/circuit without manual encoding), and strong
  accuracy/speed on tabular data of this size.
- **Explicitly not used**: deep learning / neural nets. This is moderate-sized
  structured tabular data; a neural net would add training and interpretability
  cost without a credible accuracy gain, which conflicts directly with the
  project's interpretability goals.

## 12. Evaluation metrics

Reported together, each with an explicit engineering reading:

- **MAE** — "predictions are off by X seconds on average." The primary
  number to quote to a non-ML audience.
- **RMSE** — penalizes large misses more heavily; a gap between RMSE and MAE
  flags that a few laps (incidents, SC restarts) are dominating the error.
- **R²** — variance explained. Reported **both overall and within-circuit**,
  because on a grouped/cross-circuit split, a naive overall R² can look
  artificially strong purely from explaining circuit-to-circuit differences —
  within-circuit R² is the honest measure of whether the model captures
  in-session variation.
- **Median Absolute Error** — robust companion to MAE, unaffected by the
  handful of incident-affected laps that any real dataset contains.
- **Classification variant (if pursued)**: balanced accuracy / macro-F1, given
  the expected class imbalance (most laps are "average").

## 13. Interpretability strategy

- **Permutation importance** (preferred over impurity-based importance),
  computed on the held-out grouped split — impurity-based importance is known to
  bias toward high-cardinality features like driver/circuit, which would be
  misleading here.
- **Partial dependence** for tyre age per compound — the direct visualization of
  the degradation curve shape, which is the engineering-relevant output of
  problem 2.
- **Residual analysis** (§14) as an interpretability tool, not just a QA step.
- **Explicit non-causal framing**: feature importance shows *predictive
  association*, not a causal wear mechanism. Tyre-age importance is confounded
  with race-distance/fuel-load effects that correlate with it — the report will
  say so rather than claim "tyre wear costs N seconds/lap" as an isolated causal
  fact.

## 14. Error analysis strategy

Residuals are sliced, not just aggregated:

- **By circuit** — identifies tracks the model systematically underfits (e.g.
  street circuits with higher lap-time variance).
- **By driver** — a systematic bias for one driver suggests either an unmodeled
  driver effect or a data-quality issue for that driver's laps, and the two are
  told apart by inspection, not assumed.
- **By compound** — degradation-model accuracy is expected to differ by
  compound (softer compounds have less linear degradation).
- **By session/race** — wet or safety-car-heavy sessions are expected to carry
  higher error; this is flagged and explained, not hidden by averaging it away.
- **Residual-vs-predicted plot** — checks for heteroscedasticity (e.g. larger
  errors at high-degradation, late-stint laps).
- **Manual inspection of the largest-N absolute errors** — to separate genuine
  model failure from unmodelable events (red flags, mechanical issues, crashes)
  that no feature set could have predicted.

## 15. Testing strategy

`pytest`, run fully offline (no network calls in the test suite), following the
sibling projects' pattern of testing against a small committed real-data slice
plus synthetic fixtures for edge cases:

- **Data validation rules** — schema/range/referential checks on raw and cleaned
  data.
- **Feature engineering functions** — pure, deterministic functions tested with
  known small inputs (e.g. tyre age must be non-negative and monotonically
  non-decreasing within a stint).
- **Leakage guards** — explicit tests asserting that no group (race/circuit) ID
  appears in both a fold's train and test sets, and that a historical feature for
  round *N* never reflects data from round ≥ *N*.
- **Baseline correctness** — baselines computed against a small hand-checked
  example with a known expected output.
- **Metric computation** — MAE/RMSE/R²/median-AE checked against hand-computed
  values on a toy array.
- **Split strategy correctness** — group-integrity assertions for each split
  function.
- **End-to-end integration test** — the full pipeline run on a tiny fixture
  dataset, checked for shape/type correctness of every stage's output, not
  numerical accuracy.

## 16. Repository structure

```
motorsport-performance-prediction/
├── README.md
├── LICENSE
├── pyproject.toml
├── config/
│   └── settings.toml            # season/round scope, model + split hyperparameters
├── data/                        # git-ignored, fully regenerable
│   ├── raw/                     # cached FastF1 / Jolpica-F1 pulls + manifest
│   ├── interim/                 # typed, cleaned tables
│   └── processed/                # feature matrices per problem
├── docs/
│   ├── ARCHITECTURE_REPORT.md   # this document
│   ├── 01_data_sources.md
│   ├── 02_data_model.md
│   ├── 03_feature_definitions.md
│   ├── 04_target_definitions.md
│   ├── 05_validation_and_leakage.md
│   ├── 06_model_selection.md
│   ├── 07_metrics.md
│   └── 08_limitations.md
├── notebooks/                   # exploratory / model-analysis only, not pipeline logic
├── src/
│   └── mpp/                     # "motorsport performance prediction"
│       ├── ingest/              # FastF1 / Jolpica-F1 clients, raw caching
│       ├── validation/          # data-quality rules + leakage checks
│       ├── features/            # per-category feature builders (§6)
│       ├── splits/              # GroupKFold / GroupShuffleSplit / TimeSeriesSplit wrappers
│       ├── models/              # baselines.py, regressors.py, tuning.py
│       ├── evaluation/          # metrics.py, error_analysis.py
│       ├── interpret/           # importance.py, pdp.py
│       └── pipeline.py          # stage orchestrator
└── tests/                       # mirrors src/mpp structure
```

## 17. Implementation phases

- **Phase 0 (this report)** — architecture, no code. *Complete.*
- **Phase 1** — ingestion, raw caching, data validation rules.
- **Phase 2** — cleaning, feature engineering, EDA.
- **Phase 3** — split strategy + leakage tests + baselines (infrastructure that
  every subsequent problem reuses).
- **Phase 4** — lap-time model: training, grouped CV, tuning, evaluation,
  interpretability, error analysis. This is the anchor problem; its
  infrastructure is reused, not rebuilt, for Phases 5–6.
- **Phase 5** — tyre degradation model (reuses Phase 4 infrastructure with a
  different target and per-compound slicing).
- **Phase 6** — race pace model (reuses Phase 4 features, chronological split).
- **Phase 7** — pit stop model, **gated on a feasibility check** of the ingested
  data's pit-stop granularity/volume (§3, §20).
- **Phase 8** — performance classification, derived from Phase 4/5 residuals,
  pursued only if it adds a genuinely new engineering read beyond the regression.
- **Phase 9** — documentation polish, final consolidated report.

## 18. Risks

- **Weather data gaps** — not every session exposes reliable weather telemetry;
  handled by optional/imputed features (§4, §6), not by dropping affected
  sessions.
- **Small effective group count** — grouped-by-race/circuit splitting leaves
  only ~20–24 race-groups or ~20 circuit-groups per season, which limits how
  many folds are statistically meaningful; addressed by widening season scope
  before over-trusting a single split's numbers.
- **Fuel-load / tyre-wear confound** — lap time degrades with tyre age *and*
  with reducing fuel load simultaneously; without fuel-mass telemetry these
  cannot be fully separated (the sibling statistical-analysis project's own
  finding was that raw degradation slope is dominated by fuel burn, not tyre
  wear). This project will state that confound explicitly rather than claim an
  isolated tyre-wear effect.
- **Pit-stop data granularity** varies by season/source and may not support a
  reliable per-driver-per-stop regression; this risk is why Phase 7 is gated.
- **Class imbalance** in the optional classification variant (most laps are
  "average").
- **Identifier mismatches** between FastF1 and Jolpica-F1 (driver/team/round
  naming) requiring an explicit mapping/reconciliation layer.
- **Scope creep** — five prediction problems fully polished is a large scope;
  the phase ordering (§17) exists specifically to keep lap time as a working
  end-to-end deliverable before any other problem is started.

## 19. Assumptions

- Real, session-level FastF1/Jolpica-F1 data only — no invented or simulated
  rows, consistent with the sibling projects' stated policy.
- Initial scope is a small number of recent seasons; widening scope is a
  configuration change, not an architecture change.
- This repository is **self-contained**: it does not hard-depend on the
  `motorsport-data-analytics-bi` warehouse. Naming/schema conventions are kept
  compatible with the sibling projects by choice, not by coupling, so that
  reusing an existing cleaned dataset later remains possible without being
  required now.
- Units: seconds for time, kilometers for distance, Celsius for temperature.
- Single racing series (Formula 1) — no attempt at cross-series (F2/IndyCar)
  generalization in this phase.

## 20. Deferred features

Explicitly out of scope for now, to keep the initial delivery focused:

- Full telemetry-based degradation modeling (throttle/brake/speed traces).
- Real-time or live in-session prediction serving.
- Deep learning / sequence models.
- Cross-series generalization beyond Formula 1.
- Causal isolation of tyre-wear vs. fuel-load effects (would need fuel-mass
  telemetry not available from the chosen sources).
- Pit-stop crew-level (per-mechanic/per-wheel) breakdown.
- Strategy simulation or optimization (e.g. "optimal pit window").
- A web dashboard or app layer (a natural companion project, analogous to the
  existing telemetry-app repo, but not part of this repository's scope).
- Fixed-effects driver-skill isolation.

## 21. Expected portfolio value

This project demonstrates the discipline that distinguishes an engineering-grade
ML pipeline from a typical portfolio regression notebook: grouped, leakage-aware
validation; baseline-relative (not absolute) accuracy claims; and interpretation
that stops at "predictive association" rather than overclaiming causal insight.
Alongside the sibling BI project (descriptive) and statistical-analysis project
(inferential), it completes a three-part narrative — describe, explain, predict —
that reads coherently to an interviewer across the Data Analyst → Junior Data
Scientist range of target roles, while the domain-specific handling of stints,
compounds, and pit stops speaks directly to the motorsport-specific roles.

## 22. Final readiness assessment

The repository is new and empty, with an empty remote — there is no existing
code or history to reconcile. The architecture above is ready to move into
Phase 1 (ingestion + validation). Recommended next step: implement the
lap-time-prediction path end to end (Phases 1–4) before starting any other
problem, since it establishes the shared ingestion/feature/split/baseline
infrastructure the rest of the project depends on. Pit-stop prediction (Phase 7)
should not be committed to until a short feasibility check confirms the ingested
data actually supports it at useful granularity.
