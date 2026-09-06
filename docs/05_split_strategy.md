# Split Strategy

Comparing three candidate strategies against the actual structure of this
data, then stating which one each model-ready dataset uses.

## The three candidates

**Random split** (e.g. `sklearn.model_selection.train_test_split` with
`shuffle=True`). Assumes rows are independent and identically distributed.
**Wrong for the lap-level datasets** (`lap_time`, `tyre_degradation`): laps
within a stint are near-duplicates of each other - same driver, same tyre,
adjacent lap numbers, the same race's track/weather/incident context. A
random split puts some of a stint's laps in train and others in test, so a
model can effectively memorize that specific stint/race rather than learn a
pattern that generalizes - the reported test error looks better than the
model's real predictive skill. `tests/test_leakage_review.py::test_random_split_leaks_races_across_train_and_test`
demonstrates this concretely: on a 4-race, 20-laps-each synthetic dataset, a
random 75/25 split puts laps from the *same* race on both sides of the split
every time it was checked.

**Grouped split** (`sklearn.model_selection.GroupKFold` / `GroupShuffleSplit`,
grouped by `(season, round)` or by `circuit_id`). Holds an entire group out
together, so no row in test shares a group with any row in train. This is
the correct fix for the lap-level datasets' correlation problem, and it comes
in two flavours used for two different questions:
- grouped by **race** (`(season, round)`): "how well does the model predict
  a race it has never seen any part of" - the primary evaluation split.
- grouped by **circuit**: "how well does the model generalize to a circuit
  it has never seen at all" - a harder secondary check, since a
  race-only group could still let the model see the same circuit in an
  earlier round.

**Temporal split** (chronological holdout, or `sklearn.model_selection.TimeSeriesSplit`,
ordered by `round`). Train on earlier rounds, test on later ones. Matches
a genuinely different question from the two above: not "is this an unseen
race/circuit" but "would this have been predictable *before it happened*,
using only information known at the time." Grouping by race would still let
the model train on round 6 to predict round 2, which is not a scenario that
occurs in real deployment and can flatter a model with hindsight (car
development, driver form trends) a real forecast would not have.

## Which dataset uses which, and why

| Dataset | Strategy | Reasoning |
|---|---|---|
| `lap_time` | Grouped, by `(season, round)` (`mpp.splits.group_split.race_group_shuffle_split` / `race_group_kfold`); circuit-grouped (`circuit_group_shuffle_split`) as a secondary cross-circuit-generalization check | Rows are individual laps, heavily correlated within a stint/race - random splitting would leak. There is no reason to forbid training on a later race to predict an earlier one's *lap time* specifically (unlike race pace, a lap time model is not claiming to forecast future form), so grouping (not temporal ordering) is the binding constraint. |
| `tyre_degradation` | Same as `lap_time` - grouped by `(season, round)`, circuit-grouped as a secondary check | Same stint/race correlation problem, same reasoning. |
| `race_pace` | Temporal - `mpp.splits.temporal_split.chronological_holdout_split` / `time_series_splits`, ordered by `round` | Each row is already one driver-round (no within-group row correlation to guard against), but the prediction question is explicitly forward-looking ("expected pace *before* the next race, given history so far") - a grouped or random split would ignore that direction and let the model use future rounds' realized data to predict a past one, which architecture SS8 identifies as unrealistic for this specific problem. |
| `pit_stop` | Grouped, by `(season, round)` (`race_group_shuffle_split` / `race_group_kfold`) | Multiple pit stops from the same race share that race's pit-lane conditions, safety-car timing, and strategy context - the same correlation argument as the lap-level datasets, at a coarser grain. |

## Hyperparameter tuning

Per architecture SS8: tuning uses a **nested** split - an outer grouped/
temporal split (as above) held out purely for the final reported metric, and
an inner grouped/temporal split (over the outer split's training portion
only) for choosing hyperparameters. This is not yet wired into a tuning loop
in Phase 3 (that is Phase 4's job - architecture SS17); the split functions
in `mpp.splits` are already fold-generator-shaped (`race_group_kfold`,
`time_series_splits` both yield `(train_idx, test_idx)` pairs) specifically
so a Phase 4 `GridSearchCV`/`RandomizedSearchCV` `cv=` argument can consume
them directly, nested inside an outer loop over the same functions.

## Implementation notes

- All split functions return **positional** indices (use with `.iloc`), the
  same convention scikit-learn's own splitters use, so `mpp.splits` results
  compose directly with `sklearn.model_selection` tooling (e.g. passing a
  `race_group_kfold` fold list as `cv=` to `cross_val_score`).
- `race_group_kfold` caps `n_splits` at the number of distinct races actually
  present, since `GroupKFold` cannot create more folds than there are groups
  - relevant given the current narrow 6-round scope.
- Every split function is unit tested for the property that actually matters
  (no shared group across train/test for the grouped splits; no later round
  in train than in test for the temporal splits) - see `tests/splits/`.
