from __future__ import annotations

import pandas as pd

from mpp.splits.group_split import (
    circuit_group_shuffle_split,
    race_group_kfold,
    race_group_shuffle_split,
)


def _dataset() -> pd.DataFrame:
    # 4 races across 3 circuits (bahrain repeats), 10 laps each.
    rows = []
    races = [(1, "bahrain"), (2, "jeddah"), (3, "bahrain"), (4, "monaco")]
    for round_, circuit in races:
        for lap in range(1, 11):
            rows.append({"season": 2023, "round": round_, "circuit_id": circuit, "lap_number": lap})
    return pd.DataFrame(rows)


def test_race_group_shuffle_split_never_splits_a_race_across_train_and_test():
    dataset = _dataset()
    train_idx, test_idx = race_group_shuffle_split(dataset, test_size=0.5, random_state=0)
    train_races = set(zip(dataset.iloc[train_idx]["season"], dataset.iloc[train_idx]["round"]))
    test_races = set(zip(dataset.iloc[test_idx]["season"], dataset.iloc[test_idx]["round"]))
    assert train_races.isdisjoint(test_races)


def test_circuit_group_shuffle_split_never_splits_a_circuit_across_train_and_test():
    dataset = _dataset()
    train_idx, test_idx = circuit_group_shuffle_split(dataset, test_size=0.5, random_state=0)
    train_circuits = set(dataset.iloc[train_idx]["circuit_id"])
    test_circuits = set(dataset.iloc[test_idx]["circuit_id"])
    assert train_circuits.isdisjoint(test_circuits)


def test_circuit_group_shuffle_keeps_both_bahrain_rounds_together():
    # bahrain appears at round 1 and round 3 - a circuit-grouped split must
    # put both entirely in train or entirely in test, never split between them.
    dataset = _dataset()
    train_idx, test_idx = circuit_group_shuffle_split(dataset, test_size=0.5, random_state=0)
    bahrain_rounds_in_train = set(dataset.iloc[train_idx].loc[dataset.iloc[train_idx]["circuit_id"] == "bahrain", "round"])
    bahrain_rounds_in_test = set(dataset.iloc[test_idx].loc[dataset.iloc[test_idx]["circuit_id"] == "bahrain", "round"])
    assert not (bahrain_rounds_in_train and bahrain_rounds_in_test)


def test_race_group_kfold_covers_every_row_exactly_once_as_test():
    dataset = _dataset()
    seen_as_test = []
    for _, test_idx in race_group_kfold(dataset, n_splits=4):
        seen_as_test.extend(test_idx.tolist())
    assert sorted(seen_as_test) == list(range(len(dataset)))


def test_race_group_kfold_caps_n_splits_to_the_number_of_races():
    dataset = _dataset()  # only 4 distinct (season, round) races
    folds = list(race_group_kfold(dataset, n_splits=10))
    assert len(folds) == 4
