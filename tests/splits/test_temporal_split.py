from __future__ import annotations

import pandas as pd
import pytest

from mpp.splits.temporal_split import chronological_holdout_split, time_series_splits


def _dataset() -> pd.DataFrame:
    # Rows deliberately out of round order, to check the split sorts internally.
    rounds = [3, 1, 2, 1, 3, 2, 4, 4]
    return pd.DataFrame({"round": rounds, "driver": ["VER"] * len(rounds)})


def test_chronological_holdout_puts_the_last_round_in_test_only():
    dataset = _dataset()
    train_idx, test_idx = chronological_holdout_split(dataset, n_test_rounds=1)
    assert set(dataset.iloc[test_idx]["round"]) == {4}
    assert set(dataset.iloc[train_idx]["round"]) == {1, 2, 3}


def test_chronological_holdout_train_never_contains_a_later_round_than_test():
    dataset = _dataset()
    train_idx, test_idx = chronological_holdout_split(dataset, n_test_rounds=2)
    assert dataset.iloc[train_idx]["round"].max() < dataset.iloc[test_idx]["round"].min()


def test_chronological_holdout_rejects_too_many_test_rounds():
    dataset = _dataset()  # 4 distinct rounds
    with pytest.raises(ValueError):
        chronological_holdout_split(dataset, n_test_rounds=4)


def test_time_series_splits_never_puts_a_later_round_in_train_than_in_test():
    dataset = _dataset()
    for train_idx, test_idx in time_series_splits(dataset, n_splits=3):
        assert dataset.iloc[train_idx]["round"].max() <= dataset.iloc[test_idx]["round"].min()


def test_time_series_splits_test_rounds_move_forward_across_folds():
    dataset = _dataset()
    test_round_maxima = [
        dataset.iloc[test_idx]["round"].max() for _, test_idx in time_series_splits(dataset, n_splits=3)
    ]
    assert test_round_maxima == sorted(test_round_maxima)


def test_time_series_splits_never_splits_a_round_across_train_and_test():
    # Multiple rows per round (like race_pace: one row per driver) - a naive
    # row-position split could put some of a round's rows in train and
    # others in test. Regression guard for that leak.
    rounds = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    dataset = pd.DataFrame({"round": rounds, "driver": [f"D{i}" for i in range(len(rounds))]})
    for train_idx, test_idx in time_series_splits(dataset, n_splits=3):
        train_rounds = set(dataset.iloc[train_idx]["round"])
        test_rounds = set(dataset.iloc[test_idx]["round"])
        assert train_rounds.isdisjoint(test_rounds)


def test_time_series_splits_caps_n_splits_to_distinct_rounds_minus_one():
    dataset = _dataset()  # 4 distinct rounds -> at most 3 splits
    folds = list(time_series_splits(dataset, n_splits=100))
    assert len(folds) == 3
