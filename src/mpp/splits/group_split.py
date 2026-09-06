"""GroupKFold / GroupShuffleSplit wrappers for the lap-level and pit-stop
datasets (lap_time, tyre_degradation, pit_stop).

Why not a random split: laps within a stint are near-duplicates of each
other (same driver, same tyre, adjacent lap numbers, correlated conditions),
and every lap in a race shares that race's weather/track/incident context.
A random row split puts some of a stint's laps in train and others in test,
so the model can effectively memorize that specific race/stint rather than
learn a pattern that generalizes - producing an evaluation score that looks
good but does not reflect real predictive skill (see
tests/test_leakage_review.py for a concrete before/after demonstration).

Two grouped strategies are provided because they test two different
questions:
- race_group_*: group by (season, round) - an entire race held out together.
  This is the primary strategy (architecture SS8): "how well does the model
  predict a race it has never seen any part of."
- circuit_group_*: group by circuit_id - every round at a given circuit held
  out together. A harder, secondary check: "how well does the model
  generalize to a circuit it has never seen at all," rather than merely a
  session at an already-seen circuit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def race_groups(dataset: pd.DataFrame) -> np.ndarray:
    return (dataset["season"].astype(str) + "_" + dataset["round"].astype(str)).to_numpy()


def circuit_groups(dataset: pd.DataFrame) -> np.ndarray:
    return dataset["circuit_id"].astype(str).to_numpy()


def race_group_shuffle_split(
    dataset: pd.DataFrame, test_size: float = 0.25, random_state: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """One (train_idx, test_idx) split holding out whole races. Indices are
    positional (use with .iloc), relative to dataset's own row order."""
    groups = race_groups(dataset)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    return next(splitter.split(dataset, groups=groups))


def circuit_group_shuffle_split(
    dataset: pd.DataFrame, test_size: float = 0.25, random_state: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """One (train_idx, test_idx) split holding out whole circuits. Positional indices."""
    groups = circuit_groups(dataset)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    return next(splitter.split(dataset, groups=groups))


def race_group_kfold(dataset: pd.DataFrame, n_splits: int = 5):
    """Yield (train_idx, test_idx) pairs across n_splits folds, each fold's test
    set made of whole races never seen in that fold's training data.

    n_splits is capped at the number of distinct races available, since
    GroupKFold cannot make more folds than there are groups.
    """
    groups = race_groups(dataset)
    n_splits = min(n_splits, len(set(groups)))
    splitter = GroupKFold(n_splits=n_splits)
    yield from splitter.split(dataset, groups=groups)
