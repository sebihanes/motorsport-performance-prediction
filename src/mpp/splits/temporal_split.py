"""Chronological / TimeSeriesSplit wrappers for the race_pace dataset.

race_pace's real deployment question is "predict this driver's pace before
the next race, given everything known so far" - the model is meant to be run
forward in time, never backward. A random or grouped split can hand the
model a later round to train on and an earlier one to test on, which is not
a scenario that will ever occur in deployment and would flatter the model
with information (car development, driver form) that a real forecast would
not have. A chronological split - train on earlier rounds, test on the
rounds after - is the only one of the three that matches how this dataset is
actually meant to be used (see docs/05_split_strategy.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def chronological_holdout_split(
    dataset: pd.DataFrame, n_test_rounds: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Train on every round except the last n_test_rounds (by `round`), test on those.

    Positional indices, relative to dataset's own row order.
    """
    rounds_sorted = sorted(dataset["round"].unique())
    if n_test_rounds >= len(rounds_sorted):
        raise ValueError(
            f"n_test_rounds={n_test_rounds} must be less than the number of distinct "
            f"rounds ({len(rounds_sorted)}) in the dataset"
        )
    test_rounds = set(rounds_sorted[-n_test_rounds:])
    test_mask = dataset["round"].isin(test_rounds).to_numpy()
    return np.flatnonzero(~test_mask), np.flatnonzero(test_mask)


def time_series_splits(dataset: pd.DataFrame, n_splits: int = 3):
    """Yield (train_idx, test_idx) pairs via sklearn's TimeSeriesSplit, ordered
    by `round`. Each successive fold's test set is a later slice of rounds
    than every round in that fold's training set - never the reverse.

    dataset need not be pre-sorted: this sorts internally and maps back to
    positional indices against dataset's own original row order (use with
    .iloc, exactly like mpp.splits.group_split's functions).
    """
    order = np.argsort(dataset["round"].to_numpy(), kind="stable")
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for train_pos, test_pos in splitter.split(np.zeros(len(order))):
        yield order[train_pos], order[test_pos]
