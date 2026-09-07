"""Hyperparameter tuning for lap-time candidate models (architecture SS8;
docs/05_split_strategy.md's "Hyperparameter tuning" section).

Tuning is nested: an outer grouped split
(``mpp.splits.group_split.race_group_shuffle_split``) holds out whole races
purely for the final reported metric and is never touched here. The ``cv``
argument every ``tune_*`` function takes below is expected to be
race-grouped folds (``mpp.splits.group_split.race_group_kfold``) computed
*only* over that outer split's training portion, so the same held-out race
is never used both to choose hyperparameters and to score the final result -
see ``mpp.models.train_lap_time`` for how the two are wired together.

Grids are deliberately small, per the brief's "avoid massive brute-force
searches": Ridge uses an exhaustive ``GridSearchCV`` since 5 alpha values is
cheap; the three tree-based models use ``RandomizedSearchCV`` with a capped
``n_iter``, sampling a handful of combinations from a wider but still bounded
space rather than a full grid.
"""

from __future__ import annotations

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from mpp.models.regressors import (
    gradient_boosting_pipeline,
    hist_gradient_boosting_pipeline,
    random_forest_pipeline,
    ridge_pipeline,
)

RANDOM_STATE = 0
N_ITER_RANDOM_SEARCH = 8

RIDGE_PARAM_GRID = {"model__alpha": [0.1, 1.0, 10.0, 50.0, 100.0]}

RANDOM_FOREST_PARAM_DISTRIBUTIONS = {
    "model__n_estimators": [100, 200, 300, 400],
    "model__max_depth": [None, 6, 10, 16],
    "model__min_samples_leaf": [1, 2, 4, 8],
    "model__max_features": ["sqrt", 0.5, 1.0],
}

GRADIENT_BOOSTING_PARAM_DISTRIBUTIONS = {
    "model__n_estimators": [100, 200, 300],
    "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
    "model__max_depth": [2, 3, 4],
    "model__subsample": [0.7, 0.85, 1.0],
}

HIST_GRADIENT_BOOSTING_PARAM_DISTRIBUTIONS = {
    "model__max_iter": [100, 200, 300],
    "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
    "model__max_leaf_nodes": [15, 31, 63],
    "model__l2_regularization": [0.0, 0.1, 1.0],
    "model__min_samples_leaf": [10, 20, 30],
}


def tune_ridge(cv, n_jobs: int = 1) -> GridSearchCV:
    return GridSearchCV(
        ridge_pipeline(),
        RIDGE_PARAM_GRID,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=n_jobs,
    )


def tune_random_forest(cv, n_jobs: int = 1) -> RandomizedSearchCV:
    return RandomizedSearchCV(
        random_forest_pipeline(),
        RANDOM_FOREST_PARAM_DISTRIBUTIONS,
        n_iter=N_ITER_RANDOM_SEARCH,
        cv=cv,
        scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
    )


def tune_gradient_boosting(cv, n_jobs: int = 1) -> RandomizedSearchCV:
    return RandomizedSearchCV(
        gradient_boosting_pipeline(),
        GRADIENT_BOOSTING_PARAM_DISTRIBUTIONS,
        n_iter=N_ITER_RANDOM_SEARCH,
        cv=cv,
        scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
    )


def tune_hist_gradient_boosting(cv, n_jobs: int = 1) -> RandomizedSearchCV:
    return RandomizedSearchCV(
        hist_gradient_boosting_pipeline(),
        HIST_GRADIENT_BOOSTING_PARAM_DISTRIBUTIONS,
        n_iter=N_ITER_RANDOM_SEARCH,
        cv=cv,
        scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
    )
