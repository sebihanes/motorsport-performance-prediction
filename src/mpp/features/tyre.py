"""Tyre-related feature derivations."""

from __future__ import annotations

import pandas as pd


def add_tyre_age_squared(df: pd.DataFrame) -> pd.DataFrame:
    """tyre_age^2 captures the non-linear shape of degradation (architecture SS6)."""
    df = df.copy()
    df["tyre_age_sq"] = df["tyre_age"].astype("Float64") ** 2
    return df


def add_compound_tyre_age_interaction(
    df: pd.DataFrame, compounds: tuple[str, ...], tyre_age_column: str = "tyre_age"
) -> pd.DataFrame:
    """One tyre_age_x_<compound> column per known compound: tyre_age where the
    lap ran that compound, 0.0 elsewhere.

    Softer compounds are known to degrade faster than harder ones - "does
    tyre age matter differently by compound" is a real race-engineering
    question, not an arbitrary cross term. A tree-based model already learns
    this interaction for free from compound + tyre_age alone (it can split on
    compound and then on tyre_age within each branch); these explicit columns
    exist so the same interaction is available to a linear model, which
    cannot construct it on its own. "UNKNOWN" (an unrecognized/missing
    compound, see mpp.cleaning.laps) is excluded - there is no meaningful
    degradation rate to attribute to "we don't know what tyre this was."
    """
    df = df.copy()
    tyre_age = df[tyre_age_column].astype("Float64")
    for compound in compounds:
        if compound == "UNKNOWN":
            continue
        column = f"tyre_age_x_{compound.lower()}"
        df[column] = tyre_age.where(df["compound"] == compound, 0.0)
    return df
