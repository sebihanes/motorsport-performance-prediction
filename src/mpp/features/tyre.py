"""Tyre-related feature derivations."""

from __future__ import annotations

import pandas as pd


def add_tyre_age_squared(df: pd.DataFrame) -> pd.DataFrame:
    """tyre_age^2 captures the non-linear shape of degradation (architecture SS6)."""
    df = df.copy()
    df["tyre_age_sq"] = df["tyre_age"].astype("Float64") ** 2
    return df
