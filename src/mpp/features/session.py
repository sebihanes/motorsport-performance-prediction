"""Session-context feature.

session_type is carried as an explicit column (rather than left implicit)
because the pipeline architecture always intended session type - practice,
qualifying, race - to be a feature (architecture SS6). In the current scope
only Race ('R') sessions are ingested (docs/01_data_sources.md), so this
column is constant for every row today; it is still wired through as a real
column, not hard-coded downstream, so ingesting a second session type later
(e.g. qualifying, for a qualifying-pace problem) requires no schema change
here - only a wider config/settings.toml scope.
"""

from __future__ import annotations

import pandas as pd


def add_session_type(df: pd.DataFrame, session_code: str) -> pd.DataFrame:
    df = df.copy()
    df["session_type"] = session_code
    return df
