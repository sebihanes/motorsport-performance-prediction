from __future__ import annotations

import pandas as pd

from mpp.features.session import add_session_type


def test_session_type_column_is_set_from_the_configured_session_code():
    df = pd.DataFrame({"lap_number": [1, 2, 3]})
    result = add_session_type(df, "R")
    assert list(result["session_type"]) == ["R", "R", "R"]


def test_session_type_does_not_mutate_the_input_frame():
    df = pd.DataFrame({"lap_number": [1]})
    add_session_type(df, "R")
    assert "session_type" not in df.columns
