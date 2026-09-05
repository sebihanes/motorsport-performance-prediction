from __future__ import annotations

import pandas as pd

from mpp.ingest.run import build_intermediate
from tests.conftest import populate_raw_dir_from_fixture


def test_build_intermediate_parses_real_fixture_with_correct_dtypes(test_config, fixtures_dir):
    populate_raw_dir_from_fixture(test_config.paths.raw_dir, fixtures_dir)

    intermediate = build_intermediate(test_config)

    laps = intermediate["laps"]
    assert len(laps) == 9
    assert pd.api.types.is_timedelta64_dtype(laps["LapTime"])
    assert pd.api.types.is_timedelta64_dtype(laps["Time"])
    assert laps.loc[laps["Driver"] == "VER", "LapTime"].iloc[0] == pd.Timedelta(seconds=99.019)

    sessions = intermediate["sessions"]
    assert sessions.iloc[0]["circuit_id"] == "bahrain"
    assert sessions.iloc[0]["total_laps"] == 6  # max LapNumber in the fixture slice

    assert (test_config.paths.intermediate_dir / "laps.parquet").exists()


def test_build_intermediate_is_idempotent(test_config, fixtures_dir):
    populate_raw_dir_from_fixture(test_config.paths.raw_dir, fixtures_dir)
    first = build_intermediate(test_config)
    second = build_intermediate(test_config)
    pd.testing.assert_frame_equal(first["laps"], second["laps"])
