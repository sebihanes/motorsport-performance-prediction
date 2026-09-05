from __future__ import annotations

import pandas as pd

from mpp.cleaning.laps import clean_laps

TD = pd.Timedelta


def _row(**overrides) -> dict:
    row = {
        "season": 2023,
        "round": 1,
        "Driver": "VER",
        "DriverNumber": "1",
        "Team": "Red Bull Racing",
        "LapNumber": 1,
        "Stint": 1,
        "LapTime": TD(seconds=99.0),
        "Time": TD(minutes=1, seconds=39),
        "Compound": "SOFT",
        "TyreLife": 4,
        "FreshTyre": True,
        "PitInTime": pd.NaT,
        "PitOutTime": pd.NaT,
        "Sector1Time": TD(seconds=30),
        "Sector2Time": TD(seconds=35),
        "Sector3Time": TD(seconds=34),
        "TrackStatus": "1",
        "Deleted": False,
        "DeletedReason": None,
        "IsAccurate": True,
    }
    row.update(overrides)
    return row


_TIMEDELTA_COLUMNS = [
    "LapTime",
    "Time",
    "PitInTime",
    "PitOutTime",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
]


def _df(*rows: dict) -> pd.DataFrame:
    records = list(rows)
    df = pd.DataFrame(records)
    # A column that is all-NaT is ambiguous (datetime vs. timedelta) and pandas
    # infers datetime64 by default; build it explicitly as timedelta64 to match
    # what mpp.ingest.run.build_intermediate always produces from real raw data.
    for col in _TIMEDELTA_COLUMNS:
        df[col] = pd.Series([r.get(col) for r in records], dtype="timedelta64[ns]")
    return df


def test_valid_lap_is_marked_valid(test_config):
    df = _df(_row())
    cleaned, log = clean_laps(df, test_config)
    assert cleaned.loc[0, "is_valid_lap"]
    assert pd.isna(cleaned.loc[0, "exclusion_reason"])
    assert cleaned.loc[0, "lap_time_s"] == 99.0


def test_missing_lap_time_is_excluded(test_config):
    df = _df(_row(LapTime=pd.NaT))
    cleaned, log = clean_laps(df, test_config)
    assert not cleaned.loc[0, "is_valid_lap"]
    assert cleaned.loc[0, "exclusion_reason"] == "missing_lap_time"


def test_deleted_lap_is_excluded_even_with_a_lap_time(test_config):
    df = _df(_row(Deleted=True))
    cleaned, log = clean_laps(df, test_config)
    assert not cleaned.loc[0, "is_valid_lap"]
    assert cleaned.loc[0, "exclusion_reason"] == "deleted_lap"


def test_pit_in_lap_is_excluded(test_config):
    df = _df(_row(PitInTime=TD(minutes=1, seconds=39)))
    cleaned, log = clean_laps(df, test_config)
    assert not cleaned.loc[0, "is_valid_lap"]
    assert cleaned.loc[0, "exclusion_reason"] == "pit_in_out_lap"
    assert cleaned.loc[0, "pit_in_lap"]


def test_lap_time_out_of_bounds_is_excluded(test_config):
    df = _df(_row(LapTime=TD(seconds=300)))  # way above the 160s bound
    cleaned, log = clean_laps(df, test_config)
    assert not cleaned.loc[0, "is_valid_lap"]
    assert cleaned.loc[0, "exclusion_reason"] == "lap_time_out_of_bounds"


def test_missing_lap_time_takes_priority_over_deleted(test_config):
    df = _df(_row(LapTime=pd.NaT, Deleted=True))
    cleaned, log = clean_laps(df, test_config)
    assert cleaned.loc[0, "exclusion_reason"] == "missing_lap_time"


def test_deleted_takes_priority_over_pit_in_out(test_config):
    df = _df(_row(Deleted=True, PitInTime=TD(minutes=1, seconds=39)))
    cleaned, log = clean_laps(df, test_config)
    assert cleaned.loc[0, "exclusion_reason"] == "deleted_lap"


def test_exact_duplicate_rows_are_removed_and_logged(test_config):
    df = _df(_row(), _row())
    cleaned, log = clean_laps(df, test_config)
    assert len(cleaned) == 1
    assert log["duplicate_rows_removed"] == 1


def test_corrupt_lap_number_is_dropped(test_config):
    df = _df(_row(LapNumber=0), _row(LapNumber=None))
    cleaned, log = clean_laps(df, test_config)
    assert len(cleaned) == 0
    assert log["corrupt_lap_number_rows_removed"] == 2


def test_compound_is_normalized_to_upper_case(test_config):
    df = _df(_row(Compound="soft"))
    cleaned, log = clean_laps(df, test_config)
    assert cleaned.loc[0, "compound"] == "SOFT"
    assert log["compound_rows_remapped_to_unknown"] == 0


def test_unrecognized_compound_becomes_unknown_and_is_counted(test_config):
    df = _df(_row(Compound="C1"))
    cleaned, log = clean_laps(df, test_config)
    assert cleaned.loc[0, "compound"] == "UNKNOWN"
    assert log["compound_rows_remapped_to_unknown"] == 1


def test_deleted_none_is_treated_as_not_deleted(test_config):
    # FastF1 represents "not deleted" as None, not False - a real observed shape.
    df = _df(_row(Deleted=None))
    cleaned, log = clean_laps(df, test_config)
    assert cleaned.loc[0, "deleted"] == False  # noqa: E712
    assert cleaned.loc[0, "is_valid_lap"]


def test_clean_laps_on_real_data_slice_produces_expected_lap_times(test_config, fixtures_dir):
    raw = pd.read_csv(fixtures_dir / "real_laps_slice_2023_r1.csv")
    timedelta_cols = [
        "LapTime",
        "Time",
        "PitInTime",
        "PitOutTime",
        "Sector1Time",
        "Sector2Time",
        "Sector3Time",
    ]
    for col in timedelta_cols:
        raw[col] = pd.to_timedelta(raw[col], errors="coerce")

    cleaned, log = clean_laps(raw, test_config)

    assert len(cleaned) == 9
    assert cleaned["is_valid_lap"].all()
    ver_lap1 = cleaned[(cleaned["driver"] == "VER") & (cleaned["lap_number"] == 1)].iloc[0]
    assert ver_lap1["lap_time_s"] == 99.019
    assert ver_lap1["tyre_age"] == 4
    assert ver_lap1["compound"] == "SOFT"
