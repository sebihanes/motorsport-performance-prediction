"""Client for FastF1 (session/lap/stint/tyre/weather timing data).

Source: https://github.com/theOehrly/Fast-F1 (wraps the official F1 live
timing service). See docs/01_data_sources.md for retrieval date, coverage and
licensing notes. Lap and weather frames are extracted with minimal reshaping
(column renames + identifier columns added) so the saved raw files are as
close as possible to what FastF1 returns - real, unaltered session data.
"""

from __future__ import annotations

from pathlib import Path

import fastf1
import pandas as pd

_CACHE_CONFIGURED = False


def configure_cache(cache_dir: Path) -> None:
    """Point FastF1's own on-disk cache at our raw data directory (idempotent)."""
    global _CACHE_CONFIGURED
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    _CACHE_CONFIGURED = True


def load_session(season: int, round_: int, session_code: str) -> fastf1.core.Session:
    if not _CACHE_CONFIGURED:
        raise RuntimeError("Call configure_cache() before loading a session")
    session = fastf1.get_session(season, round_, session_code)
    session.load(laps=True, telemetry=False, weather=True, messages=False)
    return session


def extract_laps(session: fastf1.core.Session, season: int, round_: int) -> pd.DataFrame:
    """Raw lap-by-lap extract: one row per driver-lap, columns as reported by FastF1."""
    laps = session.laps.copy()
    laps.insert(0, "round", round_)
    laps.insert(0, "season", season)
    columns = [
        "season",
        "round",
        "Driver",
        "DriverNumber",
        "Team",
        "LapNumber",
        "Stint",
        "LapTime",
        "Time",
        "Compound",
        "TyreLife",
        "FreshTyre",
        "PitInTime",
        "PitOutTime",
        "Sector1Time",
        "Sector2Time",
        "Sector3Time",
        "TrackStatus",
        "Deleted",
        "DeletedReason",
        "IsAccurate",
    ]
    return laps[columns].reset_index(drop=True)


def extract_weather(session: fastf1.core.Session, season: int, round_: int) -> pd.DataFrame:
    """Raw weather samples for the session (sampled periodically, not per-lap)."""
    weather = session.weather_data.copy()
    weather.insert(0, "round", round_)
    weather.insert(0, "season", season)
    return weather[["season", "round", "Time", "AirTemp", "TrackTemp", "Rainfall"]].reset_index(
        drop=True
    )


def session_total_laps(session: fastf1.core.Session) -> int:
    return int(session.laps["LapNumber"].max())
