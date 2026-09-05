"""Ingestion orchestrator: RAW DATA -> INGESTION.

fetch_raw() hits the network (FastF1 + Jolpica-F1) and writes untouched,
immutable artifacts under <raw_dir>. It is idempotent: a round already on
disk is skipped unless force=True, so re-running the pipeline does not
re-hammer the APIs.

build_intermediate() does no network I/O. It reads only the raw artifacts
back from disk and concatenates/structurally-types them into the four
intermediate tables (laps, weather, pitstops, sessions/drivers) that
validation (mpp.validation) and cleaning (mpp.cleaning) operate on. No
business-rule cleaning (numeric conversion of timings, categorical
normalization, invalid-lap handling) happens here - see mpp.cleaning.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mpp.config import PipelineConfig
from mpp.ingest import fastf1_client, jolpica_client
from mpp.ingest.manifest import RoundCoverage, build_manifest, write_manifest


def _round_dir(raw_dir: Path, season: int, round_: int) -> Path:
    return raw_dir / "fastf1" / str(season) / str(round_)


def _jolpica_dir(raw_dir: Path, season: int, round_: int | None = None) -> Path:
    return raw_dir / "jolpica" / str(season) / (str(round_) if round_ is not None else "")


def fetch_raw(config: PipelineConfig, force: bool = False) -> list[RoundCoverage]:
    """Fetch one race session + Jolpica reference data per configured round."""
    fastf1_client.configure_cache(config.paths.raw_dir / "fastf1_cache")

    schedule_dir = config.paths.raw_dir / "jolpica"
    schedule_dir.mkdir(parents=True, exist_ok=True)
    schedule_path = schedule_dir / f"{config.season}_schedule_raw.json"
    if force or not schedule_path.exists():
        schedule = jolpica_client.fetch_schedule(config.season)
        schedule_path.write_text(json.dumps(schedule, indent=2), encoding="utf-8")
    else:
        schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule_by_round = {row["round"]: row for row in schedule}

    coverage: list[RoundCoverage] = []
    for round_ in config.rounds:
        if round_ not in schedule_by_round:
            raise ValueError(f"Round {round_} not found in season {config.season} schedule")
        race_info = schedule_by_round[round_]

        fastf1_dir = _round_dir(config.paths.raw_dir, config.season, round_)
        laps_path = fastf1_dir / "laps_raw.csv"
        weather_path = fastf1_dir / "weather_raw.csv"
        jolp_dir = _jolpica_dir(config.paths.raw_dir, config.season, round_)
        drivers_path = jolp_dir / "drivers_raw.json"
        pitstops_path = jolp_dir / "pitstops_raw.json"

        if force or not (laps_path.exists() and weather_path.exists()):
            fastf1_dir.mkdir(parents=True, exist_ok=True)
            session = fastf1_client.load_session(config.season, round_, config.session)
            laps = fastf1_client.extract_laps(session, config.season, round_)
            weather = fastf1_client.extract_weather(session, config.season, round_)
            laps.to_csv(laps_path, index=False)
            weather.to_csv(weather_path, index=False)
        else:
            laps = pd.read_csv(laps_path)

        if force or not drivers_path.exists():
            jolp_dir.mkdir(parents=True, exist_ok=True)
            drivers = jolpica_client.fetch_drivers(config.season, round_)
            drivers_path.write_text(json.dumps(drivers, indent=2), encoding="utf-8")
        else:
            drivers = json.loads(drivers_path.read_text(encoding="utf-8"))

        if force or not pitstops_path.exists():
            jolp_dir.mkdir(parents=True, exist_ok=True)
            pitstops = jolpica_client.fetch_pitstops(config.season, round_)
            pitstops_path.write_text(json.dumps(pitstops, indent=2), encoding="utf-8")
        else:
            pitstops = json.loads(pitstops_path.read_text(encoding="utf-8"))

        coverage.append(
            RoundCoverage(
                round=round_,
                race_name=race_info["race_name"],
                circuit_id=race_info["circuit_id"],
                n_laps=len(laps),
                n_pitstops=len(pitstops),
                n_drivers=len(drivers),
            )
        )

    manifest = build_manifest(config.season, config.session, coverage)
    write_manifest(manifest, config.paths.raw_dir)
    return coverage


def build_intermediate(config: PipelineConfig) -> dict[str, pd.DataFrame]:
    """Read raw artifacts back from disk and assemble the four intermediate tables."""
    schedule_path = config.paths.raw_dir / "jolpica" / f"{config.season}_schedule_raw.json"
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule_by_round = {row["round"]: row for row in schedule}

    laps_frames, weather_frames, driver_frames, pitstop_frames, session_rows = [], [], [], [], []
    timedelta_cols = [
        "LapTime",
        "Time",
        "PitInTime",
        "PitOutTime",
        "Sector1Time",
        "Sector2Time",
        "Sector3Time",
    ]

    for round_ in config.rounds:
        fastf1_dir = _round_dir(config.paths.raw_dir, config.season, round_)
        jolp_dir = _jolpica_dir(config.paths.raw_dir, config.season, round_)

        laps = pd.read_csv(
            fastf1_dir / "laps_raw.csv", dtype={"TrackStatus": "string", "DriverNumber": "string"}
        )
        for col in timedelta_cols:
            laps[col] = pd.to_timedelta(laps[col], errors="coerce")
        weather = pd.read_csv(fastf1_dir / "weather_raw.csv")
        weather["Time"] = pd.to_timedelta(weather["Time"], errors="coerce")
        drivers = pd.DataFrame(json.loads((jolp_dir / "drivers_raw.json").read_text(encoding="utf-8")))
        pitstops = pd.DataFrame(
            json.loads((jolp_dir / "pitstops_raw.json").read_text(encoding="utf-8"))
        )

        laps["LapNumber"] = pd.array(laps["LapNumber"], dtype="Int64")
        laps["Stint"] = pd.array(laps["Stint"], dtype="Int64")
        laps["TyreLife"] = pd.array(laps["TyreLife"], dtype="Int64")
        laps_frames.append(laps)
        weather_frames.append(weather)
        driver_frames.append(drivers)
        if not pitstops.empty:
            pitstop_frames.append(pitstops)

        race_info = schedule_by_round[round_]
        session_rows.append(
            {
                "season": config.season,
                "round": round_,
                "race_name": race_info["race_name"],
                "circuit_id": race_info["circuit_id"],
                "circuit_name": race_info["circuit_name"],
                "locality": race_info["locality"],
                "country": race_info["country"],
                "date": race_info["date"],
                "total_laps": int(laps["LapNumber"].max()),
                "n_drivers": int(laps["Driver"].nunique()),
            }
        )

    intermediate = {
        "laps": pd.concat(laps_frames, ignore_index=True),
        "weather": pd.concat(weather_frames, ignore_index=True),
        "drivers": pd.concat(driver_frames, ignore_index=True),
        "pitstops": pd.concat(pitstop_frames, ignore_index=True)
        if pitstop_frames
        else pd.DataFrame(
            columns=["season", "round", "driver_id", "lap", "stop_number", "time_of_day", "duration_raw"]
        ),
        "sessions": pd.DataFrame(session_rows),
    }

    out_dir = config.paths.intermediate_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in intermediate.items():
        frame.to_parquet(out_dir / f"{name}.parquet", index=False)

    return intermediate
