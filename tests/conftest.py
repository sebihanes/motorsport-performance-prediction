from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from mpp.config import CleaningSettings, PipelineConfig, PipelinePaths, ValidationSettings

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


def populate_raw_dir_from_fixture(raw_dir: Path, fixtures_dir: Path) -> None:
    """Lay out the real fixture slice (tests/fixtures/real_*_2023_r1.*) as if it
    had been fetched by mpp.ingest.run.fetch_raw for season=2023, round=1."""
    fastf1_dir = raw_dir / "fastf1" / "2023" / "1"
    jolpica_round_dir = raw_dir / "jolpica" / "2023" / "1"
    fastf1_dir.mkdir(parents=True)
    jolpica_round_dir.mkdir(parents=True)

    shutil.copy(fixtures_dir / "real_laps_slice_2023_r1.csv", fastf1_dir / "laps_raw.csv")
    shutil.copy(fixtures_dir / "real_weather_slice_2023_r1.csv", fastf1_dir / "weather_raw.csv")
    shutil.copy(fixtures_dir / "real_drivers_slice_2023_r1.json", jolpica_round_dir / "drivers_raw.json")
    shutil.copy(fixtures_dir / "real_pitstops_slice_2023_r1.json", jolpica_round_dir / "pitstops_raw.json")

    schedule = [
        {
            "season": 2023,
            "round": 1,
            "race_name": "Bahrain Grand Prix",
            "circuit_id": "bahrain",
            "circuit_name": "Bahrain International Circuit",
            "locality": "Sakhir",
            "country": "Bahrain",
            "lat": 26.0325,
            "long": 50.5106,
            "date": "2023-03-05",
        }
    ]
    (raw_dir / "jolpica" / "2023_schedule_raw.json").write_text(json.dumps(schedule), encoding="utf-8")


@pytest.fixture
def test_config(tmp_path: Path) -> PipelineConfig:
    """A PipelineConfig pointed entirely at a scratch tmp_path - no repo files touched."""
    paths = PipelinePaths(
        raw_dir=tmp_path / "raw",
        intermediate_dir=tmp_path / "intermediate",
        processed_dir=tmp_path / "processed",
        features_dir=tmp_path / "features",
        model_ready_dir=tmp_path / "model_ready",
        quality_dir=tmp_path / "quality",
        models_dir=tmp_path / "models",
    )
    return PipelineConfig(
        season=2023,
        rounds=(1,),
        session="R",
        paths=paths,
        validation=ValidationSettings(
            lap_time_min_s=55.0,
            lap_time_max_s=160.0,
            valid_compounds=("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN"),
        ),
        cleaning=CleaningSettings(duplicate_key=("season", "round", "driver", "lap_number")),
        repo_root=tmp_path,
    )
