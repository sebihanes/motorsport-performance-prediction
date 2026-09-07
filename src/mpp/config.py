"""Load pipeline configuration from config/settings.toml.

Keeping scope (season/rounds) and thresholds in one TOML file means widening
the pipeline to more races is a config edit, not a code change - see
docs/ARCHITECTURE_REPORT.md SS4 and SS19.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "settings.toml"


@dataclass(frozen=True)
class PipelinePaths:
    raw_dir: Path
    intermediate_dir: Path
    processed_dir: Path
    features_dir: Path
    model_ready_dir: Path
    quality_dir: Path
    models_dir: Path


@dataclass(frozen=True)
class ValidationSettings:
    lap_time_min_s: float
    lap_time_max_s: float
    valid_compounds: tuple[str, ...]


@dataclass(frozen=True)
class CleaningSettings:
    duplicate_key: tuple[str, ...]


@dataclass(frozen=True)
class PipelineConfig:
    season: int
    rounds: tuple[int, ...]
    session: str
    paths: PipelinePaths
    validation: ValidationSettings
    cleaning: CleaningSettings
    repo_root: Path = field(default=REPO_ROOT)


def _resolve(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else repo_root / path


def load_config(config_path: Path | None = None, repo_root: Path | None = None) -> PipelineConfig:
    """Read config/settings.toml and resolve every path against the repo root."""
    config_path = config_path or DEFAULT_CONFIG_PATH
    repo_root = repo_root or REPO_ROOT

    with open(config_path, "rb") as fh:
        raw = tomllib.load(fh)

    scope = raw["scope"]
    paths_raw = raw["paths"]
    validation_raw = raw["validation"]
    cleaning_raw = raw["cleaning"]

    paths = PipelinePaths(
        raw_dir=_resolve(repo_root, paths_raw["raw_dir"]),
        intermediate_dir=_resolve(repo_root, paths_raw["intermediate_dir"]),
        processed_dir=_resolve(repo_root, paths_raw["processed_dir"]),
        features_dir=_resolve(repo_root, paths_raw["features_dir"]),
        model_ready_dir=_resolve(repo_root, paths_raw["model_ready_dir"]),
        quality_dir=_resolve(repo_root, paths_raw["quality_dir"]),
        models_dir=_resolve(repo_root, paths_raw["models_dir"]),
    )
    validation = ValidationSettings(
        lap_time_min_s=float(validation_raw["lap_time_min_s"]),
        lap_time_max_s=float(validation_raw["lap_time_max_s"]),
        valid_compounds=tuple(validation_raw["valid_compounds"]),
    )
    cleaning = CleaningSettings(duplicate_key=tuple(cleaning_raw["duplicate_key"]))

    return PipelineConfig(
        season=int(scope["season"]),
        rounds=tuple(int(r) for r in scope["rounds"]),
        session=str(scope["session"]),
        paths=paths,
        validation=validation,
        cleaning=cleaning,
        repo_root=repo_root,
    )
