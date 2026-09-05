from __future__ import annotations

from mpp.config import DEFAULT_CONFIG_PATH, REPO_ROOT, load_config


def test_load_config_reads_scope():
    config = load_config()
    assert config.season == 2023
    assert config.rounds == (1, 2, 3, 4, 5, 6)
    assert config.session == "R"


def test_load_config_resolves_paths_under_repo_root():
    config = load_config()
    assert config.paths.raw_dir == REPO_ROOT / "data" / "raw"
    assert config.paths.model_ready_dir == REPO_ROOT / "data" / "model_ready"


def test_load_config_reads_validation_thresholds():
    config = load_config()
    assert config.validation.lap_time_min_s == 55.0
    assert config.validation.lap_time_max_s == 160.0
    assert "SOFT" in config.validation.valid_compounds


def test_default_config_path_exists():
    assert DEFAULT_CONFIG_PATH.exists()
