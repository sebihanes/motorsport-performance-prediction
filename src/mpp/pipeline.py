"""Pipeline orchestrator - runs every stage in order:

RAW DATA -> INGESTION -> VALIDATION -> CLEANING -> FEATURE DATASET -> MODEL DATASET

Usage:
    python -m mpp.pipeline [--force]

--force re-fetches from FastF1/Jolpica-F1 even if raw files already exist for
a round; by default already-fetched rounds are reused (see mpp.ingest.run).
"""

from __future__ import annotations

import argparse
import sys

from mpp.cleaning.run import run_cleaning
from mpp.config import load_config
from mpp.datasets.run import run_dataset_build
from mpp.features.build import run_feature_build
from mpp.ingest.run import build_intermediate, fetch_raw
from mpp.validation.report import has_errors, run_validation


def run_pipeline(force: bool = False) -> int:
    config = load_config()

    print(f"[1/5] Ingesting season {config.season}, rounds {list(config.rounds)} ...")
    coverage = fetch_raw(config, force=force)
    for c in coverage:
        print(f"      round {c.round:>2} {c.race_name:<28} laps={c.n_laps:>4} pitstops={c.n_pitstops:>2}")
    intermediate = build_intermediate(config)

    print("[2/5] Validating intermediate tables ...")
    validation_results = run_validation(intermediate, config)
    n_errors = sum(1 for rs in validation_results.values() for r in rs if r.status == "error")
    n_warnings = sum(1 for rs in validation_results.values() for r in rs if r.status == "warning")
    print(f"      {n_errors} error(s), {n_warnings} warning(s) - see data/quality/validation_report.md")
    if has_errors(validation_results):
        print("      Validation errors found. Cleaning will still run so issues remain inspectable,")
        print("      but review data/quality/validation_report.md before trusting the datasets.")

    print("[3/5] Cleaning ...")
    processed, cleaning_log = run_cleaning(intermediate, config)
    print(
        f"      laps: {cleaning_log['laps']['n_rows_out']} rows kept "
        f"({cleaning_log['laps']['n_valid_laps']} valid), "
        f"{cleaning_log['laps']['duplicate_rows_removed']} duplicates removed"
    )
    print(f"      pitstops: {cleaning_log['pitstops']['n_rows_out']} rows kept")

    print("[4/5] Building feature dataset ...")
    laps_features = run_feature_build(processed, intermediate, config)
    print(f"      laps_features: {len(laps_features)} rows, {laps_features.shape[1]} columns")

    print("[5/5] Building model-ready datasets ...")
    datasets = run_dataset_build(laps_features, processed["pitstops"], config)
    for name, frame in datasets.items():
        print(f"      {name}: {len(frame)} rows")

    print("Done. See data/quality/ for validation, cleaning and dataset-build reports.")
    return 1 if has_errors(validation_results) else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the motorsport ML data pipeline.")
    parser.add_argument("--force", action="store_true", help="Re-fetch raw data even if cached.")
    args = parser.parse_args()
    sys.exit(run_pipeline(force=args.force))


if __name__ == "__main__":
    main()
