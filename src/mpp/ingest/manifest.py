"""Write a manifest recording exactly what was ingested and when.

This is the traceability record for the raw data: source, URL, retrieval
timestamp, coverage, and row counts per round. See docs/01_data_sources.md.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class RoundCoverage:
    round: int
    race_name: str
    circuit_id: str
    n_laps: int
    n_pitstops: int
    n_drivers: int


def build_manifest(
    season: int,
    session_code: str,
    rounds_covered: list[RoundCoverage],
    retrieved_at: datetime | None = None,
) -> dict:
    retrieved_at = retrieved_at or datetime.now(timezone.utc)
    return {
        "retrieved_at_utc": retrieved_at.isoformat(),
        "season": season,
        "session": session_code,
        "sources": [
            {
                "name": "FastF1",
                "url": "https://github.com/theOehrly/Fast-F1",
                "provides": ["laps", "stints", "tyre_compound", "tyre_age", "weather"],
                "license": "MIT (library); underlying data from the official F1 live "
                "timing service, non-commercial use per FastF1's terms",
            },
            {
                "name": "Jolpica-F1",
                "url": "https://api.jolpi.ca/ergast/f1",
                "provides": ["schedule", "circuit_reference", "driver_reference", "pit_stops"],
                "license": "Ergast-compatible public API, non-commercial use",
            },
        ],
        "rounds_covered": [asdict(r) for r in rounds_covered],
        "notes": "Real data only - no synthetic or invented records. "
        "Scope (season/rounds) is set in config/settings.toml.",
    }


def write_manifest(manifest: dict, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "manifest.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return path
