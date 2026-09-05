"""Client for the Jolpica-F1 API (Ergast-compatible REST API for F1 results).

Source: https://github.com/jolpica/jolpica-f1
Endpoint base: https://api.jolpi.ca/ergast/f1
Used for: season schedule / circuit reference, driver reference, pit stop
records. See docs/01_data_sources.md for retrieval date, coverage and
licensing notes. No data from this client is invented or altered - fields
are passed through as returned by the API, only reshaped into flat dicts.
"""

from __future__ import annotations

import time
from typing import Any

import requests

BASE_URL = "https://api.jolpi.ca/ergast/f1"
_TIMEOUT_S = 15
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_S = 2.0


def _get_json(url: str) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            response = requests.get(url, timeout=_TIMEOUT_S)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < _MAX_ATTEMPTS:
                time.sleep(_RETRY_BACKOFF_S * attempt)
    raise RuntimeError(f"Failed to fetch {url} after {_MAX_ATTEMPTS} attempts") from last_error


def fetch_schedule(season: int) -> list[dict[str, Any]]:
    """One row per round: round, race_name, circuit_id, circuit_name, locality, country, date."""
    data = _get_json(f"{BASE_URL}/{season}.json?limit=100")
    races = data["MRData"]["RaceTable"]["Races"]
    rows = []
    for race in races:
        circuit = race["Circuit"]
        rows.append(
            {
                "season": season,
                "round": int(race["round"]),
                "race_name": race["raceName"],
                "circuit_id": circuit["circuitId"],
                "circuit_name": circuit["circuitName"],
                "locality": circuit["Location"]["locality"],
                "country": circuit["Location"]["country"],
                "lat": float(circuit["Location"]["lat"]),
                "long": float(circuit["Location"]["long"]),
                "date": race["date"],
            }
        )
    return rows


def fetch_drivers(season: int, round_: int) -> list[dict[str, Any]]:
    """One row per driver who took part in this round: driver_id, code, names, nationality."""
    data = _get_json(f"{BASE_URL}/{season}/{round_}/drivers.json?limit=50")
    drivers = data["MRData"]["DriverTable"]["Drivers"]
    rows = []
    for driver in drivers:
        rows.append(
            {
                "season": season,
                "round": round_,
                "driver_id": driver["driverId"],
                "driver_code": driver.get("code"),
                "permanent_number": driver.get("permanentNumber"),
                "given_name": driver["givenName"],
                "family_name": driver["familyName"],
                "nationality": driver.get("nationality"),
            }
        )
    return rows


def fetch_pitstops(season: int, round_: int) -> list[dict[str, Any]]:
    """One row per pit stop: driver_id, lap, stop_number, local time-of-day, duration (string, seconds)."""
    data = _get_json(f"{BASE_URL}/{season}/{round_}/pitstops.json?limit=100")
    races = data["MRData"]["RaceTable"]["Races"]
    if not races:
        return []
    stops = races[0].get("PitStops", [])
    rows = []
    for stop in stops:
        rows.append(
            {
                "season": season,
                "round": round_,
                "driver_id": stop["driverId"],
                "lap": int(stop["lap"]),
                "stop_number": int(stop["stop"]),
                "time_of_day": stop.get("time"),
                "duration_raw": stop.get("duration"),
            }
        )
    return rows
