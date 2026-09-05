# Data Sources

Real Formula 1 timing data only. No record in this pipeline is invented,
simulated, or interpolated from anything other than the two sources below.

## FastF1

- **URL**: https://github.com/theOehrly/Fast-F1 (Python package `fastf1`,
  wraps the official F1 live timing service)
- **Retrieval process**: `mpp.ingest.fastf1_client` calls `fastf1.get_session(season, round, "R")`
  and `.load(laps=True, weather=True, telemetry=False, messages=False)`, then
  extracts the `laps` and `weather_data` frames with column renames only (no
  value changes) - see `extract_laps` / `extract_weather`. FastF1's own on-disk
  cache lives at `data/raw/fastf1_cache/` (git-ignored); our own immutable
  snapshot of each pull is written to `data/raw/fastf1/<season>/<round>/*.csv`.
- **Retrieval date**: first fetched 2026-09-05 (re-run at any time via
  `python -m mpp.pipeline --force`; the manifest at `data/raw/manifest.json`
  always records the actual timestamp of the last fetch).
- **Coverage**: lap-by-lap timing (lap time, sector times), stint/compound/tyre
  age, pit in/out timestamps, lap validity flags (`Deleted`, `IsAccurate`,
  `TrackStatus`), and weather samples (air/track temperature, rainfall),
  2018 onward. This pipeline uses Race ('R') sessions only (see SS4 of
  `docs/ARCHITECTURE_REPORT.md` for why: lap time, degradation, race pace and
  pit stops are all race-session concepts).
- **Limitations**: weather is sampled periodically (roughly once a minute),
  not per lap - `mpp.features.weather` joins each lap to its nearest sample
  within a 5-minute tolerance and leaves it null otherwise, it is never
  fabricated. `TyreLife` (tyre age) can start above 1 on lap 1 of a race,
  because pre-race reconnaissance/formation laps on the same tyre set count
  towards it - this is real FastF1 behaviour, not a data defect.
- **License / usage**: MIT-licensed library; the underlying timing data comes
  from the official F1 live timing service and is used here for
  non-commercial, educational purposes per FastF1's documented terms.

## Jolpica-F1

- **URL**: https://api.jolpi.ca/ergast/f1 (Ergast-compatible REST API,
  https://github.com/jolpica/jolpica-f1)
- **Retrieval process**: `mpp.ingest.jolpica_client` makes plain `GET` requests
  (3 attempts with backoff) to `/{season}.json` (schedule/circuit reference),
  `/{season}/{round}/drivers.json` (driver reference, including the FastF1-
  compatible 3-letter `code`), and `/{season}/{round}/pitstops.json` (pit stop
  records). Raw JSON responses are saved unmodified to
  `data/raw/jolpica/<season>/...`.
- **Retrieval date**: first fetched 2026-09-05 (see `data/raw/manifest.json`
  for the exact timestamp of the data currently on disk).
- **Coverage**: full season schedule and circuit identifiers back to 1950;
  pit stop lap/time/duration records from the mid-2010s onward. This pipeline
  uses the 2023 season, rounds 1-6, configured in `config/settings.toml`.
- **Limitations**: pit stop `duration` is the full stationary time Jolpica-F1
  reports; it is not decomposed into "tyre change" vs. other causes (damage
  repair, drive-through penalties served as a stop). Long stops in the data
  are real recorded events, not outliers to be removed. Circuit reference
  data provides identifier, name, and locality/coordinates only - it does not
  include physical track characteristics (length, corner count, elevation),
  which `docs/ARCHITECTURE_REPORT.md` SS6 anticipated as a feature category;
  see `docs/03_dataset_definitions.md` "Limitations" for how that gap is
  handled (deferred, not invented).
- **License / usage**: public, Ergast-compatible API, non-commercial use.

## Scope

Season and rounds are set in `config/settings.toml` (`[scope]`), currently
`season = 2023`, `rounds = [1, 2, 3, 4, 5, 6]` - Bahrain through Monaco.
Chosen to match the scope already used by the sibling
`motorsport-statistical-analysis` project for portfolio consistency, and
because six rounds is enough for the leakage-safe historical feature
(`mpp.features.historical`) to have real history to average over from round 2
onward, while keeping ingestion fast enough to iterate on. Widening scope is a
one-line config edit (`rounds = [1, ..., N]`); no code changes needed.
