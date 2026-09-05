"""Deterministic cleaning: intermediate tables -> processed tables.

Every function here is a pure, deterministic transformation of a DataFrame:
same input always produces the same output plus a JSON-serializable log of
what was changed and why. Nothing is dropped without being counted in that
log - see docs/02_data_pipeline.md "Cleaning decisions" for the full list of
exclusion reasons and thresholds.
"""
