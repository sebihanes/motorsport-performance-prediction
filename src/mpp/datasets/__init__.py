"""Model-ready dataset construction: FEATURE DATASET -> MODEL DATASET.

Each module here builds exactly one problem's dataset (see
docs/03_dataset_definitions.md for column-by-column definitions) and is
responsible for its own row-selection rule - which rows are fit for that
specific target - since "valid for lap-time regression" and "valid for
degradation regression" are different questions.
"""
