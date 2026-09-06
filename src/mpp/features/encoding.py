"""Reusable categorical-encoding pipeline for the model-ready datasets.

Nominal categoricals here (driver, team, circuit_id, compound, session_type)
have no natural order - encoding them as an arbitrary integer (e.g.
LabelEncoder) would hand a linear model a false sense of ordinal distance
(team #3 is not "between" teams #2 and #4). One-hot encoding is the correct
default. handle_unknown="ignore" matters specifically because of the grouped
split strategy this project uses (docs/05_split_strategy.md): a
GroupShuffleSplit held-out circuit or a driver who only appears in a later
round can put a category in the test fold that the encoder never saw during
fit - the encoder must degrade to an all-zero row for that category rather
than raising, since refusing to score an unseen driver/circuit is not an
option in a real deployment.

Numeric features are passed through unscaled by default: the primary
candidate models (Random Forest, HistGradientBoosting - architecture SS11)
are scale-invariant. Set scale_numeric=True to standardize them for a linear
model (Ridge/Lasso) instead.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_feature_pipeline(
    numeric_columns: list[str], nominal_columns: list[str], scale_numeric: bool = False
) -> ColumnTransformer:
    """A ColumnTransformer mapping (numeric_columns, nominal_columns) -> a model
    matrix. Any column not listed in either argument is dropped, not passed
    through silently - callers should pass every feature column they intend
    to use.
    """
    numeric_transform = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transform, list(numeric_columns)),
            (
                "nominal",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(nominal_columns),
            ),
        ]
    )
