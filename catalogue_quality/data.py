"""Competition input and submission contracts."""

import numpy as np
import pandas as pd

from .features import INPUT_COLUMNS, NUMERIC_COLUMNS


def validate_frame(df, training=False):
    if df.empty:
        raise ValueError("Input contains no rows")
    required = INPUT_COLUMNS + (["target"] if training else [])
    absent = set(required) - set(df.columns)
    if absent:
        raise ValueError(f"Missing columns: {sorted(absent)}")
    if not training and "target" in df:
        raise ValueError("Prediction input must not contain a target column")
    for col in ["id", "seller_id"]:
        if df[col].isna().any() or df[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing or empty {col}")
    if not df.id.is_unique:
        raise ValueError("Listing IDs must be unique")
    for col in NUMERIC_COLUMNS:
        converted = pd.to_numeric(df[col], errors="coerce")
        invalid = df[col].notna() & converted.isna()
        values = converted.dropna()
        if invalid.any() or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"{col} must contain positive finite numbers or missing values")
        df[col] = converted.astype(float)
    if (df.pack_size.dropna() % 1 != 0).any():
        raise ValueError("Observed pack_size must be an integer")
    if training:
        if df.target.isna().any() or not df.target.isin([0, 1]).all():
            raise ValueError("Training target must contain only 0 and 1")
        if df.target.nunique() != 2:
            raise ValueError("Training data needs both target classes")
        df["target"] = df.target.astype(int)
    return df


def read_listings(path, training=False):
    # IDs are strings even for a future dataset containing numeric identifiers.
    return validate_frame(pd.read_csv(path, dtype={"id": str, "seller_id": str}), training)


def read_sample(path, test):
    sample = pd.read_csv(path, dtype={"id": str})
    if list(sample.columns) != ["id", "target"]:
        raise ValueError("Sample submission must have columns id,target in that order")
    if sample.id.tolist() != test.id.tolist():
        raise ValueError("Sample submission IDs or ordering differ from test.csv")
    return sample


def check_competition_split(train, test):
    if set(train.id) & set(test.id):
        raise ValueError("Train/test listing IDs overlap")
    if set(train.seller_id) & set(test.seller_id):
        raise ValueError("Expected unseen test sellers; train/test seller IDs overlap")


def write_submission(test, probabilities, path):
    p = np.asarray(probabilities)
    if p.shape != (len(test),) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Submission predictions must be finite probabilities, one per test row")
    pd.DataFrame({"id": test.id.to_numpy(), "target": p}).to_csv(path, index=False)
    written = pd.read_csv(path, dtype={"id": str})
    if list(written.columns) != ["id", "target"] or written.id.tolist() != test.id.tolist():
        raise RuntimeError("Written submission failed schema or ID-order validation")
    values = written.target.to_numpy()
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise RuntimeError("Written submission contains invalid probabilities")
