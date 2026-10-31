"""Fit expected attributes on target-0 rows, with nested cross-fitting."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from .features import MISSING

REFERENCE_CATEGORIES = ["product", "variant", "brand", "condition"]
REFERENCE_NUMERICS = ["clearance", "log_main_quantity"]


def reference_input(x):
    r = x[REFERENCE_CATEGORIES + ["clearance"]].copy()
    r["log_main_quantity"] = np.log(x.main_quantity.fillna(1))
    return r


def fit_reference(x, y):
    y = np.asarray(y)
    if len(x) != len(y):
        raise ValueError("Reference rows and labels have different lengths")
    clean = x.iloc[np.flatnonzero(y == 0)]
    regressors = {}
    for col in ["price_pln", "shipping_weight_kg"]:
        observed = clean.loc[clean[col].notna()]
        if len(observed) < 2:
            raise ValueError(f"Need at least two observed target-0 values for {col}")
        prep = ColumnTransformer([
            ("categories", OneHotEncoder(handle_unknown="ignore"), REFERENCE_CATEGORIES),
            ("numeric", "passthrough", REFERENCE_NUMERICS),
        ])
        model = make_pipeline(prep, Ridge(alpha=2., solver="lsqr"))
        model.fit(reference_input(observed), np.log(observed[col]))
        regressors[col] = model
    known = clean.loc[clean['product'].ne(MISSING) & clean.declared_category.ne(MISSING)]
    counts = pd.crosstab(known['product'], known.declared_category)
    return {"regressors": regressors, "category_counts": counts, "clean_rows": len(clean)}


def reference_features(x, reference):
    f = pd.DataFrame(index=x.index)
    for col, model in reference["regressors"].items():
        expected = model.predict(reference_input(x))
        residual = np.log(x[col]) - expected
        f[col + "_expected_log"] = expected
        f[col + "_residual"] = residual
        f[col + "_abs_residual"] = np.abs(residual)
    counts = reference["category_counts"]
    probabilities = []
    for product, category in zip(x['product'], x.declared_category):
        if product not in counts.index or category == MISSING:
            probabilities.append(np.nan)
        else:
            row = counts.loc[product]
            probabilities.append((row.get(category, 0) + 1) / (row.sum() + len(counts.columns)))
    f["category_surprisal"] = -np.log(probabilities)
    return f


def crossfit_reference_features(x, y, sellers, n_splits=3):
    """Each row's reference excludes its entire seller and all its labels."""
    if pd.Series(sellers).nunique() < n_splits:
        raise ValueError("Too few training sellers for inner reference folds")
    pieces, seen = [], np.zeros(len(x), dtype=int)
    y, sellers = np.asarray(y), np.asarray(sellers)
    for fit, valid in GroupKFold(n_splits).split(x, y, sellers):
        if set(sellers[fit]) & set(sellers[valid]):
            raise RuntimeError("Inner seller leakage")
        reference = fit_reference(x.iloc[fit], y[fit])
        pieces.append(reference_features(x.iloc[valid], reference))
        seen[valid] += 1
    if not (seen == 1).all():
        raise RuntimeError("Inner reference predictions do not cover each row once")
    return pd.concat(pieces).loc[x.index]
