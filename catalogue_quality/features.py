"""Stateless extraction: no labels, vocabulary fitting, or dataset statistics."""

import re
import unicodedata

import numpy as np
import pandas as pd

FEATURE_VERSION = "parsed-v1"
MISSING = "__missing__"
INPUT_COLUMNS = [
    "id", "seller_id", "title", "description", "declared_category", "brand",
    "condition", "price_pln", "shipping_weight_kg", "pack_size",
]
NUMERIC_COLUMNS = ["price_pln", "shipping_weight_kg", "pack_size"]
CATEGORICAL_FEATURES = ["declared_category", "brand", "condition", "product", "variant"]
WORDS = {"one": 1, "two": 2, "three": 3, "five": 5}
COUNT = r"(\d+|one|two|three|five)"
DESCRIPTION_PATTERNS = [
    rf"included main items:\s*{COUNT}",
    rf"package contains\s+{COUNT}\s+pcs",
    rf"bundle quantity:\s*{COUNT}\s+units",
    rf"sold as a\s+{COUNT}-pack",
    rf"pack of\s+{COUNT}\s+main items",
    rf"{COUNT}\s+individually boxed main items",
]
TITLE_PATTERNS = [
    rf"quantity\s+{COUNT}\b", rf"pack of\s+{COUNT}\b",
    rf"\b{COUNT}\s*(?:-pack|pcs|units|main items)\b",
]


def normalize(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value)).lower()).strip()


def extract_quantity(text, description=False):
    """Only main-item contexts qualify; accessory counts are not a fallback."""
    text = normalize(text)
    for pattern in DESCRIPTION_PATTERNS if description else TITLE_PATTERNS:
        match = re.search(pattern, text)
        if match:
            token = match.group(1)
            value = WORDS[token] if token in WORDS else int(token)
            return float(value) if value > 0 else np.nan
    return np.nan


def parsed_features(df):
    x = df[["declared_category", "brand", "condition", *NUMERIC_COLUMNS]].copy()
    title, desc = df.title.map(normalize), df.description.map(normalize)
    # The fixed grammar is from the supplied synthetic dataset. Unfamiliar
    # grammar stays unknown instead of inventing an identity.
    x["product"] = desc.str.extract(
        r"(?:^|\. )([^.]*) by (?:aster|mira|noro|tenso|velto)\.", expand=False
    )
    x["variant"] = desc.str.extract(r"variant:\s*(\w+)", expand=False).astype("string")
    x["variant"] = x.variant.fillna(
        title.str.extract(r"\b(standard|compact|large)\b", expand=False).astype("string")
    )
    x["desc_quantity"] = desc.map(lambda s: extract_quantity(s, description=True))
    x["title_quantity"] = title.map(extract_quantity)
    x["main_quantity"] = x.desc_quantity.fillna(x.title_quantity)
    x["quantity_difference"] = x.pack_size - x.main_quantity
    x["title_desc_quantity_difference"] = x.title_quantity - x.desc_quantity
    x["clearance"] = desc.str.contains("clearance promotion", regex=False).astype(int)
    x["accessory"] = desc.str.contains("free accessory", regex=False).astype(int)
    x["title_length"], x["description_length"] = title.str.len(), desc.str.len()
    for col in NUMERIC_COLUMNS:
        x[col + "_missing"] = df[col].isna().astype(int)
        x["log_" + col] = np.log(df[col])
    for col in ["price_pln", "shipping_weight_kg"]:
        x[col + "_per_declared_item"] = df[col] / df.pack_size
        x[col + "_per_text_item"] = df[col] / x.main_quantity
    for col in CATEGORICAL_FEATURES:
        x[col] = x[col].fillna(MISSING).astype(str)
    return x.replace([np.inf, -np.inf], np.nan)


def model_input(features, columns=None):
    """Use the same sentinel policy as the validated research prototype."""
    x = features.copy()
    if columns is not None:
        absent = set(columns) - set(x.columns)
        if absent:
            raise ValueError(f"Missing model features: {sorted(absent)}")
        x = x.loc[:, columns]
    for col in x:
        if col in CATEGORICAL_FEATURES:
            x[col] = x[col].fillna(MISSING).astype(str)
        else:
            x[col] = x[col].replace([np.inf, -np.inf], np.nan).fillna(-999.)
    return x


def parsing_summary(df, x):
    return {
        "rows": len(df), "sellers": int(df.seller_id.nunique()),
        "missing_fraction": {c: float(df[c].isna().mean()) for c in INPUT_COLUMNS},
        "product_coverage": float(x['product'].ne(MISSING).mean()),
        "product_phrases": int(x.loc[x['product'].ne(MISSING), 'product'].nunique()),
        "main_quantity_coverage": float(x.main_quantity.notna().mean()),
        "quantity_disagreements": int(x.quantity_difference.fillna(0).ne(0).sum()),
        "category_counts": df.declared_category.fillna(MISSING).value_counts().to_dict(),
    }
