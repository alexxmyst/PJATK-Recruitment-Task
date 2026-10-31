"""Exploratory, seller-disjoint comparison for the catalogue research discussion.

No submissions are created. Clean-reference features are cross-fitted within
each outer training fold so training rows cannot supply their own reference.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/dsc-pjatk-catalogue-quality-challenge"
WORDS = {"one": 1, "two": 2, "three": 3, "five": 5}
COUNT = r"(\d+|one|two|three|five)"


def normal(s):
    return re.sub(r"\s+", " ", str(s).lower()).strip() if pd.notna(s) else ""


def quantity(s, description=False):
    # Contexts refer specifically to main items, excluding free accessories.
    patterns = [
        rf"included main items:\s*{COUNT}",
        rf"package contains\s+{COUNT}\s+pcs",
        rf"bundle quantity:\s*{COUNT}\s+units",
        rf"sold as a\s+{COUNT}-pack",
        rf"pack of\s+{COUNT}\s+main items",
        rf"{COUNT}\s+individually boxed main items",
    ] if description else [
        rf"quantity\s+{COUNT}\b", rf"pack of\s+{COUNT}\b",
        rf"\b{COUNT}\s*(?:-pack|pcs|units|main items)\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, s)
        if m:
            value = m.group(1)
            return WORDS[value] if value in WORDS else int(value)
    return np.nan


def parsed_features(df):
    x = df[["declared_category", "brand", "condition", "price_pln",
            "shipping_weight_kg", "pack_size"]].copy()
    title = df.title.map(normal)
    desc = df.description.map(normal)
    x["product"] = desc.str.extract(
        r"(?:^|\. )([^.]*) by (?:aster|mira|noro|tenso|velto)\.", expand=False
    )
    x["variant"] = desc.str.extract(r"variant:\s*(\w+)", expand=False)
    x["variant"] = x.variant.fillna(title.str.extract(r"\b(standard|compact|large)\b", expand=False))
    x["desc_quantity"] = desc.map(lambda s: quantity(s, True))
    x["title_quantity"] = title.map(quantity)
    x["main_quantity"] = x.desc_quantity.fillna(x.title_quantity)
    x["quantity_difference"] = x.pack_size - x.main_quantity
    x["title_desc_quantity_difference"] = x.title_quantity - x.desc_quantity
    x["clearance"] = desc.str.contains("clearance promotion", regex=False).astype(int)
    x["accessory"] = desc.str.contains("free accessory", regex=False).astype(int)
    x["title_length"] = title.str.len()
    x["description_length"] = desc.str.len()
    for c in ["price_pln", "shipping_weight_kg", "pack_size"]:
        x[c + "_missing"] = df[c].isna().astype(int)
        x["log_" + c] = np.log(df[c])
    for c in ["price_pln", "shipping_weight_kg"]:
        x[c + "_per_declared_item"] = df[c] / df.pack_size
        x[c + "_per_text_item"] = df[c] / x.main_quantity
    for c in x.select_dtypes("object"):
        x[c] = x[c].fillna("__missing__")
    return x


REF_CATS = ["product", "variant", "brand", "condition"]


def reference_input(x):
    # Exclude declared category and declared pack: they could themselves be wrong.
    r = x[REF_CATS + ["clearance"]].copy()
    r["log_main_quantity"] = np.log(x.main_quantity.fillna(1))
    return r


def fit_reference(x, y):
    clean = x.loc[y == 0]
    regressors = {}
    for col in ["price_pln", "shipping_weight_kg"]:
        ref = clean[clean[col].notna()]
        prep = ColumnTransformer([
            ("cats", OneHotEncoder(handle_unknown="ignore"), REF_CATS),
            ("nums", "passthrough", ["clearance", "log_main_quantity"]),
        ])
        model = make_pipeline(prep, Ridge(alpha=2, solver="lsqr"))
        model.fit(reference_input(ref), np.log(ref[col]))
        regressors[col] = model
    known = clean[(clean["product"] != "__missing__") & (clean.declared_category != "__missing__")]
    counts = pd.crosstab(known["product"], known.declared_category)
    return regressors, counts


def reference_features(x, reference):
    regressors, counts = reference
    f = pd.DataFrame(index=x.index)
    for col, model in regressors.items():
        expected = model.predict(reference_input(x))
        residual = np.log(x[col]) - expected
        f[col + "_expected_log"] = expected
        f[col + "_residual"] = residual
        f[col + "_abs_residual"] = np.abs(residual)
    probabilities = []
    for product, category in zip(x["product"], x.declared_category):
        if product not in counts.index or category == "__missing__":
            probabilities.append(np.nan)
        else:
            row = counts.loc[product]
            probabilities.append((row.get(category, 0) + 1) / (row.sum() + len(counts.columns)))
    f["category_surprisal"] = -np.log(probabilities)
    return f


def make_classifier():
    return CatBoostClassifier(
        iterations=450, depth=5, learning_rate=0.05, l2_leaf_reg=5,
        loss_function="Logloss", random_seed=42, thread_count=4,
        verbose=False, allow_writing_files=False,
    )


def model_prediction(xfit, yfit, xvalid):
    cat = xfit.select_dtypes("object").columns.to_list()
    model = make_classifier()
    model.fit(xfit.fillna({c: -999. for c in xfit if c not in cat}), yfit, cat_features=cat)
    return model.predict_proba(xvalid.fillna({c: -999. for c in xvalid if c not in cat}))[:, 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()
    train = pd.read_csv(DATA / "train.csv")
    x = parsed_features(train)
    y = train.target
    assignment = train.seller_id.map(
        lambda s: int(hashlib.sha256(f"{args.seed}:{s}".encode()).hexdigest()[:16], 16) % 5
    )
    report = {"fold_seed": args.seed,
              "reference_training": "target=0 rows in training partition only; 3 inner seller folds",
              "description_product_coverage": float((x['product'] != '__missing__').mean()),
              "product_phrases": int(x.loc[x['product'] != '__missing__', 'product'].nunique()),
              "main_quantity_coverage": float(x.main_quantity.notna().mean())}
    print(json.dumps(report), flush=True)
    names = ["parsed", "reconstruction"] if args.skip_baseline else ["provided_baseline", "parsed", "reconstruction"]
    predictions = {name: np.zeros(len(x)) for name in names}
    scores = {name: [] for name in predictions}
    nb = json.loads((DATA / "baseline_solution.ipynb").read_text(encoding="utf-8"))
    ns = {}
    exec("".join(nb["cells"][1]["source"]), ns)
    baseline_x = ns["features"](train.drop(columns="target"))
    for fold in range(5):
        fit = np.flatnonzero(assignment != fold)
        valid = np.flatnonzero(assignment == fold)
        fitx, validx = x.iloc[fit], x.iloc[valid]
        fity = y.iloc[fit]
        assert set(train.seller_id.iloc[fit]).isdisjoint(train.seller_id.iloc[valid])
        if not args.skip_baseline:
            baseline = ns["make_model"]()
            baseline.fit(baseline_x.iloc[fit], fity)
            predictions["provided_baseline"][valid] = baseline.predict_proba(baseline_x.iloc[valid])[:, 1]
        predictions["parsed"][valid] = model_prediction(fitx, fity, validx)
        inner_features = []
        for ifit, ivalid in GroupKFold(3).split(fitx, fity, train.seller_id.iloc[fit]):
            reference = fit_reference(fitx.iloc[ifit], fity.iloc[ifit])
            inner_features.append(reference_features(fitx.iloc[ivalid], reference))
        fitref = pd.concat(inner_features).loc[fitx.index]
        validref = reference_features(validx, fit_reference(fitx, fity))
        predictions["reconstruction"][valid] = model_prediction(
            pd.concat([fitx, fitref], axis=1), fity, pd.concat([validx, validref], axis=1)
        )
        result = {}
        for name in predictions:
            score = roc_auc_score(y.iloc[valid], predictions[name][valid])
            scores[name].append(score)
            result[name] = round(score, 6)
        print(json.dumps({"fold": fold + 1, "auc": result}), flush=True)
    for name, p in predictions.items():
        report[name] = {"fold_auc": scores[name], "mean_auc": float(np.mean(scores[name])),
                        "std_auc": float(np.std(scores[name])), "oof_auc": float(roc_auc_score(y, p)),
                        "oof_ap": float(average_precision_score(y, p))}
    # Descriptive evidence, not a validation metric: compatible clean categories.
    clean = train.assign(product=x["product"])[y == 0]
    counts = pd.crosstab(clean["product"], clean.declared_category)
    report["clean_wireless_headphone_categories"] = {
        p: counts.loc[p][counts.loc[p] > 0].to_dict()
        for p in ["wireless headphones", "over-ear bluetooth headset"]
        if p in counts.index
    }
    report_path = ROOT / "analysis" / f"research_seed_{args.seed}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
