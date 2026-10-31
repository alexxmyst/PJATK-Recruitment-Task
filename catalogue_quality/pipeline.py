"""Training, persistence, diagnostics, and inference for the fold ensemble."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

from catboost import CatBoostClassifier
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    log_loss, precision_score, recall_score, roc_auc_score,
)

from . import __version__
from .data import check_competition_split, read_listings, read_sample, write_submission
from .features import (
    CATEGORICAL_FEATURES, FEATURE_VERSION, MISSING, model_input,
    parsed_features, parsing_summary,
)
from .reference import crossfit_reference_features, fit_reference, reference_features


@dataclass(frozen=True)
class TrainingConfig:
    folds: int = 5
    inner_folds: int = 3
    fold_seed: int = 42
    model_seed: int = 42
    iterations: int = 450
    depth: int = 5
    learning_rate: float = 0.05
    l2_leaf_reg: float = 5.
    threads: int = 4

    def validate(self):
        if self.folds < 2 or self.inner_folds < 2:
            raise ValueError("Outer and inner fold counts must each be at least 2")
        if self.iterations < 1 or self.depth < 1 or self.threads < 1:
            raise ValueError("Iterations, depth and threads must be positive")
        if self.learning_rate <= 0 or self.l2_leaf_reg < 0:
            raise ValueError("Learning rate must be positive and regularization nonnegative")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment():
    return {"python": platform.python_version(), **{
        name: version(name) for name in ["numpy", "pandas", "scikit-learn", "catboost", "scipy", "joblib"]
    }}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False), encoding="utf-8")


def prepare_output(path):
    path = Path(path).resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def seller_assignment(sellers, n_splits=5, seed=42):
    return np.array([
        int(hashlib.sha256(f"{seed}:{s}".encode("utf-8")).hexdigest()[:16], 16) % n_splits
        for s in sellers
    ])


def build_audit(train, test, train_x=None, test_x=None):
    train_x = parsed_features(train) if train_x is None else train_x
    test_x = parsed_features(test) if test_x is None else test_x
    audit = {
        "train": parsing_summary(train, train_x), "test": parsing_summary(test, test_x),
        "positive_count": int(train.target.sum()), "positive_fraction": float(train.target.mean()),
        "seller_overlap": len(set(train.seller_id) & set(test.seller_id)),
        "id_overlap": len(set(train.id) & set(test.id)),
        "test_only_product_phrases": sorted(set(test_x['product']) - set(train_x['product'])),
        "numeric_quantiles": {},
    }
    for name, frame in [("train", train), ("test", test)]:
        audit["numeric_quantiles"][name] = frame[
            ["price_pln", "shipping_weight_kg", "pack_size"]
        ].quantile([0, .01, .5, .99, 1]).to_dict()
    return audit


def load_competition(data_dir):
    root = Path(data_dir)
    train = read_listings(root / "train.csv", training=True)
    test = read_listings(root / "test.csv")
    check_competition_split(train, test)
    read_sample(root / "sample_submission.csv", test)
    return train, test


def audit_data(data_dir, output_dir):
    train, test = load_competition(data_dir)
    report = build_audit(train, test)
    output = prepare_output(output_dir)
    write_json(output / "audit.json", report)
    return report


def score_predictions(y, probability):
    y, p = np.asarray(y), np.asarray(probability)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Metrics received invalid probabilities")
    both_classes = len(np.unique(y)) == 2
    labels = p >= .5
    return {
        "rows": len(y), "positives": int(y.sum()), "positive_fraction": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)) if both_classes else None,
        "average_precision": float(average_precision_score(y, p)) if both_classes else None,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, p)),
        "precision_at_0_5": float(precision_score(y, labels, zero_division=0)),
        "recall_at_0_5": float(recall_score(y, labels, zero_division=0)),
        "confusion_matrix_at_0_5": confusion_matrix(y, labels, labels=[0, 1]).tolist(),
    }


def calibration_table(y, probability):
    p = np.asarray(probability)
    frame = pd.DataFrame({"target": np.asarray(y), "prediction": p,
                          "bin": np.minimum((p * 10).astype(int), 9)})
    return frame.groupby("bin").agg(
        rows=("target", "size"), mean_prediction=("prediction", "mean"),
        observed_fraction=("target", "mean"),
    ).reset_index()


def diagnostic_frame(raw, x, ref, probability):
    columns = ["declared_category", "brand", "condition", "product", "variant",
               "price_pln", "shipping_weight_kg", "pack_size", "main_quantity",
               "desc_quantity", "title_quantity", "quantity_difference", "clearance"]
    result = pd.concat([raw[["id", "seller_id"]], x[columns], ref], axis=1).copy()
    result["prediction"] = probability
    result["expected_price_pln"] = np.exp(ref.price_pln_expected_log)
    result["expected_weight_kg"] = np.exp(ref.shipping_weight_kg_expected_log)
    if "target" in raw:
        result["target"] = raw.target
    return result


def slice_report(train, x, probability):
    dimensions = {
        "declared_category": train.declared_category.fillna(MISSING),
        "condition": train.condition.fillna(MISSING),
        "product_parsed": x['product'].ne(MISSING),
        "description_missing": train.description.isna(),
        "price_missing": train.price_pln.isna(),
        "weight_missing": train.shipping_weight_kg.isna(),
        "pack_missing": train.pack_size.isna(),
    }
    rows = []
    for dimension, values in dimensions.items():
        for value in values.unique():
            mask = values.eq(value).to_numpy()
            scores = score_predictions(train.target.to_numpy()[mask], np.asarray(probability)[mask])
            rows.append({"dimension": dimension, "value": str(value), **{
                k: scores[k] for k in ["rows", "positives", "roc_auc", "average_precision", "brier_score"]
            }})
    return pd.DataFrame(rows)


def train_pipeline(data_dir, output_dir, config=None):
    config = config or TrainingConfig()
    config.validate()
    started = time.perf_counter()
    train, test = load_competition(data_dir)
    x, xt = parsed_features(train), parsed_features(test)
    assignment = seller_assignment(train.seller_id, config.folds, config.fold_seed)
    y = train.target.to_numpy()
    for fold in range(config.folds):
        fit, valid = assignment != fold, assignment == fold
        if not fit.any() or not valid.any() or len(np.unique(y[valid])) < 2:
            raise ValueError(f"Fold {fold + 1} is empty or lacks a class; change the fold seed/count")
        if train.loc[fit, 'seller_id'].nunique() < config.inner_folds:
            raise ValueError("Too few training sellers for the requested inner folds")
    output = prepare_output(output_dir)
    (output / "models").mkdir()
    audit = build_audit(train, test, x, xt)
    write_json(output / "audit.json", audit)
    write_json(output / "config.json", asdict(config))
    oof, test_probability = np.full(len(train), np.nan), np.zeros(len(test))
    oof_ref, test_refs, fold_reports, importances = [], [], [], []
    fold_artifacts, feature_columns = [], None
    for fold in range(config.folds):
        fit, valid = np.flatnonzero(assignment != fold), np.flatnonzero(assignment == fold)
        if set(train.seller_id.iloc[fit]) & set(train.seller_id.iloc[valid]):
            raise RuntimeError("Outer seller leakage")
        fitx, validx = x.iloc[fit], x.iloc[valid]
        print(f"Fold {fold + 1}/{config.folds}: cross-fitting reference models...", flush=True)
        fitref = crossfit_reference_features(
            fitx, y[fit], train.seller_id.iloc[fit], config.inner_folds
        )
        reference = fit_reference(fitx, y[fit])
        validref, testref = reference_features(validx, reference), reference_features(xt, reference)
        training_features = model_input(pd.concat([fitx, fitref], axis=1))
        feature_columns = training_features.columns.to_list()
        classifier = CatBoostClassifier(
            iterations=config.iterations, depth=config.depth, learning_rate=config.learning_rate,
            l2_leaf_reg=config.l2_leaf_reg, loss_function="Logloss", random_seed=config.model_seed,
            thread_count=config.threads, verbose=False, allow_writing_files=False,
        )
        # No outer validation set is passed for early stopping or parameter selection.
        classifier.fit(training_features, y[fit], cat_features=CATEGORICAL_FEATURES)
        oof[valid] = classifier.predict_proba(
            model_input(pd.concat([validx, validref], axis=1), feature_columns)
        )[:, 1]
        test_probability += classifier.predict_proba(
            model_input(pd.concat([xt, testref], axis=1), feature_columns)
        )[:, 1] / config.folds
        metrics = score_predictions(y[valid], oof[valid])
        fold_reports.append({"fold": fold + 1, "training_rows": len(fit),
                             "training_sellers": int(train.seller_id.iloc[fit].nunique()),
                             "validation_sellers": int(train.seller_id.iloc[valid].nunique()),
                             "reference_clean_rows": reference["clean_rows"], **metrics})
        print(f"  ROC-AUC={metrics['roc_auc']:.6f}; AP={metrics['average_precision']:.6f}", flush=True)
        folder = output / "models" / f"fold_{fold + 1}"
        folder.mkdir()
        classifier.save_model(str(folder / "classifier.cbm"))
        joblib.dump(reference, folder / "reference.joblib")
        fold_artifacts.append({"fold": fold + 1,
                               "classifier": str((folder / "classifier.cbm").relative_to(output).as_posix()),
                               "reference": str((folder / "reference.joblib").relative_to(output).as_posix())})
        oof_ref.append(validref)
        test_refs.append(testref)
        importances.append(classifier.get_feature_importance())
    all_ref = pd.concat(oof_ref).loc[x.index]
    # Reference values are auxiliary ensemble summaries, not an attribution of
    # which feature caused a prediction. NaN-aware means preserve unknowns.
    mean_test_ref = pd.concat(test_refs).groupby(level=0).mean().loc[xt.index]
    oof_frame = pd.DataFrame({"id": train.id, "seller_id": train.seller_id, "target": y,
                              "prediction": oof, "fold": assignment + 1})
    oof_frame.to_csv(output / "oof.csv", index=False)
    oof_diagnostics = diagnostic_frame(train, x, all_ref, oof)
    oof_diagnostics["fold"] = assignment + 1
    oof_diagnostics.to_csv(output / "oof_diagnostics.csv", index=False)
    test_diagnostics = diagnostic_frame(test, xt, mean_test_ref, test_probability)
    test_diagnostics.to_csv(output / "test_diagnostics.csv", index=False)
    errors = oof_diagnostics.assign(error_score=np.where(y == 1, 1 - oof, oof))
    errors = errors.sort_values("error_score", ascending=False).head(100)
    errors = errors.merge(train[["id", "title", "description"]], on="id", how="left", validate="one_to_one")
    errors.to_csv(output / "review_top_errors.csv", index=False)
    slice_report(train, x, oof).to_csv(output / "slice_metrics.csv", index=False)
    calibration_table(y, oof).to_csv(output / "calibration.csv", index=False)
    pd.DataFrame({"feature": feature_columns, "mean_importance": np.mean(importances, axis=0),
                  "std_importance": np.std(importances, axis=0)}).sort_values(
        "mean_importance", ascending=False
    ).to_csv(output / "feature_importance.csv", index=False)
    write_submission(test, test_probability, output / "submission.csv")
    scores = [f["roc_auc"] for f in fold_reports]
    metrics = {"folds": fold_reports, "mean_fold_roc_auc": float(np.mean(scores)),
               "std_fold_roc_auc": float(np.std(scores)), "oof": score_predictions(y, oof),
               "elapsed_seconds": time.perf_counter() - started,
               "note": "Local CV; 0.5 threshold is diagnostic only, not optimized. No Kaggle score."}
    write_json(output / "metrics.json", metrics)
    manifest = {
        "status": "complete", "pipeline_version": __version__, "feature_version": FEATURE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "config": asdict(config),
        "environment": environment(), "feature_columns": feature_columns,
        "categorical_features": CATEGORICAL_FEATURES, "fold_models": fold_artifacts,
        "input_sha256": {name: sha256_file(Path(data_dir) / name) for name in
                          ["train.csv", "test.csv", "sample_submission.csv"]},
        "source_sha256": {p.name: sha256_file(p) for p in Path(__file__).parent.glob("*.py")},
        "fold_assignment_sha256": hashlib.sha256(oof_frame[["id", "fold"]].to_csv(
            index=False, lineterminator="\n").encode("utf-8")).hexdigest(),
        "test_prediction_summary": pd.Series(test_probability).describe().to_dict(),
    }
    for artifact in fold_artifacts:
        artifact["classifier_sha256"] = sha256_file(output / artifact["classifier"])
        artifact["reference_sha256"] = sha256_file(output / artifact["reference"])
    write_training_summary(output, metrics, audit)
    # Written last: incomplete runs cannot be mistaken for usable ensembles.
    write_json(output / "manifest.json", manifest)
    return metrics


def write_training_summary(output, metrics, audit):
    lines = ["# Catalogue training report", "",
             f"Mean seller-fold ROC-AUC: {metrics['mean_fold_roc_auc']:.6f}",
             f"Fold standard deviation: {metrics['std_fold_roc_auc']:.6f}",
             f"OOF Average Precision: {metrics['oof']['average_precision']:.6f}", "",
             "| Fold | Validation rows | ROC-AUC | Average Precision |", "|---|---:|---:|---:|"]
    for f in metrics["folds"]:
        lines.append(f"| {f['fold']} | {f['rows']} | {f['roc_auc']:.6f} | {f['average_precision']:.6f} |")
    lines += ["", "## Data and validation", "",
              f"Training rows: {audit['train']['rows']}; test rows: {audit['test']['rows']}.",
              "Outer folds hold out entire sellers. Three inner folds (by default) generate reference features for training rows.",
              "Reference estimators use target-0 rows from their training partition only.",
              "ROC-AUC measures ranking; it is not classification accuracy. This report contains local validation results.",
              "", "## Artifacts", "",
              "- submission.csv: mean probability from all outer-fold models, in test ID order.",
              "- oof.csv and oof_diagnostics.csv: predictions and reference features for held-out sellers.",
              "- test_diagnostics.csv: observed attributes, ensemble reference summaries and predictions.",
              "- review_top_errors.csv: 100 largest OOF errors, with original text for inspection.",
              "- slice_metrics.csv and calibration.csv: subgroup quality and probability reliability.",
              "- feature_importance.csv: mean and variability of CatBoost feature importance.",
              "- models/ and manifest.json: complete reloadable ensemble and provenance.", "",
              "Expected values are model estimates, not verified repairs. Original values should be preserved.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def load_ensemble(model_dir):
    root = Path(model_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Model bundle is incomplete or has an incompatible feature version")
    folds = manifest.get("fold_models", [])
    if len(folds) != manifest["config"]["folds"] or not folds:
        raise ValueError("Model bundle is missing folds")
    models = []
    for fold in folds:
        resolved = {}
        for kind in ["classifier", "reference"]:
            path = (root / fold[kind]).resolve()
            if not path.is_relative_to(root) or sha256_file(path) != fold[kind + "_sha256"]:
                raise ValueError(f"Invalid model artifact: {kind} for fold {fold['fold']}")
            resolved[kind] = path
        classifier = CatBoostClassifier(thread_count=manifest["config"]["threads"])
        classifier.load_model(str(resolved["classifier"]))
        # joblib is for trusted, locally produced bundles only.
        reference = joblib.load(resolved["reference"])
        models.append((classifier, reference))
    return manifest, models


def predict_frame(raw, model_dir):
    manifest, models = load_ensemble(model_dir)
    x = parsed_features(raw)
    probability, references = np.zeros(len(raw)), []
    for classifier, reference in models:
        ref = reference_features(x, reference)
        values = model_input(pd.concat([x, ref], axis=1), manifest["feature_columns"])
        probability += classifier.predict_proba(values)[:, 1] / len(models)
        references.append(ref)
    mean_ref = pd.concat(references).groupby(level=0).mean().loc[x.index]
    return probability, diagnostic_frame(raw, x, mean_ref, probability)


def predict_pipeline(model_dir, test_path, output_dir, sample_path=None):
    raw = read_listings(test_path)
    if sample_path:
        read_sample(sample_path, raw)
    probability, diagnostics = predict_frame(raw, model_dir)
    output = prepare_output(output_dir)
    write_submission(raw, probability, output / "submission.csv")
    diagnostics.to_csv(output / "test_diagnostics.csv", index=False)
    write_json(output / "prediction_report.json", {
        "rows": len(raw), "input_sha256": sha256_file(test_path),
        "model_manifest_sha256": sha256_file(Path(model_dir) / "manifest.json"),
        "environment": environment(), "probabilities": pd.Series(probability).describe().to_dict(),
        "parsing": parsing_summary(raw, parsed_features(raw)),
    })
    return probability
