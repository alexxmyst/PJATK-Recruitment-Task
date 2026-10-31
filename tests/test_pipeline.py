"""Checks for leakage boundaries, parsing traps, and saved-model inference."""

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from catalogue_quality.data import (
    check_competition_split, read_sample, validate_frame, write_submission,
)
from catalogue_quality.features import extract_quantity, parsed_features
from catalogue_quality.pipeline import TrainingConfig, load_ensemble, predict_frame, train_pipeline
from catalogue_quality.reference import crossfit_reference_features, fit_reference, reference_features


@pytest.mark.parametrize("text,count", [
    ("Also includes 5 pcs of free accessory pieces. Included main items: 2.", 2),
    ("Variant: large. Pack of one main items. Also includes 3 pcs of free accessory pieces.", 1),
    ("Three individually boxed main items.", 3),
    ("Sold as a 5-pack.", 5),
    ("Bundle quantity: 2 units.", 2),
    ("Package contains 3 pcs.", 3),
])
def test_main_item_context_beats_accessory_counts(text, count):
    assert extract_quantity(text, description=True) == count


def test_accessories_alone_do_not_define_pack_size():
    assert np.isnan(extract_quantity("Also includes 3 pcs of free accessory pieces.", description=True))


def example_rows(sellers=9, rows_per_seller=8):
    records = []
    for seller in range(sellers):
        for row in range(rows_per_seller):
            corrupt = int(row % 4 == 0)
            records.append({
                "id": f"offer_{seller}_{row}", "seller_id": f"seller_{seller}",
                "title": f"Noro item{seller} / standard / quantity 1",
                "description": f"Item{seller} by Noro. Variant: standard. Included main items: 1.",
                "declared_category": "computers", "brand": "Noro", "condition": "new",
                "price_pln": float(50 + seller + row / 10), "shipping_weight_kg": .4 + seller / 100,
                "pack_size": 3. if corrupt else 1., "target": corrupt,
            })
    return pd.DataFrame(records)


def test_inner_reference_cannot_see_its_own_seller_product():
    raw = example_rows(sellers=6)
    x = parsed_features(raw)
    # Each seller has a unique product. Its category reference must therefore
    # be unknown when the seller is genuinely excluded by the inner split.
    crossfit = crossfit_reference_features(x, raw.target, raw.seller_id, n_splits=3)
    assert crossfit.category_surprisal.isna().all()
    in_sample = reference_features(x, fit_reference(x, raw.target))
    assert in_sample.category_surprisal.notna().all()
    assert set(["target", "id", "seller_id"]).isdisjoint(x.columns)


def test_positive_values_cannot_influence_clean_reference():
    raw = example_rows(sellers=6)
    modified = raw.copy()
    modified.loc[modified.target == 1, "price_pln"] *= 10000
    x, changed = parsed_features(raw), parsed_features(modified)
    first = reference_features(x, fit_reference(x, raw.target))
    second = reference_features(x, fit_reference(changed, raw.target))
    np.testing.assert_allclose(first.price_pln_expected_log, second.price_pln_expected_log)


def test_invalid_values_and_seller_leakage_are_rejected():
    raw = example_rows()
    bad = raw.copy()
    bad.loc[0, "price_pln"] = -1
    with pytest.raises(ValueError, match="positive finite"):
        validate_frame(bad, training=True)
    with pytest.raises(ValueError, match="seller IDs overlap"):
        check_competition_split(raw, raw.assign(id="new_" + raw.id))


def test_submission_checks_id_order_and_probability_range(tmp_path):
    test = example_rows().drop(columns="target")
    sample = pd.DataFrame({"id": test.id.iloc[::-1], "target": .5})
    path = tmp_path / "sample.csv"
    sample.to_csv(path, index=False)
    with pytest.raises(ValueError, match="ordering"):
        read_sample(path, test)
    with pytest.raises(ValueError, match="finite probabilities"):
        write_submission(test, np.full(len(test), np.nan), tmp_path / "submission.csv")


def test_full_training_and_reload_match_predictions(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    train = example_rows(sellers=30)
    test = example_rows(sellers=2).drop(columns="target")
    test["id"], test["seller_id"] = "test_" + test.id, "test_" + test.seller_id
    train.to_csv(data / "train.csv", index=False)
    test.to_csv(data / "test.csv", index=False)
    pd.DataFrame({"id": test.id, "target": .5}).to_csv(data / "sample_submission.csv", index=False)
    output = tmp_path / "run"
    config = replace(TrainingConfig(), folds=3, inner_folds=2, iterations=25, threads=1)
    train_pipeline(data, output, config)
    saved = pd.read_csv(output / "submission.csv")
    prediction, diagnostics = predict_frame(test, output)
    np.testing.assert_allclose(prediction, saved.target, atol=1e-14)
    assert diagnostics.id.tolist() == test.id.tolist()
    assert (output / "review_top_errors.csv").exists()
    oof = pd.read_csv(output / "oof.csv")
    assert oof.groupby("seller_id").fold.nunique().eq(1).all()
    # Unknown text/products/categories must still have finite predictions.
    unfamiliar = test.iloc[:2].copy()
    unfamiliar.loc[:, ["description", "title", "brand", "declared_category"]] = [
        "Unfamiliar product wording.", "Unknown item", "NeverSeenBrand", "NeverSeenCategory"
    ]
    p, _ = predict_frame(unfamiliar, output)
    assert np.isfinite(p).all()
    with pytest.raises(ValueError, match="new or empty"):
        train_pipeline(data, output, config)
    # Corrupt bundles are rejected before deserializing the affected model.
    manifest = json.loads((output / "manifest.json").read_text())
    artifact = output / manifest["fold_models"][0]["reference"]
    artifact.write_bytes(artifact.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Invalid model artifact"):
        load_ensemble(output)
