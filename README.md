# Catalogue quality detection

A complete CPU pipeline for the DSC PJATK Catalogue Quality Challenge. It parses the templated listing text, learns expected attributes from rows without controlled corruption, and trains a CatBoost classifier on observed values and deviations from those expectations.

## Run it

Open a terminal in this project folder. Dependencies are already installed in the Python environment used for the local run. For a new environment, Python 3.13 and the pinned requirements reproduce that environment:

```powershell
python -m pip install -r requirements.txt
```

Train, validate, save the ensemble and create a submission:

```powershell
python -m catalogue_quality train --output-dir outputs/catalogue_v1
```

The default data folder is `data/dsc-pjatk-catalogue-quality-challenge`. To use another location:

```powershell
python -m catalogue_quality train --data-dir "C:/path/to/competition" --output-dir outputs/run_002
```

Each output directory must be new or empty. This avoids overwriting models or results from an earlier experiment. A completed run has `manifest.json`; partial runs do not.

Run only the input audit:

```powershell
python -m catalogue_quality audit --output-dir outputs/audit
```

Reload the saved ensemble and predict without retraining:

```powershell
python -m catalogue_quality predict --model-dir outputs/catalogue_v1 --test data/dsc-pjatk-catalogue-quality-challenge/test.csv --sample-submission data/dsc-pjatk-catalogue-quality-challenge/sample_submission.csv --output-dir outputs/reloaded_predictions
```

`submission.csv` contains `id,target`, with probability of controlled corruption in the original test order. It is checked after writing. Uploading to Kaggle is a separate action.

## How the pipeline works

1. Validate schemas, unique listing IDs, positive observed numeric values, binary labels, disjoint train/test sellers and matching submission IDs.
2. Normalize text and parse product phrase, variant, main quantity, promotions and accessory flags. Main quantities exclude free accessories. Extract ratios, log values and quantity differences.
3. Assign entire sellers to five outer folds using `SHA256(fold_seed:seller_id) modulo folds`, matching the supplied baseline's default split.
4. For each outer fold, use three inner seller folds to generate reference features for the classifier's training rows. No row or seller contributes to its own reference model.
5. Fit reference models on target-0 rows only. Two ridge regressions estimate log price and log shipping weight from product, variant, brand, condition, text quantity and promotion status. Product-conditioned category probabilities measure category compatibility.
6. Train CatBoost on original structured attributes, parsed facts and reference features: expected values, signed/absolute numeric deviations and category surprisal. Outer validation labels do not control early stopping or model settings.
7. Score unseen sellers, save each fold's classifier/reference models, and average the five test probabilities.
8. Write metrics, audit, calibration, subgroup diagnostics, largest validation errors and feature importance.

`target=0` means no controlled corruption; it still permits natural missingness and legitimate category alternatives. References therefore learn catalogue conventions rather than asserting that every row is factually perfect.

The default parameters match the exploratory model: 450 trees, depth 5, learning rate 0.05, L2 regularization 5, model seed 42, fold seed 42, four CPU threads. Reference models use ridge alpha 2. Numerical missing values use the validated classifier sentinel `-999`; categorical unknowns use `__missing__`. Unknown products remain unknown instead of receiving an invented identity. Seller and listing IDs are excluded from model features.

## Outputs

| File | Purpose |
|---|---|
| `submission.csv` | Five-fold ensemble probabilities for Kaggle |
| `report.md` / `metrics.json` | Overall and per-fold validation performance |
| `oof.csv` | Each training row's prediction from a model that held out its seller |
| `oof_diagnostics.csv` | Held-out predictions with parsed facts and reference deviations |
| `test_diagnostics.csv` | Test probabilities with observed and expected attributes |
| `review_top_errors.csv` | 100 largest OOF errors with original title/description |
| `slice_metrics.csv` | Performance by category, condition and missingness |
| `calibration.csv` | Mean predictions versus observed positive rates in ten probability bins |
| `feature_importance.csv` | Mean and standard deviation of importance across fold models |
| `audit.json` | Missingness, parser coverage, category distributions and numeric quantiles |
| `config.json` / `manifest.json` | Settings, package versions, feature schema and input/model/source hashes |
| `models/fold_N/` | CatBoost classifier and its fitted reference estimators |

Expected values are estimates, not verified corrections. Keep original values alongside them. Diagnostic ensemble reference values summarize multiple models and are not causal explanations of individual predictions. Metrics at probability 0.5 are illustrative; the official ranking metric is ROC-AUC.

The manifest records checksums and inference verifies the saved model files. Load only locally produced or otherwise trusted bundles: reference estimators use joblib serialization. Keep the training package versions for reliable reloads.

## Experiments and validation

Repeat with a different assignment of sellers:

```powershell
python -m catalogue_quality train --fold-seed 73 --output-dir outputs/seed_73
```

Inspect commands and configurable training settings:

```powershell
python -m catalogue_quality --help
python -m catalogue_quality train --help
```

Install development requirements if necessary, then run the regression checks:

```powershell
python -m pytest tests -q --basetemp .pytest_tmp
```

These checks cover accessory-count traps, exclusion of each seller from its own reference, exclusion of positive rows from reference training, schema/submission contracts, and a complete save/reload/predict round trip.

The research prototype achieved mean ROC-AUC 0.990303 on default folds and 0.990497 on a second fold seed. These are local cross-validation results, not a Kaggle score or classification accuracy. The parser is tailored to the supplied synthetic grammar; unknown future templates and product families need separate evaluation. See `analysis/research_notes.md` for research context and limitations. The output of the completed pipeline run is the authoritative result for this implementation.

The completed `outputs/catalogue_v1` run reproduces mean ROC-AUC **0.990303** and OOF Average Precision **0.978768**. Reloading the saved ensemble produces identical predictions for all 10,000 test rows, and all 12 regression checks pass. The original competition files match the downloaded ZIP byte for byte.

One substantial weak slice remains: the 680 training rows with missing descriptions have OOF ROC-AUC **0.7015**, versus **0.9921** when the description is present. The current product identity extractor relies on description grammar; title-based product recovery would be the next targeted improvement. Review this slice before using the model for an operational catalogue, where missing descriptions may be more common.

## Team walkthrough notebook

`notebooks/catalogue_quality_walkthrough.ipynb` presents the core pipeline step by step, with simpler code, worked examples, charts and executed training outputs. `notebooks/catalogue_quality_walkthrough.html` is the shareable browser version. See `notebooks/README.md` for setup and rerunning instructions; additional dependencies are listed in `requirements-notebook.txt`.

## Project layout

```text
catalogue_quality/
  data.py         Input and submission validation
  features.py     Stateless text and numeric extraction
  reference.py    Clean-reference fitting and inner cross-fitting
  pipeline.py     Training, model bundles, reports and inference
  cli.py          Command line interface
tests/            Leakage, parsing, data and round-trip checks
analysis/         Research prototype, references and earlier results
notebooks/        Executed team walkthrough and HTML report
data/             Original competition files
outputs/          Generated runs and model bundles
```
