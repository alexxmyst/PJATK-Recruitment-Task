# Catalogue quality: research and exploratory experiment

Date: 2026-10-28

## Observations from the supplied data

- The baseline notebook defines target 0 as no controlled corruption and target 1 as one controlled corruption. Target 0 does not mean no natural missingness or no ambiguous catalogue choices.
- All 15,320 non-missing training descriptions match the phrase pattern `product by brand.`. There are 180 distinct product phrases; these include aliases and are not necessarily 180 canonical product types.
- A context-sensitive quantity parser extracts a main-item quantity from title or description for 15,990 of 16,000 rows (99.9375%). Free accessory counts are excluded.
- There are 391 rows with an observed disagreement between parsed main quantity and structured pack size. All 391 are positive in this training dataset. This is a descriptive finding, not a guarantee about future data or all positive rows.
- Category compatibility is probabilistic. Among target-0 rows, `wireless headphones` occurs in audio (88) and computers (23); `gym shaker` occurs in sport (54) and kitchen (20). Expert hard rules would incorrectly flag some legitimate rows.
- Promotion text may explain legitimate numeric variation. Clean-row median prices are PLN 46.69 with the promotion phrase and PLN 80.83 without. This comparison is descriptive and confounded by product mix, not an estimate of a causal discount.

## Main experiment

The experiment compares the supplied baseline with a small structured CatBoost classifier and with the same classifier augmented by learned clean-reference features. The parsed model includes product phrase, variant, main quantity, numeric fields, ratios, quantity differences, and a few text flags. It does not vectorize the raw description.

The reference models use only target-0 rows from the current training partition. Two ridge regressions estimate log price and log shipping weight from product phrase, variant, brand, condition, log main quantity, and a promotion indicator. Their expected values, signed residuals and absolute residuals become features. Smoothed category probabilities conditional on product phrase supply a category surprisal feature.

The category being scored and the observed price/weight being reconstructed are excluded from their own respective reference inputs. Declared pack size is also excluded from the numeric reference inputs because it may be corrupted; quantity extracted from text is used instead.

Validation uses five seller-disjoint folds, assigned by SHA256(seed:seller_id) modulo 5. A further three seller-disjoint folds within each outer training partition create reference features for training the final classifier. This prevents a training row from contributing to its own clean reference. Outer validation labels are used only for scoring. Neither hidden test labels nor test inputs are used by the experiment.

| Model | Mean fold ROC-AUC, seed 42 | OOF Average Precision, seed 42 |
|---|---:|---:|
| Supplied TF-IDF/logistic baseline | 0.707412 | 0.570619 |
| Parsed features + CatBoost | 0.933991 | 0.872393 |
| Parsed + clean-reference features + CatBoost | 0.990303 | 0.978768 |

The first comparison changes both features and classifier. It does not isolate the effect of parsing alone. The second comparison keeps the final CatBoost configuration fixed and adds the reference features.

The second seller-fold seed (73) gives mean ROC-AUC 0.935128 for parsed features and 0.990497 for reconstruction features; OOF Average Precision is 0.872857 and 0.978606 respectively. This is repeat validation on the same dataset, not an independent new test set.

The classifiers use 450 iterations, depth 5, learning rate 0.05, L2 regularization 5, random seed 42, four CPU threads, and no hyperparameter search. Full scores are in `research_seed_42.json` and `research_seed_73.json`.

Run from the workspace:

```powershell
python analysis/catalogue_research_probe.py
python analysis/catalogue_research_probe.py --seed 73 --skip-baseline
```

## Relevant primary sources

1. [PatternRAG: Leveraging Product Catalog Patterns for Multilingual E-commerce Product Attribute Prediction, Amazon, EMNLP 2025](https://www.amazon.science/publications/leveraging-product-catalog-patterns-for-multilingual-e-commerce-product-attribute-prediction). Uses similar listings within the catalogue as attribute reference examples. Relevant to product peer groups, synonym resolution, and missing-attribute inference. Its task and evaluation metrics differ from this competition.
2. [Robust Variational Autoencoders for Outlier Detection and Repair of Mixed-Type Data, AISTATS 2020](https://proceedings.mlr.press/v108/eduardo20a.html). Models individual corrupted cells and a clean data distribution. Motivates conditional reconstruction and error localization. The prototype here uses supervised clean references and ridge regressions rather than an RVAE.
3. [VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain, NeurIPS 2020](https://papers.nips.cc/paper_files/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html). Learns to recover corrupted values and identify corruption masks. Motivates controlled single-field corruption augmentation with known affected fields.
4. [HoloClean: Holistic Data Repairs with Probabilistic Inference, 2017](https://arxiv.org/abs/1702.00820). Combines constraints and statistical evidence for probabilistic repair. Motivates scoring candidate repairs rather than applying universal hard constraints.

## Further experiments

- Resolve synonym pairs into canonical product identities, especially for missing descriptions. Learn alias links within training folds, and preserve unseen-product fallbacks.
- Replace one expected numeric value with conditional prediction intervals or densities. Standardize residuals by product-dependent uncertainty, with shrinkage for small groups.
- Generate realistic single-field corruptions from target-0 training rows. Use real training positives to estimate plausible corruption magnitudes and avoid treating legitimate alternative categories as errors. Generate after seller splitting and evaluate only on unmodified held-out examples.
- Fit soft specialists for category, quantity, price and weight errors. Treat the mapping from a controlled corruption to visible cells as a hypothesis to check, not proof that exactly one observed field is always anomalous.
- Evaluate candidate repair gains: how much does changing one attribute improve the probability of a coherent listing? Preserve original values and use proposed corrections as auxiliary features.
- Use catalogue retrieval or constrained language-model extraction for unresolved aliases and malformed text. Test on unfamiliar templates and held-out product families to assess dependence on the generator's grammar.
- Audit residual false positives/negatives, missing-product cases, and calibration before considering submission.

## Limits

This is an exploratory local comparison, not a Kaggle result or a claim of 99% classification accuracy. The text grammar was examined on the current training dataset, so a future template shift remains an open risk. ROC-AUC measures ranking, and calibration and operational precision require separate evaluation. No catalogue values were repaired, no submission was created, and the supplied competition files were not edited.
