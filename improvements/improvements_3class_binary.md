# What to do better: 3-class (benign / atypical / malignant) and cancer vs not

Context: results of `../02_0_vectors_to_classifier_selector.ipynb` on the official BRACS Testing set (87 slides). The tiers are currently obtained by summing the 7-class probabilities; no model is trained directly on 3 classes or 2 classes.

| | ABMIL | CLAM-MB |
|---|---|---|
| Tier accuracy | 0.736 | 0.724 |
| Tier macro F1 | 0.658 | 0.634 |
| Cancer AUROC | 0.969 | 0.966 |
| Cancer sensitivity / specificity at 0.5 | 0.875 / 0.927 | 0.938 / 0.909 |

The Testing counts implied by these rates are 32 cancer and 55 non-cancer slides (inferred, please check). The two models' sensitivities then differ by 2 slides.

## 1. Train dedicated 3-class and 2-class models

- Collapse the labels (benign = N, PB, UDH; atypical = FEA, ADH; malignant = DCIS, IC; cancer = DCIS, IC) and run the same pipeline: patient-grouped folds, early stopping on separate slides, fold ensemble.
- Compare against the summed 7-class probabilities on the out-of-fold predictions with a paired patient bootstrap (`paired_bootstrap`).
- Summing 7-class probabilities is not optimised for the tier decision, so a dedicated model is likely but not guaranteed to be better.
- Cost is low: the final fold models took about 2 to 3 minutes per architecture.

## 2. Choose the cancer threshold on the out-of-fold predictions

- Do not use 0.5. Class weighting and prevalence shift the probabilities.
- The pool has 459 out-of-fold slides against 87 test slides, so a threshold chosen there is far more stable.
- Fix the threshold before opening the Testing set, then report sensitivity and specificity with intervals (patient bootstrap or Wilson), plus PPV and NPV.
- BRACS is enriched for atypia and cancer, so PPV and NPV do not transfer to a screening population; say so.
- Rough 95% intervals for sensitivity at 0.5 are about 0.72 to 0.95 (ABMIL, 28/32) and 0.80 to 0.98 (CLAM-MB, 30/32).

## 3. Report per-tier recall and F1

- Atypical is the weak link. The BRIGHT baseline scores F1 0.43 on precancerous against 0.74 on cancerous; expect the same pattern.
- The notebook prints only the pooled tier measures. Add per-tier recall, precision and F1 with intervals, and the 3x3 confusion matrix with counts.

## 4. Reduce tuning noise

- Several selections in sections 5.2 and 5.3 are decided by differences of 0.001 to 0.01 macro F1, while the spread over folds is 0.05 to 0.08.
- Use a smaller grid, repeat the cross-validation (for example 3 x 5), and prefer the simpler setting when scores are within one standard error.
- Do not present the selected values as findings.

## 5. Add a baseline and a stronger training recipe

- Add `'linear'` (mean-pool) to `MODELS` to show what attention adds.
- Try class-balanced batch sampling (used by nnMIL) as an alternative to loss weights.
- Keep the fold ensemble; it is already part of the pipeline.

## 6. Try the ROI annotations (untested idea)

- BRACS has ROI-level labels, and the BRIGHT challenge allowed training on them.
- One BRIGHT entry used ROI data to train a patch-selection model and placed 4th.
- I have not seen this benchmarked with foundation-model features. It could help most for the atypical tier. Treat it as an experiment, not a documented gain.

## Before trusting any of this

- Scan year is associated with class (Cramer's V about 0.27) and two MPP values exist; check for a batch effect first.
- The Testing set has 87 slides, so most differences between models are within the noise. Decide the headline model and metric before looking at it.
