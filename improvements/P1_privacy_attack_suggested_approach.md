# Paper 1 — Suggested Privacy Attack Evaluation Approach

## Objective

Evaluate whether exporting approximately **2,000 CLS embedding vectors per whole-slide image (WSI)**, one per tissue tile and **without patient identifiers, clinical metadata, or raw images**, creates a *practically feasible route to patient re-identification*.

The aim is **not** to prove that all medical or tissue information is unrecoverable. The aim is to characterize re-identification and linkability under realistically available auxiliary data, computational resources, time, and techniques. GDPR Recital 26 and Brazil's LGPD Article 12 motivate this context-dependent assessment; experimental results alone do not establish legal anonymization.

This refines the P1 objective in the *StreamEmbed PhD Thesis Proposal* (pp. 2–3), which proposes measuring privacy–utility trade-offs for embeddings under active attacks.

## Data and attack surface

- **Released object:** a slide-level collection of ~2,000 CLS vectors, each derived from a tissue tile.
- **Not released:** raw tiles/WSIs, patient names or IDs, clinical metadata, and slide annotations.
- **Assumption to document:** whether vectors are ordered or include tile coordinates, slide/group identifiers, or encoder version. Any such information changes the attack surface.
- **Evaluation ground truth:** patient and slide identifiers are retained **only within the controlled evaluation environment** to measure attacks; they are never supplied in the exported representation.
- **Attacker capabilities:** evaluate at least (a) released vectors alone and (b) released vectors plus a realistically obtainable reference image gallery; document whether the encoder architecture/weights are public.

## Prioritized attack tests

### A. Reference-slide matching — primary re-identification test

**Question:** Can a set of anonymous CLS vectors be matched to a reference WSI associated with an identified patient?

**Method:**
1. Export the target slide's CLS vectors without identifiers.
2. Build a separate reference gallery of WSIs whose identity is known *within the evaluation environment*.
3. Embed gallery slides using the known encoder (and optionally evaluate an attacker with only an approximate encoder).
4. Rank gallery candidates against the anonymous slide using slide-set matching methods; include a simple mean-pooling/cosine baseline and a stronger distributional or learned matcher.
5. Repeat for increasing gallery sizes, with exact-slide and different-slide/same-patient scenarios reported separately.

**Metrics:** Top-1/Top-5 identification rate, Recall@K, false-match rate at fixed thresholds, gallery size, runtime, and hardware requirements.

**Interpretation:** Successful matching to an *identified* reference is direct evidence of a feasible identification pathway. Failure to match a small gallery is not evidence that all possible references are safe.

### B. Same-patient linkage — linkability test

**Question:** Can different anonymous WSIs be inferred to belong to the same patient?

**Method:**
1. Construct positive pairs from different slides of the same patient, and negative pairs from different patients.
2. Prevent train/test leakage at patient level; evaluate distinct tissue blocks, acquisition dates, scanners, or staining conditions when available.
3. Compare vector-set similarity baselines with a trained matching model, using only legitimately available attacker inputs.
4. Control for trivial correlations such as duplicate tissue regions, near-duplicate scans, and institution-specific artifacts.

**Metrics:** ROC-AUC, EER, TPR at a low FPR, and false-link rate.

**Interpretation:** Linking two unidentified slides establishes *linkability*, not necessarily identification by name. It still informs the anonymization risk assessment.

### C. Model inversion — supporting disclosure test

**Question:** Can tile images or identifying visual markings be recovered from CLS vectors?

**Method:**
1. Train an embedding-to-image decoder on disjoint public or otherwise authorized training data; if encoder weights are public, include a white-box reconstruction method.
2. Reconstruct sampled tile images from target CLS vectors.
3. Assess recovered tissue morphology separately from recovered *identifying* material (e.g., text or slide-label artifacts, only if such content could realistically pass the tissue extraction process).
4. Record attack cost and whether any reconstruction actually enables linkage to an identified reference.

**Metrics:** LPIPS, SSIM/PSNR as reconstruction descriptors; rate of recoverable identifying marks; runtime and hardware. **Do not** interpret high-quality tissue reconstruction alone as patient identification.

### D. Cross-institution matching — optional robustness extension

Repeat reference matching or patient linkage with different acquisition sites, scanners, or staining pipelines, **only if** suitable multi-site data and patient-level ground truth exist. Report site-specific results rather than assuming a cross-site identity overlap.

## Tests deprioritized from the original seven

| Original attack | Decision | Reason |
|---|---|---|
| Model inversion | **Keep** (supporting) | Could reveal identifying content, but tissue reconstruction alone does not identify a patient. |
| Membership inference | **Deprioritize** | Training membership is distinct from identification through exported CLS vectors; revisit only if membership in a particular cohort is itself a material privacy threat. |
| Cross-patient linkage | **Keep**, redefine as same-patient linkage | Tests whether multiple slides can be linked to the same individual. |
| Attribute inference | **Deprioritize** | Inferring diagnosis or tissue characteristics does not itself identify the patient, though it can remain a sensitive-data disclosure concern. |
| Embedding matching | **Keep — primary** | Tests linkage to an external, identified slide reference. |
| Cross-site linkage | **Conditional** | Useful when multi-site overlap and supporting ground truth exist. |
| Training data extraction | **Deprioritize** | Primarily examines model memorization, not identification from the exported vectors. |

## Privacy–utility experiment matrix

Evaluate the prioritized attacks for the **actual released representation** (tile-level CLS sets), rather than assuming that patch-level, slide-level, hierarchical, and multi-scale representations are all in scope.

| Release condition | Reference matching | Patient linkage | Inversion | Diagnostic utility |
|---|---|---|---|---|
| Raw CLS vectors (baseline) | Yes | Yes | Yes | RQ1–RQ3 |
| Reduced vector dimension | Yes | Yes | Yes | RQ1–RQ3 |
| Aggregated/pooled vectors (if permissible for downstream tasks) | Yes | Yes | Yes | RQ1–RQ3 |
| Noise-injected CLS vectors | Yes | Yes | Yes | RQ1–RQ3 |
| Combined mitigation | Yes | Yes | Yes | RQ1–RQ3 |

For noise injection, distinguish **empirical noise** from a mechanism that provides formal **differential privacy**. Do not claim a DP guarantee without defining adjacency (tile, slide, or patient), sensitivity, composition, and privacy parameters.

Measure utility using the proposal's endpoints: **lesion detection AUC (RQ1), histologic subtype macro-F1 (RQ2), and Ki67 concordance / categorical classification performance (RQ3)**. Preserve patient-disjoint evaluation and report uncertainty intervals. If data for RQ3 are unavailable, identify that limitation explicitly.

## Feasibility and reporting

For each attack, record:

- The exact information available to the attacker, including whether a reference gallery and encoder weights are accessible.
- Attack success at fixed false-positive rates and on meaningful gallery sizes, with confidence intervals and chance baselines.
- Compute resources, wall-clock time, training data needs, and approximate resource cost.
- Sensitivity to slide count, tile count, stain/scanner shift, and gallery composition.
- Whether the outcome demonstrates **identity disclosure**, **linkability without a name**, or **only tissue-information recovery**.

## Proposed P1 conclusion format

> Under the evaluated threat models and resource budgets, releasing ~2,000 tile-level CLS vectors per WSI resulted in [measured identification/linkage outcomes]. The strongest observed attack required [auxiliary data and resources]. [Mitigation] changed attack success from [X] to [Y] while changing lesion AUC, subtype macro-F1, and Ki67 performance by [measured differences]. These findings characterize practical re-identification risk under the tested conditions; they do not independently certify legal anonymization under GDPR or LGPD.

## Relevant legal texts

- [GDPR — Recital 26 (identifiability and means reasonably likely to be used)](https://eur-lex.europa.eu/eli/reg/2016/679/oj)
- [Brazil — LGPD, Law 13.709/2018, Article 12](https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm)

*Source basis: StreamEmbed PhD Thesis Proposal, March 2026, especially pp. 2–3. The attack prioritization and experimental details above are suggested refinements for the clarified CLS-vector release scenario, not claims already established in the proposal.*
