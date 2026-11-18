# phd-streamembed

Turn whole-slide pathology images (WSIs) into compact tile embeddings with the foundation model **UNI2-h**, so they can be used later for classification. Developed on the BRACS breast-cancer slide data set (547 slides, 189 patients, 7 diagnostic classes).

Slides are gigapixel images, far too large to feed to a network at once. The pipeline cuts each slide into small tiles, drops empty background, embeds each tile and writes the vectors to disk **one file per slide**, processing the data tile by tile and slide by slide so memory use stays small.

## Notebooks

The work is organised as numbered notebooks. Each one keeps its code thin; the logic lives in `utils/`. Every figure carries its own caption, and every code cell has a markdown cell before it.

| notebook | purpose | status |
|---|---|---|
| `00_data_visualization.ipynb` | the data set: labels, audit of the official split, slide properties, pyramid levels, class weights | done |
| `01_1_wsi_to_vectors.ipynb` | slides to tile embeddings, and inspection of the embeddings | done |
| `01_2_vectors_to_pca_analysis.ipynb` | PCA of the tile embeddings (explained variance, effective rank, reconstruction, cosine neighbourhood preservation) and the reduced embeddings saved as `pca_<d>/`; with `PER_FOLD` one PCA per cross-validation fold on its fit slides only (`cv_fold_<k>/pca_<d>/`) | done |
| `01_3_wsi_to_mod_vectors.ipynb` | synthetic slides from the saved real tile embeddings, saved per cross-validation fold as `.pt` files (`level_1_mod/`) with a label file `labels/BRACS_mod.csv` | done |
| `01_3_pca_to_synthetic_vectors.ipynb` | synthetic WSI bags from the PCA-reduced vectors of 01_2: class-specific (or shared) MiniBatchKMeans, blended real Training slides as composition templates (`N_TEMPLATES_PER_WSI`), per-cluster Gaussian sampling by the independent-tile `gaussian` method or by `hier_lowrank` (one offset per cluster and bag from the between-slide variance of the real cluster means, low-rank covariance), with diagnostics of the between-slide variance and of a real-vs-synthetic classifier; optional `PER_FOLD` (one generator per fold), `MIN_CLUSTER_SLIDES` (drop clusters seen in few slides, with a support table) and `COMPOSITION_MODE = 'lesion_share'` (ADH / DCIS bags from a lesion-share model); saved under `synthetic/<method>/pca_<D>/k_<K>/` with the label sheet `labels/BRACS_synthetic.csv` | done |
| `02_0_vectors_to_classifier_selector.ipynb` | PyTorch classifier (multiple-instance learning) on the saved vectors, with a full set of metrics on the official Testing set | done |
| `02_1_vector_to_classifier_experimentation.ipynb` | one CLAM-MB per label set (Benign/Atypical/Malignant, Non-cancer/Cancer, Non-cancer/DCIS/IC) with confusion matrix, ROC/AUROC, accuracy, F1, sensitivity, specificity and precision on the Testing set | done |
| `02_2_vector_to_classifier_finetunning.ipynb` | Bayesian optimisation (optuna) of the direct 3-class CLAM-MB (Non-cancer / ADH+DCIS / IC) by patient-grouped cross-validation, an ensemble of the fold models, evaluated once on the Testing set; the 4-class version (with ADH as its own class) is kept in `02_2_vector_to_classifier_finetunning_4class.ipynb` | done |
| `02_3_pca_vector_to_classifier_finetunning.ipynb` | the tuning of 02_2 on the PCA-reduced vectors of 01_2 (`DIMENSION` components, 512 by default), scored on the same folds and evaluated once on the Testing set, and compared with the 1536-d models of 02_1 and 02_2 | done |
| `02_4_synthetic_vector_to_classifier_finetunning.ipynb` | the tuning of 02_3 with the classifier fitted on the synthetic bags of 01_3 (`hier_lowrank` by default; number of clusters, dropout, `k_sample` and tile cap searched, lr / weight decay / coarse weight fixed from 02_3), stopped on real slides (optionally on fresh bags drawn in every epoch, `FRESH_BAGS`), scored by CV (default, optimistic) or on clean Validation slides (`SCORING`), with `PER_FOLD` every fold uses its own PCA space and synthetic set (leak-free CV), evaluated once on the Testing set and compared with the models of 02_3, 02_2 and 02_1 | done |
| `03_reidentification_risk_assessment.ipynb` | re-identification risk assessment | planned (empty) |

## The data set

- **Slides and labels.** The `.svs` slides go in `DATA_DIR`; `labels/BRACS.xlsx` holds, for every slide, the class, the patient and the official set (Training / Validation / Testing), plus the regions of interest (RoIs) of each slide.
- **Seven classes**, from benign to malignant: **N** normal, **PB** pathological benign, **UDH** usual ductal hyperplasia, **FEA** flat epithelial atypia, **ADH** atypical ductal hyperplasia, **DCIS** ductal carcinoma in situ, **IC** invasive carcinoma. They can be read as four tiers: no finding (N), benign (PB, UDH), atypical / high risk (FEA, ADH) and malignant (DCIS, IC). Definitions are in section 3.1 of `00_data_visualization.ipynb`.
- **A slide label is the highest-grade RoI of the slide**, so slides also contain lower-grade tissue (a slide labelled IC has about 61% IC regions on average). The label describes the tissue sample, not the person: 78 of 189 patients have slides of several classes.
- **The official split is used unchanged**, so results stay comparable with published BRACS work. Its audit (notebook 00, section 3.3): the class mix of the three sets does not differ significantly from a random assignment of patients (Cramér's V 0.15, permutation p = 0.47), but the Validation and Testing sets are small, and one patient (67) has slides in both Training and Validation.
- **Class imbalance** (Training: PB 120 and IC 100 slides against FEA 24): class weights for the loss are computed on the Training set only (`utils/class_weights.py`) and saved to `class_weights.csv`.
- **Technical properties.** All slides come from one scanner (SS7212, 40x); scan year is moderately associated with class (Cramér's V 0.27), which is a possible batch effect. Twelve slides have no pyramid level 4, and level 4 is downsampled 32x or 64x depending on the slide.

## How it works

```
.svs slide ──► tiles (224x224) ──► drop background ──► UNI2-h ──► <slide>.pt
              at one pyramid level   tissue filter       per tile:   CLS vector (1536-d)
                                                                     + optional 16x16 patch tokens
```

1. **Tile** each slide at a chosen pyramid level (`WSITileDataset`, `utils/dataset.py`).
2. **Filter** tiles that are mostly white or black (`background_filter`, `utils/clean_empty.py`).
3. **Embed** batches of tiles with UNI2-h (`utils/load_model.py`, `utils/transformer_per_model.py`).
4. **Save** features per slide (`generate_features`, `utils/generate_features.py`).
5. **Inspect** one file or all slides (`utils/inspect_features.py`, `utils/cls_overview.py`), choose the number of clusters (`utils/cluster_quality.py`) and compare the embeddings with the 7 classes (`utils/class_analysis.py`, labels from `labels/BRACS.xlsx`).

Steps 1 to 5 are in `01_1_wsi_to_vectors.ipynb`. The classifier that uses the embeddings is in `02_0_vectors_to_classifier_selector.ipynb` (see below).

## Output format

One PyTorch file per slide: `<FEATURES_DIR>/<model>/<slide_stem>.pt`, a dict with

| key | shape / type | content |
|---|---|---|
| `cls_features` | `[tiles, 1536]` | one summary vector per tile |
| `patch_features` | `[tiles, 256, 1536]` | 16x16 patch tokens per tile; only saved with `save_patch_features=True` |
| `coordinates` | `[tiles, 2]` | `(x, y)` of each tile, in pixels of the tiled pyramid level |
| `model_encoder` | `str` | model name, e.g. `UNI2-h` |
| `tile_params` | `dict` | `level`, `tile_size`, `stride` and the tile filter with its arguments |

Patch tokens make up about 99% of the file size (a slide with 355 tiles is about 560 MB with them and about 2 MB without), so `save_patch_features=False` is the sensible choice when only a classifier on CLS vectors is planned.

Load a file with `torch.load(path, weights_only=True)`.

## The classifier

A slide is a bag of tile embeddings with one label, so classification is **multiple-instance learning**. `02_0_vectors_to_classifier_selector.ipynb` trains two PyTorch models on the CLS embeddings and the 7 classes of `labels/BRACS.xlsx`:

- a **gated attention MIL model** (ABMIL, Ilse et al., 2018) that learns a weight for every tile, pools the tiles with these weights and classifies the result; its attention can be drawn on the slide;
- **CLAM-MB** (Lu et al., 2021), which has one attention branch per class, so that every class gets its own slide vector, and adds an instance-level clustering loss on the tiles that the attention selects.

The results of the two models are shown side by side. The models are listed in `MODELS` (section 2 of the notebook); the mean-pool linear model (`MeanPoolLinear`) is still in `utils/mil_models.py` and can be added there as a reference.

Protocol: the official split is used unchanged. The official Testing set is evaluated once, at the end (the training code never receives Testing labels). The official Training and Validation slides are pooled and split into patient-grouped, class-stratified folds (`utils/mil_cv.py`); every choice (loss weighting, regularisation, coarse auxiliary loss, epoch) is made by cross-validation on them, with nested early stopping so that a held-out fold is never used for fitting or stopping, and the final models are the fold models of the selected setting. The loss is the class-weighted cross-entropy with weights from the Training counts. Every model is trained with several seeds; the predictions of the seeds are averaged. One `SEED` fixes every random choice, and the code picks the GPU if there is one and the CPU otherwise.

Reported on the Testing set, with 95% patient-level bootstrap intervals: accuracy, balanced accuracy, macro and weighted F1, per-class precision, recall and F1, the confusion matrix, one-vs-rest AUROC and average precision (ROC and precision-recall curves), quadratic-weighted kappa, Matthews correlation, top-2 accuracy, log-loss, Brier score and calibration error, plus two clinical views derived from the seven class probabilities (three tiers; cancer against non-cancer) and a paired comparison of the two models. Results are shown in the notebook and not written to disk. The only file written is the best classifier (highest Validation macro-F1 over both models and seeds, never chosen on the Testing set) as `classifier/classifier_model_<date-time>.pt`, with weights, class names, feature scaler, settings and Validation result; `utils.mil_train.load_classifier` loads it.

## Setup

Requirements: Python >= 3.14, [uv](https://docs.astral.sh/uv/), the system library **libvips** (used through `pyvips`), and a GPU is recommended for feature extraction. Notebook 00 needs no GPU and no model; notebook 02 trains on a GPU if there is one and otherwise on the CPU (slower, same results up to floating-point differences).

```bash
uv sync                      # install the dependencies from pyproject.toml / uv.lock
uv run jupyter lab           # open the notebooks
```

**Model weights.** UNI2-h is gated on Hugging Face. Request access to [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h), log in with `huggingface-cli login`, then run

```bash
uv run python setup_only/model_download.py   # downloads into original_vit_model/
```

**Data.** Put the `.svs` slides in one flat folder and set `DATA_DIR` in the notebooks. Two helper scripts in `setup_only/` prepare it (edit the CONFIG block at the top of each first):

- `flatten_copy.py` copies all files of a folder tree into one flat folder.
- `rsync_upload.py` uploads the slides to a remote workstation (`--dry-run` lists what would be sent).

The label sheet `labels/BRACS.xlsx` is read with the standard library (no `openpyxl` needed). `scikit-learn` provides the evaluation metrics of notebook 02.

## Usage

Run the notebooks from the top, from the repository root.

**`00_data_visualization.ipynb`** (section 2): `DATA_DIR`, `LABELS_FILE`, `OUTPUT_DIR`, `TILE_SIZE`, `TILE_EXAMPLE_LEVEL` and `SEED`. It writes `slides.csv` (one row per slide: label, patient, set, technical properties) and `class_weights.csv` (loss weights from the Training counts) to `OUTPUT_DIR`. A full run takes about ten seconds on the local data.

**`01_1_wsi_to_vectors.ipynb`** (section 2.1):

| parameter | meaning |
|---|---|
| `PYRAMID_LEVEL` | resolution to tile from: 1 is full resolution, higher is more downsampled |
| `TILE_SIZE`, `STRIDE` | tile side in pixels (224 for UNI2-h); `STRIDE == TILE_SIZE` means no overlap |
| `BATCH_SIZE` | tiles per model call |
| `DATA_DIR`, `FEATURES_DIR` | slide folder and output folder |

**`02_0_vectors_to_classifier_selector.ipynb`** (section 2): `FEATURES_DIR`, `MODEL_NAME`, `LABELS_FILE`, `MODEL_DIR`, `SAVE_BEST_MODEL`, `SEED`, `N_SEEDS`, `EPOCHS`, `PATIENCE`, `LR`, `WEIGHT_DECAY`, `BATCH_SIZE`, `MAX_TILES_TRAIN` (tiles per slide per epoch) and `N_BOOT`. Needs the feature files from notebook 01 (with `cls_features`; patch tokens are not used) and takes minutes on a GPU for features tiled at level 3.

The number of tiles grows roughly 16-fold per pyramid level step towards level 1, and runtime and disk use grow with it. At level 3 the BRACS slides give about 94k tiles.

Helpers, usable from any notebook:

```python
from utils.dataset_overview import build_slide_table, plot_split_audit
from utils.class_weights import plot_loss_weighting, weighted_cross_entropy
from utils.plot_pyramid import plot_pyramid_levels
from utils.plot_wsi import plot_wsi
from utils.inspect_features import inspect_feature_file
from utils.cls_overview import plot_cls_overview
from utils.cluster_quality import plot_cluster_quality
from utils.class_analysis import plot_class_analysis
from utils.mil_data import load_bags
from utils.mil_train import run_seeds, predict_ensemble
from utils.classification_metrics import evaluate_predictions, patient_bootstrap

table = build_slide_table("labels/BRACS.xlsx")                          # slide, patient, class, set
plot_split_audit(table)                                                 # is the official split sound?
plot_loss_weighting(y_train, ["N", "PB", "UDH", "FEA", "ADH", "DCIS", "IC"])   # class weights and their effect
plot_pyramid_levels("data/BRACS_1003658.svs")                           # same tissue at levels 1 to 4
plot_wsi(dataset, slide=0)                                              # tiling of one slide
inspect_feature_file(features_dir=FEATURES_DIR, model_name="UNI2-h")    # one .pt file
plot_cls_overview(FEATURES_DIR, "UNI2-h", n_clusters=7)                 # all slides
plot_cluster_quality(FEATURES_DIR, "UNI2-h")                            # which k? (silhouette, stability)
plot_class_analysis(FEATURES_DIR, "UNI2-h", "labels/BRACS.xlsx")       # embeddings vs the 7 classes, linear probe
data = load_bags(FEATURES_DIR, "UNI2-h", table)                         # bags of tile embeddings, one per slide
runs = run_seeds("abmil", [0, 1, 2], data, None, device)                # train the attention model (see notebook 02)
```

## Data pipeline per notebook

One diagram per notebook, in notebook order, drawn from its inputs to its outputs. Cylinders are data on disk, boxes are processing steps (numbers are the notebook sections). Add the diagram of each new notebook here when it is created.

### `00_data_visualization.ipynb`: the data set (labels, split, slides)

```mermaid
flowchart TD
    subgraph IN["Inputs"]
        X[("labels/BRACS.xlsx<br/>class, patient, set, RoIs")]
        SVS[("BRACS slides<br/>.svs files in DATA_DIR")]
    end

    subgraph S3["3. Labels and split"]
        L1["3.1 build_slide_table + validate_labels"]
        L2["3.2 plot_class_distribution"]
        L3["3.3 plot_split_audit<br/>class x set test, patient overlap"]
        L4["3.4 plot_roi_composition<br/>weak slide labels"]
        L5["3.5 plot_loss_weighting<br/>class weights from Training only"]
    end

    subgraph S4["4. Slide properties"]
        P1["4.1 scan_slides<br/>levels, MPP, scanner, date"]
        P2["4.2 plot_slide_properties<br/>batch screen"]
    end

    subgraph S5["5. Visual inspection"]
        V1["5.1 plot_class_gallery"]
        V2["5.2 plot_pyramid_levels<br/>levels 1 to 4"]
        V3["5.3 plot_tile_examples"]
    end

    OUT[("OUTPUT_DIR<br/>slides.csv, class_weights.csv")]

    X --> L1
    L1 --> L2
    L1 --> L3
    L1 --> L4
    L1 --> L5
    SVS --> P1
    L1 --> P1
    P1 --> P2
    P1 --> V1
    P1 --> V2
    P1 --> V3
    P1 --> OUT
    L5 --> OUT
```

The official Training/Validation/Testing split is used unchanged (comparable with published BRACS results). Its properties are documented in section 3.3: the class mix of the sets does not differ significantly from a random assignment of patients, one patient has slides in two sets, and 78 of 189 patients have slides of more than one class. The loss weights are computed on the Training set only.

- **Input:** `labels/BRACS.xlsx` and the `.svs` slides (headers only; pixels are read for the figures in section 5).
- **Output:** `slides.csv` and `class_weights.csv`, plus the figures documenting the data. Later notebooks read these tables, so the split and the weights are the same everywhere.

### `01_1_wsi_to_vectors.ipynb`: slides to tile embeddings

```mermaid
flowchart TD
    subgraph IN["Inputs"]
        SVS[("BRACS slides<br/>.svs files in DATA_DIR")]
        W[("UNI2-h weights<br/>original_vit_model/")]
    end

    subgraph S2["2. Data"]
        P["2.1 Parameters<br/>level, tile size, stride, batch size"]
        DS["2.2 WSITileDataset<br/>index tiles + tissue filter"]
        CHK["2.3 and 2.4 Checks<br/>tile count, plot_wsi"]
        DL["2.5 DataLoader<br/>ordered batches, worker processes"]
    end

    subgraph S3["3. Feature extraction"]
        M["3.1 Load UNI2-h<br/>+ model transform"]
        GF["3.2 generate_features<br/>CLS vectors, optional patch tokens"]
    end

    OUT[("FEATURES_DIR/UNI2-h/<br/>one .pt file per slide")]

    subgraph S4["4. Inspection"]
        I1["4.1 inspect_feature_file<br/>one slide"]
        I2["4.2 plot_cls_overview<br/>all slides"]
        I3["4.3 plot_cluster_quality<br/>choice of k"]
        I4["4.4 plot_class_analysis<br/>vs 7 classes + linear probe"]
    end

    SVS --> DS
    P --> DS
    DS --> CHK
    DS --> DL
    DL --> GF
    W --> M
    M --> GF
    GF --> OUT
    OUT --> I1
    OUT --> I2
    OUT --> I3
    OUT --> I4
    LAB[("labels/BRACS.xlsx<br/>class, patient, set")] --> I4
```

- **Input:** `.svs` slides, the UNI2-h checkpoint and, for section 4.4, `labels/BRACS.xlsx`.
- **Output:** per-slide `.pt` files (`cls_features`, `coordinates`, `model_encoder`, `tile_params`, optional `patch_features`); the figures in section 4 are for checking only.

### `01_2_vectors_to_pca_analysis.ipynb`: PCA of the tile embeddings

```mermaid
flowchart TD
    subgraph IN["Inputs"]
        FEAT[("level_1/UNI2-h/<br/>one .pt file per slide (from 01_1)")]
        LAB[("labels/BRACS.xlsx<br/>class, patient, official set")]
    end

    subgraph S3["3. Data"]
        D1["3.1 split_fit_eval<br/>Training fits, Validation evaluates"]
        D2["3.2 sample_tiles + sampling_stats"]
    end

    subgraph S4["4-8. Analyses"]
        P1["4 fit_pca"]
        P2["5 variance_table"]
        P3["6 spectrum_stats"]
        P4["7 reconstruction_metrics"]
        P5["8 neighbour_preservation"]
    end

    subgraph S9["9. Comparison"]
        C1["final_comparison + recommend_dimensions"]
    end

    subgraph S10["10. Storage"]
        W1["save_pca_embeddings"]
    end

    subgraph S12["12. PER_FOLD"]
        F1["cv_folds_for_features<br/>the folds of 02_3 / 02_4"]
        F2["fit_fold_pca_embeddings<br/>one PCA per fold, fit slides only"]
    end

    OUT[("pca_d/UNI2-h/<br/>slide .pt, pca_model.pt, metadata.json")]
    FOUT[("cv_fold_k/pca_d/UNI2-h/<br/>slide .pt, pca_model.pt, metadata.json with the fold")]

    FEAT --> D1
    LAB --> D1
    D1 --> D2
    D2 --> P1
    P1 --> P2
    P1 --> P3
    P1 --> P4
    P1 --> P5
    P2 --> C1
    P4 --> C1
    P5 --> C1
    P1 --> W1
    W1 --> OUT
    FEAT --> F1
    LAB --> F1
    F1 --> F2
    F2 --> FOUT
```

- **Input:** the per-slide `.pt` files of notebook 01_1 and `labels/BRACS.xlsx`.
- **Output:** `<PCA_OUTPUT_DIR>/pca_<d>/UNI2-h/<slide>.pt` (`cls_features` `[tiles, d]`, `coordinates`, `model_encoder`, `tile_params`, `pca`, `slide`, `label`, `patient`, `set`; tile order and coordinates as in the original), `pca_<d>/pca_model.pt` (mean, components, explained variance) and `pca_<d>/metadata.json`. The PCA is fitted on Training tiles only; Testing slides are only transformed. With `PER_FOLD = True` (section 12) one PCA per fold of 02_3 / 02_4 is fitted on the fit slides of that fold only (not on its early-stopping or held-out slides, nor on Testing) and written to `<PCA_OUTPUT_DIR>/cv_fold_<k>/pca_<d>/UNI2-h/` (stored sets `Training`, `Validation`, `HeldOut`, `Testing`; `metadata.json` records the fold and its `fit_slides` / `stop_slides` / `held_slides`). No figure is exported.

### `01_3_pca_to_synthetic_vectors.ipynb`: synthetic WSI bags from the PCA vectors

```mermaid
flowchart TD
    PCA[("pca_D/UNI2-h/<br/>one .pt file per slide (from 01_2)<br/>PER_FOLD: cv_fold_K/pca_D/")] --> SEL["3. build_sources + select_training<br/>Training slides only, validated<br/>PER_FOLD: fit slides of one fold"]
    LAB[("labels/BRACS.xlsx")] --> SEL
    SEL --> KM["4. sample_class_tiles + fit_class_clusters or fit_shared_clusters<br/>MiniBatchKMeans per class and k (or one shared codebook)"]
    KM --> STAT["5. collect_class_statistics<br/>composition templates, Gaussian sums, per-slide cluster means"]
    STAT --> SUP["5.1 cluster_support_table<br/>slides per cluster (read-only)"]
    STAT --> PRUNE["5.2 prune_state<br/>MIN_CLUSTER_SLIDES: drop sparse clusters,<br/>statistics collected again"]
    PRUNE --> GAUSS["6. GaussianGenerator.fit or HierarchicalGenerator.fit<br/>within / between-slide variance, low-rank factors"]
    PRUNE --> LES["6.1 lesion_enrichment + fit_lesion_models<br/>COMPOSITION_MODE lesion_share: lesion clusters,<br/>Beta lesion share, lognormal size"]
    PRUNE --> GEN["7. generate_configuration<br/>blend of several templates (Dirichlet), or lesion-share bags for ADH / DCIS;<br/>one offset per cluster and bag"]
    GAUSS --> GEN
    LES --> GEN
    GEN --> DIAG["8. diagnostics (tables, plots, between-slide variance, real-vs-synthetic AUC,<br/>8.8 lesion share and bag-level AUC)"]
    GEN --> OUT[("synthetic/method+tag/pca_D/k_K/<br/>SYN_*.pt, metadata.json, cluster_models.pt, labels.csv<br/>PER_FOLD: cv_fold_K/synthetic/")]
    GEN --> SHEET[("labels/BRACS_synthetic.csv<br/>PER_FOLD: BRACS_synthetic_cv_fold_K.csv")]
    GEN --> FOLDS["10.1 run_source for the other folds<br/>PER_FOLD"]
```

- **Input:** the reduced vectors `<PCA_INPUT_DIR>/pca_<D>/UNI2-h/<slide>.pt` of 01_2 and the official split of `labels/BRACS.xlsx`. Only Training slides are used for clusters, Gaussians and templates.
- **Output:** `<SYNTHETIC_OUTPUT_DIR>/<method>/pca_<D>/k_<K>/SYN_<method>_pca<D>_k<K>_<class>_<i>.pt` (the keys of the PCA files; `set` is `Synthetic`, no real identifiers), `metadata.json` (with the generator options), `cluster_models.pt` (centres, generator parameters and, for fresh bags in 02_4, the `composition` templates of each class without slide ids), `labels.csv`, and the CSV `labels/BRACS_synthetic.csv` (columns of `BRACS.xlsx` plus `method`, `pca_dim`, `k`, `n_tiles`, `n_templates`, `max_template_weight`). Methods: `gaussian` (independent tiles) and `hier_lowrank` (per-bag offsets of the cluster means, `diag` / `full` / `lowrank` covariance, `OFFSET_SCALE` 0 to 1, `COVARIANCE_SHRINKAGE`, optional `SHARED_CODEBOOK`); `GeneratorBase` (with the per-bag hook `start_bag`) is the extension point. Each method writes to its own folder (the generation method plus `OUTPUT_TAG`, plus `_lesion` in the lesion-share mode), so existing sets are not touched. The three new options default to the earlier behaviour: `PER_FOLD` (one pipeline run per CV fold from the fit slides recorded by 01_2, output in `cv_fold_<k>/synthetic/`), `MIN_CLUSTER_SLIDES` (removed clusters recorded in `metadata.json`) and `COMPOSITION_MODE` (`lesion_share` needs `SHARED_CODEBOOK`; adds `lesion_share.csv`, the lesion rule and clusters in `metadata.json` and the lesion model in `cluster_models.pt`; not available with `FRESH_BAGS`). The notebook constants `SHARED_CODEBOOK = True`, `K_VALUES = [100, 150, 250]`, `MIN_SLIDES = 10`, `COVARIANCE_SHRINKAGE = 0.2` are starting values, not validated; the pipeline is in `utils/synthetic_pipeline.py`.

### `01_3_wsi_to_mod_vectors.ipynb`: synthetic slides to vectors

```mermaid
flowchart TD
    subgraph IN["Inputs"]
        FEAT[("level_1/UNI2-h/<br/>one .pt file per real slide (from 01)")]
        LAB[("labels/BRACS.xlsx<br/>class, patient, official set")]
    end

    subgraph S3["3. Data"]
        D1["3.1 build_slide_table + load_bags"]
        D2["3.2 relabel_table<br/>Non-cancer / atypia-neoplasia / IC"]
        D3["3.3 make_cv_folds<br/>patient-grouped, stratified"]
    end

    subgraph S4["4. Synthetic slides"]
        R1["4.1 make_synthetic_folds<br/>N_TOTAL recipes per fold,<br/>fit and early-stopping slides only"]
        R2["4.2 plot_synthetic_summary"]
    end

    subgraph S5["5. Saving"]
        W1["save_synthetic_folds<br/>one .pt per synthetic slide"]
        W2["write_mod_labels"]
    end

    subgraph S6["6. Verification"]
        V1["reload + checks<br/>tiles, sources, folds"]
    end

    REC[("level_1_mod/recipes/<br/>synthetic_recipes.pt")]
    OUT[("level_1_mod/UNI2-h/fold_k/<br/>SYN_*.pt")]
    CSV[("labels/BRACS_mod.csv")]

    FEAT --> D1
    LAB --> D1
    D1 --> D2
    D2 --> D3
    D3 --> R1
    R1 --> REC
    R1 --> R2
    R1 --> W1
    R1 --> W2
    W1 --> OUT
    W2 --> CSV
    OUT --> V1
    CSV --> V1
```

- **Input:** the per-slide `.pt` files of notebook 01 and `labels/BRACS.xlsx`.
- **Output:** `<OUT_DIR>/UNI2-h/fold_<k>/SYN_*.pt` (same keys as the files of 01, plus `label`, `sources`, `counts`, `source_indices`, `source_ids`; zero `coordinates`), the recipes in `<OUT_DIR>/recipes/synthetic_recipes.pt` and `labels/BRACS_mod.csv` (columns `WSI Filename`, `Patient Id`, `WSI label`, `Set`, `fold`, `part`, `size`, `n_sources`, `max_share`, `sources`). The synthetic slides are made per fold, only from the fit / early-stopping slides of the fold, so the held-out patients leak nothing.

### `02_0_vectors_to_classifier_selector.ipynb`: embeddings to a slide classifier

```mermaid
flowchart TD
    subgraph IN["Inputs"]
        FEAT[("FEATURES_DIR/UNI2-h/<br/>one .pt file per slide (from 01)")]
        LAB[("labels/BRACS.xlsx<br/>class, patient, official set")]
    end

    subgraph S3["3. Data"]
        D1["3.1 build_slide_table<br/>official split"]
        D2["3.2 load_bags<br/>bag of tile embeddings per slide"]
        D3["3.3 plot_bag_sizes"]
        D4["3.4 class weights<br/>from Training only"]
    end

    subgraph S4["4. Models"]
        M1["gated attention MIL<br/>(ABMIL)"]
        M2["CLAM-MB<br/>class-wise attention + instance loss"]
    end

    subgraph S5["5. Training (Training + Validation, cross-validated)"]
        F0["5.0 make_cv_folds<br/>patient-grouped, stratified"]
        T1["5.1 cv_compare<br/>loss weighting"]
        T1b["5.2 cv_compare<br/>dropout, weight decay, tile dropout"]
        T1c["5.3 cv_compare<br/>coarse tier loss"]
        T2["5.4 cross_validate<br/>fold models, early stopping inside each fold"]
        T2b["5.5 out-of-fold performance"]
        T3["5.6 plot_training_curves"]
    end

    subgraph S6["6. Evaluation (Testing set, used once)"]
        E1["6.1 predict_ensemble + evaluate_predictions<br/>+ patient bootstrap"]
        E2["6.2 to 6.7 confusion, ROC and PR, per class,<br/>calibration, clinical views, model comparison"]
    end

    A["7. plot_attention<br/>tiles each model relies on"]
    DEC["8.2 select_best_run + plot_model_decision<br/>architecture graph and specification"]
    OUT[("8.3 classifier/<br/>classifier_model_date-time.pt<br/>(best on early-stopping slides)")]

    FEAT --> D2
    LAB --> D1
    D1 --> D2
    D2 --> D3
    D2 --> D4
    D2 --> F0
    F0 --> T1
    D4 --> T1
    M1 --> T1
    M1 --> T2
    M2 --> T2
    T1 --> T1b
    T1b --> T1c
    T1c --> T2
    T2 --> T2b
    T2 --> T3
    T2 --> E1
    E1 --> E2
    E1 --> A
    E1 --> DEC
    T2 --> DEC
    DEC --> OUT
```

- **Input:** the per-slide feature files of notebook 01 and `labels/BRACS.xlsx`.
- **Output:** the results are shown in the notebook; the only file written is the best classifier with its configuration (seeds, selected weighting, tiling, versions, device). The Testing labels are used only in section 6.

### `02_1_vector_to_classifier_experimentation.ipynb`: one CLAM-MB per label set

```mermaid
flowchart TD
    subgraph IN["Inputs"]
        FEAT[("FEATURES_DIR/UNI2-h/<br/>one .pt file per slide (from 01)")]
        LAB[("labels/BRACS.xlsx<br/>class, patient, official set")]
    end

    subgraph S3["3. Data"]
        D1["3.1 build_slide_table + load_bags"]
        D2["3.2 make_holdout_split<br/>fit / early stopping (patient-grouped)"]
        D3["3.3 relabel_table<br/>three label sets"]
    end

    subgraph S4["4. Training (one CLAM-MB per label set)"]
        T1["train_final<br/>constants fixed from 02"]
        T2["plot_training_curves"]
    end

    subgraph S5["5. Evaluation (Testing set, used once)"]
        E1["predict_testing"]
        E2["class_views + bootstrap_views<br/>accuracy, F1, recall, precision,<br/>sensitivity, specificity, AUROC"]
        E3["plot_final_results<br/>confusion matrix + ROC"]
    end

    subgraph S7["7-8. Two-stage (Cancer, then DCIS / IC)"]
        H1["make_stage2_table + train_final<br/>DCIS / IC on cancer slides only"]
        H2["cascade_predict<br/>Cancer? then DCIS / IC; 3x3 confusion matrix<br/>(7.5: from the stage decisions)"]
        H3["class_views + paired_bootstrap_views<br/>vs the flat 3-class model"]
        H4["7.6 ADH experiment: direct 4-class vs<br/>two-stage (ADH as cancer, then ADH / DCIS / IC)"]
        H5["10 Direct 3-class: Non-cancer / ADH&DCIS / IC<br/>vs the 4-class model with ADH and DCIS merged"]
    end

    OUT[("classifier/final/label set/<br/>classifier_model_date-time.pt")]

    FEAT --> D1
    LAB --> D1
    D1 --> D2
    D2 --> D3
    D3 --> T1
    T1 --> T2
    T1 --> E1
    E1 --> E2
    E1 --> E3
    T1 --> OUT
    D2 --> H1
    T1 --> H2
    H1 --> H2
    H2 --> H3
    D2 --> H4
    D2 --> H5
    H4 --> H5
    H4 --> OUT
    H1 --> OUT
```

### Planned notebooks

| notebook | purpose | diagram |
|---|---|---|
| `03_reidentification_risk_assessment.ipynb` | re-identification risk assessment | to be added |

### `02_3_pca_vector_to_classifier_finetunning.ipynb`: tuning on PCA-reduced vectors

```mermaid
flowchart TD
    subgraph IN["Inputs"]
        PCA[("pca_DIMENSION/UNI2-h/<br/>one .pt file per slide (from 01_2)")]
        LAB[("labels/BRACS.xlsx<br/>class, patient, official set")]
        FULL[("level_1/UNI2-h/<br/>1536-d vectors, Testing slides only")]
        OLD[("classifier/final/<br/>models of 02_1 and 02_2")]
    end

    subgraph S3["3. Data"]
        D1["3.1 build_slide_table + load_bags"]
        D2["3.2 relabel_table<br/>Non-cancer / atypia-neoplasia / IC"]
        D3["3.3 make_cv_folds<br/>patient-grouped, stratified"]
    end

    subgraph S4["4. Bayesian optimisation"]
        B1["objective_factory + run_study<br/>TPE sampler, one trial = one 5-fold CV"]
        B2["trials_table + plot_tuning"]
    end

    subgraph S5["5-6. Selection and confirmation"]
        C1["select_setting<br/>recall floor, best macro-F1,<br/>one-standard-error rule"]
        C2["cross_validate, 2 repeats,<br/>new seeds, top settings"]
    end

    subgraph S7["7. Ensemble"]
        N1["oof_by_repeat + single_vs_ensemble<br/>out-of-fold, no Testing"]
    end

    subgraph S8["8. Testing set, used once"]
        E1["predict_ensemble<br/>all fold models"]
        E2["class_views + bootstrap_views<br/>plot_final_results"]
        E3["paired_bootstrap_views<br/>vs the 1536-d models of 02_1 and 02_2"]
    end

    OUT[("classifier/final/noncancer_noninvasive_invasive_pcaDIMENSION/<br/>ensemble_date-time.pt")]

    PCA --> D1
    LAB --> D1
    D1 --> D2
    D2 --> D3
    D3 --> B1
    B1 --> B2
    B2 --> C1
    C1 --> C2
    C2 --> N1
    N1 --> E1
    E1 --> E2
    E1 --> E3
    OLD --> E3
    FULL --> E3
    N1 --> OUT
```

- **Input:** the reduced vectors `<FEATURES_ROOT>/pca_<DIMENSION>/UNI2-h/<slide>.pt` written by notebook 01_2 (the PCA was fitted on Training tiles only; the constant `DIMENSION` is 512 by default and can be 256, 1024, ...). The original vectors (`level_1/`) are read for the Testing slides only, for the comparison.
- **Search:** that of 02_2 (including the tile cap); the study `three_class_pca<DIMENSION>` is in `outputs/02_3_tuning/study.db` and one study exists per dimension.
- **Output:** `classifier/final/noncancer_noninvasive_invasive_pca<DIMENSION>/ensemble_<date-time>.pt`; the Testing results are compared with the 02_1 model and the 02_2 ensemble (1536-d) by a paired patient bootstrap.

### `02_4_synthetic_vector_to_classifier_finetunning.ipynb`: tuning on synthetic vectors

```mermaid
flowchart TD
    subgraph IN["Inputs"]
        PCA[("pca_DIMENSION/UNI2-h/<br/>real PCA vectors (01_2)<br/>PER_FOLD: cv_fold_K/pca_DIMENSION/")]
        SYN[("synthetic/METHOD/pca_DIMENSION/k_K/<br/>labels/BRACS_synthetic.csv (01_3)<br/>PER_FOLD: cv_fold_K/synthetic/")]
        LAB[("labels/BRACS.xlsx<br/>class, patient, official set")]
        FULL[("level_1/UNI2-h/<br/>1536-d vectors, Testing slides only")]
        OLD[("classifier/final/<br/>models of 02_1, 02_2, 02_3")]
    end

    subgraph S3["3. Data"]
        D1["3.1 build_slide_table + load_bags<br/>real bags: stop, held-out, Testing"]
        D2["3.2 relabel_table<br/>Non-cancer / atypia-neoplasia / IC"]
        D3["3.3 make_cv_folds<br/>the folds of 02_3"]
        D3b["3.3b check_fold_metadata + load_fold_data<br/>PER_FOLD: bags of every fold space"]
        D4["3.4 load_synthetic_sets (PER_FOLD: _per_fold)<br/>one set per k; FRESH_BAGS: generators +<br/>templates of cluster_models.pt"]
        D5["3.5 validation_mask<br/>score of SCORING"]
    end

    subgraph S4["4. Bayesian optimisation"]
        B1["synthetic_objective_factory + run_study<br/>fit on synthetic bags of syn_k (saved, or<br/>FreshSyntheticBags: new draw every epoch),<br/>stop on real stop slides, score on held-out"]
        B2["trials_table + plot_tuning"]
    end

    subgraph S5["5-7. Selection, confirmation, ensemble"]
        C1["select_setting + confirmation<br/>synthetic_cross_validate, 2 repeats"]
        N1["oof_by_repeat + single_vs_ensemble"]
    end

    subgraph S8["8. Testing set, used once"]
        E1["predict_ensemble<br/>real Testing slides"]
        E2["class_views + bootstrap_views"]
        E3["summary_across_sets + paired_bootstrap_views<br/>vs 02_3, 02_1 and 02_2"]
    end

    OUT[("classifier/final/noncancer_noninvasive_invasive_synpcaDIMENSION/<br/>ensemble_date-time.pt")]

    PCA --> D1
    LAB --> D1
    D1 --> D2 --> D3
    SYN --> D4
    D3 --> D3b
    D3b --> B1
    D3 --> D5
    D3 --> B1
    D4 --> B1
    D5 --> B1
    B1 --> B2 --> C1 --> N1 --> E1 --> E2
    E1 --> E3
    OLD --> E3
    FULL --> E3
    N1 --> OUT
```

- **Input:** the real PCA vectors `<FEATURES_ROOT>/pca_<DIMENSION>/UNI2-h/` (early-stopping, held-out and Testing slides), the synthetic bags `<FEATURES_ROOT>/synthetic/<SYNTHETIC_METHOD>/pca_<DIMENSION>/k_<K>/` (`hier_lowrank` by default) with the sheet `labels/BRACS_synthetic.csv` (notebook 01_3), and the models of 02_1, 02_2 and 02_3 for the comparison.
- **Design:** the folds of 02_3; with `FRESH_BAGS = True` the bags are drawn anew in every epoch (`FreshSyntheticBags`, seeded by seed, epoch and slide; `train_model` calls `set_epoch` and takes the scaler from one fixed draw), which needs the `composition` key of a re-run 01_3; every fold model is fitted on all synthetic slides of the sampled `syn_k`, stopped on the real early-stopping slides of its fold and scored on the real held-out slides. `SCORING = 'cv'` (default) pools every held-out slide (optimistic: the generator saw all official Training slides, so it is comparable only with other synthetic sets, not with 02_3); `'validation'` scores only the held-out official Validation slides of patients without Training slides. The search space is restricted in this notebook only: `dropout`, `k_sample` in {16, 32}, `max_tiles` in {8192, None} and `syn_k`; lr, weight decay and the coarse-loss weight are fixed at the 02_3 values (`constants_from_02_3`, fallback `DEFAULT_CONSTANTS`).
- **Per-fold mode (`PER_FOLD = True`, default `False`):** removes the leak of the global PCA and generator. The PCA (01_2) and the synthetic set (01_3) of every fold are fitted on the fit slides of that fold only; each fold is fitted, stopped and scored in its own PCA space (`training_folds` with one synthetic set and one real data dict per fold, `eval_data` and `dim` in the fold dict; `cross_validate` stores the fold's data in each run, so `predict_ensemble`, `oof_by_repeat` and the saved ensemble use the right space) and `SCORING` must be `'cv'`, which then scores every held-out slide without that leak. The folds are checked against those recorded by 01_2. `LAZY_FOLDS = True` keeps one fold's bags in memory at a time. The comparison with the global-PCA models of 02_3 is not exactly paired, because the fold models do not share one PCA basis.
- **Output:** `classifier/final/noncancer_noninvasive_invasive_synpca<DIMENSION>/ensemble_<date-time>.pt`; the study `three_class_syn_pca<DIMENSION>_<SCORING>` (suffix `_fresh` with `FRESH_BAGS`, then `_perfold` with `PER_FOLD`; the ensemble folder gets the same suffixes) is in `outputs/02_4_tuning/study.db`. With `PER_FOLD` each model in the file records its `fold` and `features_dir`.

## Repository layout

```
00_data_visualization.ipynb   data set: labels, split audit, slide properties, class weights
01_wsi_to_vectors.ipynb       tiling -> embedding -> inspection
02_vectors_to_classifier.ipynb  MIL classifier on the embeddings, metrics on the Testing set
021_vector_to_final_classifier.ipynb  one CLAM-MB per label set, metrics on the Testing set
03_reidentification_risk_assessment.ipynb   planned (empty)
labels/BRACS.xlsx             slide labels, patients, official split, RoIs
outputs/                      tables written by notebook 00 (slides.csv, class_weights.csv)
classifier/                   best classifier saved by notebook 02 (classifier_model_<date-time>.pt)
utils/                        code, by role:
    pipeline          dataset, clean_empty, transformer_basic, transformer_per_model, dataloader,
                      load_model, device, generate_features
    classifier        mil_data, mil_models, mil_train, classification_metrics, plot_classification,
                      mil_cv, mil_tuning, final_classifier, synthetic_slides, synthetic_training, per_fold
    synthetic bags    synthetic_gaussian, synthetic_pipeline (stages of 01_3, also per fold), synthetic_lesion
                      (lesion-share composition), pca_analysis (01_2, incl. the PCA per fold)
    data set          labels, slide_stats, dataset_overview, class_weights, plot_pyramid
    embeddings        plot_wsi, inspect_features, cls_data, cls_overview, cluster_quality, class_analysis
    figures           figure_captions (the caption under each panel)
setup_only/                   one-off scripts: model download, flatten data, rsync upload
original_vit_model/           UNI2-h weights (not versioned)
data/                         slides (not versioned)
```

## Notes and limitations

- Tiles of a slide must arrive together and in order (`shuffle=False`), because features are written slide by slide.
- Slides without the requested pyramid level are skipped (12 slides at level 4); tiles with damaged image data are left out. Both are reported as warnings after indexing.
- Re-running overwrites existing `.pt` files of the same slide name.
- Augmentation and the model-specific transform settings are not stored in the output files.
- Slide labels are weak labels for tiles (see above); class separation is therefore assessed at slide level.
- The class weights fix the loss, not the evaluation: report balanced accuracy and per-class recall on the untouched Validation and Testing sets.
- Notebook 02 evaluates on the official Testing set only once; with 87 Testing slides and classes of 7 to 20 slides, intervals are wide and small differences between models cannot be resolved. Slides without a feature file (for example without pyramid level 4) shrink the sets and are reported.
- The loss weights shift the output probabilities towards the rare classes, so the probabilities are not calibrated to the true class frequencies.
- Scan year is associated with class (Cramér's V 0.27); check for batch effects before interpreting classifier results.

### `02_2_vector_to_classifier_finetunning.ipynb`: tuning of the direct three-class model

```mermaid
flowchart TD
    subgraph IN["Inputs"]
        FEAT[("FEATURES_DIR/UNI2-h/<br/>one .pt file per slide (from 01)")]
        LAB[("labels/BRACS.xlsx<br/>class, patient, official set")]
        OLD[("classifier/final/noncancer_adhdcis_ic/<br/>direct 3-class model of 02_1")]
    end

    subgraph S3["3. Data"]
        D1["3.1 build_slide_table + load_bags"]
        D2["3.2 relabel_table<br/>Non-cancer / atypia-neoplasia / IC"]
        D3["3.3 make_cv_folds<br/>patient-grouped, stratified"]
    end

    subgraph S4["4. Bayesian optimisation"]
        B1["objective_factory + run_study<br/>TPE sampler, one trial = one 5-fold CV"]
        B2["trials_table + plot_tuning"]
    end

    subgraph S5["5-6. Selection and confirmation"]
        C1["select_setting<br/>recall floor, best macro-F1,<br/>one-standard-error rule"]
        C2["cross_validate, 2 repeats,<br/>new seeds, top settings"]
    end

    subgraph S7["7. Ensemble"]
        N1["oof_by_repeat + single_vs_ensemble<br/>out-of-fold, no Testing"]
    end

    subgraph S8["8. Testing set, used once"]
        E1["predict_ensemble<br/>all fold models"]
        E2["class_views + bootstrap_views<br/>plot_final_results"]
        E3["paired_bootstrap_views<br/>vs the 02_1 model"]
    end

    OUT[("classifier/final/noncancer_noninvasive_invasive_tuned/<br/>ensemble_date-time.pt")]

    FEAT --> D1
    LAB --> D1
    D1 --> D2
    D2 --> D3
    D3 --> B1
    B1 --> B2
    B2 --> C1
    C1 --> C2
    C2 --> N1
    N1 --> E1
    E1 --> E2
    E1 --> E3
    OLD --> E3
    N1 --> OUT
```

- **Classes:** Non-cancer (N, PB, UDH, FEA), non-invasive atypia/neoplasia (ADH, DCIS), invasive carcinoma (IC). ADH is joined to DCIS because the direct four-class model could not separate them (notebook `02_2_vector_to_classifier_finetunning_4class.ipynb`).
- **Search:** optuna TPE over `lr`, `weight_decay`, `dropout`, `coarse_weight` (two tiers: Non-cancer vs lesion), `k_sample` and `max_tiles` (2048, 4096, 8192, None); 20 trials, each a 5-fold patient-grouped cross-validation scored by the pooled out-of-fold macro-F1. The study (`three_class`) is stored in `outputs/02_2_tuning/study.db` and can be resumed.
- **Choice:** recall floor per class (`MIN_CLASS_RECALL`), best macro-F1, one-standard-error rule, confirmation of the top 2 settings with 2 repeats and new seeds. The Testing set is used once, at the end.
- **Output:** the ensemble of the fold models of the chosen setting (5 folds x 2 repeats), `classifier/final/noncancer_noninvasive_invasive_tuned/ensemble_<date-time>.pt` (`utils.mil_tuning.save_ensemble` / `load_ensemble`).

