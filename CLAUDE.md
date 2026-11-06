# CLAUDE.md

PhD research code on the BRACS breast-pathology data set (547 whole-slide images, 7 classes): slides -> tile embeddings with UNI2-h, saved as one `.pt` file per slide, for a later classifier. See `README.md` for the overview, the data set facts, the output format and the pipeline diagrams.

## Commands

```bash
uv sync                                   # install dependencies (Python >= 3.14, uv.lock)
uv run jupyter lab                        # run the notebooks
uv run python setup_only/model_download.py   # fetch UNI2-h weights into original_vit_model/
```

There is no test suite and no linter configured. Verify changes by running the code on real or synthetic data (the label sheet and the slides in `data/` exist locally; feature files may not). When writing throwaway scripts, put them in the session scratchpad, not in the repo. Run Python from the repository root (`utils` is imported as a top-level package; `.venv/bin/python` works without `uv run`).

## Notebooks

Numbered, in order. Keep each one thin (logic belongs in `utils/`), with a markdown cell before every code cell and the figure explanation in the figure captions.

- `00_data_visualization.ipynb`: the data set. Labels and class definitions, audit of the official split, RoI composition, class weights, slide properties, pyramid levels, tile examples. Writes `slides.csv` and `class_weights.csv` to `outputs/00_data_overview/`.
- `01_wsi_to_vectors.ipynb`: tiling -> UNI2-h embeddings -> inspection (one file, all slides, choice of k, embeddings vs the 7 classes).
- `02_vectors_to_classifier.ipynb`: multiple-instance classifier (mean-pool linear baseline and gated ABMIL) on the saved CLS embeddings; official split, Validation for every choice, Testing evaluated once; full metrics with patient-bootstrap intervals; results are only shown on screen; the one file it writes is the best classifier (chosen on Validation macro-F1, never on Testing) to `classifier/classifier_model_<date-time>.pt`.
- `03_reidentification_risk_assessment.ipynb`: planned, empty. When a notebook is created or changed, update its Mermaid diagram in `README.md` (diagrams are in notebook order).

## Layout

- Pipeline: `utils/dataset.py` (`WSITileDataset`, `get_config()` returns the tiling parameters), `generate_features.py` (model run, per-slide `.pt` files, `save_patch_features` flag, background saving thread), `dataloader.py`, `clean_empty.py`, `transformer_basic.py`, `transformer_per_model.py`, `load_model.py`, `device.py`.
- Data set (notebook 00): `utils/labels.py` (reads `labels/BRACS.xlsx` with the standard library: `load_bracs_labels`, `load_bracs_roi_distribution`, `BRACS_CLASSES`), `slide_stats.py` (`scan_slides`, header scan), `dataset_overview.py` (`build_slide_table`, `validate_labels`, `split_audit_stats`, `plot_class_distribution`, `plot_split_audit`, `plot_roi_composition`, `plot_slide_properties`, `build_summary`), `class_weights.py` (`compute_class_weights`, `weighted_cross_entropy`, `plot_loss_weighting`), `plot_pyramid.py` (`plot_pyramid_levels`, `plot_class_gallery`, `plot_tile_examples`).
- Embeddings (notebook 01): `utils/plot_wsi.py`, `inspect_features.py`, `cls_overview.py`, `cluster_quality.py`, `class_analysis.py`; `cls_data.py` holds what they share (loading, PCA, k-means, silhouette, Davies-Bouldin, ARI, NMI).
- Classifier (notebook 02): `utils/mil_data.py` (`load_bags`, padded/masked batches, scaler fitted on Training), `mil_models.py` (`MeanPoolLinear`, `GatedABMIL`, `build_model`), `mil_train.py` (`set_seed`, `train_model`, `run_seeds`, `predict_ensemble`, `compare_weightings`, `save_best_model`, `load_classifier`), `classification_metrics.py` (scikit-learn metrics, `clinical_views`, `patient_bootstrap`, `paired_bootstrap`), `plot_classification.py` (all figures of notebook 02).
- `utils/figure_captions.py`: `split_with_notes` writes the caption under each panel; keep captions and the notebook markdown in sync.
- `setup_only/`: one-off helper scripts with a CONFIG block at the top; not part of the pipeline.
- `labels/BRACS.xlsx`: versioned. `original_vit_model/`, `data/`: large, git-ignored. `outputs/`: tables written by notebook 00 (decide with the user whether to version them).

## Code style (match the existing files)

- Section banners: `# IMPORTS ****...`, `# Internal imports ****...`, `# Main function ****...`, each 100 columns wide.
- Inside functions, number the logic with `# STEP n: ...` comments.
- Docstrings list parameters under `* Arguments:` as `name: description`.
- Type hints use modern syntax (`str | None`). Prefer small private helpers with a leading underscore.
- Plot functions take `show: bool = True`, return the figure (or a dict containing it), and style themselves with `plt.rc_context` so global matplotlib settings are untouched.
- Charts use the validated dataviz palette (categorical blue/orange/aqua, blue sequential ramp, blue-orange diverging for residuals); the first 3 categorical colours are colour-blind-safe in scatter and map plots. `plot_cls_overview` accepts any `n_clusters`, with extra non-validated colours beyond 3 (the user's choice).
- **Captions and notebook text are written for a PhD thesis**: concise academic prose that states what is plotted in statistical terms, how to interpret it, and the caveat or consequence. No "what / how to read" or "good / warning" labels, no informal rules of thumb without a source. Separate paragraphs with a blank line. Cite the standard reference when a measure has one (silhouette: Rousseeuw 1987; Davies-Bouldin 1979; adjusted Rand index: Hubert and Arabie 1985; class-balanced loss: Cui et al. 2019). Do not state results in a caption that depend on the data unless they are computed into it.

## Gotchas

- **Slide order matters.** `generate_features` writes a slide's file when the next slide starts, so the loader must use `shuffle=False` and tiles must stay slide-major. DataLoader workers keep batch order, so `num_workers > 0` is fine.
- **Pickling.** Python 3.14 uses `forkserver`, so the dataset is pickled for worker processes. `WSITileDataset.__getstate__` drops the cached pyvips image. The `tile_filter` must be picklable (a `functools.partial` of a module-level function works, a lambda does not) whenever `index_workers > 1`.
- **Pyramid levels.** Level 1 is full resolution; higher is more downsampled. 12 of the 547 slides have only 3 levels (skipped at level 4), and the downsample of level 4 is 32x or 64x depending on the slide. Tile counts grow about 16x per level step toward level 1; a full-resolution level cannot be loaded as one array (read regions lazily, see `plot_pyramid._read_region`).
- **Damaged data.** Some SVS files have unreadable JPEG tiles (for example `BRACS_1791.svs` at level 2). Indexing skips those tiles when a tile filter is set and records them in `dataset.unreadable_tiles`.
- **Output size.** Patch tokens are about 99% of a feature file. Default to `save_patch_features=False` unless patch-level work is planned.
- **Old feature files** may lack `tile_params`; the inspection code handles that.
- **Official split.** Use the BRACS Training/Validation/Testing split unchanged (comparability with published work). Known caveats, all reported by notebook 00: patient 67 has slides in Training and Validation; 78 of 189 patients have slides of several classes; Validation and Testing classes are small. Group by patient in any cross-validation inside Training; never split by tile.
- **Labels are weak.** A slide label is its highest-grade RoI and describes the tissue, not the person. Tile labels inherited from the slide are noisy.
- **Class weights come from the Training set only.** Evaluate with unweighted balanced accuracy and per-class recall on untouched Validation/Testing. `class_analysis` uses the same `compute_class_weights`.
- **Test-set discipline (notebook 02).** `train_model` is given labels of Training and Validation slides only (Testing labels are not even available inside it); loss weighting and the epoch are chosen on Validation; the Testing set is first used in section 6 and nothing is changed afterwards. Keep it that way in any new code.
- **Seed and device.** One `SEED` (`mil_train.set_seed`) drives every random choice; run i of a model uses `SEED + i`. Code is device-agnostic: the device comes from `utils.device.get_device()`, there are no `.cuda()` calls, and bf16 autocast is enabled only when `device.type == "cuda"`. This machine has no GPU and no feature files (they are on the workstation): test with synthetic features that carry real BRACS slide names (as done in the session scratchpad) on the CPU, and say that the GPU path was not run here.
- **Possible batch effect.** Scan year is associated with class (Cramer's V about 0.27) and two MPP values exist (0.2519, 0.2524); check this before interpreting classifier results.
- **Dependencies.** scikit-learn (and its dependency scipy) was added with the user's approval for the classification metrics of notebook 02 (`classification_metrics.py`); the earlier analyses (permutation tests, own k-means, ARI, NMI, silhouette) stay in numpy/torch. seaborn and openpyxl are not installed; do not add any other package without asking. The xlsx is parsed with the standard library.
- Data and features live on a remote workstation (`/mnt/Data/bracs/data`, `/mnt/Scratch/phd_streamembed_features`); paths in the notebooks may not exist on this machine (locally the slides are in `data/`), so confirm the host before assuming a file is missing.
- Editing a notebook: it is edited on disk as JSON (keep `indent=1`, `ensure_ascii=False`, new cells need an `id`). Warn the user to reload it in Jupyter so an open copy does not overwrite the edit. Clear the outputs of cells whose code changed.
- `utils/__pycache__/*.pyc` files show up in `git status` (no `__pycache__` entry in `.gitignore`); do not add them to commits.
- `todo_list.py` is the user's plain list of TODO comments, not code to run.

## Conventions for changes

- Keep behaviour backwards compatible for existing flags and file formats unless asked; new flags default to the old behaviour.
- Anything that changes how features are produced should also be reflected in `tile_params` or the output keys, so a file stays self-describing.
- Do not commit unless asked.
