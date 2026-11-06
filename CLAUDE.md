# CLAUDE.md

PhD research code: whole-slide images (BRACS) -> tile embeddings with UNI2-h, saved as one `.pt` file per slide. See `README.md` for the overview and output format.

## Commands

```bash
uv sync                                   # install dependencies (Python >= 3.14, uv.lock)
uv run jupyter lab                        # run 01_wsi_to_vectors.ipynb
uv run python setup_only/model_download.py   # fetch UNI2-h weights into original_vit_model/
```

There is no test suite and no linter configured. Verify changes by running the code on a few real or synthetic slides. When writing throwaway scripts, put them in the session scratchpad, not in the repo. Run Python from the repository root (`utils` is imported as a top-level package).

## Layout

- `01_wsi_to_vectors.ipynb`: the workflow. Keep it thin; logic belongs in `utils/`.
- `utils/dataset.py`: `WSITileDataset` (indexes tiles with pyvips, `get_config()` returns the tiling parameters).
- `utils/generate_features.py`: runs the model and writes per-slide `.pt` files (`save_patch_features` flag, background saving thread).
- `utils/dataloader.py`, `clean_empty.py`, `transformer_basic.py`, `transformer_per_model.py`, `load_model.py`, `device.py`: the other pipeline pieces.
- `utils/plot_wsi.py`, `inspect_features.py`, `cls_overview.py`: inspection plots, each exposes one function.
- `setup_only/`: one-off helper scripts with a CONFIG block at the top; not part of the pipeline.
- `original_vit_model/`, `data/`: large, git-ignored.

## Code style (match the existing files)

- Section banners: `# IMPORTS ****...`, `# Internal imports ****...`, `# Main function ****...`, each 100 columns wide.
- Inside functions, number the logic with `# STEP n: ...` comments.
- Docstrings list parameters under `* Arguments:` as `name: description`.
- Type hints use modern syntax (`str | None`). Prefer small private helpers with a leading underscore.
- Plot functions take `show: bool = True`, return the figure (or a dict containing it), and style themselves with `plt.rc_context` so global matplotlib settings are untouched.
- Charts use the validated dataviz palette (categorical blue/orange/aqua, blue sequential ramp); categorical colours are only colour-blind-safe up to 3 groups in scatter and map plots.

## Gotchas

- **Slide order matters.** `generate_features` writes a slide's file when the next slide starts, so the loader must use `shuffle=False` and tiles must stay slide-major. DataLoader workers keep batch order, so `num_workers > 0` is fine.
- **Pickling.** Python 3.14 uses `forkserver`, so the dataset is pickled for worker processes. `WSITileDataset.__getstate__` drops the cached pyvips image. The `tile_filter` must be picklable (a `functools.partial` of a module-level function works, a lambda does not) whenever `index_workers > 1`.
- **Pyramid levels.** Level 1 is full resolution; higher is more downsampled. Slides differ in how many levels and which downsample factors they have, and slides without the requested level are skipped. Tile counts grow about 16x per level step toward level 1; a full-resolution level cannot be loaded as one array.
- **Damaged data.** Some SVS files have unreadable JPEG tiles (for example `BRACS_1791.svs` at level 2). Indexing skips those tiles when a tile filter is set and records them in `dataset.unreadable_tiles`.
- **Output size.** Patch tokens are about 99% of a feature file. Default to `save_patch_features=False` unless patch-level work is planned.
- **Old feature files** may lack `tile_params`; the inspection code handles that.
- Data and features live on a remote workstation (`/mnt/Data/bracs/data`, `/mnt/Scratch/phd_streamembed_features`); paths in the notebook may not exist on this machine, so confirm the host before assuming a file is missing.
- Editing the notebook: it is edited on disk as JSON (keep `indent=1`). Warn the user to reload it in Jupyter so an open copy does not overwrite the edit. Clear the outputs of cells whose code changed.
- `todo_list.py` is a plain list of TODO comments, not code to run.

## Conventions for changes

- Keep behaviour backwards compatible for existing flags and file formats unless asked; new flags default to the old behaviour.
- Anything that changes how features are produced should also be reflected in `tile_params` or the output keys, so a file stays self-describing.
- Do not commit unless asked.
