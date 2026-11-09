# IMPORTS *****************************************************************************************
# External libraries
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

# Internal imports ***************************************************************************
from utils.figure_captions import split_with_notes

# Helper functions ********************************************************************************
def pca_rgb(x, k=3):
    """Project rows of x on their top-k principal components, scaled to [0, 1] per component."""
    x = x - x.mean(0, keepdim=True)
    _, _, v = torch.pca_lowrank(x, q=k, center=False)
    z = (x @ v[:, :k]).numpy()
    lo, hi = np.percentile(z, 1, axis=0), np.percentile(z, 99, axis=0)
    return np.clip((z - lo) / (hi - lo + 1e-8), 0, 1)


# Main function ***********************************************************************************
def inspect_feature_file(
    feature_file: str | Path | None = None,
    features_dir: str | Path | None = None,
    model_name: str | None = None,
    n_show: int = 8,
    dataset=None,
    show: bool = True,
):
    """
    Print a summary of one feature file and plot its CLS embeddings.

    The summary covers the path, size, shapes, dtypes, value statistics, tiling parameters and the
    first tile coordinates. The figure has three panels, all computed from the CLS vectors: the slide
    atlas (top-3 PCA components as RGB), the distribution of the CLS norm (the Euclidean length of
    each tile's vector) and the spread of every embedding dimension across the tiles. Patch tokens,
    if the file has them, are not plotted.

    * Arguments:
    feature_file: The .pt file to inspect. When None, the first file in features_dir / model_name
    features_dir: The folder that holds one subfolder per model (used when feature_file is None)
    model_name: The model subfolder name (used when feature_file is None)
    n_show: The number of first tile coordinates printed in the text summary
    dataset: Optional WSITileDataset, used for the tiling parameters of files that do not store them
    show: Whether to call plt.show() (the figure is returned either way)

    Returns the figure.
    """

    # STEP 1: Resolve the file to inspect
    if feature_file is None:
        if features_dir is None or model_name is None:
            raise ValueError("Pass feature_file, or both features_dir and model_name.")

        folder = Path(features_dir) / model_name
        files = sorted(folder.glob("*.pt"))

        if not files:
            raise FileNotFoundError(f"No .pt files found in {folder.resolve()}")

        feature_file = files[0]

    feature_file = Path(feature_file)

    # STEP 3: Load the file and pull out its parts (float32 copies for stable statistics)
    data = torch.load(feature_file, map_location="cpu", weights_only=True)
    cls = data["cls_features"].float()          # [T, D]
    patch = data["patch_features"].float() if "patch_features" in data else None   # [T, P, D] (absent in CLS-only files)
    coords = data["coordinates"].numpy()        # [T, 2] (x, y) in pixels of the tiled level
    params = data.get("tile_params")
    params_source = "stored in the file"

    # Older files have no tile_params: fall back to this notebook's dataset (may differ from when the file was made)
    if params is None and dataset is not None and hasattr(dataset, "get_config"):
        params = dataset.get_config()
        params_source = "NOT in the file; taken from this notebook's current `dataset`"
    T, D = cls.shape
    P = patch.shape[1] if patch is not None else 0
    grid = round(P ** 0.5)
    n = min(n_show, T)

    # STEP 4: Text summary
    print(f"Path      : {feature_file.resolve()}")
    print(f"Modified  : {datetime.fromtimestamp(feature_file.stat().st_mtime):%Y-%m-%d %H:%M:%S}")
    print(f"File      : {feature_file.name}  ({feature_file.stat().st_size / 1e6:.1f} MB)")
    print(f"Encoder   : {data['model_encoder']}")
    print(f"Tiles     : {T}   |   embedding dim: {D}   |   " + (f"patch grid: {grid}x{grid} ({P} tokens)" if patch is not None else "no patch features (CLS-only file)"))
    print(f"Dtypes    : cls={data['cls_features'].dtype}, patch={data['patch_features'].dtype if patch is not None else None}, coords={data['coordinates'].dtype}")
    print(f"Finite    : {bool(torch.isfinite(cls).all() and (patch is None or torch.isfinite(patch).all()))}")
    print(f"CLS stats : mean={cls.mean():+.3f}  std={cls.std():.3f}  min={cls.min():+.2f}  max={cls.max():+.2f}")
    if params:
        flt = params["tile_filter"]
        print(f"Tiling    : level={params['level']}  tile_size={params['tile_size']}  stride={params['stride']}  ({params_source})")
        print(f"Filter    : {flt['name']} {flt['params']}" if flt else "Filter    : none")
    else:
        print("Tiling    : unknown (not in the file, and no `dataset` with get_config() in this notebook)")
    print("First coords (x, y):", coords[:n].tolist())

    # STEP 5: Derived quantities
    cls_norm = cls.norm(dim=1).numpy()

    # Slide atlas: one pixel per tile, coloured by the top-3 PCA components of its CLS vector
    stride = params["stride"] if params else int(np.gcd.reduce(np.abs(np.diff(np.unique(coords[:, 0])))) or 1)
    gx, gy = coords[:, 0] // stride, coords[:, 1] // stride
    gx, gy = gx - gx.min(), gy - gy.min()
    atlas = np.ones((gy.max() + 1, gx.max() + 1, 3))
    atlas[gy, gx] = pca_rgb(cls)

    # STEP 6: Figure (styled with rc_context so the global matplotlib settings are left alone)
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        fig = plt.figure(figsize=(16, 7.5), layout="constrained")
        fig.suptitle(f"{feature_file.stem}  ·  {data['model_encoder']}  ·  {T} tiles × {D}-d", fontsize=16, weight="bold")

        # Captions under each panel: what the result means and how to act on it
        notes = [
            (0, 1, "Each tile is drawn at its position on the slide and coloured by the projection of its CLS embedding onto the first three principal components (PCA fitted on this slide; each component rescaled to [0, 1] between its 1st and 99th percentile and mapped to red, green and blue).\n"
                   "Similar colours denote similar embeddings. Spatially coherent regions that follow the tissue morphology indicate that the embedding encodes tissue structure; colour noise unrelated to the tissue layout indicates that it does not.\n"
                   "Colours are relative to this slide's own PCA and cannot be compared between slides."),
            (1, 2, "Distribution of the CLS norm over the tiles of the slide; the CLS norm is the Euclidean (L2) length of the CLS embedding, ‖h‖₂ = (Σ_d h_d²)^½. Dashed line, median. The norm depends on image content; it is not a measure of image quality or of diagnostic relevance.\n"
                   "A narrow unimodal distribution indicates homogeneous tissue; several modes or a heavy tail indicate distinct tissue types or atypical tiles. An accumulation at low values may reflect mostly blank or out-of-focus tiles that passed the tissue filter. Extreme tiles (pen marks, folds, blur, glass) should be inspected before they enter later analyses."),
            (2, 3, "Standard deviation of each embedding dimension across the tiles of this slide, sorted in decreasing order.\n"
                   "A steep decline means that the variation between tiles is concentrated in few dimensions; a flat curve means that it is spread over many. Dimensions near zero are practically constant on this slide.\n"
                   "For classification, the relevant variation is the one between slides and classes (Sections 4.2 and 4.4), not within one slide."),
        ]

        # One row: slide atlas, CLS norm histogram, spread of every dimension
        panels = split_with_notes(fig, notes, [1.2, 1, 1], 16, plot_height=4.2, note_height=3.3)
        ax_atlas, ax_hist, ax_std = panels.subplots(1, 3, width_ratios=[1.2, 1, 1])

        ax_atlas.imshow(atlas, interpolation="nearest")
        ax_atlas.set_title("Slide atlas: tissue appearance as colour")
        ax_atlas.set_xticks([]); ax_atlas.set_yticks([])
        ax_atlas.spines[:].set_visible(False)

        ax_hist.hist(cls_norm, bins=40, color="#4c78a8", edgecolor="white")
        ax_hist.axvline(np.median(cls_norm), color="#e45756", ls="--", label=f"median {np.median(cls_norm):.1f}")
        ax_hist.set_title("How varied are the tiles? (CLS norm)"); ax_hist.set_xlabel("norm"); ax_hist.set_ylabel("tiles")
        ax_hist.legend(frameon=False)

        ax_std.plot(np.sort(cls.std(0).numpy())[::-1], color="#54a24b")
        ax_std.set_title("Which dimensions vary between tiles?"); ax_std.set_xlabel("dimension rank"); ax_std.set_ylabel("std")

    if show:
        plt.show()

    return fig
