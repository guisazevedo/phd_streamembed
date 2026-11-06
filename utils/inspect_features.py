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
    Print a summary of one feature file and plot its tiles' embeddings.

    The summary covers the path, size, shapes, dtypes, value statistics, tiling parameters and the
    first tile coordinates. The figure shows the slide atlas (CLS top-3 PCA as RGB), the tile
    positions by CLS norm, the CLS norm distribution, patch-token PCA maps of the first tiles
    (only when the file has patch features), and the CLS vectors, their cosine similarity and the
    spread of every dimension.

    * Arguments:
    feature_file: The .pt file to inspect. When None, the first file in features_dir / model_name
    features_dir: The folder that holds one subfolder per model (used when feature_file is None)
    model_name: The model subfolder name (used when feature_file is None)
    n_show: The number of first tiles shown in the "first N" panels
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
    cls_unit = torch.nn.functional.normalize(cls, dim=1)
    sim = (cls_unit[:n] @ cls_unit[:n].T).numpy()


    # Slide atlas: one pixel per tile, coloured by the top-3 PCA components of its CLS vector
    stride = params["stride"] if params else int(np.gcd.reduce(np.abs(np.diff(np.unique(coords[:, 0])))) or 1)
    gx, gy = coords[:, 0] // stride, coords[:, 1] // stride
    gx, gy = gx - gx.min(), gy - gy.min()
    atlas = np.ones((gy.max() + 1, gx.max() + 1, 3))
    atlas[gy, gx] = pca_rgb(cls)

    # Patch maps: joint PCA over the first n tiles, so colours mean the same thing across tiles
    patch_rgb = pca_rgb(patch[:n].reshape(-1, patch.shape[-1])).reshape(n, grid, grid, 3) if patch is not None else None


    # STEP 6: Figure (styled with rc_context so the global matplotlib settings are left alone)
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}):
        fig = plt.figure(figsize=(16, 18), layout="constrained")
        fig.suptitle(f"{feature_file.stem}  ·  {data['model_encoder']}  ·  {T} tiles × {D}-d", fontsize=14, weight="bold")
        rows = fig.subfigures(3, 1, height_ratios=[6.7, 4.6, 6.6])

        # Captions under each panel: what the result means and how to act on it
        notes_atlas = [
            (0, 1, "Each tile is drawn at its position on the slide and coloured by the projection of its CLS embedding onto the first three principal components (PCA fitted on this slide; each component rescaled to [0, 1] between its 1st and 99th percentile and mapped to red, green and blue).\n"
                   "Similar colours denote similar embeddings. Spatially coherent regions that follow the tissue morphology indicate that the embedding encodes tissue structure; colour noise unrelated to the tissue layout indicates that it does not.\n"
                   "Colours are relative to this slide's own PCA and cannot be compared between slides."),
            (1, 2, "Tiles coloured by the Euclidean (L2) norm of their CLS embedding. The norm depends on image content; it is not a measure of image quality or of diagnostic relevance.\n"
                   "Spatially isolated extremes, or a band along the slide edge, usually correspond to artefacts (pen marks, folds, blur, glass) and should be inspected in the tile view before such tiles enter later analyses."),
            (2, 3, "Distribution of the CLS norm over all tiles of the slide; dashed line, median.\n"
                   "A narrow unimodal distribution indicates homogeneous tissue; several modes or a heavy tail indicate distinct tissue types or atypical tiles, which can be located on the map to the left. An accumulation at low values may reflect mostly blank or out-of-focus tiles that passed the tissue filter."),
        ]
        notes_patch = [
            (0, n, "Patch tokens of the first tiles (a 16 × 16 grid per tile, each token covering 14 × 14 pixels of the 224 × 224 tile), projected onto three principal components computed jointly over these tiles and shown as RGB. The shared projection makes colours comparable between the tiles shown, but not with other slides.\n"
                   "Coherent regions with sharp boundaries that follow visible structures (nuclei, cytoplasm, stroma, background) indicate that the model resolves tissue components within a tile. A uniform colour or noise indicates a blank or blurred tile, or patch features that carry little information."),
        ]
        notes_cls = [
            (0, 1, "CLS embeddings of the first tiles (rows) across the embedding dimensions (columns); red, positive; blue, negative; colour scale clipped at the 99th percentile of the absolute values. Single dimensions are not interpretable; the informative pattern is the similarity between rows.\n"
                   "Similar rows correspond to similar tiles, and columns that keep the same sign in every row are dimensions that vary little within the slide. Near-identical rows for visibly different tiles would indicate collapsed embeddings."),
            (1, 2, "Pairwise cosine similarity between the embeddings on the left, an angular measure that ignores vector length: 1, same direction; 0, orthogonal; negative, opposite. The diagonal equals 1 by construction.\n"
                   "The first tiles are neighbours on the slide, so comparatively high similarity is expected, and a row with low values marks a tile that differs from its surroundings. Values near 1 for all pairs would mean that the model barely distinguishes tiles; values near 0, that it finds no shared structure even between adjacent tiles."),
            (2, 3, "Standard deviation of each embedding dimension across the tiles of this slide, sorted in decreasing order.\n"
                   "A steep decline means that the variation between tiles is concentrated in few dimensions; a flat curve means that it is spread over many. Dimensions near zero are practically constant on this slide.\n"
                   "For classification, the relevant variation is the one between slides and classes (Sections 4.2 and 4.4), not within one slide."),
        ]

        # Row 1: slide atlas, CLS norm map, CLS norm histogram
        top = split_with_notes(rows[0], notes_atlas, [1.2, 1, 1], 16, plot_height=4.2, note_height=2.5)
        ax_atlas, ax_pos, ax_hist = top.subplots(1, 3, width_ratios=[1.2, 1, 1])
        ax_atlas.imshow(atlas, interpolation="nearest")
        ax_atlas.set_title("Slide atlas: tissue appearance as colour")
        ax_atlas.set_xticks([]); ax_atlas.set_yticks([])
        ax_atlas.spines[:].set_visible(False)

        sc = ax_pos.scatter(coords[:, 0], coords[:, 1], c=cls_norm, s=14, marker="s", cmap="magma")
        ax_pos.invert_yaxis(); ax_pos.set_aspect("equal")
        ax_pos.set_title("Where are strong / weak responses? (CLS norm)")
        top.colorbar(sc, ax=ax_pos, shrink=0.7)

        ax_hist.hist(cls_norm, bins=40, color="#4c78a8", edgecolor="white")
        ax_hist.axvline(np.median(cls_norm), color="#e45756", ls="--", label=f"median {np.median(cls_norm):.1f}")
        ax_hist.set_title("How varied are the tiles? (CLS norm)"); ax_hist.set_xlabel("norm"); ax_hist.set_ylabel("tiles")
        ax_hist.legend(frameon=False)

        # Row 2: patch-token PCA maps of the first n tiles
        if patch_rgb is None:
            rows[1].text(0.5, 0.5, "No patch features in this file (CLS-only)", ha="center", va="center", fontsize=12, color="gray")
        else:
            axes = np.atleast_1d(split_with_notes(rows[1], notes_patch, [1] * n, 16, plot_height=3.0, note_height=1.6).subplots(1, n))
            for i, ax in enumerate(axes):
                ax.imshow(patch_rgb[i], interpolation="nearest")
                ax.set_title(f"tile {i}\n({coords[i, 0]}, {coords[i, 1]})", fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
            axes[0].set_ylabel(f"patch tokens\n{grid}×{grid} PCA", fontsize=9)
            axes[0].figure.suptitle("What does the model see inside a tile? Patch-token PCA maps", fontsize=10, weight="bold", x=0.005, ha="left")

        # Row 3: CLS vectors of the first n tiles, similarity between them, per-dimension spread
        bottom = split_with_notes(rows[2], notes_cls, [1.4, 1, 1], 16, plot_height=3.4, note_height=3.2)
        ax_vec, ax_sim, ax_std = bottom.subplots(1, 3, width_ratios=[1.4, 1, 1])
        vmax = float(cls[:n].abs().quantile(0.99))
        im = ax_vec.imshow(cls[:n].numpy(), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax_vec.set_title(f"The summary vectors of the first {n} tiles"); ax_vec.set_xlabel("embedding dimension"); ax_vec.set_ylabel("tile")
        bottom.colorbar(im, ax=ax_vec, shrink=0.8)

        im = ax_sim.imshow(sim, cmap="viridis", vmin=min(sim.min(), 0), vmax=1)
        ax_sim.set_title("How alike are those tiles?"); ax_sim.set_xlabel("tile"); ax_sim.set_ylabel("tile")
        bottom.colorbar(im, ax=ax_sim, shrink=0.8)

        ax_std.plot(np.sort(cls.std(0).numpy())[::-1], color="#54a24b")
        ax_std.set_title("Which dimensions vary between tiles?"); ax_std.set_xlabel("dimension rank"); ax_std.set_ylabel("std")

    if show:
        plt.show()

    return fig
