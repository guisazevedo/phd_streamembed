# IMPORTS *****************************************************************************************
# External libraries
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

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
        fig = plt.figure(figsize=(16, 11), layout="constrained")
        fig.suptitle(f"{feature_file.stem}  ·  {data['model_encoder']}  ·  {T} tiles × {D}-d", fontsize=14, weight="bold")
        rows = fig.subfigures(3, 1, height_ratios=[1.4, 1.0, 1.0])

        # Row 1: slide atlas, CLS norm map, CLS norm histogram
        ax_atlas, ax_pos, ax_hist = rows[0].subplots(1, 3, width_ratios=[1.2, 1, 1])
        ax_atlas.imshow(atlas, interpolation="nearest")
        ax_atlas.set_title("Slide atlas: CLS top-3 PCA as RGB (1 px = 1 tile)")
        ax_atlas.set_xticks([]); ax_atlas.set_yticks([])
        ax_atlas.spines[:].set_visible(False)

        sc = ax_pos.scatter(coords[:, 0], coords[:, 1], c=cls_norm, s=14, marker="s", cmap="magma")
        ax_pos.invert_yaxis(); ax_pos.set_aspect("equal")
        ax_pos.set_title("Tile positions, coloured by CLS norm")
        rows[0].colorbar(sc, ax=ax_pos, shrink=0.7)

        ax_hist.hist(cls_norm, bins=40, color="#4c78a8", edgecolor="white")
        ax_hist.axvline(np.median(cls_norm), color="#e45756", ls="--", label=f"median {np.median(cls_norm):.1f}")
        ax_hist.set_title("CLS L2-norm distribution"); ax_hist.set_xlabel("norm"); ax_hist.set_ylabel("tiles")
        ax_hist.legend(frameon=False)

        # Row 2: patch-token PCA maps of the first n tiles
        if patch_rgb is None:
            rows[1].text(0.5, 0.5, "No patch features in this file (CLS-only)", ha="center", va="center", fontsize=12, color="gray")
        else:
            axes = np.atleast_1d(rows[1].subplots(1, n))
            for i, ax in enumerate(axes):
                ax.imshow(patch_rgb[i], interpolation="nearest")
                ax.set_title(f"tile {i}\n({coords[i, 0]}, {coords[i, 1]})", fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
            axes[0].set_ylabel(f"patch tokens\n{grid}×{grid} PCA", fontsize=9)

        # Row 3: CLS vectors of the first n tiles, similarity between them, per-dimension spread
        ax_vec, ax_sim, ax_std = rows[2].subplots(1, 3, width_ratios=[1.4, 1, 1])
        vmax = float(cls[:n].abs().quantile(0.99))
        im = ax_vec.imshow(cls[:n].numpy(), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax_vec.set_title(f"CLS vectors, first {n} tiles"); ax_vec.set_xlabel("embedding dimension"); ax_vec.set_ylabel("tile")
        rows[2].colorbar(im, ax=ax_vec, shrink=0.8)

        im = ax_sim.imshow(sim, cmap="viridis", vmin=min(sim.min(), 0), vmax=1)
        ax_sim.set_title("Cosine similarity (first tiles)"); ax_sim.set_xlabel("tile"); ax_sim.set_ylabel("tile")
        rows[2].colorbar(im, ax=ax_sim, shrink=0.8)

        ax_std.plot(np.sort(cls.std(0).numpy())[::-1], color="#54a24b")
        ax_std.set_title("Per-dimension std across tiles (sorted)"); ax_std.set_xlabel("dimension rank"); ax_std.set_ylabel("std")

    if show:
        plt.show()

    return fig
