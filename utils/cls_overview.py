# IMPORTS *****************************************************************************************
# External libraries
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# Internal imports ***************************************************************************
from utils.cls_data import fit_pca, kmeans, load_cls_features
from utils.figure_captions import split_with_notes

# Design tokens ***********************************************************************************
# Validated reference palette: categorical slots 1-3, blue sequential ramp, light surface
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
# Cluster colours: the three palette colours first, then extra distinct hues (not colour-blind-validated)
CLUSTER_COLORS = SERIES + ["#8e5bd0", "#e0b020", "#d6469a", "#2aa5c9", "#7a5c3e", "#8a8f98", "#b23a3a", "#6b8e23", "#1f3a8a"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
STYLE = ({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


# Helper functions ********************************************************************************
def cluster_colors(k):
    """Return k colours for k clusters; beyond the named list, cycle through matplotlib's tab20."""
    extra = [plt.get_cmap("tab20")(i % 20) for i in range(max(0, k - len(CLUSTER_COLORS)))]
    return (CLUSTER_COLORS + extra)[:k]


# Main function ***********************************************************************************
def plot_cls_overview(
    features_dir: str | Path,
    model_name: str,
    max_slides: int | None = None,
    n_clusters: int = 3,
    max_fit: int = 30000,
    max_scatter: int = 10000,
    n_maps: int = 3,
    n_bars: int = 12,
    seed: int = 0,
    show: bool = True,
) -> dict:
    """
    Summarize the CLS embeddings of many slides: density, clusters, variance, spatial maps and slide effect.

    Only the CLS vectors and tile coordinates are read (files are memory-mapped, so patch tokens
    are never loaded). PCA and k-means are fitted on a random subset of tiles and then applied to
    all of them. Clusters are numbered by size, largest first.

    * Arguments:
    features_dir: The folder that holds one subfolder per model
    model_name: The model subfolder with the per-slide .pt files
    max_slides: The maximum number of slides to load (None = all)
    n_clusters: The number of k-means clusters (any number; colours beyond 3 are not colour-blind-safe)
    max_fit: The number of tiles used to fit PCA and k-means
    max_scatter: The number of tiles drawn in the cluster scatter plot
    n_maps: The number of slides drawn as cluster maps
    n_bars: The number of slides in the cluster-mix bars
    seed: The random seed (subsampling, k-means start)
    show: Whether to call plt.show() (the figure is returned either way)

    Returns a dict with the figure and the main results (cluster labels per tile, slide names,
    explained variance, cluster sizes and the two cosine-similarity samples).
    """

    # STEP 1: Validate the settings
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2.")

    colors = cluster_colors(n_clusters)

    # STEP 2: Load every slide's CLS vectors and tile coordinates
    data = load_cls_features(features_dir, model_name, max_slides)
    X, coords, slide_id, names, counts, starts = (data[k] for k in ("X", "coords", "slide_id", "names", "counts", "starts"))
    N, D = X.shape
    rng = np.random.default_rng(seed)

    # STEP 3: PCA fitted on a random subset, then applied to all tiles
    pca = fit_pca(X, max_fit, seed)
    Z, explained, fit_idx, g = pca["Z"], pca["explained"], pca["fit_idx"], pca["generator"]
    q = Z.shape[1]

    # STEP 4: k-means (Lloyd, k-means++ start) on the PCA coordinates; clusters are numbered by size, largest first
    centers = kmeans(Z[fit_idx.numpy()], n_clusters, g)
    labels = torch.cdist(torch.as_tensor(Z), centers).argmin(1).numpy()
    order = np.argsort(-np.bincount(labels, minlength=n_clusters))
    labels = np.argsort(order)[labels]                      # relabel: 0 = largest cluster
    cluster_size = np.bincount(labels, minlength=n_clusters)

    # STEP 5: Cosine similarity of random tile pairs, from the same slide vs from different slides
    Xn = torch.nn.functional.normalize(X, dim=1)
    n_pairs = 20000
    i = rng.integers(0, N, n_pairs)
    j_same = starts[slide_id[i]] + (rng.random(n_pairs) * counts[slide_id[i]]).astype(int)
    j_diff = rng.integers(0, N, n_pairs)
    keep = slide_id[j_diff] != slide_id[i]
    cos_same = (Xn[i] * Xn[j_same]).sum(1).numpy()
    cos_diff = (Xn[i[keep]] * Xn[j_diff[keep]]).sum(1).numpy()

    # STEP 6: Text summary
    n90 = int(np.searchsorted(np.cumsum(explained), 90)) + 1
    print(f"Slides {len(names)}  |  tiles {N:,}  |  CLS dim {D}  |  tiles per slide: median {int(np.median(counts))}, min {counts.min()}, max {counts.max()}")
    print(f"PCA       : PC1 {explained[0]:.1f}%, PC2 {explained[1]:.1f}% of variance; {n90} PCs reach 90% (of the {q} computed)")
    print("Clusters  : " + ", ".join(f"C{c + 1} {cluster_size[c]:,} ({100 * cluster_size[c] / N:.0f}%)" for c in range(n_clusters)))
    print(f"Cosine    : same-slide pairs {cos_same.mean():.3f}  vs  different-slide pairs {cos_diff.mean():.3f}")

    # STEP 7: Figure (styled with rc_context so the global matplotlib settings are left alone)
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(17, 20), layout="constrained")
        fig.suptitle(f"CLS embeddings · {model_name} · {N:,} tiles from {len(names)} slides", fontsize=16, weight="bold", x=0.01, ha="left")
        rows = fig.subfigures(3, 1, height_ratios=[7.1, 6.8, 5.7])

        # Captions under each panel: what the result means and how to act on it
        notes_pca = [
            (0, 1, "Density of all tiles in the plane of the first two principal components (PCA fitted on a random subset of tiles and applied to all); colour, number of tiles per hexagonal bin on a logarithmic scale.\n"
                   "Separate high-density regions indicate distinct groups of tile appearance; a single mass indicates a uniform or continuous distribution. Isolated distant tiles are candidate outliers (artefacts, debris). Only two components are shown; the variance panel gives the fraction of the variance they represent."),
            (1, 2, "The same plane, with tiles coloured by k-means cluster (clustering in the 50-component PCA space; clusters numbered by size, C1 the largest). The clustering is unsupervised: clusters are groups of similar appearance, not diagnostic categories.\n"
                   "Clusters that overlap in this projection may still be separable in the remaining components. Their meaning has to be established from the cluster maps and example tiles, and the choice of k is assessed in Section 4.3."),
            (2, 3, "Fraction of the total variance explained by each principal component (bars) and cumulatively (line); dashed line, 90%.\n"
                   "If few components reach 90%, the embedding is effectively low-dimensional and the two-dimensional projections are representative. A slow increase means that the information is spread over many directions and these projections show only a small part of it.\n"
                   "A dominant first component can reflect a slide-level effect (staining, scanner) instead of tissue; see the slide-effect panel."),
        ]
        notes_maps = [
            (0, n_maps, "Cluster assignment of every tile, drawn at its position on three slides. If a cluster follows recognisable morphological regions (for example tumour, stroma, fat, background) consistently across slides, the embedding separates these structures without supervision.\n"
                   "Spatially scattered assignments, or clusters that occur on a single slide only, suggest that they reflect noise or slide-specific appearance (stain, scanner) rather than tissue. A cluster can be named by inspecting its tiles with plot_wsi."),
            (n_maps, n_maps + 1, "Composition of each slide as the fraction of its tiles in each cluster.\n"
                   "Slides with similar composition consist of similar tissue types. Differences in composition are useful for classification only if they follow the diagnosis; if slides group by scanner, site or staining batch instead, a classifier may exploit these confounders rather than the disease."),
        ]
        notes_effect = [
            (0, 1, "Distribution of the cosine similarity between 20,000 random pairs of tiles drawn either from the same slide (orange) or from different slides (blue).\n"
                   "Higher similarity within than between slides indicates a slide-specific component in the embedding. Part of it is expected, because each slide has its own tissue content, but it can also reflect staining, scanner or patient effects.\n"
                   "Training, validation and test sets should therefore be split by slide or patient, never by tile; otherwise a classifier can reach high accuracy by recognising the slide, and its performance will not generalise. Overlapping distributions indicate little slide-specific signal."),
            (1, 2, "Number of tiles per slide; dashed line, median.\n"
                   "Slides with many more tiles contribute proportionally more to every tile-level statistic above (PCA, clustering) and to a tile-level training loss. In classification, tiles should be pooled or sampled per slide, or slides weighted, so that large slides do not dominate."),
        ]

        cluster_names = [f"C{c + 1}" for c in range(n_clusters)]
        sub = rng.choice(N, min(N, max_scatter), replace=False)

        # Row 1: density, clusters, variance
        top = split_with_notes(rows[0], notes_pca, [1, 1, 0.9], 17, plot_height=4.6, note_height=2.5)
        ax_d, ax_c, ax_v = top.subplots(1, 3, width_ratios=[1, 1, 0.9])
        hb = ax_d.hexbin(Z[:, 0], Z[:, 1], gridsize=45, cmap=SEQ, mincnt=1, bins="log", linewidths=0)
        top.colorbar(hb, ax=ax_d, shrink=0.8, label="tiles per cell (log)")
        ax_d.set_title("Where do the tiles live? (density)"); ax_d.set_xlabel("PC1"); ax_d.set_ylabel("PC2")

        for c in range(n_clusters):
            m = labels[sub] == c
            ax_c.scatter(Z[sub][m, 0], Z[sub][m, 1], s=7, color=colors[c], alpha=0.45, linewidths=0)
        ax_c.set_title(f"Which tiles group together? (k-means, k={n_clusters})"); ax_c.set_xlabel("PC1"); ax_c.set_ylabel("PC2")
        ax_c.legend(handles=[Line2D([], [], marker="o", ls="", color=colors[c], label=f"{cluster_names[c]} · {100 * cluster_size[c] / N:.0f}%") for c in range(n_clusters)],
                    frameon=False, loc="best")
        ax_c.set_xlim(ax_d.get_xlim()); ax_c.set_ylim(ax_d.get_ylim())

        k_show = min(20, len(explained))
        ax_v.bar(np.arange(1, k_show + 1), explained[:k_show], color=SERIES[0], width=0.7, label="per PC")
        ax_v.plot(np.arange(1, k_show + 1), np.cumsum(explained)[:k_show], color=INK_2, lw=2, label="cumulative")
        ax_v.axhline(90, color=GRID, lw=1.2, ls="--")
        ax_v.set_title("How many directions matter? (variance)"); ax_v.set_xlabel("principal component"); ax_v.set_ylabel("% of variance")
        ax_v.legend(frameon=False, loc="center right")
        ax_v.set_ylim(0, 100)
        ax_v.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Row 2: cluster maps of three slides + composition of the first slides
        middle = split_with_notes(rows[1], notes_maps, [1] * n_maps + [1.25], 17, plot_height=4.2, note_height=2.6)
        map_axes = middle.subplots(1, n_maps + 1, width_ratios=[1] * n_maps + [1.25])
        for ax, k in zip(map_axes[:n_maps], range(min(n_maps, len(names)))):
            sel = slide_id == k
            ax.scatter(coords[sel, 0], coords[sel, 1], c=[colors[l] for l in labels[sel]], s=10, marker="s", linewidths=0)
            ax.invert_yaxis(); ax.set_aspect("equal"); ax.grid(False)
            ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)
            ax.set_title(f"{names[k]} · {sel.sum()} tiles")

        ax_b = map_axes[-1]
        nb = min(n_bars, len(names))
        frac = np.stack([np.bincount(labels[slide_id == k], minlength=n_clusters) / max(counts[k], 1) for k in range(nb)])
        left = np.zeros(nb)
        for c in range(n_clusters):
            ax_b.barh(np.arange(nb), frac[:, c], left=left, color=colors[c], edgecolor=SURFACE, linewidth=1.5, label=cluster_names[c])
            left += frac[:, c]
        ax_b.set_yticks(np.arange(nb)); ax_b.set_yticklabels([names[k] for k in range(nb)], fontsize=9)
        ax_b.invert_yaxis(); ax_b.set_xlim(0, 1); ax_b.grid(False)
        ax_b.set_title("What is each slide made of? (cluster mix)"); ax_b.set_xlabel("share of tiles")
        ax_b.legend(frameon=False, ncol=min(n_clusters, 6), loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=10)

        # Row 3: slide effect and dataset size
        bottom = split_with_notes(rows[2], notes_effect, [1.4, 1], 17, plot_height=3.7, note_height=2.0)
        ax_s, ax_n = bottom.subplots(1, 2, width_ratios=[1.4, 1])
        bins = np.linspace(min(cos_same.min(), cos_diff.min()), max(cos_same.max(), cos_diff.max()), 60)
        ax_s.hist(cos_diff, bins=bins, density=True, histtype="step", lw=2, color=SERIES[0], label=f"different slides (mean {cos_diff.mean():.2f})")
        ax_s.hist(cos_same, bins=bins, density=True, histtype="step", lw=2, color=SERIES[1], label=f"same slide (mean {cos_same.mean():.2f})")
        ax_s.set_title("Do tiles from one slide look alike? (slide effect)")
        ax_s.set_xlabel("cosine similarity"); ax_s.set_ylabel("density"); ax_s.legend(frameon=False)

        ax_n.hist(counts, bins=min(30, max(5, len(names) // 2)), color=SERIES[0], edgecolor=SURFACE, linewidth=1.5)
        ax_n.axvline(np.median(counts), color=INK_2, ls="--", lw=1.5)
        ax_n.set_title("How big is each slide? (tiles per slide)"); ax_n.set_xlabel("tiles"); ax_n.set_ylabel("slides")
        ax_n.text(np.median(counts), ax_n.get_ylim()[1] * 0.95, f" median {int(np.median(counts))}", color=INK_2, va="top")

    if show:
        plt.show()

    return {
        "figure": fig,
        "slide_names": names,
        "slide_id": slide_id,
        "coordinates": coords,
        "labels": labels,
        "cluster_sizes": cluster_size,
        "explained_variance": explained,
        "cos_same": cos_same,
        "cos_diff": cos_diff,
    }
