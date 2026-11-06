# IMPORTS *****************************************************************************************
# External libraries
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from tqdm.auto import tqdm

# Design tokens ***********************************************************************************
# Validated reference palette: categorical slots 1-3, blue sequential ramp, light surface
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
STYLE = ({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


# Helper functions ********************************************************************************
def _kmeans(z, k, generator, iters=40):
    z = torch.as_tensor(z)
    centers = z[torch.randint(len(z), (1,), generator=generator)]
    for _ in range(k - 1):
        dist = torch.cdist(z, centers).min(1).values ** 2
        centers = torch.cat([centers, z[torch.multinomial(dist / dist.sum(), 1, generator=generator)]])
    for _ in range(iters):
        labels = torch.cdist(z, centers).argmin(1)
        centers = torch.stack([z[labels == c].mean(0) if (labels == c).any() else centers[c] for c in range(k)])
    return centers


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
    n_clusters: The number of k-means clusters (colour-blind-safe up to 3)
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
    if n_clusters > len(SERIES):
        raise ValueError(
            f"n_clusters={n_clusters} is above {len(SERIES)}, the number of colour-blind-safe "
            "categorical colours for scatter and map plots."
        )

    files = sorted((Path(features_dir) / model_name).glob("*.pt"))[:max_slides]

    if not files:
        raise FileNotFoundError(f"No .pt files found in {(Path(features_dir) / model_name).resolve()}")

    # STEP 2: Load every slide's CLS vectors and tile coordinates
    cls_list, coord_list, slide_id = [], [], []
    for k, f in enumerate(tqdm(files, desc="Loading CLS vectors")):
        d = torch.load(f, map_location="cpu", weights_only=True, mmap=True)
        cls_list.append(d["cls_features"].float())
        coord_list.append(d["coordinates"].numpy())
        slide_id.append(np.full(len(d["cls_features"]), k))
    X = torch.cat(cls_list)                      # [N, D]
    coords = np.concatenate(coord_list)          # [N, 2]
    slide_id = np.concatenate(slide_id)          # [N]
    names = [f.stem for f in files]
    N, D = X.shape
    counts = np.bincount(slide_id, minlength=len(files))
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])

    # STEP 3: PCA fitted on a random subset, then applied to all tiles
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    fit_idx = torch.randperm(N, generator=g)[: min(N, max_fit)]
    mu = X[fit_idx].mean(0, keepdim=True)
    q = min(50, D, len(fit_idx) - 1)
    _, S_, V = torch.pca_lowrank(X[fit_idx] - mu, q=q, center=False, niter=4)
    total_var = ((X[fit_idx] - mu) ** 2).sum() / (len(fit_idx) - 1)
    explained = (S_ ** 2 / (len(fit_idx) - 1) / total_var).numpy() * 100        # % variance per PC
    Z = ((X - mu) @ V).numpy()                                                    # [N, q]

    # STEP 4: k-means (Lloyd, k-means++ start) on the PCA coordinates; clusters are numbered by size, largest first
    centers = _kmeans(Z[fit_idx.numpy()], n_clusters, g)
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
    print(f"Slides {len(files)}  |  tiles {N:,}  |  CLS dim {D}  |  tiles per slide: median {int(np.median(counts))}, min {counts.min()}, max {counts.max()}")
    print(f"PCA       : PC1 {explained[0]:.1f}%, PC2 {explained[1]:.1f}% of variance; {n90} PCs reach 90% (of the {q} computed)")
    print("Clusters  : " + ", ".join(f"C{c + 1} {cluster_size[c]:,} ({100 * cluster_size[c] / N:.0f}%)" for c in range(n_clusters)))
    print(f"Cosine    : same-slide pairs {cos_same.mean():.3f}  vs  different-slide pairs {cos_diff.mean():.3f}")

    # STEP 7: Figure (styled with rc_context so the global matplotlib settings are left alone)
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(17, 13), layout="constrained")
        fig.suptitle(f"CLS embeddings · {model_name} · {N:,} tiles from {len(files)} slides", fontsize=14, weight="bold", x=0.01, ha="left")
        rows = fig.subfigures(3, 1, height_ratios=[1.15, 1.0, 0.9])
        cluster_names = [f"C{c + 1}" for c in range(n_clusters)]
        sub = rng.choice(N, min(N, max_scatter), replace=False)

        # Row 1: density, clusters, variance
        ax_d, ax_c, ax_v = rows[0].subplots(1, 3, width_ratios=[1, 1, 0.9])
        hb = ax_d.hexbin(Z[:, 0], Z[:, 1], gridsize=45, cmap=SEQ, mincnt=1, bins="log", linewidths=0)
        rows[0].colorbar(hb, ax=ax_d, shrink=0.8, label="tiles per cell (log)")
        ax_d.set_title("Where tiles live: PC1 vs PC2 density"); ax_d.set_xlabel("PC1"); ax_d.set_ylabel("PC2")

        for c in range(n_clusters):
            m = labels[sub] == c
            ax_c.scatter(Z[sub][m, 0], Z[sub][m, 1], s=7, color=SERIES[c], alpha=0.45, linewidths=0)
        ax_c.set_title(f"k-means clusters (k={n_clusters}) on the same plane"); ax_c.set_xlabel("PC1"); ax_c.set_ylabel("PC2")
        ax_c.legend(handles=[Line2D([], [], marker="o", ls="", color=SERIES[c], label=f"{cluster_names[c]} · {100 * cluster_size[c] / N:.0f}%") for c in range(n_clusters)],
                    frameon=False, loc="best")
        ax_c.set_xlim(ax_d.get_xlim()); ax_c.set_ylim(ax_d.get_ylim())

        k_show = min(20, len(explained))
        ax_v.bar(np.arange(1, k_show + 1), explained[:k_show], color=SERIES[0], width=0.7, label="per PC")
        ax_v.plot(np.arange(1, k_show + 1), np.cumsum(explained)[:k_show], color=INK_2, lw=2, label="cumulative")
        ax_v.axhline(90, color=GRID, lw=1.2, ls="--")
        ax_v.set_title("How many directions matter"); ax_v.set_xlabel("principal component"); ax_v.set_ylabel("% of variance")
        ax_v.legend(frameon=False, loc="center right")
        ax_v.set_ylim(0, 100)
        ax_v.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Row 2: cluster maps of three slides + composition of the first slides
        map_axes = rows[1].subplots(1, n_maps + 1, width_ratios=[1] * n_maps + [1.25])
        for ax, k in zip(map_axes[:n_maps], range(min(n_maps, len(files)))):
            sel = slide_id == k
            ax.scatter(coords[sel, 0], coords[sel, 1], c=[SERIES[l] for l in labels[sel]], s=10, marker="s", linewidths=0)
            ax.invert_yaxis(); ax.set_aspect("equal"); ax.grid(False)
            ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)
            ax.set_title(f"{names[k]} · {sel.sum()} tiles")

        ax_b = map_axes[-1]
        nb = min(n_bars, len(files))
        frac = np.stack([np.bincount(labels[slide_id == k], minlength=n_clusters) / max(counts[k], 1) for k in range(nb)])
        left = np.zeros(nb)
        for c in range(n_clusters):
            ax_b.barh(np.arange(nb), frac[:, c], left=left, color=SERIES[c], edgecolor=SURFACE, linewidth=1.5, label=cluster_names[c])
            left += frac[:, c]
        ax_b.set_yticks(np.arange(nb)); ax_b.set_yticklabels([names[k] for k in range(nb)], fontsize=7)
        ax_b.invert_yaxis(); ax_b.set_xlim(0, 1); ax_b.grid(False)
        ax_b.set_title("Cluster mix per slide"); ax_b.set_xlabel("share of tiles")
        ax_b.legend(frameon=False, ncol=n_clusters, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)

        # Row 3: slide effect and dataset size
        ax_s, ax_n = rows[2].subplots(1, 2, width_ratios=[1.4, 1])
        bins = np.linspace(min(cos_same.min(), cos_diff.min()), max(cos_same.max(), cos_diff.max()), 60)
        ax_s.hist(cos_diff, bins=bins, density=True, histtype="step", lw=2, color=SERIES[0], label=f"different slides (mean {cos_diff.mean():.2f})")
        ax_s.hist(cos_same, bins=bins, density=True, histtype="step", lw=2, color=SERIES[1], label=f"same slide (mean {cos_same.mean():.2f})")
        ax_s.set_title("Do tiles from one slide look alike? Cosine similarity of random tile pairs")
        ax_s.set_xlabel("cosine similarity"); ax_s.set_ylabel("density"); ax_s.legend(frameon=False)

        ax_n.hist(counts, bins=min(30, max(5, len(files) // 2)), color=SERIES[0], edgecolor=SURFACE, linewidth=1.5)
        ax_n.axvline(np.median(counts), color=INK_2, ls="--", lw=1.5)
        ax_n.set_title("Tiles per slide"); ax_n.set_xlabel("tiles"); ax_n.set_ylabel("slides")
        ax_n.text(np.median(counts), ax_n.get_ylim()[1] * 0.95, f" median {int(np.median(counts))}", color=INK_2, va="top")

        plt.show()

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
