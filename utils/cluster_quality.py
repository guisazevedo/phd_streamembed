# IMPORTS *****************************************************************************************
# External libraries
from itertools import combinations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

# Internal imports ********************************************************************************
from utils.cls_data import adjusted_rand_index, assign, davies_bouldin, fit_pca, kmeans, load_cls_features, silhouette
from utils.cls_overview import INK_2, SERIES, STYLE
from utils.figure_captions import split_with_notes


# Main function ***********************************************************************************
def plot_cluster_quality(
    features_dir: str | Path,
    model_name: str,
    k_values: range | list[int] = range(2, 13),
    max_slides: int | None = None,
    fit_size: int = 10000,
    n_eval: int = 4000,
    n_repeats: int = 5,
    seed: int = 0,
    show: bool = True,
) -> dict:
    """
    Help choose the number of k-means clusters and report how trustworthy each choice is.

    For every k, k-means is fitted n_repeats times on different random subsamples of the tiles (in PCA
    space). All repeats are scored on the same held-out sample of tiles with four measures: inertia
    (elbow), silhouette, Davies-Bouldin index, and stability (adjusted Rand index between repeats).
    Values are shown as mean +/- standard deviation over the repeats.

    * Arguments:
    features_dir: The folder that holds one subfolder per model
    model_name: The model subfolder with the per-slide .pt files
    k_values: The numbers of clusters to compare
    max_slides: The maximum number of slides to load (None = all)
    fit_size: The number of tiles each repeat fits k-means on
    n_eval: The number of tiles used to score every repeat (silhouette needs all pairwise distances)
    n_repeats: The number of repeats per k (at least 2 to measure stability)
    seed: The random seed
    show: Whether to call plt.show() (the figure is returned either way)

    Returns a dict with the figure, k_values and, per measure, an array [len(k_values), n_repeats]
    ("inertia", "silhouette", "davies_bouldin") or [len(k_values), n_pairs] ("stability").
    """

    # STEP 1: Validate the settings
    k_values = list(k_values)
    if n_repeats < 2:
        raise ValueError("n_repeats must be at least 2 to measure stability.")
    if min(k_values) < 2:
        raise ValueError("k_values must all be at least 2.")

    # STEP 2: Load the CLS vectors, project with PCA, and fix one evaluation sample for every run
    data = load_cls_features(features_dir, model_name, max_slides)
    X, names = data["X"], data["names"]
    N = len(X)
    Z = fit_pca(X, seed=seed)["Z"]
    rng = np.random.default_rng(seed)
    eval_idx = rng.choice(N, min(N, n_eval), replace=False)
    Ze = Z[eval_idx]

    # STEP 3: For every k and repeat, fit on a fresh subsample and score on the evaluation sample
    metrics = {key: np.zeros((len(k_values), n_repeats)) for key in ("inertia", "silhouette", "davies_bouldin")}
    stability = np.zeros((len(k_values), n_repeats * (n_repeats - 1) // 2))
    for ik, k in enumerate(k_values):
        runs = []
        for r in range(n_repeats):
            g = torch.Generator().manual_seed(seed * 1000 + 17 * k + r)
            fit = rng.choice(N, min(N, fit_size), replace=False)
            centers = kmeans(Z[fit], k, g)
            labels = assign(Ze, centers)
            runs.append(labels)
            metrics["inertia"][ik, r] = float((torch.cdist(torch.as_tensor(Ze), centers).min(1).values ** 2).mean())
            if len(np.unique(labels)) > 1:
                metrics["silhouette"][ik, r] = silhouette(Ze, labels).mean()
                metrics["davies_bouldin"][ik, r] = davies_bouldin(Ze, labels)
            else:
                metrics["silhouette"][ik, r], metrics["davies_bouldin"][ik, r] = np.nan, np.nan
        stability[ik] = [adjusted_rand_index(a, b) for a, b in combinations(runs, 2)]

    # STEP 4: Text summary (best k by each measure; these are hints, not decisions)
    sil_mean = np.nanmean(metrics["silhouette"], 1)
    db_mean = np.nanmean(metrics["davies_bouldin"], 1)
    print(f"Slides {len(names)}  |  tiles {N:,}  |  scored on {len(eval_idx):,} held-out tiles  |  {n_repeats} repeats per k")
    print(f"{'k':>3} {'inertia':>9} {'silhouette':>14} {'Davies-Bouldin':>16} {'stability (ARI)':>17}")
    for ik, k in enumerate(k_values):
        print(f"{k:>3} {metrics['inertia'][ik].mean():>9.2f} {sil_mean[ik]:>8.3f} ±{np.nanstd(metrics['silhouette'][ik]):.3f} "
              f"{db_mean[ik]:>9.3f} ±{np.nanstd(metrics['davies_bouldin'][ik]):.3f} {stability[ik].mean():>9.3f} ±{stability[ik].std():.3f}")
    print(f"Best k by silhouette: {k_values[int(np.nanargmax(sil_mean))]}  |  by Davies-Bouldin: {k_values[int(np.nanargmin(db_mean))]}  |  most stable: {k_values[int(stability.mean(1).argmax())]}")

    # STEP 5: Figure (styled with rc_context so the global matplotlib settings are left alone)
    notes_top = [
        (0, 1, "Within-cluster inertia (mean squared distance of a held-out tile to its cluster centre) as a function of k; line, mean over repeats; band, ± one standard deviation.\n"
               "Inertia decreases monotonically with k, so its minimum is uninformative; the relevant feature is a bend (elbow) after which additional clusters bring little reduction. The absence of a clear elbow indicates that the data have no pronounced natural number of clusters. k should then be set by the research question (for example k = 7 for the seven classes) and reported as a modelling choice."),
        (1, 2, "Mean silhouette coefficient (Rousseeuw, 1987) of held-out tiles: for each tile, the difference between its mean distance to the nearest other cluster and to its own cluster, divided by the larger of the two. It lies in [−1, 1]: values near 1 indicate well separated clusters, values near 0 overlapping clusters, and negative values tiles closer to another cluster.\n"
               "A maximum suggests a natural k. By the conventional reading of Kaufman and Rousseeuw (1990), values up to 0.25 indicate no substantial structure, 0.26 to 0.50 weak, 0.51 to 0.70 reasonable and above 0.70 strong structure. Continuous tissue appearance often yields low values."),
    ]
    notes_bottom = [
        (0, 1, "Davies-Bouldin index (Davies and Bouldin, 1979): the mean, over clusters, of the largest ratio between the sum of the average spreads of two clusters and the distance between their centres. Lower values indicate compact and well separated clusters; a minimum suggests the best k.\n"
               "The index uses cluster centres only, whereas the silhouette uses all pairwise distances. Agreement between the two supports the choice of k; disagreement indicates weak cluster structure."),
        (1, 2, "Adjusted Rand index (Hubert and Arabie, 1985) between the assignments of the held-out tiles produced by k-means fitted on different random subsamples; one value per pair of repeats; line, mean; band, ± one standard deviation. A value of 1 means identical partitions and about 0 agreement at chance level. Dotted line, 0.8, a conventional threshold for a reproducible partition.\n"
               "Low stability means that the partition depends on the subsample and seed and should not be interpreted as a property of the data. A well supported k combines a high silhouette with high stability."),
    ]

    ks = np.array(k_values)
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(14, 13), layout="constrained")
        fig.suptitle(f"Choosing k · {model_name} · {N:,} tiles from {len(names)} slides", fontsize=14, weight="bold", x=0.01, ha="left")
        rows = fig.subfigures(2, 1)

        def line(ax, values, label, best=None):
            mean, sd = np.nanmean(values, 1), np.nanstd(values, 1)
            ax.fill_between(ks, mean - sd, mean + sd, color=SERIES[0], alpha=0.18, linewidth=0)
            ax.plot(ks, mean, color=SERIES[0], lw=2, marker="o", ms=5)
            if best is not None:
                ax.axvline(ks[best], color=INK_2, ls="--", lw=1.2)
                ax.text(ks[best], ax.get_ylim()[1], f" k={ks[best]}", color=INK_2, va="top", fontsize=8)
            ax.set_xlabel("number of clusters k"); ax.set_ylabel(label)
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        top = split_with_notes(rows[0], notes_top, [1, 1], 14, plot_height=3.4, note_height=2.3)
        ax1, ax2 = top.subplots(1, 2)
        line(ax1, metrics["inertia"], "mean squared distance to centre"); ax1.set_title("Elbow: how tight are the clusters?")
        line(ax2, metrics["silhouette"], "silhouette (higher = better)", best=int(np.nanargmax(sil_mean))); ax2.set_title("Silhouette: how well separated?")

        bottom = split_with_notes(rows[1], notes_bottom, [1, 1], 14, plot_height=3.4, note_height=2.4)
        ax3, ax4 = bottom.subplots(1, 2)
        line(ax3, metrics["davies_bouldin"], "Davies-Bouldin (lower = better)", best=int(np.nanargmin(db_mean))); ax3.set_title("Davies-Bouldin: compact and apart?")
        line(ax4, stability, "adjusted Rand index between runs"); ax4.set_title("Stability: is the grouping reproducible?")
        ax4.axhline(0.8, color=INK_2, ls=":", lw=1.2); ax4.set_ylim(min(0, np.nanmin(stability)), 1.02)

    if show:
        plt.show()

    return {"figure": fig, "k_values": k_values, "stability": stability, **metrics}
