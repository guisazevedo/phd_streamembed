# IMPORTS *****************************************************************************************
# External libraries
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

# Internal imports ********************************************************************************
from utils.class_weights import compute_class_weights
from utils.cls_data import (adjusted_rand_index, assign, contingency, fit_pca, kmeans, load_cls_features,
                            normalized_mutual_info, silhouette)
from utils.cls_overview import GRID, INK_2, SEQ, STYLE, SURFACE, cluster_colors
from utils.figure_captions import split_with_notes
from utils.labels import BRACS_CLASSES, load_bracs_labels


# Helper functions ********************************************************************************
def _patient_folds(patients: np.ndarray, n_folds: int, rng: np.random.Generator) -> np.ndarray:
    """Assign every slide to a fold so that all slides of one patient share a fold (balanced by slide count)."""
    unique, inverse, sizes = np.unique(patients, return_inverse=True, return_counts=True)
    fold_of_patient = np.zeros(len(unique), dtype=int)
    load = np.zeros(n_folds)
    for p in sorted(rng.permutation(len(unique)), key=lambda i: -sizes[i]):
        f = int(load.argmin())
        fold_of_patient[p] = f
        load[f] += sizes[p]
    return fold_of_patient[inverse]


def _fit_logistic(X: torch.Tensor, y: torch.Tensor, n_classes: int, l2: float, iters: int = 100):
    """Class-balanced multinomial logistic regression (L-BFGS); returns a function that gives class scores."""
    mean, std = X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp(min=1e-6)
    Xs = (X - mean) / std
    weight = compute_class_weights(torch.bincount(y, minlength=n_classes).numpy(), "inverse")
    W = torch.zeros(X.shape[1], n_classes, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xs @ W + b, y, weight=weight) + l2 * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return lambda Xnew: (((Xnew - mean) / std) @ W + b).detach()


def _probe(M, y, train_idx, test_idx, n_classes, l2):
    """Train the probe on train_idx and return the predicted classes of test_idx."""
    score = _fit_logistic(M[train_idx], y[train_idx], n_classes, l2)
    return score(M[test_idx]).argmax(1)


def _cross_val_predict(M, y, patients, n_classes, n_folds, l2, rng):
    """Out-of-fold predictions with folds that never split a patient's slides."""
    folds = _patient_folds(patients, n_folds, rng)
    pred = torch.zeros(len(y), dtype=torch.long)
    for f in range(n_folds):
        test = torch.as_tensor(np.where(folds == f)[0])
        train = torch.as_tensor(np.where(folds != f)[0])
        if len(test):
            pred[test] = _probe(M, y, train, test, n_classes, l2)
    return pred


def _confusion(y_true, y_pred, n_classes):
    return contingency(np.asarray(y_true), np.asarray(y_pred), n_classes, n_classes)


def _balanced_accuracy(y_true, y_pred, n_classes):
    cm = _confusion(y_true, y_pred, n_classes)
    present = cm.sum(1) > 0
    return float((np.diag(cm)[present] / cm.sum(1)[present]).mean())


def _macro_f1(cm):
    precision = np.diag(cm) / np.maximum(cm.sum(0), 1)
    recall = np.diag(cm) / np.maximum(cm.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(f1[cm.sum(1) > 0].mean())


# Main function ***********************************************************************************
def plot_class_analysis(
    features_dir: str | Path,
    model_name: str,
    labels_file: str | Path,
    class_order: list[str] = BRACS_CLASSES,
    n_clusters: int = 7,
    max_slides: int | None = None,
    max_fit: int = 30000,
    n_eval: int = 4000,
    n_folds: int = 5,
    n_boot: int = 1000,
    probe_l2: float = 0.05,
    seed: int = 0,
    show: bool = True,
) -> dict:
    """
    Relate the CLS embeddings to the slide-level class labels (how much does the embedding know about the classes?).

    Tile-level: how classes and unsupervised clusters overlap (contingency table, NMI, ARI) and how well
    separated each class is (silhouette). Every tile inherits its slide's label, which is a weak label
    (a slide labelled IC also contains normal tissue), so tile-level separation is expected to be modest.
    Slide-level: a linear probe (class-balanced logistic regression on the mean CLS vector of each slide)
    evaluated with patient-grouped cross-validation, compared with three baselines: chance, always predicting
    the biggest class, and the same pipeline with shuffled labels. The probe is also evaluated on the
    official Training -> Testing split of the dataset. The probe results are printed and returned, but not
    plotted: the figure has four panels (classes in the embedding, class balance, classes against clusters,
    silhouette per class).

    * Arguments:
    features_dir: The folder that holds one subfolder per model
    model_name: The model subfolder with the per-slide .pt files
    labels_file: The BRACS.xlsx spreadsheet (see utils.labels.load_bracs_labels)
    class_order: The classes in display order; slides with other labels are dropped
    n_clusters: The number of k-means clusters compared with the classes
    max_slides: The maximum number of slides to load (None = all)
    max_fit: The number of tiles used to fit PCA
    n_eval: The number of tiles scored for the silhouette per class
    n_folds: The number of patient-grouped cross-validation folds
    n_boot: The number of bootstrap resamples (over slides) for the confidence interval
    probe_l2: The L2 penalty of the linear probe (fixed, not tuned, so no validation data is needed)
    seed: The random seed
    show: Whether to call plt.show() (the figure is returned either way)

    Returns a dict with the figure, classes, the tile-level tables, the slide-level confusion matrix and the
    probe metrics (not plotted).
    """

    # STEP 1: Load the embeddings and attach the labels; slides without a known label are dropped
    data = load_cls_features(features_dir, model_name, max_slides)
    lookup = load_bracs_labels(labels_file)
    classes = [c for c in class_order if any(v["label"] == c for v in lookup.values())]
    K = len(classes)
    slide_ok = np.array([n in lookup and lookup[n]["label"] in classes for n in data["names"]])
    if slide_ok.sum() < K or K < 2:
        raise ValueError(f"Only {slide_ok.sum()} of {len(slide_ok)} slides have a label of {classes}; check the slide names.")
    if not slide_ok.all():
        print(f"Warning: {int((~slide_ok).sum())} slides without a usable label were dropped.")

    keep_tile = slide_ok[data["slide_id"]]
    X = data["X"][torch.as_tensor(keep_tile)]
    new_index = np.cumsum(slide_ok) - 1
    slide_id = new_index[data["slide_id"][keep_tile]]
    coords = data["coords"][keep_tile]
    names = [n for n, ok in zip(data["names"], slide_ok) if ok]
    S, N = len(names), len(X)
    y_slide = np.array([classes.index(lookup[n]["label"]) for n in names])
    patients = np.array([lookup[n]["patient"] for n in names])
    split = np.array([lookup[n]["set"] for n in names])
    y_tile = y_slide[slide_id]
    slides_per_class = np.bincount(y_slide, minlength=K)
    tiles_per_class = np.bincount(y_tile, minlength=K)
    rng = np.random.default_rng(seed)

    # STEP 2: PCA, k-means and the class-versus-cluster comparison (all tiles)
    pca = fit_pca(X, max_fit, seed)
    Z, g = pca["Z"], pca["generator"]
    centers = kmeans(Z[pca["fit_idx"].numpy()], n_clusters, g)
    cluster = assign(Z, centers)
    cluster = np.argsort(np.argsort(-np.bincount(cluster, minlength=n_clusters)))[cluster]    # 0 = largest
    table = contingency(y_tile, cluster, K, n_clusters)
    class_to_cluster = table / table.sum(1, keepdims=True).clip(min=1) * 100
    nmi = normalized_mutual_info(y_tile, cluster)
    ari = adjusted_rand_index(y_tile, cluster)

    # STEP 3: Silhouette of every class label (tile level, on a class-balanced sample)
    per_class = max(2, n_eval // K)
    sample = np.concatenate([rng.choice(np.where(y_tile == c)[0], min(per_class, tiles_per_class[c]), replace=False) for c in range(K)])
    sil_tile = silhouette(Z[sample], y_tile[sample])
    sil_class = np.array([sil_tile[y_tile[sample] == c].mean() for c in range(K)])

    # STEP 4: Slide-level linear probe with patient-grouped cross-validation
    M = torch.stack([X[torch.as_tensor(slide_id == s)].mean(0) for s in range(S)])           # mean CLS per slide
    y_t = torch.as_tensor(y_slide)
    n_folds = max(2, min(n_folds, len(np.unique(patients))))
    pred = _cross_val_predict(M, y_t, patients, K, n_folds, probe_l2, rng).numpy()
    cm = _confusion(y_slide, pred, K)
    acc, bal_acc, f1 = float((pred == y_slide).mean()), _balanced_accuracy(y_slide, pred, K), _macro_f1(cm)
    recall = np.diag(cm) / np.maximum(cm.sum(1), 1)
    boots = np.array([_balanced_accuracy(y_slide[i], pred[i], K) for i in (rng.integers(0, S, S) for _ in range(n_boot))])
    ci = np.percentile(boots, [2.5, 97.5])

    # STEP 5: Baselines: chance, majority class, shuffled labels (same pipeline), official split
    majority_acc = float(slides_per_class.max() / S)
    shuffled = torch.as_tensor(rng.permutation(y_slide))
    pred_shuf = _cross_val_predict(M, shuffled, patients, K, n_folds, probe_l2, rng).numpy()
    bal_shuf = _balanced_accuracy(shuffled.numpy(), pred_shuf, K)
    train_idx, test_idx = np.where(split == "Training")[0], np.where(split == "Testing")[0]
    bal_official = None
    if len(train_idx) and len(test_idx):
        pred_off = _probe(M, y_t, torch.as_tensor(train_idx), torch.as_tensor(test_idx), K, probe_l2).numpy()
        bal_official = _balanced_accuracy(y_slide[test_idx], pred_off, K)

    # STEP 6: Text summary
    print(f"Slides {S}  |  patients {len(np.unique(patients))}  |  tiles {N:,}  |  classes {K}")
    print("Per class : " + ", ".join(f"{c} {slides_per_class[i]} slides/{tiles_per_class[i]:,} tiles" for i, c in enumerate(classes)))
    print(f"Tile level: class vs {n_clusters} clusters  NMI {nmi:.3f}  ARI {ari:.3f}   |   mean silhouette by class label {sil_class.mean():+.3f}")
    print(f"Probe     : {n_folds}-fold patient-grouped CV  accuracy {acc:.3f}  balanced accuracy {bal_acc:.3f} (95% CI {ci[0]:.3f}-{ci[1]:.3f})  macro-F1 {f1:.3f}")
    print(f"Baselines : chance {1 / K:.3f}  majority-class accuracy {majority_acc:.3f}  shuffled-label balanced accuracy {bal_shuf:.3f}"
          + (f"  |  official Training->Testing balanced accuracy {bal_official:.3f}" if bal_official is not None else ""))

    # STEP 7: Figure (styled with rc_context so the global matplotlib settings are left alone)
    notes_a = [
        (0, 1, "Tiles in the plane of the first two principal components, coloured by the class label of their slide (random subset of up to 10,000 tiles).\n"
               "If the classes occupy different regions, the embedding separates them without supervision. Strong mixing is expected: most slides contain several tissue types and each tile inherits the slide label, so the labels are weak at tile level. Class separation is therefore assessed at slide level (linear probe, reported in the text output). Two components represent only part of the variance, so overlap here does not exclude separability in other directions."),
        (1, 2, "Number of slides (bars) and tiles (text) per class.\n"
               "Strong class imbalance makes accuracy dominated by the large classes, so balanced accuracy (mean per-class recall) and per-class results are reported. Classes with few slides give noisy estimates, which appear as wide confidence intervals."),
    ]
    notes_b = [
        (0, 1, f"Row-normalised contingency table between the slide class of each tile (rows) and its k-means cluster (columns); each row sums to 100%. The agreement between the two partitions is summarised by the normalised mutual information (NMI = {nmi:.2f}) and the adjusted Rand index (ARI = {ari:.2f}); both are 0 for independent partitions and 1 for identical ones.\n"
               f"A class concentrated in few clusters indicates that these clusters capture appearance typical of that class. An even spread indicates that the unsupervised structure follows tissue appearance rather than diagnosis, the usual outcome, because the clusters are learned without labels."),
        (1, 2, "Mean silhouette coefficient of the tiles of each class, with the class label as grouping (class-balanced sample, PCA space). Positive values mean that tiles are closer to their own class than to other classes, values near 0 that classes overlap, and negative values that tiles resemble another class more than their own.\n"
               "Values near 0 are expected with weak tile labels and do not by themselves show that the features are uninformative; the slide-level probe is the more relevant test. The classes with the lowest values are the most likely to be confused, as would be expected for neighbouring grades."),
    ]
    colors = cluster_colors(K)
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(16, 13.5), layout="constrained")
        fig.suptitle(f"CLS embeddings vs classes · {model_name} · {N:,} tiles, {S} slides, {K} classes", fontsize=16, weight="bold", x=0.01, ha="left")
        rows = fig.subfigures(2, 1, height_ratios=[5.2, 5.6])

        # Row 1: classes in the embedding, class balance
        r1 = split_with_notes(rows[0], notes_a, [1.3, 1], 16, plot_height=3.8, note_height=1.5)
        ax_s, ax_b = r1.subplots(1, 2, width_ratios=[1.3, 1])
        sub = rng.choice(N, min(N, 10000), replace=False)
        for c in range(K):
            m = y_tile[sub] == c
            ax_s.scatter(Z[sub][m, 0], Z[sub][m, 1], s=6, color=colors[c], alpha=0.4, linewidths=0, label=classes[c])
        ax_s.set_xlabel("PC1"); ax_s.set_ylabel("PC2"); ax_s.set_title("Do the classes occupy different regions?")
        ax_s.legend(frameon=False, markerscale=3, ncol=2, fontsize=10, title="slide class", title_fontsize=10)

        ypos = np.arange(K)
        ax_b.barh(ypos, slides_per_class, color=colors, edgecolor=SURFACE, linewidth=1.5)
        for i in range(K):
            ax_b.text(slides_per_class[i], i, f"  {slides_per_class[i]} slides · {tiles_per_class[i]:,} tiles", va="center", fontsize=10, color=INK_2)
        ax_b.set_yticks(ypos); ax_b.set_yticklabels(classes); ax_b.invert_yaxis(); ax_b.grid(False)
        ax_b.set_xlim(0, slides_per_class.max() * 1.7); ax_b.set_xlabel("slides"); ax_b.set_title("How balanced are the classes?")

        # Row 2: classes against unsupervised clusters, silhouette per class
        r2 = split_with_notes(rows[1], notes_b, [1.3, 1], 16, plot_height=3.6, note_height=2.0)
        ax_h, ax_sil = r2.subplots(1, 2, width_ratios=[1.3, 1])
        im = ax_h.imshow(class_to_cluster, cmap=SEQ, vmin=0, vmax=max(class_to_cluster.max(), 1), aspect="auto")
        ax_h.set_xticks(range(n_clusters)); ax_h.set_xticklabels([f"C{c + 1}" for c in range(n_clusters)])
        ax_h.set_yticks(range(K)); ax_h.set_yticklabels(classes); ax_h.grid(False)
        ax_h.set_xlabel("k-means cluster"); ax_h.set_ylabel("slide class"); ax_h.set_title("Which clusters do the tiles of each class fall in? (% of class)")
        if K * n_clusters <= 100:
            for i in range(K):
                for j in range(n_clusters):
                    v = class_to_cluster[i, j]
                    ax_h.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=10, color="white" if v > 0.55 * class_to_cluster.max() else INK_2)
        r2.colorbar(im, ax=ax_h, shrink=0.8, label="% of the class's tiles")

        ax_sil.barh(ypos, sil_class, color=colors, edgecolor=SURFACE, linewidth=1.5)
        ax_sil.axvline(0, color=INK_2, lw=1)
        ax_sil.set_yticks(ypos); ax_sil.set_yticklabels(classes); ax_sil.invert_yaxis(); ax_sil.grid(False)
        ax_sil.set_xlabel("mean silhouette of the class's tiles"); ax_sil.set_title("How separate is each class?")

    if show:
        plt.show()

    return {
        "figure": fig, "classes": classes, "slide_names": names, "slide_labels": y_slide,
        "class_to_cluster": class_to_cluster, "nmi": nmi, "ari": ari, "silhouette_by_class": sil_class,
        "confusion": cm, "predictions": pred, "accuracy": acc, "balanced_accuracy": bal_acc,
        "balanced_accuracy_ci": tuple(ci), "macro_f1": f1, "shuffled_balanced_accuracy": bal_shuf,
        "official_balanced_accuracy": bal_official, "majority_accuracy": majority_acc,
    }
