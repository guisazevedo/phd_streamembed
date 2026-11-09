# IMPORTS *****************************************************************************************
# External libraries
from pathlib import Path
import textwrap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvips
from sklearn.metrics import precision_recall_curve, roc_curve

# Internal imports ********************************************************************************
from utils.dataset_overview import SET_COLORS, SETS
from utils.classification_metrics import HEADLINE, clinical_views, expected_calibration_error
from utils.cls_overview import INK_2, SEQ, SERIES, STYLE, SURFACE, cluster_colors
from utils.dataset import WSITileDataset
from utils.figure_captions import split_with_notes
from utils.plot_pyramid import _read_region

# Constants ***************************************************************************************
MODEL_LABELS = {"linear": "mean-pool linear", "abmil": "gated ABMIL", "clam_mb": "CLAM-MB"}
MODEL_COLORS = {"linear": SERIES[1], "abmil": SERIES[0], "clam_mb": SERIES[2]}
MEASURE_LABELS = {
    "accuracy": "accuracy", "balanced_accuracy": "balanced accuracy", "macro_f1": "macro F1", "macro_auroc": "macro AUROC",
    "macro_ap": "macro average precision", "quadratic_kappa": "quadratic kappa", "mcc": "Matthews correlation",
}


# Helper functions ********************************************************************************
def _label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def _color(model: str, position: int = 0) -> str:
    return MODEL_COLORS.get(model, SERIES[position % len(SERIES)])


def _finish(fig, show: bool):
    if show:
        plt.show()
    return fig


def _architecture_boxes(name: str, in_dim: int, n_classes: int, kw: dict) -> list[dict]:
    """The boxes of the forward pass of a model, top to bottom (see plot_model_decision)."""
    hidden, att, p = kw.get("hidden", 512), kw.get("attention_dim", 256), kw.get("dropout", 0.25)
    head = [dict(text=f"Tile embeddings\nN tiles × {in_dim}  (UNI2-h CLS token)"), dict(text="Feature scaler\n(mean and s.d. of the fit slides)")]
    if name == "linear":
        return head + [dict(text="Mean over the real tiles\nslide vector"), dict(text=f"Dropout({p}) · Linear({in_dim} → {n_classes})"), dict(text=f"{n_classes} logits → softmax", out=True)]
    scores = n_classes if name == "clam_mb" else 1
    trunk = head + [dict(text=f"Shared projection\nLinear({in_dim} → {hidden}) · ReLU · Dropout({p})"),
                    dict(gate=(f"tanh branch\nLinear({hidden} → {att})", f"sigmoid branch\nLinear({hidden} → {att})")),
                    dict(text=f"Gated product → Linear({att} → {scores})\n" + ("one attention score per class" if scores > 1 else "one attention score per tile")),
                    dict(text="Softmax over the real tiles\n" + ("(separately for each class)" if scores > 1 else "attention weights sum to 1"))]
    if name == "clam_mb":
        return trunk + [dict(text=f"Attention-weighted sum of the projections\n{n_classes} slide vectors of size {hidden}"),
                        dict(text=f"{n_classes} classifiers Linear({hidden} → 1)\none logit per class"), dict(text=f"{n_classes} logits → softmax", out=True)]
    return trunk + [dict(text=f"Attention-weighted sum of the projections\nslide vector of size {hidden}"), dict(text=f"Linear({hidden} → {n_classes})"), dict(text=f"{n_classes} logits → softmax", out=True)]


def _draw_architecture(ax, name: str, in_dim: int, n_classes: int, kw: dict):
    """Boxes and arrows of the forward pass; the training-only instance loss of CLAM-MB is dashed."""
    boxes = _architecture_boxes(name, in_dim, n_classes, kw)
    n = len(boxes)
    pitch = 0.95 / n
    height, gap = 0.68 * pitch, 0.32 * pitch
    top = 0.97
    centre = 0.36 if name == "clam_mb" else 0.5
    width = 0.58
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ys = [top - i * (height + gap) for i in range(n)]
    for box, y in zip(boxes, ys):
        if "gate" in box:
            for text, x in zip(box["gate"], (centre - width / 4 - 0.01, centre + width / 4 + 0.01)):
                ax.add_patch(plt.Rectangle((x - width / 4, y - height), width / 2 - 0.01, height, fc=SURFACE, ec=_color(name), lw=1.4))
                ax.text(x, y - height / 2, text, ha="center", va="center", fontsize=8, color="#0b0b0b")
            continue
        colour = _color(name) if box.get("out") else INK_2
        ax.add_patch(plt.Rectangle((centre - width / 2, y - height), width, height, fc="#e9f1fc" if box.get("out") else SURFACE, ec=colour, lw=1.4))
        ax.text(centre, y - height / 2, box["text"], ha="center", va="center", fontsize=8.5, color="#0b0b0b")
    for y in ys[1:]:
        ax.annotate("", xy=(centre, y), xytext=(centre, y + gap), arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=1.2))
    if name == "clam_mb":
        # Training-only branch: instance-clustering loss on the tiles that the attention selects
        k, bag = kw.get("k_sample", 8), kw.get("bag_weight", 0.7)
        y = ys[3] - 0.01
        ax.add_patch(plt.Rectangle((0.69, y - 0.2), 0.3, 0.2, fc=SURFACE, ec=_color(name), lw=1.4, ls="--"))
        ax.text(0.84, y - 0.1, f"Training only\ninstance loss: {n_classes} classifiers\nLinear({kw.get('hidden', 512)} → 2) on the\n{k} top (and, for the true\nclass, {k} bottom) attention\ntiles; loss = {bag} × slide\n+ {round(1 - bag, 2)} × instance", ha="center", va="center", fontsize=7.5, color="#0b0b0b")
        ax.annotate("", xy=(0.69, y - 0.1), xytext=(centre + width / 2, y - 0.1), arrowprops=dict(arrowstyle="-|>", color=_color(name), lw=1.2, ls="--"))


def _wrap(value, width: int) -> list[str]:
    return textwrap.wrap(str(value), width, break_long_words=True) or [""]


def _draw_specification(ax, groups: list[tuple[str, list[tuple[str, str]]]]):
    """A three-group specification table drawn as text (group headings, then name and value rows)."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    y, line = 0.99, 0.0295
    for title, rows in groups:
        ax.text(0, y, title, fontsize=10, fontweight="bold", color="#0b0b0b", va="top")
        y -= line * 1.25
        ax.plot([0, 1], [y + line * 0.25, y + line * 0.25], color="#e6e5e1", lw=1)
        for key, value in rows:
            lines = _wrap(value, 62)
            ax.text(0, y, key, fontsize=8.5, color=INK_2, va="top")
            ax.text(0.33, y, "\n".join(lines), fontsize=8.5, color="#0b0b0b", va="top", linespacing=1.25)
            y -= line * len(lines)
        y -= line * 0.6


# Main functions **********************************************************************************
def plot_bag_sizes(data: dict, show: bool = True):
    """
    Number of tiles per slide (the size of each bag), by official set.

    * Arguments:
    data: The output of mil_data.load_bags
    show: Whether to call plt.show() (the figure is returned either way)
    """

    table = data["table"]
    sizes = np.array([len(data["bags"][s]) for s in table["slide"]])
    notes = [(0, 1, "Number of tiles per slide (bag size) in each official set; dashed line, the median of all slides.\n"
                    "Bag sizes vary by an order of magnitude, which is why the models pool a variable number of tiles with a mask and why long bags are subsampled during training. "
                    "The three sets should have similar distributions; a set with systematically larger bags would suggest a difference in slide size or tissue amount between them.")]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(9, 5.6), layout="constrained")
        top = split_with_notes(fig, notes, [1], 9, plot_height=3.4, note_height=1.6)
        ax = top.subplots(1, 1)
        bins = np.linspace(sizes.min(), sizes.max(), 30)
        for s in SETS:
            ax.hist(sizes[(table["set"] == s).to_numpy()], bins=bins, histtype="step", lw=2, color=SET_COLORS[s], label=f"{s} (n = {int((table['set'] == s).sum())})", density=True)
        ax.axvline(np.median(sizes), color=INK_2, ls="--", lw=1.2)
        ax.set_xlabel("tiles per slide"); ax.set_ylabel("density"); ax.set_title("How large are the bags?"); ax.legend(frameon=False)

    return _finish(fig, show)


def plot_training_curves(histories: dict[str, list[pd.DataFrame]], show: bool = True):
    """
    Training and early-stopping (Validation) curves of every model, averaged over seeds or fold models.

    * Arguments:
    histories: {model: [history DataFrame of each run]} from mil_train.train_model (seeds, or the fold models of a cross-validation)
    show: Whether to call plt.show() (the figure is returned either way)
    """

    notes = [(0, 3, "Training loss (class-weighted cross-entropy), early-stopping loss (unweighted cross-entropy) and early-stopping macro-averaged F1 per epoch; the early-stopping slides "
                    "are the slides on which the epoch is selected (the Validation slides, or in cross-validation a part of each fold that is held out from the fit). "
                    "Lines, mean over runs; bands, ± one standard deviation; triangles, the epoch selected for each run (best early-stopping macro-F1).\n"
                    "A training loss that keeps falling while the early-stopping loss rises and its F1 stalls indicates overfitting; selecting the epoch "
                    "on the F1 and restoring its weights limits the damage. The early-stopping set is small, so the selected epoch is itself uncertain, which "
                    "is one reason for averaging several runs. The training loss is weighted and the early-stopping loss is not, so the two are not on the same scale.")]
    panels = [("train_loss", "training loss (weighted)", "Does the model fit the Training set?"),
              ("val_loss", "early-stopping loss (unweighted)", "Does it generalise to the early-stopping slides?"),
              ("val_macro_f1", "early-stopping macro F1", "Which epoch is selected?")]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(15, 5.6), layout="constrained")
        top = split_with_notes(fig, notes, [1, 1, 1], 15, plot_height=3.4, note_height=1.5)
        axes = top.subplots(1, 3)
        for ax, (column, ylabel, title) in zip(axes, panels):
            for i, (model, runs) in enumerate(histories.items()):
                length = max(len(h) for h in runs)
                stacked = np.full((len(runs), length), np.nan)
                for r, h in enumerate(runs):
                    stacked[r, : len(h)] = h[column].to_numpy()
                mean, sd = np.nanmean(stacked, 0), np.nanstd(stacked, 0)
                epochs = np.arange(1, length + 1)
                ax.fill_between(epochs, mean - sd, mean + sd, color=_color(model, i), alpha=0.18, linewidth=0)
                ax.plot(epochs, mean, color=_color(model, i), lw=2, label=_label(model))
                if column == "val_macro_f1":
                    for h in runs:
                        best = int(h["val_macro_f1"].idxmax()) if len(h) else 0
                        ax.scatter(h["epoch"].iloc[best], h["val_macro_f1"].iloc[best], marker="v", s=36, color=_color(model, i), edgecolor=SURFACE, zorder=3)
            ax.set_xlabel("epoch"); ax.set_ylabel(ylabel); ax.set_title(title)
        axes[0].legend(frameon=False)

    return _finish(fig, show)


def plot_confusion(confusions: dict[str, pd.DataFrame], title: str = "Testing set", show: bool = True):
    """
    Confusion matrices (rows true class, columns predicted class) of one or more models.

    * Arguments:
    confusions: {model: confusion DataFrame of counts} from classification_metrics.evaluate_predictions
    title: The set that the matrices describe (used in the title)
    show: Whether to call plt.show() (the figure is returned either way)
    """

    n = len(confusions)
    notes = [(0, n, f"Confusion matrices on the {title}: rows are the true classes, columns the predicted classes; colour, the share of the true class; numbers, the share (%) and the number of slides. "
                    "A dominant diagonal means that classes are recognised; mass next to the diagonal is confusion between adjacent grades, which is clinically plausible and also occurs between "
                    "pathologists, while mass far from the diagonal (for example benign slides predicted as invasive carcinoma) is a failure to examine slide by slide.\n"
                    "The classes have few slides in the Testing set, so one slide moves a row percentage by several points; read the counts together with the percentages.")]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(7.4 * n, 8.6), layout="constrained")
        top = split_with_notes(fig, notes, [1] * n, 7.4 * n, plot_height=6.2, note_height=1.6)
        axes = np.atleast_1d(top.subplots(1, n))
        for ax, (model, cm) in zip(axes, confusions.items()):
            counts = cm.to_numpy()
            share = counts / np.maximum(counts.sum(1, keepdims=True), 1) * 100
            im = ax.imshow(share, cmap=SEQ, vmin=0, vmax=100)
            K = len(cm)
            ax.set_xticks(range(K)); ax.set_xticklabels(cm.columns); ax.set_yticks(range(K)); ax.set_yticklabels(cm.index); ax.grid(False)
            for i in range(K):
                for j in range(K):
                    ax.text(j, i, f"{share[i, j]:.0f}%\n({counts[i, j]})", ha="center", va="center", fontsize=10, color="white" if share[i, j] > 55 else INK_2)
            ax.set_xlabel("predicted class"); ax.set_ylabel("true class"); ax.set_title(_label(model))
        top.colorbar(im, ax=list(axes), shrink=0.7, label="% of the true class")

    return _finish(fig, show)


def plot_roc_pr(y: np.ndarray, probabilities: dict[str, np.ndarray], classes: list[str], show: bool = True):
    """
    One-vs-rest ROC and precision-recall curves of every class, for every model.

    * Arguments:
    y: The true class index of every slide
    probabilities: {model: predicted probabilities [n_slides, n_classes]}
    classes: The class names (the classes of BRACS.xlsx)
    show: Whether to call plt.show() (the figure is returned either way)
    """

    n, colors = len(probabilities), cluster_colors(len(classes))
    notes_roc = [(0, n, "One-vs-rest ROC curve of each class (that class against all the others): sensitivity against 1 − specificity as the decision threshold on the class probability varies. "
                        "The legend gives the area under the curve (AUROC); the thick dashed line is the macro-average over classes, and the diagonal is chance (AUROC 0.5).\n"
                        "AUROC measures how well a class is ranked above the others and does not depend on the threshold or on the class frequency. A curve hugging the upper left corner is good; "
                        "classes with few slides give coarse, step-like curves.")]
    notes_pr = [(0, n, "One-vs-rest precision-recall curve of each class; the legend gives the average precision (AP), the area under the curve.\n"
                       "Unlike the ROC curve, precision depends on how common the class is: a classifier without information has an AP equal to the prevalence of the class, so AP must be read "
                       "against that value (rare classes can have a low AP even when well ranked). Both views are reported because the classes are imbalanced.")]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(7.4 * n, 16.5), layout="constrained")
        rows = fig.subfigures(2, 1)
        for row, notes, kind in ((rows[0], notes_roc, "roc"), (rows[1], notes_pr, "pr")):
            top = split_with_notes(row, notes, [1] * n, 7.4 * n, plot_height=5.8, note_height=1.6)
            axes = np.atleast_1d(top.subplots(1, n))
            for ax, (model, proba) in zip(axes, probabilities.items()):
                grid, curves = np.linspace(0, 1, 201), []
                for k, name in enumerate(classes):
                    positive = (y == k).astype(int)
                    if not 0 < positive.sum() < len(y):
                        continue
                    if kind == "roc":
                        fpr, tpr, _ = roc_curve(positive, proba[:, k])
                        auc = np.trapezoid(tpr, fpr)
                        ax.plot(fpr, tpr, color=colors[k], lw=1.6, label=f"{name}  {auc:.2f}")
                        curves.append(np.interp(grid, fpr, tpr))
                    else:
                        precision, recall, _ = precision_recall_curve(positive, proba[:, k])
                        ap = -np.sum(np.diff(recall) * precision[:-1])
                        ax.plot(recall, precision, color=colors[k], lw=1.6, label=f"{name}  {ap:.2f} (prev. {positive.mean():.2f})")
                if kind == "roc":
                    macro = np.mean(curves, 0)
                    ax.plot(grid, macro, color=INK_2, lw=3, ls="--", label=f"macro  {np.trapezoid(macro, grid):.2f}")
                    ax.plot([0, 1], [0, 1], color=INK_2, lw=1, ls=":")
                    ax.set_xlabel("1 − specificity"); ax.set_ylabel("sensitivity")
                else:
                    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_ylim(0, 1.02)
                ax.set_xlim(0, 1); ax.set_title(f"{_label(model)}: {'ROC (AUROC)' if kind == 'roc' else 'precision-recall (AP)'}")
                ax.legend(frameon=False, fontsize=10, loc="lower right" if kind == "roc" else "upper right")

    return _finish(fig, show)


def plot_per_class(per_class: dict[str, pd.DataFrame], intervals: dict[str, dict], show: bool = True):
    """
    F1 and AUROC of every class with 95% intervals, for every model.

    * Arguments:
    per_class: {model: per_class DataFrame} from classification_metrics.evaluate_predictions
    intervals: {model: patient_bootstrap(..., per_class=True) result}
    show: Whether to call plt.show() (the figure is returned either way)
    """

    first = next(iter(per_class.values()))
    classes, support = list(first.index), first["support"].astype(int).to_numpy()
    notes = [(0, 1, "F1 score of each class (harmonic mean of precision and recall at the decision rule 'most probable class') with the 95% patient-bootstrap interval; the classes are labelled with their number of Testing slides.\n"
                    "Intervals are wide for the small classes, so differences between classes, or between models, that are smaller than the interval cannot be interpreted. F1 depends on the "
                    "decision rule and on the class frequency."),
             (1, 2, "AUROC of each class against the rest, with the 95% patient-bootstrap interval.\n"
                    "A class with a high AUROC and a low F1 is ranked well but is not assigned by the decision rule, which happens when probabilities are shifted by class weighting or by "
                    "overlap with a neighbouring class; the threshold, not the representation, is then the problem.")]
    x = np.arange(len(classes))
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(15, 6.4), layout="constrained")
        top = split_with_notes(fig, notes, [1, 1], 15, plot_height=3.8, note_height=1.6)
        axes = top.subplots(1, 2)
        width = 0.8 / len(per_class)
        for ax, (column, ci_key, title) in zip(axes, (("f1", "f1", "How well is each class classified? (F1)"), ("auroc", "auroc", "How well is each class ranked? (AUROC)"))):
            for j, (model, table) in enumerate(per_class.items()):
                values = table[column].to_numpy()
                low, high = intervals[model]["per_class_ci"][ci_key]
                position = x - 0.4 + width * (j + 0.5)
                ax.bar(position, values, width=width, color=_color(model, j), edgecolor=SURFACE, linewidth=1, label=_label(model))
                ax.errorbar(position, values, yerr=[np.clip(values - low, 0, None), np.clip(high - values, 0, None)], fmt="none", ecolor=INK_2, capsize=3, lw=1.2)
            ax.set_xticks(x); ax.set_xticklabels([f"{c}\n(n = {n})" for c, n in zip(classes, support)]); ax.set_ylim(0, 1.05); ax.set_ylabel(column.upper() if column == "auroc" else "F1")
            ax.set_title(title)
        axes[0].legend(frameon=False)
        axes[1].axhline(0.5, color=INK_2, ls=":", lw=1)

    return _finish(fig, show)


def plot_calibration(y: np.ndarray, probabilities: dict[str, np.ndarray], n_bins: int = 10, show: bool = True):
    """
    Reliability diagram and confidence histogram of the most probable class.

    * Arguments:
    y: The true class index of every slide
    probabilities: {model: predicted probabilities [n_slides, n_classes]}
    n_bins: The number of confidence bins
    show: Whether to call plt.show() (the figure is returned either way)
    """

    notes = [(0, 1, "Reliability diagram: slides are binned by the probability that the model gives to its predicted class (confidence); each point compares the mean confidence of a bin with the "
                    "observed accuracy in it (marker area, number of slides). The legend gives the expected calibration error (ECE).\n"
                    "Points on the diagonal mean that the probabilities can be read as frequencies; points below it mean overconfidence. The loss weighting moves probability towards the rare "
                    "classes on purpose, so calibration with respect to the true class frequencies is not expected; if the probabilities are to be used as risks, they should be recalibrated "
                    "on the Validation set (for example by temperature scaling)."),
             (1, 2, "Distribution of the confidence of the predicted class.\n"
                    "A model whose confidence is concentrated near 1 while its accuracy is lower is overconfident; a confidence spread over the whole range indicates uncertain predictions. "
                    "With few Testing slides the bins are sparsely populated and the diagram is only indicative.")]
    edges = np.linspace(0, 1, n_bins + 1)
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(14, 6.8), layout="constrained")
        top = split_with_notes(fig, notes, [1, 1], 14, plot_height=3.8, note_height=2.2)
        ax_r, ax_h = top.subplots(1, 2)
        ax_r.plot([0, 1], [0, 1], color=INK_2, ls=":", lw=1)
        for j, (model, proba) in enumerate(probabilities.items()):
            confidence, correct = proba.max(1), (proba.argmax(1) == y).astype(float)
            bins = np.clip(np.digitize(confidence, edges[1:-1]), 0, n_bins - 1)
            filled = [b for b in range(n_bins) if (bins == b).any()]
            xs = [confidence[bins == b].mean() for b in filled]
            ys = [correct[bins == b].mean() for b in filled]
            sizes = [30 + 330 * (bins == b).sum() / len(y) for b in filled]
            ax_r.plot(xs, ys, color=_color(model, j), lw=1.5)
            ax_r.scatter(xs, ys, s=sizes, color=_color(model, j), edgecolor=SURFACE, zorder=3, label=f"{_label(model)} (ECE {expected_calibration_error(y, proba, n_bins):.2f})")
            ax_h.hist(confidence, bins=edges, histtype="step", lw=2, color=_color(model, j), label=_label(model))
        ax_r.set_xlim(0, 1); ax_r.set_ylim(0, 1.02); ax_r.set_xlabel("mean confidence of the bin"); ax_r.set_ylabel("observed accuracy of the bin")
        ax_r.set_title("Are the probabilities calibrated?"); ax_r.legend(frameon=False, loc="upper left")
        ax_h.set_xlabel("confidence of the predicted class"); ax_h.set_ylabel("slides"); ax_h.set_title("How confident is the model?"); ax_h.legend(frameon=False)

    return _finish(fig, show)


def plot_clinical_views(y: np.ndarray, probabilities: dict[str, np.ndarray], classes: list[str], main: str | None = None, specificity: float = 0.90, show: bool = True,
                        threshold: float = 0.5) -> dict:
    """
    Clinical summaries of the 7-class probabilities: three tiers, and cancer against non-cancer.

    Two rows: the confusion matrices (three tiers, then cancer or not at a fixed threshold) and the ROC and
    precision-recall curves for cancer.

    * Arguments:
    y: The true class index of every slide
    probabilities: {model: predicted probabilities [n_slides, n_classes]}
    classes: The class names (the classes of BRACS.xlsx)
    main: The only model whose confusion matrices are shown (None = one pair of matrices for every model)
    specificity: The specificity at which the sensitivity for cancer is reported
    show: Whether to call plt.show() (the figure is returned either way)
    threshold: A slide is called cancer when the summed probability of DCIS and IC is at least this value (fixed, not tuned)

    Returns a dict with the figure and the clinical views per model.
    """

    views = {m: clinical_views(y, p, classes, specificity, threshold) for m, p in probabilities.items()}
    matrix_models = [main] if main is not None else list(views)
    k = len(matrix_models)
    names_text = " and ".join(_label(m) for m in matrix_models)
    notes_matrices = [
        (0, k, "Three clinical tiers obtained by adding the probabilities of the classes of each tier (benign: N, PB, UDH; atypical: FEA, ADH; malignant: DCIS, IC), for "
               f"{names_text}. Rows are the true tiers; numbers are the share (%) of the true tier and the number of slides" + ("; all matrices share one colour scale." if k > 1 else ".") + "\n"
               "Grouping removes the confusions inside a tier, so the tier results are better than the seven-class ones; the atypical tier remains the hardest because it lies between "
               "two others."),
        (k, 2 * k, f"Cancer (DCIS or IC) against everything else, for {names_text}: a slide is called cancer when the summed probability of DCIS and IC is at least {threshold:g}. "
                   "Rows are the true groups; numbers are the share (%) of the true group and the number of slides.\n"
                   "The matrix is the discrete counterpart of one point of the ROC curve below: the threshold sets the balance between the cancer slides that are found (sensitivity) and the "
                   "non-cancer slides that are cleared (specificity). The threshold is fixed in advance and not tuned on the Testing set; the class weighting of the loss and the prevalence "
                   "of cancer shift the probabilities, so it is not necessarily a suitable operating point, which for any application would be chosen on the Validation set. "
                   "Predictive values depend on the prevalence of cancer in the Testing set, which is not a screening population."),
    ]
    notes_curves = [
        (0, 1, f"ROC curve for cancer (DCIS or IC) against everything else, using the summed probability of DCIS and IC as score. Circles mark the sensitivity at {specificity:.0%} specificity.\n"
               "This is the clinically decisive boundary. It is easier than the seven-class problem because the hard confusions between neighbouring grades do not count; the operating "
               "point for use should be chosen on the Validation set, not on the Testing set."),
        (1, 2, "Precision-recall curve for the same task; the dotted line is the prevalence of cancer in the Testing set, the performance of a classifier without information.\n"
               "Precision shows how many slides flagged as cancer are cancer, which depends on the prevalence in the population where the model is used; the Testing set is not a screening population."),
    ]
    with plt.rc_context(STYLE):
        width = 17 if k > 1 else 14
        fig = plt.figure(figsize=(width, 13), layout="constrained")
        rows = fig.subfigures(2, 1, height_ratios=[6.3, 6.3])

        # Row 1: confusion matrices, three tiers then cancer or not, one per model
        top = split_with_notes(rows[0], notes_matrices, [1] * (2 * k), width, plot_height=3.9, note_height=2.4)
        axes = np.atleast_1d(top.subplots(1, 2 * k))
        for ax, model in zip(axes[:k], matrix_models):
            tiers = views[model]["tiers"]
            counts = tiers["confusion"].to_numpy(); names = tiers["names"]
            im = _confusion_panel(ax, counts, names, "predicted tier", "true tier", f"Benign, atypical or malignant?\n({_label(model)})")
        for ax, model in zip(axes[k:], matrix_models):
            _confusion_panel(ax, views[model]["cancer"]["confusion"], ["non-cancer", "cancer"], "predicted", "true", f"Cancer or not?\n({_label(model)}, threshold {threshold:g})")
        top.colorbar(im, ax=list(axes), shrink=0.8, label="% of the true class")

        # Row 2: ROC and precision-recall curves for cancer
        bottom = split_with_notes(rows[1], notes_curves, [1, 1], width, plot_height=3.9, note_height=2.0)
        ax_r, ax_p = bottom.subplots(1, 2)
        for j, (model, view) in enumerate(views.items()):
            c = view["cancer"]
            ax_r.plot(c["fpr"], c["tpr"], color=_color(model, j), lw=2, label=f"{_label(model)}  AUROC {c['auroc']:.2f}")
            ax_r.scatter([1 - specificity], [c["sensitivity_at_specificity"]], s=60, color=_color(model, j), edgecolor=SURFACE, zorder=3)
            precision, recall, _ = precision_recall_curve(c["y"], c["score"])
            ax_p.plot(recall, precision, color=_color(model, j), lw=2, label=f"{_label(model)}  AP {c['ap']:.2f}")
        ax_r.plot([0, 1], [0, 1], color=INK_2, ls=":", lw=1)
        ax_r.set_xlabel("1 − specificity"); ax_r.set_ylabel("sensitivity"); ax_r.set_title("Cancer or not? (ROC)"); ax_r.legend(frameon=False, loc="lower right")
        prevalence = next(iter(views.values()))["cancer"]["y"].mean()
        ax_p.axhline(prevalence, color=INK_2, ls=":", lw=1)
        ax_p.set_xlabel("recall (sensitivity)"); ax_p.set_ylabel("precision"); ax_p.set_ylim(0, 1.02); ax_p.set_title("Cancer or not? (precision-recall)"); ax_p.legend(frameon=False, loc="lower left")

    _finish(fig, show)

    return {"figure": fig, "views": views}


def _confusion_panel(ax, counts, names: list[str], xlabel: str, ylabel: str, title: str):
    """Draw a confusion matrix as row shares (%) with the counts, on a fixed 0-100% scale; returns the image."""
    counts = np.asarray(counts)
    share = counts / np.maximum(counts.sum(1, keepdims=True), 1) * 100
    im = ax.imshow(share, cmap=SEQ, vmin=0, vmax=100)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names); ax.set_yticks(range(len(names))); ax.set_yticklabels(names); ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{share[i, j]:.0f}%\n({counts[i, j]})", ha="center", va="center", fontsize=11, color="white" if share[i, j] > 55 else INK_2)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)

    return im


def plot_model_comparison(intervals: dict[str, pd.DataFrame], per_seed: dict[str, pd.DataFrame], paired: pd.DataFrame, a: str, b: str, show: bool = True):
    """
    Headline measures of the models with 95% intervals, and the paired difference between two of them.

    * Arguments:
    intervals: {model: ci DataFrame} from classification_metrics.patient_bootstrap (point, low, high per measure)
    per_seed: {model: DataFrame with one row per seed and one column per measure}
    paired: The output of classification_metrics.paired_bootstrap(a, b)
    a, b: The names of the models compared (the difference is a minus b)
    show: Whether to call plt.show() (the figure is returned either way)
    """

    notes = [(0, 1, "Headline measures on the Testing set. Large markers, the seed-ensemble with the 95% patient-bootstrap interval; small dots, the single models (seeds, or fold models and repeats).\n"
                    "Chance level is 0 for kappa and the Matthews correlation, 0.5 for AUROC and the inverse of the number of classes for balanced accuracy. Dots that spread more than the "
                    "interval show that the result depends on the training data and the random initialisation, and that a single run would be unreliable."),
             (1, 2, f"Difference {_label(a)} minus {_label(b)} with the 95% interval of a paired patient bootstrap: both models are scored on the same resamples.\n"
                    "An interval that contains 0 means that, with this number of Testing slides, there is no evidence that the two models differ; an interval to the right of 0 supports "
                    f"{_label(a)}. With 87 Testing slides only fairly large differences can be detected.")]
    measures = [m for m in HEADLINE if m in intervals[a].index]
    y = np.arange(len(measures))[::-1]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(14, 7.2), layout="constrained")
        top = split_with_notes(fig, notes, [1.3, 1], 14, plot_height=4.4, note_height=1.8)
        ax_m, ax_d = top.subplots(1, 2, width_ratios=[1.3, 1])
        for j, (model, table) in enumerate(intervals.items()):
            offset = (j - (len(intervals) - 1) / 2) * 0.3
            pos = y + offset
            ax_m.errorbar(table.loc[measures, "point"], pos, xerr=[table.loc[measures, "point"] - table.loc[measures, "low"], table.loc[measures, "high"] - table.loc[measures, "point"]],
                          fmt="o", ms=8, color=_color(model, j), ecolor=_color(model, j), capsize=3, lw=1.8, label=_label(model))
            for i, m in enumerate(measures):
                if model in per_seed:
                    ax_m.scatter(per_seed[model][m], np.full(len(per_seed[model]), pos[i]), s=10, color=_color(model, j), alpha=0.45, linewidths=0)
        ax_m.set_yticks(y); ax_m.set_yticklabels([MEASURE_LABELS[m] for m in measures]); ax_m.grid(axis="y", visible=False); ax_m.set_xlabel("value")
        ax_m.set_title("How do the models compare?"); ax_m.legend(frameon=False, loc="lower left")

        ax_d.errorbar(paired.loc[measures, "difference"], y, xerr=[paired.loc[measures, "difference"] - paired.loc[measures, "low"], paired.loc[measures, "high"] - paired.loc[measures, "difference"]],
                      fmt="o", ms=8, color=SERIES[0], ecolor=SERIES[0], capsize=3, lw=1.8)
        ax_d.axvline(0, color=INK_2, lw=1.2, ls="--")
        ax_d.set_yticks(y); ax_d.set_yticklabels([]); ax_d.grid(axis="y", visible=False); ax_d.set_xlabel(f"{_label(a)} − {_label(b)}")
        ax_d.set_title("Is the difference real?")

    return _finish(fig, show)


def plot_model_decision(run: dict, config: dict, model_dir, show: bool = True):
    """
    The model chosen for saving: its architecture as a graph and its specification as a table.

    * Arguments:
    run: The selected train_model result (mil_train.select_best_run)
    config: The configuration of the run (the dict that mil_train.save_best_model stores in the file)
    model_dir: The folder the classifier is saved to (the file name is only known when it is written)
    show: Whether to call plt.show() (the figure is returned either way)
    """

    # STEP 1: Quantities read from the run and the configuration
    name, kw, classes = run["model_name"], run["model_kwargs"], run["classes"]
    args, base, ntrain = config.get("train_args", {}), config.get("seed", 0), config.get("n_slides", {})
    repeat, fold = divmod(run["seed"] - base, 1000)
    parameters = sum(p.numel() for p in run["model"].parameters())
    best = run["best"]
    sizes = ", ".join(f"{s} {n}" for s, n in ntrain.items())
    cuda = "bf16 autocast on CUDA, float32 otherwise"

    # STEP 2: The specification table
    groups = [
        ("Location and use", [("file", f"{Path(model_dir)}/classifier_model_<date-time>.pt"), ("loading", "utils.mil_train.load_classifier(path)"),
                              ("contents", "weights, class names, feature scaler, architecture settings, early-stopping result, configuration of the run")]),
        ("Architecture", [("model", f"{_label(name)}  ({parameters:,} parameters)"), ("input", f"{run['in_dim']}-dimensional CLS embedding per tile"),
                          ("outputs", f"{len(classes)} classes: {', '.join(classes)}"), ("settings", ", ".join(f"{k} = {v}" for k, v in kw.items()) or "defaults")]),
        ("Training", [("optimiser", "AdamW, linear warm-up then cosine decay"), ("learning rate", args.get("lr", "")), ("weight decay", run["weight_decay"]),
                      ("batch / tiles", f"{args.get('batch_size', '')} slides per batch, at most {args.get('max_tiles', '')} tiles per slide"),
                      ("tile dropout", run["tile_dropout"]), ("coarse loss weight", run["coarse_weight"]), ("loss weighting", config.get("weighting", "")),
                      ("epochs", f"selected {run['best_epoch']} of {run['epochs_run']} run (maximum {args.get('epochs', '')}, patience {args.get('patience', '')})"), ("precision", cuda)]),
        ("Data and selection", [("features", f"{config.get('model_name', '')}; tiling {config.get('tile_params', '')}"), ("slides", sizes),
                                ("selected run", f"fold {fold + 1} of {config.get('n_folds', '')}, repeat {repeat + 1}, seed {run['seed']}"),
                                ("early-stopping result", f"macro F1 {best['val_macro_f1']:.3f}, balanced accuracy {best['val_balanced_accuracy']:.3f}"),
                                ("rule", "highest macro F1 on the early-stopping slides of the fold, over all fold models of all models; the Testing set plays no part")]),
    ]

    # STEP 3: Draw the graph, the table and the captions
    notes = [(0, 1, f"Forward pass of the selected model ({_label(name)}). Each box is one operation on the tiles of a slide; N is the number of tiles of the slide, which varies, and padding is masked so that it takes no part in the softmax or the sums. "
                    + {"clam_mb": "The gated attention network (Ilse et al., 2018) scores every tile; CLAM-MB (Lu et al., 2021) uses one attention branch and one classifier per class, and adds an instance-level clustering loss during training only, which is not used at prediction time.",
                       "abmil": "The gated attention network (Ilse et al., 2018) scores every tile, and the slide vector is the attention-weighted sum of the tile projections.",
                       "linear": "The slide vector is the plain mean of the tile embeddings, so no tile is weighted."}.get(name, "")),
             (1, 2, "Specification of the file written in section 8.3. The selected model is one fold model of the seed ensemble: it has seen the fit slides of its fold only, so its performance on the Testing set is not the ensemble performance "
                    "reported in section 6. The early-stopping result is computed on slides that also select the epoch and the model, and is therefore optimistic.")]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(15, 11.5), layout="constrained")
        top = split_with_notes(fig, notes, [1.1, 1], 15, plot_height=8.6, note_height=1.5)
        ax_a, ax_b = top.subplots(1, 2, width_ratios=[1.1, 1])
        ax_a.set_title("Architecture"); ax_b.set_title("Specification")
        _draw_architecture(ax_a, name, run["in_dim"], len(classes), kw)
        _draw_specification(ax_b, groups)

    return _finish(fig, show)


def plot_attention(
    slides: list[str],
    data: dict,
    attention: dict[str, np.ndarray],
    predictions: pd.DataFrame,
    data_dir: str | Path | None = None,
    top_k: int = 6,
    show: bool = True,
    model: str = "abmil",
):
    """
    Attention of an attention model over the tiles of a few slides, and the tiles with the highest attention.

    * Arguments:
    slides: The slides to show
    data: The output of mil_data.load_bags (coordinates and tile_params)
    attention: {slide: attention weight of every tile}
    predictions: The output of classification_metrics.predictions_table (true, pred, confidence)
    data_dir: The folder with the .svs files; the tiles are shown only when the files can be read
    top_k: The number of highest-attention tiles shown per slide
    show: Whether to call plt.show() (the figure is returned either way)
    model: The name of the model that produced the attention ("abmil" or "clam_mb"); used in the caption
    """

    params = data["tile_params"] if isinstance(data["tile_params"], dict) else None
    n = len(slides)
    what = ("the weight that the model gives it when it forms the slide vector" if model != "clam_mb" else
            "the weight that the attention branch of the predicted class gives it when that class's slide vector is formed")
    notes = [(0, n, f"Attention of the {_label(model)} model: each square is a tile, drawn at its position on the slide and coloured by {what} "
                    "(weights sum to 1 over the slide; the colour scale stops at the 99.5th percentile). Below each map, the tiles with the highest attention.\n"
                    "Attention shows which tiles the model weighs most; it is not a causal explanation and not a lesion segmentation. High attention on artefacts, slide edges or stained debris "
                    "would show that the model uses features unrelated to the lesion, and has to be checked by a pathologist. Titles give the true class, the predicted class and its probability.")]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(5.2 * n, 9.6), layout="constrained")
        fig.suptitle(f"Attention of the {_label(model)} model over the tiles", fontsize=16, weight="bold", x=0.005, ha="left")
        top = split_with_notes(fig, notes, [1] * n, 5.2 * n, plot_height=7.2, note_height=2.4)
        columns = np.atleast_1d(top.subfigures(1, n))
        for column, slide in zip(columns, slides):
            row = predictions[predictions["slide"] == slide].iloc[0]
            weights, xy = attention[slide], data["coords"][slide].numpy()
            grid = column.add_gridspec(2, 1, height_ratios=[3, 1.35])
            ax = column.add_subplot(grid[0])
            stride = params["stride"] if params else int(np.gcd.reduce(np.abs(np.diff(np.unique(xy[:, 0])))) or 1)
            gx, gy = xy[:, 0] // stride, xy[:, 1] // stride
            gx, gy = gx - gx.min(), gy - gy.min()
            raster = np.full((gy.max() + 1, gx.max() + 1), np.nan)
            raster[gy, gx] = weights
            sc = ax.imshow(raster, cmap=SEQ.with_extremes(bad=SURFACE), vmin=0, vmax=np.percentile(weights, 99.5), interpolation="nearest")
            ax.grid(False); ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)
            ax.set_title(f"{slide}\ntrue {row['true']} · predicted {row['pred']} ({row['confidence']:.2f})", fontsize=11)
            column.colorbar(sc, ax=ax, shrink=0.6, label="attention")
            inner = grid[1].subgridspec(2, max(top_k // 2, 1))
            best = np.argsort(-weights)[:top_k]
            path = Path(data_dir) / f"{slide}.svs" if data_dir is not None else None
            page = WSITileDataset._get_level_pages(path)[params["level"] - 1] if (path is not None and path.exists() and params) else None
            for i, tile_index in enumerate(best):
                axt = column.add_subplot(inner[i // max(top_k // 2, 1), i % max(top_k // 2, 1)])
                axt.set_xticks([]); axt.set_yticks([]); axt.grid(False); axt.spines[:].set_visible(False)
                tile = None
                if page is not None:
                    x, y = (int(v) for v in xy[tile_index])
                    try:
                        tile = _read_region(path, page, x, y, params["tile_size"])
                    except pyvips.Error:                 # coordinates outside the slide (files from another tiling) or damaged data
                        tile = None
                if tile is not None:
                    axt.imshow(tile)
                else:
                    axt.text(0.5, 0.5, f"({xy[tile_index][0]}, {xy[tile_index][1]})", ha="center", va="center", fontsize=9, color=INK_2, transform=axt.transAxes)

    return _finish(fig, show)
