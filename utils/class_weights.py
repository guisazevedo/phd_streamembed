# IMPORTS *****************************************************************************************
# External libraries
from collections.abc import Sequence
import matplotlib.pyplot as plt
import numpy as np
import torch

# Internal imports ********************************************************************************
from utils.cls_overview import GRID, INK_2, SERIES, STYLE, SURFACE
from utils.figure_captions import split_with_notes

# Constants ***************************************************************************************
METHODS = ("inverse", "sqrt", "effective")
UNWEIGHTED_COLOR = "#8a8f98"


# Main functions **********************************************************************************
def class_counts(labels: Sequence[int] | np.ndarray, n_classes: int) -> np.ndarray:
    """Return the number of samples of each class for integer labels in [0, n_classes)."""
    return np.bincount(np.asarray(labels, dtype=int), minlength=n_classes)


def compute_class_weights(counts: Sequence[int] | np.ndarray, method: str = "inverse", beta: float = 0.999) -> torch.Tensor:
    """
    Compute one loss weight per class from the number of training samples of each class.

    Weights are rescaled so that the mean weight over samples is 1, that is sum(n_c * w_c) = N. The loss
    then keeps its scale and only the relative importance of the classes changes. Classes without samples
    get weight 0.

    * Arguments:
    counts: The number of training samples n_c of each class (compute it on the Training set only)
    method: "inverse" (w ~ 1 / n_c, every class contributes equally to the loss, the 'balanced' rule),
        "sqrt" (w ~ 1 / sqrt(n_c), a milder correction) or "effective" (class-balanced loss of
        Cui et al., 2019: w ~ (1 - beta) / (1 - beta ** n_c))
    beta: The parameter of the "effective" method, between 0 and 1 (closer to 1 = closer to "inverse")

    Returns a float32 tensor of shape [n_classes].
    """

    # STEP 1: Raw weight of every class with samples
    n = np.asarray(counts, dtype=float)
    present = n > 0
    raw = np.zeros_like(n)

    if method == "inverse":
        raw[present] = 1.0 / n[present]
    elif method == "sqrt":
        raw[present] = 1.0 / np.sqrt(n[present])
    elif method == "effective":
        if not 0 < beta < 1:
            raise ValueError(f"beta must be in (0, 1), got {beta}")
        raw[present] = (1.0 - beta) / (1.0 - beta ** n[present])
    else:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    # STEP 2: Rescale so that the weights average to 1 over the samples
    return torch.as_tensor(raw * n.sum() / (raw * n).sum(), dtype=torch.float32)


def weighted_cross_entropy(weights: torch.Tensor, label_smoothing: float = 0.0) -> torch.nn.CrossEntropyLoss:
    """
    Return the cross-entropy loss with per-class weights.

    PyTorch averages a batch as sum(w_y * loss) / sum(w_y) over the samples, so the value is a weighted
    mean. Compute the weights on the Training set only. Use the weighted loss to train, and report
    unweighted evaluation metrics (balanced accuracy, per-class recall) on the untouched Validation and
    Testing sets.

    * Arguments:
    weights: The class weights from compute_class_weights (a tensor or an array)
    label_smoothing: The label smoothing of the loss (0 = none)
    """
    return torch.nn.CrossEntropyLoss(weight=torch.as_tensor(weights, dtype=torch.float32), label_smoothing=label_smoothing)


def plot_loss_weighting(
    labels_train: Sequence[int] | np.ndarray,
    classes: list[str],
    methods: Sequence[str] = METHODS,
    beta: float = 0.999,
    show: bool = True,
) -> dict:
    """
    Show how class weights change the loss under class imbalance.

    Four panels: class frequency, the weight of each class per method, the share of the total loss that
    each class contributes (unweighted versus weighted), and the cross-entropy of three reference
    predictors (the class prior, a uniform predictor and one that favours the majority class) under each
    weighting. The reference predictors need no model: they show what each loss rewards.

    * Arguments:
    labels_train: The integer class label of every Training sample (slide or tile)
    classes: The class names, in label order
    methods: The weighting methods to compare (see compute_class_weights)
    beta: The parameter of the "effective" method
    show: Whether to call plt.show() (the figure is returned either way)

    Returns a dict with the figure, counts, weights (per method), loss_share (per method, in %) and
    reference_loss ({predictor: {method or "unweighted": value}}).
    """

    # STEP 1: Counts, weights and the share of the loss that each class contributes (n_c * w_c / N)
    K = len(classes)
    counts = class_counts(labels_train, K)
    N = counts.sum()
    weights = {m: compute_class_weights(counts, m, beta).numpy() for m in methods}
    weights_all = {"unweighted": np.ones(K), **weights}
    share = {m: 100 * counts * w / (counts * w).sum() for m, w in weights_all.items()}

    # STEP 2: Cross-entropy of reference predictors on the Training labels, checked against PyTorch
    prior = counts / N
    majority = np.full(K, 0.1 / (K - 1)); majority[counts.argmax()] = 0.9
    predictors = {"class prior\n(n_c / N)": prior, "uniform\n(1 / K)": np.full(K, 1.0 / K), f"favours {classes[counts.argmax()]}\n(p = 0.9)": majority}
    labels_t = torch.as_tensor(np.asarray(labels_train, dtype=int))
    reference = {}
    for name, p in predictors.items():
        logits = torch.log(torch.as_tensor(p, dtype=torch.float32)).repeat(len(labels_t), 1)
        reference[name] = {"unweighted": float(torch.nn.functional.cross_entropy(logits, labels_t))}
        for m in methods:
            reference[name][m] = float(weighted_cross_entropy(torch.as_tensor(weights[m]))(logits, labels_t))

    # STEP 3: Text summary
    print(f"Samples {N:,}  |  classes {K}  |  imbalance ratio (largest / smallest class) {counts.max() / max(counts[counts > 0].min(), 1):.1f}")
    print(f"{'class':>6} {'n':>6} " + " ".join(f"{m:>10}" for m in methods))
    for c in range(K):
        print(f"{classes[c]:>6} {counts[c]:>6} " + " ".join(f"{weights[m][c]:>10.3f}" for m in methods))

    # STEP 4: Figure (styled with rc_context so the global matplotlib settings are left alone)
    colors = {"unweighted": UNWEIGHTED_COLOR, **{m: SERIES[i] for i, m in enumerate(methods)}}
    notes_top = [
        (0, 1, "Number of Training samples per class. The imbalance ratio between the largest and the smallest class sets how strongly an "
               "unweighted loss favours the frequent classes.\n"
               "Weights are computed on the Training set only: using Validation or Testing counts would leak information about the "
               "evaluation sets into training."),
        (1, 2, "Loss weight of each class for each method, rescaled so that the mean weight over samples is 1 and the loss keeps its scale. "
               "Inverse-frequency weights make every class contribute equally; the square-root rule is a milder correction; the "
               "class-balanced weights of Cui et al. (2019) are based on the effective number of samples, (1 − β^n) / (1 − β), and "
               "approach the inverse rule as β tends to 1.\n"
               "Large weights on rare classes raise the variance of the gradient estimate, which is why the milder rules are an option."),
    ]
    notes_bottom = [
        (0, 1, "Share of the total loss contributed by each class, n_c·w_c normalised to 100%. Without weights it equals the class frequency, "
               "so the frequent classes dominate the gradient.\n"
               "Inverse weights equalise the shares (100/K per class); the other methods lie in between. The shares show what the optimiser "
               "is asked to prioritise; they do not guarantee that rare classes will be learned, since this also depends on how separable they are."),
        (1, 2, "Cross-entropy of three fixed predictors on the Training labels, without any model: the class prior, a uniform distribution, and "
               "a predictor that assigns probability 0.9 to the majority class.\n"
               "Under the unweighted loss the prior scores best, i.e. a model with uninformative features minimises the loss by reproducing the "
               "class frequencies. Under the inverse-weighted loss the uniform predictor scores best (its loss equals log K) and the prior and "
               "the majority predictor are penalised, so exploiting the imbalance no longer lowers the loss.\n"
               "Weighting changes the objective but adds no information: report balanced accuracy and per-class recall, and keep the test set unweighted."),
    ]

    x = np.arange(K)
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(15, 12.5), layout="constrained")
        fig.suptitle(f"Class imbalance and loss weighting · {N:,} Training samples, {K} classes", fontsize=14, weight="bold", x=0.01, ha="left")
        rows = fig.subfigures(2, 1, height_ratios=[1, 1.15])

        top = split_with_notes(rows[0], notes_top, [1, 1.3], 15, plot_height=3.4, note_height=2.0)
        ax_n, ax_w = top.subplots(1, 2, width_ratios=[1, 1.3])
        ax_n.bar(x, counts, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5, width=0.7)
        for i in range(K):
            ax_n.text(i, counts[i], f"{counts[i]}\n{100 * counts[i] / N:.0f}%", ha="center", va="bottom", fontsize=8, color=INK_2)
        ax_n.set_xticks(x); ax_n.set_xticklabels(classes); ax_n.set_ylim(0, counts.max() * 1.2); ax_n.set_ylabel("samples")
        ax_n.set_title("How imbalanced is the Training set?")

        width = 0.8 / len(methods)
        for j, m in enumerate(methods):
            ax_w.bar(x - 0.4 + width * (j + 0.5), weights[m], width=width, color=colors[m], edgecolor=SURFACE, linewidth=1, label=m)
        ax_w.axhline(1, color=INK_2, ls="--", lw=1)
        ax_w.set_xticks(x); ax_w.set_xticklabels(classes); ax_w.set_ylabel("weight (mean over samples = 1)")
        ax_w.set_title("How much weight does each class get?"); ax_w.legend(frameon=False, ncol=len(methods))

        bottom = split_with_notes(rows[1], notes_bottom, [1, 1.3], 15, plot_height=3.8, note_height=2.6)
        ax_s, ax_r = bottom.subplots(1, 2, width_ratios=[1, 1.3])
        names = ["unweighted", *methods]
        width = 0.8 / len(names)
        for j, m in enumerate(names):
            ax_s.bar(x - 0.4 + width * (j + 0.5), share[m], width=width, color=colors[m], edgecolor=SURFACE, linewidth=1, label=m)
        ax_s.axhline(100 / K, color=INK_2, ls="--", lw=1)
        ax_s.set_xticks(x); ax_s.set_xticklabels(classes); ax_s.set_ylabel("share of the total loss (%)")
        ax_s.set_title("Which classes drive the loss?"); ax_s.legend(frameon=False, ncol=2, fontsize=8)

        px = np.arange(len(predictors))
        for j, m in enumerate(names):
            vals = [reference[name][m] for name in predictors]
            ax_r.bar(px - 0.4 + width * (j + 0.5), vals, width=width, color=colors[m], edgecolor=SURFACE, linewidth=1, label=m)
            for xi, v in zip(px - 0.4 + width * (j + 0.5), vals):
                ax_r.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=INK_2)
        ax_r.axhline(np.log(K), color=INK_2, ls=":", lw=1)
        ax_r.text(px[-1] + 0.45, np.log(K), " log K", va="bottom", ha="right", fontsize=8, color=INK_2)
        ax_r.set_xticks(px); ax_r.set_xticklabels(list(predictors)); ax_r.set_ylabel("cross-entropy on Training labels")
        ax_r.set_title("Which predictor does each loss reward?"); ax_r.legend(frameon=False, ncol=2, fontsize=8)

    if show:
        plt.show()

    return {"figure": fig, "counts": counts, "weights": weights, "loss_share": share, "reference_loss": reference}
