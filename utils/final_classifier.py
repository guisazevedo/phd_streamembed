# IMPORTS *****************************************************************************************
# External libraries
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

# Internal imports ********************************************************************************
from utils.cls_overview import INK_2, SEQ, STYLE, cluster_colors
from utils.figure_captions import split_with_notes
from utils.mil_train import predict_ensemble, train_model, training_class_weights

# Constants ***************************************************************************************
POOL_SETS = ("Training", "Validation")
VIEW_METRICS = ["accuracy", "sensitivity", "specificity", "precision", "f1", "auroc"]
MACRO = "macro"


# Helper functions ********************************************************************************
def _view_array(y: np.ndarray, proba: np.ndarray, pred: np.ndarray | None = None) -> np.ndarray:
    """
    One-vs-rest measures of every class as an array [n_classes + 1, 6] (columns: see VIEW_METRICS).

    The last row is the macro average over classes (NaN entries are skipped), except for the accuracy,
    which is the multi-class accuracy. A measure is NaN when its denominator is 0 (a class without
    positives or without negatives in the sample). The predicted class is `pred` (default: the class with the
    highest probability); the AUROC always uses the probabilities (scores).
    """
    K, n = proba.shape[1], len(y)
    pred = proba.argmax(1) if pred is None else pred
    cm = np.bincount(y * K + pred, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(float)
    fn, fp = cm.sum(1) - tp, cm.sum(0) - tp
    tn = n - tp - fn - fp
    divide = lambda a, b: np.divide(a, b, out=np.full(K, np.nan), where=b > 0)
    auroc = np.full(K, np.nan)
    for k in range(K):
        positive = (y == k).astype(int)
        if 0 < positive.sum() < n:
            auroc[k] = roc_auc_score(positive, proba[:, k])
    values = np.stack([divide(tp + tn, np.full(K, float(n))), divide(tp, tp + fn), divide(tn, tn + fp), divide(tp, tp + fp), divide(2 * tp, 2 * tp + fp + fn), auroc], axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        macro = np.nanmean(values, axis=0)
    macro[0] = tp.sum() / n

    return np.vstack([values, macro])


# Main functions **********************************************************************************
def relabel_table(table: pd.DataFrame, mapping: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Replace the 7-class label of every slide by its group.

    * Arguments:
    table: The slide table (columns slide, patient, label, set); the label may be a Categorical
    mapping: {BRACS class: group name}; the order of its values (first appearance) is the class order of the model

    Returns the relabelled copy (label as str) and the ordered list of group names.
    """
    missing = set(table["label"].astype(str)) - set(mapping)
    if missing:
        raise ValueError(f"Classes without a group in the mapping: {sorted(missing)}")
    result = table.copy()
    result["label"] = result["label"].astype(str).map(mapping)

    return result, list(dict.fromkeys(mapping.values()))


def make_holdout_split(table: pd.DataFrame, early_stop_fraction: float = 0.2, seed: int = 0) -> pd.DataFrame:
    """
    Split the Training and Validation slides into fit and early-stopping slides, patient-grouped and stratified
    on the original 7-class label (so every label set gets the same split).

    The returned table keeps the official Testing slides as "Testing", marks the fit slides "Training" and the
    early-stopping slides "Validation", so train_model can be used unchanged. All slides of a patient are in the
    same part, which also removes the patient with slides in both official sets.

    * Arguments:
    table: The slide table of mil_data.load_bags with the original 7-class label
    early_stop_fraction: The approximate share of the pool used for early stopping
    seed: The seed of the split
    """

    # STEP 1: The pool (never the official Testing slides)
    result = table.copy()
    pool = result[result["set"].isin(POOL_SETS)]
    if pool.empty:
        raise ValueError("No Training or Validation slides with features.")

    # STEP 2: One patient-grouped, stratified split; the first fold is the early-stopping part
    splitter = StratifiedGroupKFold(n_splits=max(2, round(1 / early_stop_fraction)), shuffle=True, random_state=seed)
    _, stop_index = next(iter(splitter.split(pool, pool["label"].astype(str), pool["patient"])))
    result.loc[pool.index, "set"] = "Training"
    result.loc[pool.index[stop_index], "set"] = "Validation"

    # STEP 3: No patient is in both parts
    shared = set(result.loc[result["set"] == "Training", "patient"]) & set(result.loc[result["set"] == "Validation", "patient"])
    if shared:
        raise ValueError(f"Patients in both the fit and the early-stopping slides: {sorted(shared)}")

    return result


def train_final(
    data: dict,
    table: pd.DataFrame,
    classes: list[str],
    device: torch.device,
    seed: int,
    weighting: str | None,
    model_args: dict,
    train_args: dict,
    dropout: float,
    tile_dropout: float,
    verbose: bool = True,
) -> dict:
    """
    Train one CLAM-MB on a (relabelled) table whose `set` column comes from make_holdout_split.

    The labels of the Testing slides are never used by train_model. The coarse auxiliary loss is not used: it
    is defined on the 7 classes. The class weights come from the fit slides and weight the training loss and
    the early-stopping loss alike (macro F1 selects the epoch, the weighted early-stopping loss breaks ties).

    * Arguments:
    data: The output of mil_data.load_bags (the bags are shared, not copied)
    table: The relabelled table with the split (columns slide, patient, label, set)
    classes: The group names, in the order of the model outputs
    device: The device to run on
    seed: The seed of the run
    weighting: The class weighting ("sqrt", "inverse", "effective" or None)
    model_args: The CLAM-MB settings (k_sample, bag_weight)
    train_args: The arguments of train_model (epochs, patience, lr, weight_decay, batch_size, max_tiles)
    dropout: The dropout rate of the model
    tile_dropout: The fraction of the tiles of a training bag removed at random in every epoch
    verbose: Whether train_model prints its progress
    """
    local = {**data, "table": table, "classes": classes}

    return train_model("clam_mb", local, training_class_weights(local, weighting), device, seed, verbose=verbose,
                       tile_dropout=tile_dropout, dropout=dropout, weighted_val_loss=True, **model_args, **train_args)


def predict_testing(run: dict, data: dict, table: pd.DataFrame, classes: list[str], device: torch.device) -> dict:
    """
    Probabilities of the official Testing slides (their labels are read here for the first time).

    Returns a dict with test (the Testing rows of the table), y (class index), proba [n, K] and patients.
    """
    test = table[table["set"] == "Testing"].reset_index(drop=True)
    proba = predict_ensemble([run], data, test["slide"].tolist(), device)["proba"]

    return {"test": test, "y": np.array([classes.index(l) for l in test["label"]]), "proba": proba, "patients": test["patient"].to_numpy()}


def make_stage2_table(split: pd.DataFrame, mapping: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Table for the second stage of a two-stage classifier: only the slides whose 7-class label is in the mapping.

    The official Testing slides are dropped, so the second-stage model cannot see their labels. The fit and
    early-stopping parts of make_holdout_split are kept as they are.

    * Arguments:
    split: The table of make_holdout_split (the original 7-class label)
    mapping: {BRACS class: group name} of the classes handled by the second stage (e.g. DCIS and IC)

    Returns the filtered, relabelled table and the ordered group names.
    """
    keep = split[split["label"].astype(str).isin(mapping) & (split["set"] != "Testing")]

    return relabel_table(keep, mapping)


def combine_two_stage(proba_first: np.ndarray, proba_second: np.ndarray) -> np.ndarray:
    """
    Scores of a two-stage classifier over [rest, second-stage classes...] by the chain rule.

    The first stage gives P(rest) and P(positive) (two columns, the positive class last); the second stage
    splits the positive class: P(class j) = P(positive) * P(j | positive). The rows sum to 1. Use these as
    scores (ROC, AUROC), not for the decision: the argmax of the product can send a slide that the first
    stage calls positive back to the rest class; the decision is cascade_predict.

    * Arguments:
    proba_first: First-stage probabilities [n, 2] (rest, positive)
    proba_second: Second-stage probabilities [n, J], computed for every slide
    """
    return np.concatenate([proba_first[:, 0:1], proba_first[:, 1:2] * proba_second], axis=1)


def cascade_predict(proba_first: np.ndarray, proba_second: np.ndarray) -> np.ndarray:
    """
    Predicted class of a two-stage classifier: the first stage decides rest or positive (its argmax), and only
    the slides it calls positive are assigned by the second stage.

    So the rest class is decided by the first stage alone, and the second stage only divides the positive
    slides among its classes.

    * Arguments:
    proba_first: First-stage probabilities [n, 2] (rest, positive)
    proba_second: Second-stage probabilities [n, J]

    Returns the class index [n] over [rest, second-stage classes...] (0 = rest).
    """
    positive = proba_first.argmax(1) == 1

    return np.where(positive, 1 + proba_second.argmax(1), 0)


def class_views(y: np.ndarray, proba: np.ndarray, classes: list[str], pred: np.ndarray | None = None) -> pd.DataFrame:
    """
    Accuracy, sensitivity (recall), specificity, precision, F1 and AUROC of every class against the rest, and their macro average.

    The accuracy of a class row is the one-vs-rest accuracy; the macro row holds the multi-class accuracy and
    the macro average of the other measures. For two classes, the row of the positive class (the last one) is
    the usual binary view.

    * Arguments:
    y: The true class index of every slide
    proba: The predicted probabilities [n_slides, n_classes]
    classes: The class names, in the order of the probability columns
    pred: The predicted class of every slide; None = the class with the highest probability. Give it when the
        decision is not the argmax of proba (a cascade); the AUROC still uses proba

    Returns a DataFrame indexed by class name and "macro" with one column per measure (VIEW_METRICS).
    """
    return pd.DataFrame(_view_array(y, proba, pred), index=[*classes, MACRO], columns=VIEW_METRICS)


def bootstrap_views(y: np.ndarray, proba: np.ndarray, patients: np.ndarray, classes: list[str], n_boot: int = 1000, seed: int = 0, pred: np.ndarray | None = None) -> dict[str, pd.DataFrame]:
    """
    95% percentile intervals of class_views from a bootstrap that resamples patients, with all of their slides.

    Slides of one patient are not independent, so resampling slides would give intervals that are too narrow.
    A measure undefined in a resample (a class absent from it) is left out of that measure's interval.

    Returns {"low": DataFrame, "high": DataFrame}, with the layout of class_views.
    """
    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(patients, return_inverse=True)
    members = [np.where(inverse == i)[0] for i in range(len(unique))]
    samples = []
    for _ in range(n_boot):
        index = np.concatenate([members[i] for i in rng.integers(0, len(unique), len(unique))])
        samples.append(_view_array(y[index], proba[index], None if pred is None else pred[index]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        low, high = np.nanpercentile(samples, [2.5, 97.5], axis=0)
    names = [*classes, MACRO]

    return {"low": pd.DataFrame(low, index=names, columns=VIEW_METRICS), "high": pd.DataFrame(high, index=names, columns=VIEW_METRICS)}


def format_views(views: pd.DataFrame, intervals: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """The measures as text, "value [low, high]" when intervals (from bootstrap_views) are given."""
    if intervals is None:
        return views.map(lambda v: f"{v:.3f}")
    low, high = intervals["low"], intervals["high"]

    return pd.DataFrame({c: [f"{views.loc[r, c]:.3f} [{low.loc[r, c]:.3f}, {high.loc[r, c]:.3f}]" for r in views.index] for c in views.columns}, index=views.index)


def paired_bootstrap_views(y: np.ndarray, proba_a: np.ndarray, pred_a: np.ndarray, proba_b: np.ndarray, pred_b: np.ndarray, patients: np.ndarray,
                           classes: list[str], n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """
    Difference A - B of the measures of class_views between two models on the same slides, with a paired patient bootstrap.

    Both models are scored on the same resamples, so the interval reflects the difference and not the
    uncertainty of each model separately. The decisions are pred_a and pred_b (they need not be the argmax).

    Returns a DataFrame with one row per class and "macro", and for every measure the columns
    <measure> difference, low, high (95% interval) and the share of resamples in which A is better, as a
    MultiIndex on the columns (measure, statistic).
    """
    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(patients, return_inverse=True)
    members = [np.where(inverse == i)[0] for i in range(len(unique))]
    observed = _view_array(y, proba_a, pred_a) - _view_array(y, proba_b, pred_b)
    samples = []
    for _ in range(n_boot):
        index = np.concatenate([members[i] for i in rng.integers(0, len(unique), len(unique))])
        samples.append(_view_array(y[index], proba_a[index], pred_a[index]) - _view_array(y[index], proba_b[index], pred_b[index]))
    samples = np.array(samples)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        low, high = np.nanpercentile(samples, [2.5, 97.5], axis=0)
        better = np.nanmean(samples > 0, axis=0)
    names = [*classes, MACRO]
    columns = pd.MultiIndex.from_product([VIEW_METRICS, ["difference", "low", "high", "share_a_better"]])
    values = np.stack([observed, low, high, better], axis=2).reshape(len(names), -1)

    return pd.DataFrame(values, index=names, columns=columns)


def format_differences(differences: pd.DataFrame) -> pd.DataFrame:
    """The output of paired_bootstrap_views as text, "difference [low, high]" for every class and measure."""
    return pd.DataFrame({m: [f"{differences.loc[r, (m, 'difference')]:+.3f} [{differences.loc[r, (m, 'low')]:+.3f}, {differences.loc[r, (m, 'high')]:+.3f}]" for r in differences.index]
                         for m in VIEW_METRICS}, index=differences.index)


def summary_across_sets(views: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The macro row of every label set (one row per set); for the binary set read the positive class from its own table."""
    return pd.DataFrame({name: v.loc[MACRO] for name, v in views.items()}).T


def plot_final_results(
    name: str,
    confusion: pd.DataFrame,
    y: np.ndarray,
    proba: np.ndarray,
    classes: list[str],
    show: bool = True,
):
    """
    Confusion matrix (left) and one-vs-rest ROC curves with the AUROC (right) of one label set on the Testing set.

    * Arguments:
    name: The name of the label set (used in the titles)
    confusion: The confusion matrix of counts (rows true class, columns predicted class)
    y: The true class index of every Testing slide
    proba: The predicted probabilities [n_slides, n_classes]
    classes: The class names, in the order of the probability columns
    show: Whether to call plt.show() (the figure is returned either way)
    """
    K, colors = len(classes), cluster_colors(len(classes))
    binary = K == 2
    notes = [
        (0, 1, f"Confusion matrix of the {name} model on the Testing set: rows are the true classes, columns the predicted classes (the class with the highest probability); colour, the share of the true class; "
               "numbers, the share (%) and the number of slides.\n"
               "A dominant diagonal means that the classes are recognised. The Testing set has few slides in some classes, so one slide moves a row percentage by several points; read the counts with the percentages."),
        (1, 2, ("ROC curve of the positive class (" + classes[-1] + "): sensitivity against 1 − specificity as the decision threshold on its probability varies; the diagonal is chance (AUROC 0.5). "
                if binary else
                "One-vs-rest ROC curve of each class (that class against all the others): sensitivity against 1 − specificity as the decision threshold on the class probability varies. "
                "The thick dashed line is the average of the class curves (macro), and the diagonal is chance (AUROC 0.5). ")
               + "The legend gives the area under the curve (AUROC), which measures how well a class is ranked above the others and does not depend on the threshold or on the class frequency.\n"
               "Classes with few slides give coarse, step-like curves."),
    ]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(15, 8.8), layout="constrained")
        top = split_with_notes(fig, notes, [1, 1], 15, plot_height=6.2, note_height=1.9)
        ax_cm, ax_roc = top.subplots(1, 2)

        # STEP 1: Confusion matrix
        counts = confusion.to_numpy()
        share = counts / np.maximum(counts.sum(1, keepdims=True), 1) * 100
        im = ax_cm.imshow(share, cmap=SEQ, vmin=0, vmax=100)
        ax_cm.set_xticks(range(K)); ax_cm.set_xticklabels(confusion.columns); ax_cm.set_yticks(range(K)); ax_cm.set_yticklabels(confusion.index); ax_cm.grid(False)
        for i in range(K):
            for j in range(K):
                ax_cm.text(j, i, f"{share[i, j]:.0f}%\n({counts[i, j]})", ha="center", va="center", fontsize=11, color="white" if share[i, j] > 55 else INK_2)
        ax_cm.set_xlabel("predicted class"); ax_cm.set_ylabel("true class"); ax_cm.set_title(f"{name}: confusion matrix")
        top.colorbar(im, ax=ax_cm, shrink=0.7, label="% of the true class")

        # STEP 2: ROC curves (the positive class only for two classes, where the other curve is its mirror image)
        grid, curves = np.linspace(0, 1, 201), []
        for k in ([K - 1] if binary else range(K)):
            positive = (y == k).astype(int)
            if not 0 < positive.sum() < len(y):
                continue
            fpr, tpr, _ = roc_curve(positive, proba[:, k])
            ax_roc.plot(fpr, tpr, color=colors[k], lw=2, label=f"{classes[k]}  AUROC {np.trapezoid(tpr, fpr):.2f}")
            curves.append(np.interp(grid, fpr, tpr))
        if not binary and curves:
            macro = np.mean(curves, 0)
            ax_roc.plot(grid, macro, color=INK_2, lw=3, ls="--", label=f"macro  AUROC {np.trapezoid(macro, grid):.2f}")
        ax_roc.plot([0, 1], [0, 1], color=INK_2, lw=1, ls=":")
        ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1.02); ax_roc.set_xlabel("1 − specificity"); ax_roc.set_ylabel("sensitivity")
        ax_roc.set_title(f"{name}: ROC"); ax_roc.legend(frameon=False, loc="lower right")
    if show:
        plt.show()

    return fig
