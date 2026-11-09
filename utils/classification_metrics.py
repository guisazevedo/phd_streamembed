# IMPORTS *****************************************************************************************
# External libraries
import functools
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, log_loss, matthews_corrcoef, precision_recall_fscore_support, roc_auc_score, roc_curve,
                             top_k_accuracy_score)

# Constants ***************************************************************************************
# Clinical groupings of the BRACS classes (see section 3.1 of 00_data_visualization.ipynb)
TIERS = {"benign": ["N", "PB", "UDH"], "atypical": ["FEA", "ADH"], "malignant": ["DCIS", "IC"]}
CANCER = ["DCIS", "IC"]
HEADLINE = ["accuracy", "balanced_accuracy", "macro_f1", "macro_auroc", "macro_ap", "quadratic_kappa", "mcc"]


# Helper functions ********************************************************************************
def _quiet(function):
    """Silence warnings inside a function: a bootstrap resample can miss a class, and its measures are then left out of the averages."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return function(*args, **kwargs)
    return wrapper


def _one_vs_rest(y: np.ndarray, proba: np.ndarray, function) -> np.ndarray:
    """Apply an AUC-type function to every class against the rest; NaN where the class has no positives or no negatives."""
    values = np.full(proba.shape[1], np.nan)
    for k in range(proba.shape[1]):
        positive = (y == k).astype(int)
        if 0 < positive.sum() < len(y):
            values[k] = function(positive, proba[:, k])

    return values


def expected_calibration_error(y: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error: bin the predictions by confidence and average |accuracy - confidence| weighted by bin size."""
    confidence, correct = proba.max(1), (proba.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.digitize(confidence, edges[1:-1]), 0, n_bins - 1)

    return float(sum(abs(correct[bins == b].mean() - confidence[bins == b].mean()) * (bins == b).mean() for b in range(n_bins) if (bins == b).any()))


def headline_metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    """The headline measures (see HEADLINE); used for the bootstrap, so it is kept cheap."""
    pred = proba.argmax(1)
    classes = range(proba.shape[1])

    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, labels=classes, average="macro", zero_division=0),
        "macro_auroc": float(np.nanmean(_one_vs_rest(y, proba, roc_auc_score))),
        "macro_ap": float(np.nanmean(_one_vs_rest(y, proba, average_precision_score))),
        "quadratic_kappa": cohen_kappa_score(y, pred, weights="quadratic"),
        "mcc": matthews_corrcoef(y, pred),
    }


# Main functions **********************************************************************************
def evaluate_predictions(y: np.ndarray, proba: np.ndarray, classes: list[str]) -> dict:
    """
    All classification measures for a set of slides.

    * Arguments:
    y: The true class index of every slide (index into classes)
    proba: The predicted class probabilities [n_slides, n_classes]
    classes: The class names, in the order of the probability columns (the classes of BRACS.xlsx)

    Returns a dict with summary (Series of scalar measures: accuracy, balanced accuracy, macro and weighted
    F1, macro precision and recall, macro AUROC, macro AP, quadratic-weighted kappa, MCC, top-2 accuracy,
    log-loss, Brier score, ECE), per_class (DataFrame indexed by class name: precision, recall, F1, support,
    AUROC, AP) and confusion (DataFrame of counts, rows true class, columns predicted class).
    """

    # STEP 1: Scalar measures
    K = len(classes)
    pred = proba.argmax(1)
    onehot = np.eye(K)[y]
    summary = {
        **headline_metrics(y, proba),
        "weighted_f1": f1_score(y, pred, labels=range(K), average="weighted", zero_division=0),
        "macro_precision": precision_recall_fscore_support(y, pred, labels=range(K), average="macro", zero_division=0)[0],
        "macro_recall": precision_recall_fscore_support(y, pred, labels=range(K), average="macro", zero_division=0)[1],
        "top2_accuracy": top_k_accuracy_score(y, proba, k=2, labels=range(K)) if K > 2 else 1.0,
        "log_loss": log_loss(y, np.clip(proba, 1e-12, 1), labels=range(K)),
        "brier": float(((proba - onehot) ** 2).sum(1).mean()),
        "ece": expected_calibration_error(y, proba),
    }

    # STEP 2: Per-class measures
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=range(K), zero_division=0)
    per_class = pd.DataFrame({
        "precision": precision, "recall": recall, "f1": f1, "support": support,
        "auroc": _one_vs_rest(y, proba, roc_auc_score), "ap": _one_vs_rest(y, proba, average_precision_score),
    }, index=classes)

    # STEP 3: Confusion matrix
    confusion = pd.DataFrame(confusion_matrix(y, pred, labels=range(K)), index=classes, columns=classes)
    confusion.index.name, confusion.columns.name = "true", "predicted"

    return {"summary": pd.Series(summary), "per_class": per_class, "confusion": confusion}


def collapse_probabilities(proba: np.ndarray, classes: list[str], groups: dict[str, list[str]]) -> tuple[np.ndarray, list[str]]:
    """Sum the probabilities of the classes of every group; groups with no class in `classes` are dropped. Returns (proba, group names)."""
    names = [g for g, members in groups.items() if any(m in classes for m in members)]
    columns = [[classes.index(m) for m in groups[g] if m in classes] for g in names]

    return np.stack([proba[:, c].sum(1) for c in columns], axis=1), names


def collapse_labels(y: np.ndarray, classes: list[str], groups: dict[str, list[str]]) -> np.ndarray:
    """Map class indices to the index of their group (see collapse_probabilities)."""
    names = [g for g, members in groups.items() if any(m in classes for m in members)]
    lookup = {classes.index(m): names.index(g) for g in names for m in groups[g] if m in classes}

    return np.array([lookup[i] for i in y])


def clinical_views(y: np.ndarray, proba: np.ndarray, classes: list[str], specificity: float = 0.90, threshold: float = 0.5) -> dict:
    """
    Two clinical summaries of the 7-class probabilities, with no extra training.

    Tiers: benign (N, PB, UDH), atypical (FEA, ADH) and malignant (DCIS, IC) by summing the probabilities
    of the classes of each tier. Cancer: probability of DCIS or IC against the rest, with AUROC, average
    precision and the sensitivity at a given specificity. A slide is also called cancer when its summed
    probability of DCIS and IC is at least `threshold` (0.5 is the argmax of the two collapsed probabilities;
    it is fixed in advance and not tuned), which gives a 2 x 2 confusion matrix and its rates.

    Returns a dict with tiers (dict: names, y, proba, summary, confusion) and cancer (dict: y, score,
    auroc, ap, sensitivity_at_specificity, specificity_level, fpr, tpr, threshold, confusion [rows true
    non-cancer/cancer, columns predicted], sensitivity, specificity, ppv, npv; a rate is NaN when its
    denominator is 0).
    """

    # STEP 1: Three tiers
    tier_proba, tier_names = collapse_probabilities(proba, classes, TIERS)
    tier_y = collapse_labels(y, classes, TIERS)
    tiers = evaluate_predictions(tier_y, tier_proba, tier_names)

    # STEP 2: Cancer against the rest
    cancer_columns = [classes.index(c) for c in CANCER if c in classes]
    score = proba[:, cancer_columns].sum(1)
    positive = np.isin(y, cancer_columns).astype(int)
    fpr, tpr, _ = roc_curve(positive, score)

    # STEP 3: The 2 x 2 confusion matrix at the threshold and its rates
    called = (score >= threshold).astype(int)
    tp, fn = int(((called == 1) & (positive == 1)).sum()), int(((called == 0) & (positive == 1)).sum())
    fp, tn = int(((called == 1) & (positive == 0)).sum()), int(((called == 0) & (positive == 0)).sum())
    rate = lambda num, den: num / den if den else float("nan")

    return {
        "tiers": {"names": tier_names, "y": tier_y, "proba": tier_proba, **tiers},
        "cancer": {
            "y": positive, "score": score, "auroc": roc_auc_score(positive, score), "ap": average_precision_score(positive, score),
            "sensitivity_at_specificity": float(tpr[fpr <= 1 - specificity].max()), "specificity_level": specificity, "fpr": fpr, "tpr": tpr,
            "threshold": threshold, "confusion": np.array([[tn, fp], [fn, tp]]),
            "sensitivity": rate(tp, tp + fn), "specificity": rate(tn, tn + fp), "ppv": rate(tp, tp + fp), "npv": rate(tn, tn + fn),
        },
    }


@_quiet
def patient_bootstrap(y: np.ndarray, proba: np.ndarray, patients: np.ndarray, n_boot: int = 1000, seed: int = 0, per_class: bool = False) -> dict:
    """
    Bootstrap confidence intervals that resample patients, with all of their slides, instead of slides.

    Slides of one patient are not independent, so resampling slides would give intervals that are too
    narrow. Classes missing from a resample are left out of that resample's macro averages.

    * Arguments:
    y: The true class index of every slide
    proba: The predicted probabilities [n_slides, n_classes]
    patients: The patient of every slide
    n_boot: The number of resamples
    seed: The random seed
    per_class: Whether to also bootstrap the per-class recall, F1 and AUROC

    Returns a dict with samples (DataFrame, one row per resample, headline measures), ci (DataFrame indexed
    by measure: point, low, high; the 95% percentile interval) and, if per_class, per_class_ci (DataFrame
    with a column per measure and class).
    """

    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(patients, return_inverse=True)
    members = [np.where(inverse == i)[0] for i in range(len(unique))]
    K = proba.shape[1]
    rows, class_rows = [], []
    for _ in range(n_boot):
        index = np.concatenate([members[i] for i in rng.integers(0, len(unique), len(unique))])
        rows.append(headline_metrics(y[index], proba[index]))
        if per_class:
            pred = proba[index].argmax(1)
            _, recall, f1, _ = precision_recall_fscore_support(y[index], pred, labels=range(K), zero_division=0)
            class_rows.append(np.concatenate([recall, f1, _one_vs_rest(y[index], proba[index], roc_auc_score)]))

    samples = pd.DataFrame(rows)
    point = pd.Series(headline_metrics(y, proba))
    ci = pd.DataFrame({"point": point, "low": samples.quantile(0.025), "high": samples.quantile(0.975)})
    result = {"samples": samples, "ci": ci}
    if per_class:
        array = np.array(class_rows)
        result["per_class_samples"] = array
        result["per_class_ci"] = {name: np.nanpercentile(array[:, i * K:(i + 1) * K], [2.5, 97.5], axis=0) for i, name in enumerate(["recall", "f1", "auroc"])}

    return result


@_quiet
def paired_bootstrap(y: np.ndarray, proba_a: np.ndarray, proba_b: np.ndarray, patients: np.ndarray, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """
    Difference between two models on the same slides (A minus B) with a paired patient bootstrap.

    Both models are scored on the same resamples, so the interval reflects the difference and not the
    uncertainty of each model separately.

    Returns a DataFrame indexed by measure: difference, low, high (95% interval) and the share of
    resamples in which A is better than B.
    """

    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(patients, return_inverse=True)
    members = [np.where(inverse == i)[0] for i in range(len(unique))]
    differences = []
    for _ in range(n_boot):
        index = np.concatenate([members[i] for i in rng.integers(0, len(unique), len(unique))])
        a, b = headline_metrics(y[index], proba_a[index]), headline_metrics(y[index], proba_b[index])
        differences.append({k: a[k] - b[k] for k in a})
    samples = pd.DataFrame(differences)
    observed = pd.Series(headline_metrics(y, proba_a)) - pd.Series(headline_metrics(y, proba_b))

    return pd.DataFrame({"difference": observed, "low": samples.quantile(0.025), "high": samples.quantile(0.975), "share_a_better": (samples > 0).mean()})


def predictions_table(table: pd.DataFrame, classes: list[str], proba: np.ndarray) -> pd.DataFrame:
    """
    One row per slide with the class names of the label sheet: slide, patient, set, true, pred, confidence
    and one probability column p_<class> per class.

    * Arguments:
    table: The slides, in the order of the rows of proba (columns slide, patient, set, label)
    classes: The class names, in the order of the probability columns
    proba: The predicted probabilities [n_slides, n_classes]
    """
    result = table[["slide", "patient", "set"]].copy().reset_index(drop=True)
    result["true"] = table["label"].astype(str).to_numpy()
    result["pred"] = np.array(classes)[proba.argmax(1)]
    result["confidence"] = proba.max(1)
    for k, c in enumerate(classes):
        result[f"p_{c}"] = proba[:, k]

    return result


def per_seed_metrics(y: np.ndarray, per_seed_proba: list[np.ndarray]) -> pd.DataFrame:
    """The headline measures of every single seed (one row per seed)."""
    return pd.DataFrame([headline_metrics(y, p) for p in per_seed_proba])


def results_table(evaluations: dict[str, dict], intervals: dict[str, pd.DataFrame], per_seed: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    One table of results for the thesis: a row per measure, and per model the seed-ensemble value with its
    95% patient-bootstrap interval (headline measures) and the mean ± standard deviation over single seeds.

    * Arguments:
    evaluations: {model: evaluate_predictions result of the seed-ensemble}
    intervals: {model: ci DataFrame from patient_bootstrap}
    per_seed: {model: DataFrame from per_seed_metrics}
    """
    rows = {}
    for measure in evaluations[next(iter(evaluations))]["summary"].index:
        row = {}
        for model, evaluation in evaluations.items():
            value = evaluation["summary"][measure]
            if measure in intervals[model].index:
                ci = intervals[model].loc[measure]
                row[model] = f"{value:.3f} [{ci['low']:.3f}, {ci['high']:.3f}]"
                row[f"{model} (seeds)"] = f"{per_seed[model][measure].mean():.3f} ± {per_seed[model][measure].std(ddof=1) if len(per_seed[model]) > 1 else 0:.3f}"
            else:
                row[model], row[f"{model} (seeds)"] = f"{value:.3f}", ""
        rows[measure] = row

    return pd.DataFrame(rows).T
